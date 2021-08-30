import os
import os.path as osp
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from multiple_datasets import MultipleDatasets
from demo_dataset import CropDataset
from core.config import cfg

from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images
from models import hmr
from smpl import SMPL

from vis_utils import save_obj, visualize_box
from score_utils import REBA, RULA
from coord_utils import axis_angle_to_euler_angle

class DataProcessing:
    def __init__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.tracker = MPT(
                device=device,
                batch_size=cfg.DATASET.batch_size,
                display=False,
                detector_type='yolo',
                output_format='dict',
                yolo_img_size=416,
            )


    def __call__(self, input_path, output_path):
        image_path = osp.join(output_path, 'tmp')
        os.system(f'rm -rf {image_path}')

        print()
        print("===> Data preprocessing...")
        file_num, fps = self.get_images(input_path, image_path)
        min_frame_num = file_num * cfg.DATASET.min_frame_ratio

        # tracking    
        print() 
        print("===> Get human tracking results...")   
        tracking_results = self.tracker(image_path) 

        filtered_results = []
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] >= min_frame_num:
                filtered_results.append(tracking_results[person_id])
        tracking_results = filtered_results
    
        idx = self.select_target_id(tracking_results)
        #print("!!!")
        #idx = 2
        result = tracking_results[idx]
        return image_path, file_num, fps, result['bbox'], result['frames']


    def select_target_id(self, results):
        areas = []

        for result in results:
            bbox = result['bbox']
            area = (bbox[:,2] * bbox[:,3]).mean()
            areas.append(area)
        
        areas = np.array(areas)
        return np.argmax(areas)

    def get_images(self, file_name, tmp_path):
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        os.makedirs(tmp_path, exist_ok=True)

        idx = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if ret == False:
                break

            cv2.imwrite(osp.join(tmp_path, '{0:09d}.jpg'.format(idx)),frame)
            idx += 1

            #if idx == 100:
            #    print("!!!")
            #    break

        cap.release()
        cv2.destroyAllWindows()
        del cap
        return idx, fps


class Predictor:
    def __init__(self, args):
        self.data_loader = DataProcessing()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.smpl_model = SMPL()
        self.spin_model = hmr(cfg.SPIN.SMPL_MEAN_PARAMS).to(self.device)
        checkpoint = torch.load(cfg.SPIN.checkpoint)
        self.spin_model.load_state_dict(checkpoint['model'], strict=False)

        self.reba, self.rula = REBA(), RULA()

        score_type = args.type
        scores = score_type.replace(' ', '').upper().split(',')
        if 'REBA' in scores:
            self.run_reba = True
        else:
            self.run_reba = False

        if 'RULA' in scores:
            self.run_rula = True
        else:
            self.run_rula = False

    def __call__(self, input_path, info_path, output_path, debug_frame):
        # data processing (tracking)
        image_folder, img_num, fps, bboxes, frames = self.data_loader(input_path, output_path)
        start_id, end_id = frames[0], frames[-1]+1
        timestamp = (0, frames, img_num)  

        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=None,
            scale=cfg.DATASET.bbox_scale,
        ) 

        crop_dataloader = DataLoader(dataset, batch_size=cfg.DATASET.batch_size, num_workers=cfg.DATASET.workers)
        
        self.spin_model.eval()
        images = []
        result = []
        debug_result=[]
        print()
        print("===> Estimate human pose...")  
        with torch.no_grad():
            for i, batch in tqdm(enumerate(crop_dataloader)):
                batch = batch.to(self.device)
                pred_rotmat, pred_betas, pred_camera = self.spin_model(batch)
                
                pred_rotmat = pred_rotmat.cpu().numpy()
                res = []
                de_res = []
                for rotmat in pred_rotmat:
                    pose = []
                    for p in rotmat:
                        pp = cv2.Rodrigues(p)[0].reshape(-1)
                        pose.append(pp)
                    pose = np.stack(pose)
                    de_res.append(pose)
                    pose = axis_angle_to_euler_angle(pose)
                    res.append(pose)
                res = np.stack(res)
                result.append(res)
                debug_result.append(de_res)
                images.append(batch.cpu().numpy())

        result = np.concatenate(result)
        images = np.concatenate(images)
        debug_result = np.concatenate(debug_result)

        if debug_frame > 0:
            idx = np.where(frames==debug_frame)[0][0]

            pose = torch.tensor(debug_result[idx]).view(1, -1).float()
            shape = torch.zeros((1,10)).float()
            
            smpl_mesh_coord, _ = self.smpl_model.layer['neutral'](pose, shape)
            smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3) * 1000
            save_obj(smpl_mesh_coord, self.smpl_model.face, osp.join(output_path, 'smpl_model.obj'))
            print()
            print("Debug file is saved in ", output_path)   
            os.system(f'rm -rf {image_folder}')
            return 

        if osp.isfile(info_path):
            with open(info_path, 'r') as f:
                add_info = json.load(f)
        else:
            with open(cfg.DATASET.default_information, 'r') as f:
                add_info = json.load(f)
    
        if self.run_reba:
            reba_results, reba_joint_names = self.reba(result, add_info)
        
            print()
            print("===> Post Processing...")  

            final_score, scores, group_a, group_b, logs = \
                self.post_processing_result(reba_results, reba_joint_names, timestamp, output_path, title="REBA")

            self.visualize_result(image_folder, bboxes, timestamp, fps, final_score, scores, group_a, group_b, reba_joint_names, logs, output_path, title="REBA")

            os.system(f'rm -rf {image_folder}')

            f = open(osp.join(output_path, 'reba_result.txt'), 'w')
            data = f"AVG Score: {final_score[0]} \n%50 Score: {final_score[1]} \n%10 Score: {final_score[2]} \nMAX Score: {final_score[3]}"
            f.write(data)
            f.close()

        print()
        print()
        print("===> DONE!")
        print("Result files saved in ", output_path)    

        if self.run_reba:
            print()
            print("----- REBA -----")
            print("AVG Score: ", final_score[0])
            print("%50 Score: ", final_score[1])
            print("%10 Score: ", final_score[2])
            print("MAX Score: ", final_score[3])


    def post_processing_result(self, results, joint_names, timestamp, output_path, title=''):
        scores = []
        group_a = []
        group_b = []
        logs = []

        for result in results:
            scores.append(result['score'])
            group_a.append(result['group_a'])
            group_b.append(result['group_b'])
            logs.append(result['log_score'])

        scores = np.array(scores)
        group_a = np.array(group_a)
        group_b = np.array(group_b)
        logs = np.array(logs)

        if True:
            x_axis = timestamp[1]

            plt.title(title+' Score')
            plt.xlim([timestamp[0], timestamp[2]])
            plt.xlabel('frames')
            plt.ylabel('score')
            plt.plot(x_axis, scores)
            plt.savefig(osp.join(output_path, title+'_score.png'))
            plt.clf()

            plt.title(title+' Group A Score')
            plt.xlim([timestamp[0], timestamp[2]])
            plt.xlabel('frames')
            plt.ylabel('score')
            plt.plot(x_axis, group_a)
            plt.savefig(osp.join(output_path, title+'_group_a.png'))
            plt.clf()

            plt.title(title+' Group B Score')
            plt.xlim([timestamp[0], timestamp[2]])
            plt.xlabel('frames')
            plt.ylabel('score')
            plt.plot(x_axis, group_b)
            plt.savefig(osp.join(output_path, title+'_group_b.png'))
            plt.clf()
            
            plt.title(title+' Group A Score Log')
            plt.xlim([timestamp[0], timestamp[2]])
            plt.xlabel('frames')
            plt.ylabel('score')
            for i, joint in enumerate(joint_names[:3]):
                plt.plot(x_axis, logs[:,i], label=joint)
            plt.legend() 
            plt.savefig(osp.join(output_path, title+'_group_a_log.png'))
            plt.clf()

            plt.title(title+' Group B Score Log')
            plt.xlim([timestamp[0], timestamp[2]])
            plt.xlabel('frames')
            plt.ylabel('score')
            for i, joint in enumerate(joint_names[3:]):
                plt.plot(x_axis, logs[:,3+i], label=joint)
            plt.legend() 
            plt.savefig(osp.join(output_path, title+'_group_b_log.png'))
            plt.clf()

        scores_log = np.copy(scores)
        scores.sort()
        scores = scores[::-1]
        score_avg = round(scores.mean(),3)
        score50 = round(scores[:len(scores)//2].mean(),3)
        score10 = round(scores[:len(scores)//10].mean(),3)
        score_max = round(scores.max(),3)
        return (score_avg, score50, score10, score_max), scores_log, group_a, group_b, logs

    
    def visualize_result(self, image_folder, bboxes, timestamp, fps, final_score, scores, group_a, group_b, joint_names, logs, output_path, title="REBA"):
        image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]

        image_file_names = sorted(image_file_names)
        img = cv2.imread(image_file_names[0])

        height,width,c = img.shape
        
        resize_w = 720
        resize_h = int(height * resize_w / width)
        canvas_w = resize_w + 280
        canvas_h =resize_h

        video_writer = cv2.VideoWriter(osp.join(output_path, title+'_video.mp4'), 0x7634706d, fps, (canvas_w, canvas_h))
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,255,255)
        font_size = 0.5
        for i, file_path in enumerate(image_file_names):
            canvas = np.zeros((canvas_h, canvas_w, 3))
            img = cv2.imread(file_path)
            
            
            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]
                idx = idx // 4 * 4
                bbox = bboxes[idx]
                img = visualize_box(img, bbox[None,:])

                cv2.putText(canvas, "frame: " + str(i), (resize_w+15, canvas_h-15), font, 0.4, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, title+" Score: " + str(scores[idx]), (resize_w+15, 30), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "Group A Score: " + str(group_a[idx]), (resize_w+15, 55), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "Group B Score: " + str(group_b[idx]), (resize_w+15, 80), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "- Score per Joints ", (resize_w+15, 150), font, font_size, color, 1, cv2.LINE_AA)
                for j, joint in enumerate(joint_names):
                    cv2.putText(canvas, joint + ": " + str(logs[idx][j]), (resize_w+15, 175 + 20*j), font, font_size, color, 1, cv2.LINE_AA)

            img = cv2.resize(img, (resize_w, resize_h), interpolation = cv2.INTER_AREA)
            canvas[:resize_h, :resize_w, :] = img

            video_writer.write(np.uint8(canvas))
        video_writer.release()

        


            
            
                
  