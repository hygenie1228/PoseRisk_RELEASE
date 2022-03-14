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
from scipy.stats import mode
import csv
from glob import glob

from multiple_datasets import MultipleDatasets
from demo_dataset import CropDataset
from core.config import cfg

from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images
from models import hmr
from smpl import SMPL

from vis_utils import save_obj, visualize_box, report_pose, vis_3d_pose
from reba import REBA
from rula import RULA
from coord_utils import axis_angle_to_euler_angle, rot_to_angle, get_joint_cam

class DataProcessing:
    def __init__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.tracker = MPT(
                device=device,
                batch_size=cfg.DATASET.batch_size,
                display=False,
                detection_threshold=0.1,
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

        if min_frame_num > 1000:
            min_frame_num = 1000

        # tracking    
        print() 
        print("===> Get human tracking results...") 
        tracking_results = self.tracker(image_path) 

        filtered_results = []
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] >= min_frame_num:
                filtered_results.append(tracking_results[person_id])

        if len(filtered_results) == 0:
            for person_id in list(tracking_results.keys()):
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

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if width > 800:
            height = int(height * 800 / width)
            width = 800
        elif height > 450:
            width = int(width * 450 / height)
            height = 450

        idx = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if ret == False:
                break

            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(osp.join(tmp_path, '{0:09d}.jpg'.format(idx)),frame)
            idx += 1

            #if idx == 1500:
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
        
        self.debugging = args.debug
        self.smpl_joint_names = ('L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
        'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist', 'L_Hand', 'R_Hand')

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
                    pose = rot_to_angle(rotmat)
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
        joint_cam = get_joint_cam(debug_result, self.smpl_model)

        # For debug - joint_cam
        if self.debugging:
            print()
            print("===> Visualize Estimated Results...")  
            self.visualize_joint_cam(joint_cam, timestamp, output_path)

        # For debug - mesh
        if self.debugging and debug_frame>=0:
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

        pose_str = self.pose_to_str(result)

        print()
        print("===> Post Processing...")  
        if self.run_reba:
            reba_results, reba_joint_names = self.reba(result, joint_cam, add_info)

            final_score_reba, scores, group_a, group_b, logs = \
                self.post_processing_result(reba_results, reba_joint_names, timestamp, output_path, title="REBA")

            self.visualize_result(image_folder, bboxes, timestamp, fps, final_score_reba, scores, group_a, group_b, reba_joint_names, logs, add_info["REBA"], output_path, title="REBA")
            self.save_csv(pose_str, image_folder, bboxes, timestamp, fps, final_score_reba, scores, group_a, group_b, reba_joint_names, logs, add_info["REBA"], output_path, title="REBA")

            reba_action_level, reba_action_name = self.reba.action_level(final_score_reba[4])
            f = open(osp.join(output_path, 'reba_result.txt'), 'w')
            data = f"AVG Score: {final_score_reba[0]} \n%50 Score: {final_score_reba[1]} \n%10 Score: {final_score_reba[2]} \
                    \nMAX Score: {final_score_reba[3]} \nMODE Score: {final_score_reba[4]} \nAction level: {reba_action_level} \nAction: {reba_action_name} "
            f.write(data)
            f.close()

        if self.run_rula:
            rula_results, rula_joint_names = self.rula(result, joint_cam, add_info)

            final_score_rula, scores, group_a, group_b, logs = \
                self.post_processing_result(rula_results, rula_joint_names, timestamp, output_path, title="RULA")

            self.visualize_result(image_folder, bboxes, timestamp, fps, final_score_rula, scores, group_a, group_b, rula_joint_names, logs, add_info["RULA"], output_path, title="RULA")
            self.save_csv(pose_str, image_folder, bboxes, timestamp, fps, final_score_rula, scores, group_a, group_b, rula_joint_names, logs, add_info["RULA"], output_path, title="RULA")
            rula_action_level, rula_action_name = self.rula.action_level(final_score_rula[4])

            f = open(osp.join(output_path, 'rula_result.txt'), 'w')
            data = f"AVG Score: {final_score_rula[0]} \n%50 Score: {final_score_rula[1]} \n%10 Score: {final_score_rula[2]} \
                    \nMAX Score: {final_score_rula[3]} \nMODE Score: {final_score_rula[4]} \nAction level: {rula_action_level} \nAction: {rula_action_name}"
            f.write(data)
            f.close()

        os.system(f'rm -rf {image_folder}')

        print()
        print()
        print("===> DONE!")
        print("Result files saved in ", output_path)    

        if self.run_reba:
            print()
            print("----- REBA -----")
            print("AVG Score:\t", final_score_reba[0])
            print("%50 Score:\t", final_score_reba[1])
            print("%10 Score:\t", final_score_reba[2])
            print("MAX Score:\t", final_score_reba[3])
            print("MODE Score:\t", final_score_reba[4])
            print("\nAction Level:\t", reba_action_level)
            print("Action:\t\t", reba_action_name)
            print()

        if self.run_rula:
            print()
            print("----- RULA -----")
            print("AVG Score:\t", final_score_rula[0])
            print("%50 Score:\t", final_score_rula[1])
            print("%10 Score:\t", final_score_rula[2])
            print("MAX Score:\t", final_score_rula[3])
            print("MODE Score:\t", final_score_rula[4])
            print("\nAction Level:\t", rula_action_level)
            print("Action:\t\t", rula_action_name)
            print()

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
        score_mode = mode(scores).mode.item()
        return (score_avg, score50, score10, score_max, score_mode), scores_log, group_a, group_b, logs

    def visualize_joint_cam(self, joint_cam, timestamp, output_path):
        image_folder = osp.join(output_path, 'debug')
        os.system(f'rm -rf {image_folder}')
        os.makedirs(image_folder, exist_ok=True)
        
        img_paths = []
        for j, i in tqdm(enumerate(timestamp[1])):
            joint_cam_i = joint_cam[j//2*2]
            vis_3d_pose(joint_cam_i, self.smpl_model.skeleton, 'smpl', osp.join(output_path, 'debug', f'joint_cam_{i}.png'), frame=i)
            img_paths.append(osp.join(output_path, 'debug', f'joint_cam_{i}.png'))

        img = cv2.imread(img_paths[0])
        height,width,_ = img.shape
        video_writer = cv2.VideoWriter(osp.join(output_path, 'estimation_result.mp4'), 0x7634706d, 20, (width, height))
        
        for i in range(len(img_paths)):
            canvas = cv2.imread(img_paths[i])
            canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
            canvas = canvas.copy()
            video_writer.write(np.uint8(canvas))
        video_writer.release()
        os.system(f'rm -rf {image_folder}')

            
    def visualize_result(self, image_folder, bboxes, timestamp, fps, final_score, scores, group_a, group_b, joint_names, logs, add_info, output_path, title="REBA"):
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
        font_size2 = 0.4
        for i, file_path in enumerate(image_file_names):
            canvas = np.zeros((canvas_h, canvas_w, 3))
            img = cv2.imread(file_path)
            
            cv2.putText(canvas, "frame: " + str(i), (resize_w+15, canvas_h-12), font, 0.4, color, 1, cv2.LINE_AA)

            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]
                idx = idx // 4 * 4
                bbox = bboxes[idx]
                img = visualize_box(img, bbox[None,:])
                
                cv2.putText(canvas, title+" Score: " + str(scores[idx]), (resize_w+15, 30), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "Group A Score: " + str(group_a[idx]), (resize_w+15, 55), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "Group B Score: " + str(group_b[idx]), (resize_w+15, 80), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "- Score per Joints ", (resize_w+15, 135), font, font_size, color, 1, cv2.LINE_AA)
                for j, joint in enumerate(joint_names):
                    cv2.putText(canvas, joint + ": " + str(logs[idx][j]), (resize_w+15, 155 + 20*j), font, font_size, color, 1, cv2.LINE_AA)
                for j, (k, v) in enumerate(add_info.items()):
                    if "Legs_bilateral_weight_" in k:
                        k = "Legs_bearing/walking/sitting"
                    cv2.putText(canvas, k + ": " + str(v), (resize_w+15, 285 + 18*j), font, font_size2, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "Unable to provide score", (resize_w+15, canvas_h-65), font, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, "score due to human detection error", (resize_w+15, canvas_h-45), font, 0.5, color, 1, cv2.LINE_AA)

            img = cv2.resize(img, (resize_w, resize_h), interpolation = cv2.INTER_AREA)
            canvas[:resize_h, :resize_w, :] = img

            video_writer.write(np.uint8(canvas))
        video_writer.release()

    def save_csv(self, pose_str, image_folder, bboxes, timestamp, fps, final_score, scores, group_a, group_b, joint_names, logs, add_info, output_path, title="REBA"):
        image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]

        image_file_names = sorted(image_file_names)
        
        f = open(osp.join(output_path, title+'_log.csv'),'w', newline='')
        wr = csv.writer(f)
        title = ['frame','final_score', 'group_a_score', 'group_b_score', 'Joint Score']
        
        for joint_name in joint_names:
            title.append(joint_name)

        title.append('Joint Angle(BEND1,BEND2,TWIST)')

        for joint_name in self.smpl_joint_names:
            title.append(joint_name)

        wr.writerow(title)

        for i, file_path in enumerate(image_file_names):
            row = []
            row.append(i)

            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]
                row.append(str(scores[idx]))
                row.append(str(group_a[idx]))
                row.append(str(group_b[idx]))
                
                row.append('')
                for j, joint in enumerate(joint_names):
                    row.append(logs[idx][j])

                row.append('')
                for j, joint in enumerate(self.smpl_joint_names):
                    row.append(str(pose_str[idx][j]))

            wr.writerow(row)

        f.close()


    def pose_to_str(self, poses):
        pose_log = []
        for i, pose in enumerate(poses[:,1:,:]):
            str_list = []
            for j, pose_i in enumerate(pose):
                joint_name = self.smpl_joint_names[j]
                if joint_name not in ('L_Thorax', 'R_Thorax', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'):
                    str_list.append(f"({pose_i[0]:.3f}, {pose_i[2]:.3f}, {pose_i[1]:.3f})")
                else:
                    str_list.append(f"({pose_i[1]:.3f}, {pose_i[2]:.3f}, {pose_i[0]:.3f})")
            pose_log.append(str_list)
        return pose_log