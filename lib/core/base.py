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

from funcs_utils import select_target_id, get_images
from coord_utils import axis_angle_to_euler_angle, rot_to_angle, get_joint_cam
from vis_utils import save_obj, visualize_box, vis_3d_pose, pose_to_str

from reba import REBA
from rula import RULA


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

        print("\n===> Data preprocessing...")
        file_num, fps = get_images(input_path, image_path, debug=False)
        min_frame_num = file_num * cfg.DATASET.min_frame_ratio

        if min_frame_num > 1000: min_frame_num = 1000

        # tracking    
        print("\n===> Get human tracking results...") 
        tracking_results = self.tracker(image_path) 

        filtered_results = []
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] >= min_frame_num:
                filtered_results.append(tracking_results[person_id])

        if len(filtered_results) == 0:
            for person_id in list(tracking_results.keys()):
                filtered_results.append(tracking_results[person_id])

        tracking_results = filtered_results

        idx = select_target_id(tracking_results)
        result = tracking_results[idx]
        return image_path, file_num, fps, result['bbox'], result['frames']

class Predictor:
    def __init__(self, args):
        self.data_loader = DataProcessing()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.smpl_model = SMPL()
        self.spin_model = hmr(cfg.SPIN.SMPL_MEAN_PARAMS).to(self.device)

        checkpoint = torch.load(cfg.SPIN.checkpoint)
        self.spin_model.load_state_dict(checkpoint['model'], strict=False)

        self.reba, self.rula = REBA(args.debug), RULA(args.debug)

        scores = args.type.replace(' ', '').upper().split(',')
        if 'REBA' in scores: self.run_reba = True
        else: self.run_reba = False

        if 'RULA' in scores: self.run_rula = True
        else: self.run_rula = False

        self.debugging = args.debug
        self.debug_frame = args.debug_frame
        debug_joints = args.debug_joints.replace(' ', '').split(',')

        if debug_joints == ['']:
            self.debug_joints = None
        else:
            for joint in debug_joints:
                if joint.upper() not in self.smpl_model.joints_name_upper:
                    print("\n\nInvalid Joint name!\n\n")
                    assert 0
            self.debug_joints = debug_joints

    def __call__(self, input_path, info_path, output_path):
        # data processing (tracking)
        image_folder, img_num, fps, bboxes, frames = self.data_loader(input_path, output_path)
        start_id, end_id = frames[0], frames[-1]+1
        timestamp = (0, frames, img_num)  
        debug_path = osp.join(output_path, 'debug')
        os.system(f'rm -rf {debug_path}; mkdir {debug_path}')

        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=None,
            scale=cfg.DATASET.bbox_scale,
        ) 
        crop_dataloader = DataLoader(dataset, batch_size=cfg.DATASET.batch_size, num_workers=cfg.DATASET.workers)

        # get estimation results
        result, joint_cam, images, debug_result = self.get_pose_estimation_results(crop_dataloader)

        # For debug - mesh
        if self.debugging and self.debug_frame>=0:
            print(f"\n===> Debug Result at frame #{self.debug_frame}")  
            self.visualize_joint_cam_mesh(debug_result, joint_cam, frames, debug_path)

            print("\n Debug files are saved in : ", debug_path)   
            os.system(f'rm -rf {image_folder}')
            return 

        if osp.isfile(info_path):
            with open(info_path, 'r') as f:
                add_info = json.load(f)
        else:
            with open(cfg.DATASET.default_information, 'r') as f:
                add_info = json.load(f)

        pose_str = pose_to_str(result)
        if self.debugging and self.debug_joints is not None:
            self.save_csv_pose_log(pose_str, timestamp, debug_path)


        print("\n===> Post Processing...")  
        if self.run_reba:
            reba_results = self.reba(result, joint_cam, add_info)

            final_score_reba, scores, logs = \
                self.post_processing(reba_results, self.reba.eval_items, timestamp, output_path, title="REBA")

            self.visualize_result(image_folder, bboxes, timestamp, fps, final_score_reba, scores, self.reba.eval_items, logs, add_info["REBA"], output_path, title="REBA")
            if self.debugging:
                self.save_csv(pose_str, timestamp, scores, self.reba.eval_items, logs, self.reba.log, debug_path, title="REBA")

            reba_action_level, reba_action_name = self.reba.action_level(final_score_reba[4])
            f = open(osp.join(output_path, 'reba_result.txt'), 'w')
            data = f"AVG Score: {final_score_reba[0]} \n%50 Score: {final_score_reba[1]} \n%10 Score: {final_score_reba[2]} \
                    \nMAX Score: {final_score_reba[3]} \nMODE Score: {final_score_reba[4]} \nAction level: {reba_action_level} \nAction: {reba_action_name} "
            f.write(data)
            f.close()

        if self.run_rula:
            rula_results = self.rula(result, joint_cam, add_info)

            final_score_rula, scores, logs = \
                self.post_processing(rula_results, self.rula.eval_items, timestamp, output_path, title="RULA")

            self.visualize_result(image_folder, bboxes, timestamp, fps, final_score_rula, scores, self.rula.eval_items, logs, add_info["RULA"], output_path, title="RULA")
            if self.debugging:
                self.save_csv(pose_str, timestamp, scores, self.rula.eval_items, logs, self.rula.log, debug_path, title="RULA")
            rula_action_level, rula_action_name = self.rula.action_level(final_score_rula[4])

            f = open(osp.join(output_path, 'rula_result.txt'), 'w')
            data = f"AVG Score: {final_score_rula[0]} \n%50 Score: {final_score_rula[1]} \n%10 Score: {final_score_rula[2]} \
                    \nMAX Score: {final_score_rula[3]} \nMODE Score: {final_score_rula[4]} \nAction level: {rula_action_level} \nAction: {rula_action_name}"
            f.write(data)
            f.close()

        os.system(f'rm -rf {image_folder}')

        print("\n\n===> DONE!")
        print("Result files saved in ", output_path)    

        if self.run_reba:
            print("\n----- REBA -----")
            print("AVG Score:\t", final_score_reba[0])
            print("%50 Score:\t", final_score_reba[1])
            print("%10 Score:\t", final_score_reba[2])
            print("MAX Score:\t", final_score_reba[3])
            print("MODE Score:\t", final_score_reba[4])
            print("\nAction Level:\t", reba_action_level)
            print("Action:\t\t", reba_action_name)
            print()

        if self.run_rula:
            print("\n----- RULA -----")
            print("AVG Score:\t", final_score_rula[0])
            print("%50 Score:\t", final_score_rula[1])
            print("%10 Score:\t", final_score_rula[2])
            print("MAX Score:\t", final_score_rula[3])
            print("MODE Score:\t", final_score_rula[4])
            print("\nAction Level:\t", rula_action_level)
            print("Action:\t\t", rula_action_name)
            print()

    def get_pose_estimation_results(self, crop_dataloader):
        self.spin_model.eval()
        images = []
        result = []
        debug_result=[]
        print("\n===> Estimate human pose...")  
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
        return result, joint_cam, images, debug_result

    def post_processing(self, results, joint_names, timestamp, output_path, title=''):
        scores = []
        logs = []

        for result in results:
            scores.append(result['score'])
            logs.append(result['log_score'])

        scores = np.array(scores)
        logs = np.array(logs)

        # Plot Graph
        x_axis = timestamp[1]
        plt.title(title+' Score')
        plt.xlim([timestamp[0], timestamp[2]])
        plt.xlabel('frames')
        plt.ylabel('score')
        plt.plot(x_axis, scores)
        plt.savefig(osp.join(output_path, title+'_score.png'))
        plt.clf()
            
        scores_log = np.copy(scores)
        scores.sort()
        scores = scores[::-1]
        score_avg = round(scores.mean(),3)
        score50 = round(scores[:len(scores)//2].mean(),3)
        score10 = round(scores[:len(scores)//10].mean(),3)
        score_max = round(scores.max(),3)
        score_mode = mode(scores).mode.item()
        return (score_avg, score50, score10, score_max, score_mode), scores_log, logs

    def visualize_joint_cam_mesh(self, debug_result, joint_cam, frames, output_path):
        idx = np.where(frames==self.debug_frame)[0][0]
        
        pose = torch.tensor(debug_result[idx]).view(1, -1).float()
        shape = torch.zeros((1,10)).float()
        
        smpl_mesh_coord, _ = self.smpl_model.layer['neutral'](pose, shape)
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3) * 1000
        save_obj(smpl_mesh_coord, self.smpl_model.face, osp.join(output_path, 'smpl_model.obj'))
        vis_3d_pose(joint_cam[idx], self.smpl_model.skeleton, 'smpl', osp.join(output_path, f'joint_3d.png'), frame=self.debug_frame)
            
    def visualize_result(self, image_folder, bboxes, timestamp, fps, final_score, scores, joint_names, logs, add_info, output_path, title="REBA"):
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
        for i, file_path in enumerate(image_file_names):
            canvas = np.zeros((canvas_h, canvas_w, 3))
            img = cv2.imread(file_path)
            
            cv2.putText(canvas, "frame: " + str(i), (resize_w+15, canvas_h-14), font, 0.5, color, 1, cv2.LINE_AA)

            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]
                idx = idx // 2 * 2
                bbox = bboxes[idx]
                img = visualize_box(img, bbox[None,:])
                
                cv2.putText(canvas, title+" Score: " + str(scores[idx]), (resize_w+15, 35), font, 0.7, (0,255,0), 1, cv2.LINE_AA)
                cv2.putText(canvas, "- Score per Joints ", (resize_w+15, 122), font, 0.6, color, 1, cv2.LINE_AA)
                for j, joint in enumerate(joint_names):
                    cv2.putText(canvas, joint + ": " + str(logs[idx][j]), (resize_w+15, 153 + 24*j), font, 0.5, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "Not detected target", (resize_w+15, canvas_h-65), font, 0.6, color, 1, cv2.LINE_AA)

            img = cv2.resize(img, (resize_w, resize_h), interpolation = cv2.INTER_AREA)
            canvas[:resize_h, :resize_w, :] = img

            video_writer.write(np.uint8(canvas))
        video_writer.release()

    def save_csv_pose_log(self, pose_str, timestamp, output_path):
        f = open(osp.join(output_path, 'pose_log.csv'),'w', newline='')
        wr = csv.writer(f)
        csv_title = ['Frame', 'Joint Pose']
        
        for joint_name in self.debug_joints:
            csv_title.append(joint_name)

        wr.writerow(csv_title)
        for i in range(timestamp[0], timestamp[-1]):
            row = []
            row.append(i)

            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]
                row.append('')
                for joint in self.debug_joints:
                    joint_idx = self.smpl_model.joints_name_upper.index(joint.upper())
                    row.append(str(pose_str[idx][joint_idx]))
            wr.writerow(row)
        f.close()

    def save_csv(self, pose_str, timestamp, scores, joint_names, logs, pose_logs, output_path, title="REBA"):
        f = open(osp.join(output_path, title+'_score_log.csv'),'w', newline='')
        wr = csv.writer(f)
        csv_title = ['Frame','Final_score','Joint Score']
        
        for joint_name in joint_names:
            csv_title.append(joint_name)

        wr.writerow(csv_title)
        for i in range(timestamp[0], timestamp[-1]):
            row = []
            row.append(i)

            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]
                row.append(str(scores[idx]))

                row.append('')
                for j, joint in enumerate(joint_names):
                    row.append(str(logs[idx][j]))

            wr.writerow(row)
        f.close()

        #### eval_pose_log
        f = open(osp.join(output_path, title+'_eval_pose_log.csv'),'w', newline='')
        wr = csv.writer(f)
        csv_title = ['Frame','']
        
        eval_names = pose_logs[0].keys()
        for eval_name in eval_names:
            csv_title.append(eval_name)

        wr.writerow(csv_title)
        for i in range(timestamp[0], timestamp[-1]):
            row = []
            row.append(i)

            if i in timestamp[1]:
                idx = np.where(timestamp[1]==i)[0][0]

                row.append('')
                for j, eval_name in enumerate(eval_names):
                    row.append(str(pose_logs[idx][eval_name]))

            wr.writerow(row)
        f.close()
        
    def visualize_joint_cam(self, joint_cam, debug_frame, output_path):
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