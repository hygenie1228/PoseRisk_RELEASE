import numpy as np
import torch
import os.path as osp
import json
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

class SMPL(object):
    def __init__(self):
        self.model_path = osp.join('data', 'base_data', 'human_models')
        self.layer = {'male': self.get_layer('male'), 'female': self.get_layer('female'), 'neutral': self.get_layer('neutral')}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].th_faces.numpy()
        self.joint_regressor = self.layer['neutral'].th_J_regressor.numpy().astype(np.float32)  # smpl joint regressor

        # add nose, L/R eye, L/R ear
        self.face_kps_vertex = (331, 2802, 6262, 3489, 3990)  # mesh vertex idx
        nose_onehot = np.array([1 if i == 331 else 0 for i in range(self.joint_regressor.shape[1])],
                               dtype=np.float32).reshape(1, -1)
        left_eye_onehot = np.array([1 if i == 2802 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)
        right_eye_onehot = np.array([1 if i == 6262 else 0 for i in range(self.joint_regressor.shape[1])],
                                    dtype=np.float32).reshape(1, -1)
        left_ear_onehot = np.array([1 if i == 3489 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)
        right_ear_onehot = np.array([1 if i == 3990 else 0 for i in range(self.joint_regressor.shape[1])],
                                    dtype=np.float32).reshape(1, -1)
        self.joint_regressor = np.concatenate(
            (self.joint_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))

        self.joint_num = 24 
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 
        'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.joints_name_upper = [i.upper() for i in self.joints_name]
        self.part_segments_color = ('silver', 'blue', 'green', 'salmon', 'turquoise', 'olive', 'lavender', 'darkblue', 'lime', 'khaki', 'cyan', 'darkgreen',
                                    'beige', 'coral', 'crimson', 'red', 'aqua', 'chartreuse', 'indigo', 'teal', 'violet', 'orchid', 'orange', 'gold')
        self.flip_pairs = (
        (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
        self.skeleton = (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
        (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.root_joint_idx = self.joints_name.index('Pelvis')

    def get_layer(self, gender):
        return SMPL_Layer(gender=gender, model_root=self.model_path)
