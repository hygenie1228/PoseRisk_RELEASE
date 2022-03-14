import numpy as np
import torch
import os.path as osp
import json


class REBA:
    def __init__(self):
        self.joint_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 
            'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
            'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')

        self.table_a = np.array(
            [[[1,2,3,4],[1,2,3,4],[3,3,5,6]],
             [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
             [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
             [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
             [[4,6,7,8],[6,7,8,9],[7,8,9,9]]
            ])

        self.table_b = np.array(
            [[[1,2,2],[1,2,3]],
             [[1,2,3],[2,3,4]],
             [[3,4,5],[4,5,5]],
             [[4,5,5],[5,6,7]],
             [[6,7,8],[7,8,8]],
             [[7,8,8],[8,9,9]]
            ])

        self.table_c = np.array([
            [1,1,1,2,3,3,4,5,6,7,7,7],
            [1,2,2,3,4,4,5,6,6,7,7,8],
            [2,3,3,3,4,5,6,7,7,8,8,8],
            [3,4,4,4,5,6,7,8,8,9,9,9],
            [4,4,4,5,6,7,8,8,9,9,9,9],
            [6,6,6,7,8,8,9,9,10,10,10,10],
            [7,7,7,8,9,9,9,10,10,11,11,11],
            [8,8,8,9,10,10,10,10,10,11,11,11],
            [9,9,9,10,10,10,11,11,11,12,12,12],
            [10,10,10,11,11,11,11,12,12,12,12,12],
            [11,11,11,11,12,12,12,12,12,12,12,12],
            [12,12,12,12,12,12,12,12,12,12,12,12]
        ])

        self.sagittal = np.array([0,0,0])
        self.coronal = np.array([0,0,0])

    def __call__(self, poses, joint_cams, add_info):
        results = []

        for ii in range(len(poses)):
            pose = poses[ii]
            joint_cam = joint_cams[ii]

            # Group A
            group_a_score, group_a_list = self.group_a(pose, joint_cam, add_info)
            group_a_score = group_a_score + add_info["REBA"]["Load/Force Score"]
            
            # Group B
            group_b_score, group_b_list = self.group_b(pose, joint_cam, add_info)
            group_b_score = group_b_score + add_info["REBA"]["Coupling"]

            # Final Score
            group_a_score = int(np.clip(group_a_score, 1, 12))
            group_b_score = int(np.clip(group_b_score, 1, 12))
            final_score = self.table_c[group_a_score-1][group_b_score-1] + add_info["REBA"]["Activity_Score"]

            data = {
                'score': final_score,
                'group_a': group_a_score,
                'group_b': group_b_score,
                'log_score': group_a_list+group_b_list
            }
            results.append(data)

        return results, ['trunk', 'neck', 'leg', 'upper_arm', 'lower_arm', 'wrist']
            
    def action_level(self, score):
        score = round(score)
        action_level = None
        action_name = None

        if score in [1]:
            action_level = 1
            action_name = "Negligible risk"
        elif score in [2,3]:
            action_level = 2
            action_name = "Low risk. Change may be needed."
        elif score in [4,5,6,7]:
            action_level = 3
            action_name = "Medium risk. Further Investigate. Change Soon."
        elif score in [8,9,10]:
            action_level = 4
            action_name = "High risk. Investigate and implement change"
        elif score >= 11:
            action_level = 5
            action_name = "Very high risk. Implement change"
        
        return action_level, action_name

    def group_a(self, pose, joint_cam, add_info):
        trunk, neck, leg = 0,0,0
        trunk += self.trunk_bending(pose, joint_cam)
        trunk += self.trunk_twisted(pose, joint_cam)
        trunk += self.trunk_side_bending(pose, joint_cam)
        neck += self.neck_bending(pose, joint_cam)
        neck += self.neck_side_bending_twisted(pose, joint_cam)
        leg += add_info["REBA"]["Legs_bilateral_weight_bearing/walking"]
        leg += self.leg_bending(pose, joint_cam, add_info)

        trunk = int(np.clip(trunk, 1, 5))
        neck = int(np.clip(neck, 1, 3))
        leg = int(np.clip(leg, 1, 4))
        return self.table_a[trunk-1][neck-1][leg-1], [trunk, neck, leg]
      

    def group_b(self, pose, joint_cam, add_info):
        upper_arm, lower_arm, wrist = 0,0,0
        upper_arm_bending = self.upper_arm_bending(pose, joint_cam)
        upper_arm += upper_arm_bending
        upper_arm += self.shoulder_rise(pose, joint_cam)
        upper_arm += self.upper_arm_aduction(pose, joint_cam)
        upper_arm -= add_info["REBA"]["Arm_supported_leaning"]
        if upper_arm_bending>1: lower_arm += self.lower_arm_bending(pose, joint_cam)
        wrist += self.wrist_bending(pose, joint_cam)
        wrist += self.wrist_side_bending_twisted(pose, joint_cam)

        upper_arm = int(np.clip(upper_arm, 1, 6))
        lower_arm = int(np.clip(lower_arm, 1, 2))
        wrist = int(np.clip(wrist, 1, 3))
        return self.table_b[upper_arm-1][lower_arm-1][wrist-1], [upper_arm, lower_arm, wrist]

    def trunk_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][0]

        if abs(angle)<5: return 1
        elif (angle>5 and angle<20) or (angle>-20 and angle<-5): return 2
        elif (angle>20 and angle<60) or (angle<-20): return 3
        elif angle>60: return 4
        else: return 1

    def trunk_side_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][2]

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 0
        else: return 0

    def trunk_twisted(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][1]

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 1
        else: return 0

    def neck_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Neck')][0]

        if angle>-5 and angle<20: return 1
        elif angle<20 or angle<-5: return 2
        else: return 1

    def neck_side_bending_twisted(self, pose, joint_cam):
        angle1 = pose[self.joint_name.index('Neck')][2]
        angle2 = pose[self.joint_name.index('Neck')][1]

        if abs(angle1)<10 and abs(angle2)<10: return 0
        elif abs(angle1)>10 or abs(angle2)>10: return 1
        else: return 0

    def leg_bending(self, pose, joint_cam, add_info):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Knee')][0]

        if angle<30: score1=0
        elif angle>30 and angle<60: score1=1
        elif angle>60 and add_info["REBA"]["sitting"] > 0 : score1=2
        else: score1=0

        angle = pose[self.joint_name.index('R_Knee')][0]

        if angle<30: score2=0
        elif angle>30 and angle<60: score2=1
        elif angle>60 and add_info["REBA"]["sitting"] > 0 : score2=2
        else: score2=0

        return max(score1, score2)

    def upper_arm_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Shoulder')][2]
        angle2 = pose[self.joint_name.index('L_Shoulder')][1]

        if angle1>-70 and angle1<110:
            if abs(angle2)<20: score1=1
            elif angle2>20 or (angle2>-45 and angle2<-20): score1=2
            elif angle2>-90 and angle2<=-45: score1=3
            elif angle2<-90: score1=4
            else: score1=1
        else: score1=1

        angle1 = pose[self.joint_name.index('R_Shoulder')][2]
        angle2 = pose[self.joint_name.index('R_Shoulder')][1]

        if angle1>-70 and angle1<110:
            if abs(angle2)<20: score2=1
            elif angle2<-20 or (angle2>20 and angle2<=45): score2=2
            elif angle2>45 and angle2<=90: score2=3
            elif angle2>90: score2=4
            else: score2=1
        else: score2=1

        return max(score1, score2)

    def shoulder_rise(self, pose, joint_cam):
        score1, score2 = 0, 0

        lshoulder = joint_cam[self.joint_name.index('L_Shoulder')]
        chest = joint_cam[self.joint_name.index('Chest')]
        neck = joint_cam[self.joint_name.index('Neck')]

        vec1= lshoulder - chest
        vec2 = neck - chest
        angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi

        if angle<=90: score1=0
        elif angle>90: score1=1
        else: score1=0

        rshoulder = joint_cam[self.joint_name.index('R_Shoulder')]
        chest = joint_cam[self.joint_name.index('Chest')]
        neck = joint_cam[self.joint_name.index('Neck')]

        vec1= rshoulder - chest
        vec2 = neck - chest
        angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi

        if angle<=90: score2=0
        elif angle>90: score2=1
        else: score2=0

        return max(score1, score2)

    def upper_arm_aduction(self, pose, joint_cam):
        score1, score2 = 0, 0

        angle1 = pose[self.joint_name.index('L_Shoulder')][2]
        angle2 = pose[self.joint_name.index('L_Shoulder')][0]

        if angle1<45 and angle2<10: score1=0
        elif angle1>45 or angle2>10: score1=1
        else: score1=0

        angle1 = pose[self.joint_name.index('R_Shoulder')][2]
        angle2 = pose[self.joint_name.index('R_Shoulder')][0]

        if angle1>-45 and angle2<10: score2=0
        elif angle1<-45 or angle2>10: score2=1
        else: score2=0

        return max(score1, score2)

    def lower_arm_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Elbow')][1]

        if angle>-100 and angle<-60: score1=1
        elif angle<-100 or (angle>-60 and angle<0): score1=2
        else: score1=1

        angle = pose[self.joint_name.index('R_Elbow')][1]

        if angle>60 and angle<100: score2=1
        elif angle>100 or (angle>0 and angle<60): score2=2
        else: score2=1

        return max(score1, score2)

    def wrist_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Wrist')][2]

        if abs(angle)<15: score1=1
        elif abs(angle)>15: score1=2
        else: score1=1
        
        angle = pose[self.joint_name.index('R_Wrist')][2]

        if abs(angle)<15: score2=1
        elif abs(angle)>15: score2=2
        else: score2=1
        
        return max(score1, score2)

    def wrist_side_bending_twisted(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Wrist')][1]
        angle2 = pose[self.joint_name.index('L_Wrist')][0]

        if abs(angle1)<10 and abs(angle2)<10: score1=0
        elif abs(angle1)>10 or abs(angle2)>10: score1=1
        else: score1=0

        angle1 = pose[self.joint_name.index('R_Wrist')][1]
        angle2 = pose[self.joint_name.index('R_Wrist')][0]

        if abs(angle1)<10 and abs(angle2)<10: score2=0
        elif abs(angle1)>10 or abs(angle2)>10: score2=1
        else: score2=0
        
        return max(score1, score2)
