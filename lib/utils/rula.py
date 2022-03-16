import numpy as np
import torch
import os.path as osp
import json


class RULA:
    def __init__(self):
        self.joint_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 
            'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
            'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')

        self.table_a = np.array([
            [
            [[1,2],[2,2],[2,3],[3,3]],
            [[2,2],[2,2],[3,3],[3,3]],
            [[2,3],[3,3],[3,3],[4,4]]
            ],[
            [[2,3],[3,3],[3,4],[4,4]],
            [[3,3],[3,3],[3,4],[4,4]],
            [[3,4],[4,4],[4,4],[5,5]]
            ],[
            [[3,3],[4,4],[4,4],[5,5]],
            [[3,4],[4,4],[4,4],[5,5]],
            [[4,4],[4,4],[4,5],[5,5]]
            ],[
            [[4,4],[4,4],[4,5],[5,5]],
            [[4,4],[4,4],[4,5],[5,5]], 
            [[4,4],[4,5],[5,5],[6,6]]  
            ],[
            [[5,5],[5,5],[5,6],[6,7]],
            [[5,6],[6,6],[6,7],[7,7]],
            [[6,6],[6,7],[7,7],[7,8]]
            ],[
            [[7,7],[7,7],[7,8],[8,9]],
            [[8,8],[8,8],[8,9],[9,9]],
            [[9,9],[9,9],[9,9],[9,9]]
            ]
            ])

        self.table_b = np.array([
            [[1,3],[2,3],[3,4],[5,5],[6,6],[7,7]],
            [[2,3],[2,3],[4,5],[5,5],[6,7],[7,7]],
            [[3,3],[3,4],[4,5],[5,5],[6,7],[7,7]],
            [[5,5],[5,6],[6,7],[7,7],[7,7],[8,8]],
            [[7,7],[7,7],[7,8],[8,8],[8,8],[8,8]],
            [[8,8],[8,8],[8,8],[8,9],[9,9],[9,9]],
            ])

        self.table_c = np.array([
            [1,2,3,3,4,5,5],
            [2,2,3,4,4,5,5],
            [3,3,3,4,4,5,6],
            [3,3,3,4,5,6,6],
            [4,4,4,5,6,7,7],
            [5,5,6,6,7,7,7],
            [5,5,6,7,7,7,7]
        ])

        self.eval_items = ['Upper_arm (L,R)', 'Lower_arm (L,R)', 'Wrist (L,R)', 'Wrist_twist (L,R)', 'Neck', 'Trunk', 'Leg']

    def __call__(self, poses, joint_cams, add_info):
        results = []

        for ii in range(len(poses)):
            pose = poses[ii]
            joint_cam = joint_cams[ii]

            # Group A
            group_a_score_L, group_a_score_R, group_a_list = self.group_a(pose, joint_cam, add_info)
            group_a_score_L = group_a_score_L + add_info["RULA"]["A_Muscle_use_L"] + add_info["RULA"]["A_Load/Force_L"]
            group_a_score_R = group_a_score_R + add_info["RULA"]["A_Muscle_use_R"] + add_info["RULA"]["A_Load/Force_R"]
            
            # Group B
            group_b_score, group_b_list = self.group_b(pose, joint_cam, add_info)
            group_b_score = group_b_score + add_info["RULA"]["B_Muscle_use"] + add_info["RULA"]["B_Load/Force"]

            # Final Score
            group_a_score_L = int(np.clip(group_a_score_L, 1, 7))
            group_a_score_R = int(np.clip(group_a_score_R, 1, 7))
            group_b_score = int(np.clip(group_b_score, 1, 7))
            
            final_score_L = self.table_c[group_a_score_L-1][group_b_score-1]
            final_score_R = self.table_c[group_a_score_R-1][group_b_score-1]
            final_score = max(final_score_L, final_score_R)
            
            data = {
                'score': final_score,
                'log_score': group_a_list + group_b_list
            }
            results.append(data)

        return results

    def action_level(self, score):
        score = round(score)
        action_level = None
        action_name = None
        
        if score in [1,2]:
            action_level = 1
            action_name = "Acceptable posture"
        elif score in [3,4]:
            action_level = 2
            action_name = "Further investigation, change may be needed"
        elif score in [5,6]:
            action_level = 3
            action_name = "Further investigation, change soon"
        elif score >= 7:
            action_level = 4
            action_name = "Investigate and implement change"
        
        return action_level, action_name

    def group_a(self, pose, joint_cam, add_info):
        upper_arm, lower_arm, wrist, wrist_twist = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])

        upper_arm += self.upper_arm_bending(pose, joint_cam, add_info)
        upper_arm += self.shoulder_rise(pose, joint_cam)
        upper_arm += self.upper_arm_aduction(pose, joint_cam)
        lower_arm += self.lower_arm_bending(pose, joint_cam)
        lower_arm += self.lower_arm_cross_out(pose, joint_cam)
        wrist += self.wrist_bending(pose, joint_cam)
        wrist += self.wrist_side_bending(pose, joint_cam)
        wrist_twist += self.wrist_twist(pose, joint_cam)

        upper_arm = np.clip(upper_arm, 1, 6)
        lower_arm = np.clip(lower_arm, 1, 3)
        wrist = np.clip(wrist, 1, 4)
        wrist_twist = np.clip(wrist_twist, 1, 2)

        group_a_score_L = self.table_a[upper_arm[0]-1][lower_arm[0]-1][wrist[0]-1][wrist_twist[0]-1]
        group_a_score_R = self.table_a[upper_arm[1]-1][lower_arm[1]-1][wrist[1]-1][wrist_twist[1]-1]
        group_a_list = [f'{upper_arm[0]},{upper_arm[1]}', f'{lower_arm[0]},{lower_arm[1]}', f'{wrist[0]},{wrist[1]}', f'{wrist_twist[0]},{wrist_twist[1]}']

        return group_a_score_L, group_a_score_R, group_a_list


    def group_b(self, pose, joint_cam, add_info):
        neck, trunk, leg = 0,0,0

        neck += self.neck_bending(pose, joint_cam)
        neck += self.neck_side_bending_twisted(pose, joint_cam)
        trunk += self.trunk_bending(pose, joint_cam)
        trunk += self.trunk_twisted(pose, joint_cam)
        trunk += self.trunk_side_bending(pose, joint_cam)
        leg += add_info["RULA"]["Legs_bilateral_weight_bearing"]

        neck = int(np.clip(neck, 1, 6))
        trunk = int(np.clip(trunk, 1, 6))
        leg = int(np.clip(leg, 1, 2))
        return self.table_b[neck-1][trunk-1][leg-1], [neck, trunk, leg]

    def upper_arm_bending(self, pose, joint_cam, add_info):
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
        score1 -= add_info["RULA"]["Arm_supported_leaning_L"]

        angle1 = pose[self.joint_name.index('R_Shoulder')][2]
        angle2 = pose[self.joint_name.index('R_Shoulder')][1]

        if angle1>-70 and angle1<110:
            if abs(angle2)<20: score2=1
            elif angle2<-20 or (angle2>20 and angle2<=45): score2=2
            elif angle2>45 and angle2<=90: score2=3
            elif angle>90: score2=4
            else: score2=1
        else: score2=1
        score2 -= add_info["RULA"]["Arm_supported_leaning_R"]

        return np.array([score1, score2])

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

        return np.array([score1, score2])

    def upper_arm_aduction(self, pose, joint_cam):
        score1, score2 = 0, 0

        angle = pose[self.joint_name.index('L_Shoulder')][2]

        if angle<45: score1=0
        elif angle>45: score1=1
        else: score1=0

        angle = pose[self.joint_name.index('R_Shoulder')][2]

        if angle<45: score2=0
        elif angle>45: score2=1
        else: score2=0

        return np.array([score1, score2])

    def lower_arm_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Elbow')][1]
        angle2 = pose[self.joint_name.index('L_Elbow')][2]
        angle = max(angle1, angle2)

        if angle>-100 and angle<-60: score1=1
        elif angle<-100 or (angle>-60 and angle<0): score1=2
        else: score1=1

        angle1 = pose[self.joint_name.index('R_Elbow')][1]
        angle2 = pose[self.joint_name.index('R_Elbow')][2]
        angle = max(angle1, angle2)

        if angle>60 and angle<100: score2=1
        elif angle>100 or (angle>0 and angle<60): score2=2
        else: score2=1

        return np.array([score1, score2])

    def lower_arm_cross_out(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Thorax')][0]

        if angle<10 or (angle>-45 and angle<-10): score1=0
        elif angle>10 or angle<-45: score1=1
        else: score1=0

        angle = pose[self.joint_name.index('R_Thorax')][0]

        if angle>-10 or (angle>10 and angle<45): score2=0
        elif angle<-10 or angle>45: score2=1
        else: score2=0

        return np.array([score1, score2])

    def wrist_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Wrist')][2]

        if abs(angle) < 1: score1=1
        elif abs(angle)>1 and abs(angle)<15: score1=2
        elif abs(angle)>15: score1=3
        else: score1=1
        
        angle = pose[self.joint_name.index('R_Wrist')][2]

        if abs(angle) < 1: score2=1
        elif abs(angle)>1 and abs(angle)<15: score2=2
        elif abs(angle)>15: score2=3
        else: score2=1
        
        return np.array([score1, score2])

    def wrist_side_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Wrist')][1]

        if abs(angle) < 10: score1=0
        elif abs(angle)>10: score1=1
        else: score1=0
        
        angle = pose[self.joint_name.index('R_Wrist')][1]

        if abs(angle) < 10: score2=0
        elif abs(angle)>10: score2=1
        else: score2=0
        
        return np.array([score1, score2])
    
    def wrist_twist(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle = pose[self.joint_name.index('L_Wrist')][0]

        if abs(angle) < 45: score1=1
        elif abs(angle)>45: score1=2
        else: score1=1
        
        angle = pose[self.joint_name.index('R_Wrist')][0]

        if abs(angle) < 45: score2=1
        elif abs(angle)>45: score2=2
        else: score2=1
        
        return np.array([score1, score2])

    def trunk_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][0]

        if abs(angle)<5: return 1
        elif angle>5 and angle<20: return 2
        elif angle>20 and angle<60: return 3
        elif angle>60: return 4
        else: return 1

    def trunk_side_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][2]

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 1
        else: return 0
    
    def trunk_twisted(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][1]

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 1
        else: return 0

    def neck_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Neck')][0]

        if angle>-5 and angle<10: return 1
        elif angle>10 and angle<20: return 2
        elif angle>20: return 3
        elif angle<-5: return 4
        else: return 1

    def neck_side_bending_twisted(self, pose, joint_cam):
        angle1 = pose[self.joint_name.index('Neck')][2]
        angle2 = pose[self.joint_name.index('Neck')][1]

        if abs(angle1)<10 and abs(angle2)<10: return 0
        elif abs(angle1)>10 or abs(angle2)>10: return 1
        else: return 0

   