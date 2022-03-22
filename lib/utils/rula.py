import numpy as np
import torch
import os.path as osp
import json


class RULA:
    def __init__(self, debug=False):
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
        
        self.debugging = debug
        self.angle_log = {}
        self.log = []

    def __call__(self, poses, joint_cams, add_info):
        results = []

        for ii in range(len(poses)):
            pose = poses[ii]
            joint_cam = joint_cams[ii]

            # Group A
            group_a_score_L, group_a_score_R, group_a_list = self.group_a(pose, joint_cam, add_info)
            group_a_score_L = group_a_score_L + add_info["RULA"]["A_Muscle_use_L"] + add_info["RULA"]["A_Load/Force_L"]
            group_a_score_R = group_a_score_R + add_info["RULA"]["A_Muscle_use_R"] + add_info["RULA"]["A_Load/Force_R"]
            group_a_score = max(group_a_score_L, group_a_score_R)

            # Group B
            group_b_score, group_b_list = self.group_b(pose, joint_cam, add_info)
            group_b_score = group_b_score + add_info["RULA"]["B_Muscle_use"] + add_info["RULA"]["B_Load/Force"]

            # Final Score
            group_a_score = int(np.clip(group_a_score, 1, 7))
            group_b_score = int(np.clip(group_b_score, 1, 7))
            final_score = self.table_c[group_a_score-1][group_b_score-1]
            
            data = {
                'score': final_score,
                'log_score': group_a_list + group_b_list
            }
            results.append(data)

            if self.debugging:
                self.log.append(self.angle_log)
                self.angle_log = {}

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
        upper_arm += self.upper_arm_abducted(pose, joint_cam)
        lower_arm += self.lower_arm_bending(pose, joint_cam)
        lower_arm += self.bent_from_midline_or_out_to_side(pose, joint_cam)
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

        angle3 = pose[self.joint_name.index('R_Shoulder')][2]
        angle4 = pose[self.joint_name.index('R_Shoulder')][1]

        if angle3>-70 and angle3<110:
            if abs(angle4)<20: angle4=1
            elif angle4<-20 or (angle4>20 and angle4<=45): score2=2
            elif angle4>45 and angle4<=90: score2=3
            elif angle4>90: score2=4
            else: score2=1
        else: score2=1
        score2 -= add_info["RULA"]["Arm_supported_leaning_R"]

        self.angle_log['upper_arm_bending'] = f'L {angle1:.1f},{angle2:.1f} R {angle3:.1f},{angle4:.1f}'
        return np.array([score1, score2])

    def shoulder_rise(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Thorax')][2]
        
        if abs(angle1)<10: score1=0
        elif abs(angle1)>=10: score1=1
        else: score1=0

        angle2 = pose[self.joint_name.index('R_Thorax')][2]

        if abs(angle2)<10: score2=0
        elif abs(angle2)>=10: score2=1
        else: score2=0

        self.angle_log['shoulder_rise'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])
        

    '''def shoulder_rise(self, pose, joint_cam):
        score1, score2 = 0, 0

        lshoulder = joint_cam[self.joint_name.index('L_Shoulder')]
        chest = joint_cam[self.joint_name.index('Chest')]
        neck = joint_cam[self.joint_name.index('Neck')]

        vec1= lshoulder - chest
        vec2 = neck - chest
        angle1 = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi

        if angle1<=90: score1=0
        elif angle1>90: score1=1
        else: angle1=0

        rshoulder = joint_cam[self.joint_name.index('R_Shoulder')]
        chest = joint_cam[self.joint_name.index('Chest')]
        neck = joint_cam[self.joint_name.index('Neck')]

        vec1= rshoulder - chest
        vec2 = neck - chest
        angle2 = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi

        if angle2<=90: score2=0
        elif angle2>90: score2=1
        else: score2=0

        self.angle_log['shoulder_rise'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])'''

    def upper_arm_abducted(self, pose, joint_cam):
        score1, score2 = 0, 0

        angle1 = pose[self.joint_name.index('L_Shoulder')][2]

        if angle1<45: score1=0
        elif angle1>45: score1=1
        else: score1=0

        angle2 = pose[self.joint_name.index('R_Shoulder')][2]

        if angle2<45: score2=0
        elif angle2>45: score2=1
        else: score2=0

        self.angle_log['upper_arm_abducted'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])

    def lower_arm_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle_1 = pose[self.joint_name.index('L_Elbow')][1]
        angle_2 = pose[self.joint_name.index('L_Elbow')][2]
        angle1 = max(angle_1, angle_2)

        if angle1>-100 and angle1<-60: score1=1
        elif angle1<-100 or (angle1>-60 and angle1<0): score1=2
        else: score1=1

        angle_1 = pose[self.joint_name.index('R_Elbow')][1]
        angle_2 = pose[self.joint_name.index('R_Elbow')][2]
        angle2 = max(angle_1, angle_2)

        if angle2>60 and angle2<100: score2=1
        elif angle2>100 or (angle2>0 and angle2<60): score2=2
        else: score2=1

        self.angle_log['lower_arm_bending'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])

    def bent_from_midline_or_out_to_side(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Thorax')][0]

        if angle1<10 or (angle1>-45 and angle1<-10): score1=0
        elif angle1>10 or angle1<-45: score1=1
        else: score1=0

        angle2 = pose[self.joint_name.index('R_Thorax')][0]

        if angle2>-10 or (angle2>10 and angle2<45): score2=0
        elif angle2<-10 or angle2>45: score2=1
        else: score2=0

        self.angle_log['bent_from_midline_or_out_to_side'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])

    def wrist_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Wrist')][2]

        if abs(angle1) < 1: score1=1
        elif abs(angle1)>1 and abs(angle1)<15: score1=2
        elif abs(angle1)>15: score1=3
        else: score1=1
        
        angle2 = pose[self.joint_name.index('R_Wrist')][2]

        if abs(angle2) < 1: score2=1
        elif abs(angle2)>1 and abs(angle2)<15: score2=2
        elif abs(angle2)>15: score2=3
        else: score2=1
        
        self.angle_log['wrist_bending'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])

    def wrist_side_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Wrist')][1]

        if abs(angle1) < 10: score1=0
        elif abs(angle1)>10: score1=1
        else: score1=0
        
        angle2 = pose[self.joint_name.index('R_Wrist')][1]

        if abs(angle2) < 10: score2=0
        elif abs(angle2)>10: score2=1
        else: score2=0
        
        self.angle_log['wrist_side_bending'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])
    
    def wrist_twist(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Wrist')][0]

        if abs(angle1) < 45: score1=1
        elif abs(angle1)>45: score1=2
        else: score1=1
        
        angle2 = pose[self.joint_name.index('R_Wrist')][0]

        if abs(angle2) < 45: score2=1
        elif abs(angle2)>45: score2=2
        else: score2=1
        
        self.angle_log['wrist_twist'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])

    def trunk_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][0]
        self.angle_log['trunk_bending'] = f'{angle:.1f}'

        if abs(angle)<5: return 1
        elif angle>5 and angle<20: return 2
        elif angle>20 and angle<60: return 3
        elif angle>60: return 4
        else: return 1

    def trunk_side_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][2]
        self.angle_log['trunk_side_bending'] = f'{angle:.1f}'

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 1
        else: return 0
    
    def trunk_twisted(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][1]
        self.angle_log['trunk_twisted'] = f'{angle:.1f}'

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 1
        else: return 0

    def neck_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Neck')][0]
        self.angle_log['neck_bending'] = f'{angle:.1f}'

        if angle>-5 and angle<10: return 1
        elif angle>10 and angle<20: return 2
        elif angle>20: return 3
        elif angle<-5: return 4
        else: return 1

    def neck_side_bending_twisted(self, pose, joint_cam):
        angle1 = pose[self.joint_name.index('Neck')][2]
        angle2 = pose[self.joint_name.index('Neck')][1]

        self.angle_log['neck_side_bending_twisted'] = f'{angle1:.1f}, {angle2:.1f}'

        if abs(angle1)<10 and abs(angle2)<10: return 0
        elif abs(angle1)>10 or abs(angle2)>10: return 1
        else: return 0

   