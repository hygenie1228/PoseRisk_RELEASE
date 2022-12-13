import numpy as np
import torch
import os.path as osp
import json


class REBA:
    def __init__(self, debug=False):
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

        self.eval_items = ['Trunk', 'Neck', 'Leg', 'Upper_arm (L,R)', 'Lower_arm (L,R)', 'Wrist (L,R)']
        self.debugging = debug
        self.angle_log = {}
        self.log = []

    def __call__(self, poses, joint_cams, add_info):
        results = []

        for ii in range(len(poses)):
            pose = poses[ii]
            joint_cam = joint_cams[ii]

            # Group A
            group_a_score, group_a_list = self.group_a(pose, joint_cam, add_info)
            group_a_score = group_a_score + add_info["REBA"]["Load/Force Score"]
            
            # Group B
            group_b_score_L, group_b_score_R, group_b_list = self.group_b(pose, joint_cam, add_info)
            group_b_score = max(group_b_score_L, group_b_score_R)
            group_b_score = group_b_score + add_info["REBA"]["Coupling"]

            # Final Score
            group_a_score = int(np.clip(group_a_score, 1, 12))
            group_b_score = int(np.clip(group_b_score, 1, 12))
            final_score = self.table_c[group_a_score-1][group_b_score-1] + add_info["REBA"]["Activity_Score"]

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
        trunk += self.trunk_twist(pose, joint_cam)
        trunk += self.trunk_side_bending(pose, joint_cam)
        neck += self.neck_bending(pose, joint_cam)
        neck += self.neck_twist(pose, joint_cam)
        leg += add_info["REBA"]["Legs_bilateral_weight_bearing/walking"]
        leg += self.leg_bending(pose, joint_cam, add_info)

        trunk = int(np.clip(trunk, 1, 5))
        neck = int(np.clip(neck, 1, 3))
        leg = int(np.clip(leg, 1, 4))
        return self.table_a[trunk-1][neck-1][leg-1], [trunk, neck, leg]
      

    def group_b(self, pose, joint_cam, add_info):
        upper_arm, lower_arm, wrist = np.array([0,0]), np.array([0,0]), np.array([0,0])
        upper_arm += self.upper_arm_bending(pose, joint_cam, add_info)
        upper_arm += self.shoulder_rise(pose, joint_cam)
        upper_arm += self.upper_arm_abducted_rotated(pose, joint_cam)
        lower_arm += self.lower_arm_bending(pose, joint_cam)
        wrist += self.wrist_bending(pose, joint_cam)
        wrist += self.wrist_side_bending_or_twisted(pose, joint_cam)

        upper_arm = np.clip(upper_arm, 1, 6)
        lower_arm = np.clip(lower_arm, 1, 2)
        wrist = np.clip(wrist, 1, 3)

        group_b_score_L = self.table_b[upper_arm[0]-1][lower_arm[0]-1][wrist[0]-1]
        group_b_score_R = self.table_b[upper_arm[1]-1][lower_arm[1]-1][wrist[1]-1]
        group_b_list = [f'{upper_arm[0]},{upper_arm[1]}', f'{lower_arm[0]},{lower_arm[1]}', f'{wrist[0]},{wrist[1]}']
        return group_b_score_L, group_b_score_R, group_b_list

    def trunk_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][0]
        self.angle_log['trunk_bending'] = f'{angle:.1f}'

        if abs(angle)<5: return 1
        elif (angle>5 and angle<20) or (angle>-20 and angle<-5): return 2
        elif (angle>20 and angle<60) or (angle<-20): return 3
        elif angle>60: return 4
        else: return 1

    def trunk_side_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][2]
        self.angle_log['trunk_side_bending'] = f'{angle:.1f}'

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 0
        else: return 0

    def trunk_twist(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Torso')][1]
        self.angle_log['trunk_twist'] = f'{angle:.1f}'

        if abs(angle)<10: return 0
        elif abs(angle)>10: return 1
        else: return 0

    def neck_bending(self, pose, joint_cam):
        angle = pose[self.joint_name.index('Neck')][0]
        self.angle_log['neck_bending'] = f'{angle:.1f}'

        if angle>-5 and angle<20: return 1
        elif angle<20 or angle<-5: return 2
        else: return 1

    def neck_twist(self, pose, joint_cam):
        angle1 = pose[self.joint_name.index('Neck')][2]
        angle2 = pose[self.joint_name.index('Neck')][1]
        self.angle_log['neck_twist'] = f'{angle1:.1f},{angle2:.1f}'

        if abs(angle1)<10 and abs(angle2)<10: return 0
        elif abs(angle1)>10 or abs(angle2)>10: return 1
        else: return 0

    def leg_bending(self, pose, joint_cam, add_info):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Knee')][0]

        if angle1<30: score1=0
        elif angle1>30 and angle1<60: score1=1
        elif angle1>60 and add_info["REBA"]["Sitting"] > 0 : score1=2
        else: score1=0

        angle2 = pose[self.joint_name.index('R_Knee')][0]

        if angle2<30: score2=0
        elif angle2>30 and angle2<60: score2=1
        elif angle2>60 and add_info["REBA"]["Sitting"] > 0 : score2=2
        else: score2=0

        self.angle_log['leg_bending'] = f'L {angle1:.1f} R {angle2:.1f}'
        return max(score1, score2)

    def upper_arm_bending(self, pose, joint_cam, add_info):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Shoulder')][2]
        angle2 = pose[self.joint_name.index('L_Shoulder')][1]

        if angle1>-110 and angle1<-20:
            if abs(angle2)<20: score1=1
            elif angle2>20 or (angle2>-45 and angle2<-20): score1=2
            elif angle2>-90 and angle2<=-45: score1=3
            elif angle2<-90: score1=4
            else: score1=1
        elif angle1>-20:
            if abs(angle2)<20: score1=1
            elif angle2>20 or angle2<70: score1=2
            elif angle2>70: score1=2
            elif angle2>-70 and angle2<-20: score1=4
            elif angle2<-70: score1=4
            else: score1=1
        else: score1=1
        score1 -= add_info["REBA"]["Arm_supported_leaning_L"]

        angle3 = pose[self.joint_name.index('R_Shoulder')][2]
        angle4 = pose[self.joint_name.index('R_Shoulder')][1]

        if angle3>20 and angle3<110:
            if abs(angle4)<20: score2=1
            elif angle4<-20 or (angle4>20 and angle4<=45): score2=2
            elif angle4>45 and angle4<=90: score2=3
            elif angle4>90: score2=4
            else: score2=1
        elif angle1>-20:
            if abs(angle2)<20: score2=1
            elif angle2>20 or angle2<70: score2=2
            elif angle2>70: score2=2
            elif angle2>-70 and angle2<-20: score2=4
            elif angle2<-70: score2=4
            else: score2=1
        else: score2=1
        score2 -= add_info["REBA"]["Arm_supported_leaning_R"]

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
        else: score1=0

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

    def upper_arm_abducted_rotated(self, pose, joint_cam):
        score1, score2 = 0, 0

        angle1 = pose[self.joint_name.index('L_Shoulder')][2]
        angle2 = pose[self.joint_name.index('L_Shoulder')][0]
        angle3 = pose[self.joint_name.index('L_Shoulder')][1]

        if angle1>-110 and angle1<-20:
            if angle1<45 and abs(angle2)<10: score1=0
            elif angle1>45 or abs(angle2)>10: score1=1
            else: score1=0
        elif angle1>-20:
            if abs(angle3)<20: score1=1
            elif angle3>20 or angle3<70: score1=1
            elif angle3>70: score1=0
            elif angle3>-70 and angle3<-20: score1=1
            elif angle3<-70: score1=0
            else: score1=0

            if abs(angle2)>10:score1+=1
        else:
            score1 = 0

        angle4 = pose[self.joint_name.index('R_Shoulder')][2]
        angle5 = pose[self.joint_name.index('R_Shoulder')][0]
        angle6 = pose[self.joint_name.index('R_Shoulder')][1]

        if angle4>20 and angle4<110:
            if angle4>45 and abs(angle5)<10: score2=0
            elif angle4<45 or abs(angle5)>10: score2=1
            else: score2=0
        elif angle4<20:
            if abs(angle6)<20: score2=1
            elif angle6>-70 and angle6<-20: score2=1
            elif angle6<-70: score2=0
            elif angle6>20 and angle6<70: score2=1
            elif angle6>70: score2=0
            else: score2=0

            if abs(angle5)>10:score1+=1
        else: score2=0

        self.angle_log['upper_arm_abducted_rotated'] = f'L {angle1:.1f},{angle2:.1f} R {angle3:.1f},{angle4:.1f}'
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

    def wrist_bending(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Wrist')][2]

        if abs(angle1)<15: score1=1
        elif abs(angle1)>15: score1=2
        else: score1=1
        
        angle2 = pose[self.joint_name.index('R_Wrist')][2]

        if abs(angle2)<15: score2=1
        elif abs(angle2)>15: score2=2
        else: score2=1

        self.angle_log['wrist_bending'] = f'L {angle1:.1f} R {angle2:.1f}'
        return np.array([score1, score2])

    def wrist_side_bending_or_twisted(self, pose, joint_cam):
        score1, score2 = 0, 0
        angle1 = pose[self.joint_name.index('L_Wrist')][1]
        angle2 = pose[self.joint_name.index('L_Wrist')][0]

        if abs(angle1)<10 and abs(angle2)<10: score1=0
        elif abs(angle1)>10 or abs(angle2)>10: score1=1
        else: score1=0

        angle3 = pose[self.joint_name.index('R_Wrist')][1]
        angle4 = pose[self.joint_name.index('R_Wrist')][0]

        if abs(angle3)<10 and abs(angle4)<10: score2=0
        elif abs(angle3)>10 or abs(angle4)>10: score2=1
        else: score2=0
        
        self.angle_log['wrist_side_bending_or_twisted'] = f'L {angle1:.1f},{angle2:.1f} R {angle3:.1f},{angle4:.1f}'
        return np.array([score1, score2])
