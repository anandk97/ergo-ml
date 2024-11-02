#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Calculation of RULA score
#%% Import Statements/Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
#%% Functions
def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle = angle_radians*(180/np.pi)
    return angle
def rotate_vec_along_axis(v,u,theta_in_deg):
  # rotate along an axis u (expressed as a unit vector) by an angle θ
  ux,uy,uz = u

  theta_in_rad = theta_in_deg/180*np.pi
  cost = np.cos(theta_in_rad)
  sint = np.sin(theta_in_rad)
  
  R = np.array([[cost+ux*ux*(1-cost), ux*uy*(1-cost)-uz*sint, ux*uz*(1-cost)+uy*sint],
                [uy*ux*(1-cost)+uz*sint, cost+uy*uy*(1-cost), uy*uz*(1-cost)-ux*sint],
                [uz*ux*(1-cost)-uy*sint, uz*uy*(1-cost)+ux*sint, cost+uz*uz*(1-cost)]])
  R = np.transpose(R)
  v_new = np.matmul(R,v.reshape(3,1))
  return v_new.reshape(3)

def upper_arm_score_from(shoulder,elbow):
    upper_arm_vec = elbow-shoulder
    down_vec = np.array([0,0,-1])
    z_vec = np.array([shoulder[0],shoulder[1],-1])
    upper_arm_angle = angle_between(upper_arm_vec,down_vec)
    # upper_arm_angle = angle_between(upper_arm_vec,z_vec)
    if upper_arm_angle < 20:
        score = 1
    else:
        if upper_arm_vec[0] < 0:
            score = 2
        else:
            if upper_arm_angle < 45:
                score = 2
            elif upper_arm_angle >=45 and upper_arm_angle <90:
                score = 3
            else:
                score = 4
    return score

def lower_arm_score_from(shoulder,elbow,wrist):
    upper_arm_vec = elbow-shoulder
    lower_arm_vec = wrist-elbow
    lower_arm_angle = angle_between(upper_arm_vec,lower_arm_vec)
    if lower_arm_angle < 60 or lower_arm_angle > 100:
        score = 2
    else:
        score = 1
    return score

def wrist_score_from(gonio_1,gonio_2,initial_angle):
    # Wrist score from gonio_1
    # Wrist twist from gonio_2
    gonio_2 = gonio_2 - initial_angle
    if gonio_2 > -3 and gonio_2 < 3:
        wrist_score = 1
    elif gonio_2 >= -15 and gonio_2 <=15:
        wrist_score = 2
    else:
        wrist_score = 3
    
    if gonio_1 > -30 and gonio_1 < 30:
        wrist_twist = 1
    else:
        wrist_twist = 2

    return wrist_score,wrist_twist

def tableA_posture_score_from(upper_arm_score,lower_arm_score,wrist_score,wrist_twist):
    uas = upper_arm_score
    las = lower_arm_score
    ws = wrist_score
    wt = wrist_twist
    if uas == 1:
        if las == 1:
            if ws == 1:
                if wt == 1:
                    score = 1
                elif wt == 2:
                    score = 2
            elif ws == 2:
                if wt == 1:
                    score = 2
                elif wt == 2:
                    score = 2
            elif ws == 3:
                if wt == 1:
                    score = 2
                elif wt == 2:
                    score = 3
            elif ws == 4:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
        elif las == 2:
            if ws == 1:
                if wt == 1:
                    score = 2
                elif wt == 2:
                    score = 2
            elif ws == 2:
                if wt == 1:
                    score = 2
                elif wt == 2:
                    score = 2
            elif ws == 3:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 4:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
        elif las == 3:
            if ws == 1:
                if wt == 1:
                    score = 2
                elif wt == 2:
                    score = 3
            elif ws == 2:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 3:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 4:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
    elif uas == 2:
        if las == 1:
            if ws == 1:
                if wt == 1:
                    score = 2
                elif wt == 2:
                    score = 3
            elif ws == 2:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 3:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 4
            elif ws == 4:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
        elif las == 2:
            if ws == 1:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 2:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 3:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 4
            elif ws == 4:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
        elif las == 3:
            if ws == 1:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 4
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 3:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 4:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
    elif uas == 3:
        if las == 1:
            if ws == 1:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 3
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 3:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 4:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
        elif las == 2:
            if ws == 1:
                if wt == 1:
                    score = 3
                elif wt == 2:
                    score = 4
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 3:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 4:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
        elif las == 3:
            if ws == 1:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 3:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 5
            elif ws == 4:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
    elif uas == 4:
        if las == 1:
            if ws == 1:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 3:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 5
            elif ws == 4:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
        elif las == 2:
            if ws == 1:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 3:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 5
            elif ws == 4:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
        elif las == 3:
            if ws == 1:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 4
            elif ws == 2:
                if wt == 1:
                    score = 4
                elif wt == 2:
                    score = 5
            elif ws == 3:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
            elif ws == 4:
                if wt == 1:
                    score = 6
                elif wt == 2:
                    score = 6
    elif uas == 5:
        if las == 1:
            if ws == 1:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
            elif ws == 2:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 5
            elif ws == 3:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 6
            elif ws == 4:
                if wt == 1:
                    score = 6
                elif wt == 2:
                    score = 7
        elif las == 2:
            if ws == 1:
                if wt == 1:
                    score = 5
                elif wt == 2:
                    score = 6
            elif ws == 2:
                if wt == 1:
                    score = 6
                elif wt == 2:
                    score = 6
            elif ws == 3:
                if wt == 1:
                    score = 6
                elif wt == 2:
                    score = 7
            elif ws == 4:
                if wt == 1:
                    score = 7
                elif wt == 2:
                    score = 7
        elif las == 3:
            if ws == 1:
                if wt == 1:
                    score = 6
                elif wt == 2:
                    score = 6
            elif ws == 2:
                if wt == 1:
                    score = 6
                elif wt == 2:
                    score = 7
            elif ws == 3:
                if wt == 1:
                    score = 7
                elif wt == 2:
                    score = 7
            elif ws == 4:
                if wt == 1:
                    score = 7
                elif wt == 2:
                    score = 8
    elif uas == 6:
        if las == 1:
            if ws == 1:
                if wt == 1:
                    score = 7
                elif wt == 2:
                    score = 7
            elif ws == 2:
                if wt == 1:
                    score = 7
                elif wt == 2:
                    score = 7
            elif ws == 3:
                if wt == 1:
                    score = 7
                elif wt == 2:
                    score = 8
            elif ws == 4:
                if wt == 1:
                    score = 8
                elif wt == 2:
                    score = 9
        elif las == 2:
            if ws == 1:
                if wt == 1:
                    score = 8
                elif wt == 2:
                    score = 8
            elif ws == 2:
                if wt == 1:
                    score = 8
                elif wt == 2:
                    score = 8
            elif ws == 3:
                if wt == 1:
                    score = 8
                elif wt == 2:
                    score = 9
            elif ws == 4:
                if wt == 1:
                    score = 9
                elif wt == 2:
                    score = 9
        elif las == 3:
            if ws == 1:
                if wt == 1:
                    score = 9
                elif wt == 2:
                    score = 9
            elif ws == 2:
                if wt == 1:
                    score = 9
                elif wt == 2:
                    score = 9
            elif ws == 3:
                if wt == 1:
                    score = 9
                elif wt == 2:
                    score = 9
            elif ws == 4:
                if wt == 1:
                    score = 9
                elif wt == 2:
                    score = 9
    return score

def arm_wrist_muscle_use_score_from():
    score = 1 # Based on typical Hand Layup Motion observed
    return score

def force_load_score_from(total_load):
    if total_load < 19.57: #Other number is 97.86
        score = 0
    elif total_load >= 19.57 and total_load < 97.86:
        score = 2
    else:
        score = 3
    return score

def neck_score_from(l_shoulder,r_shoulder,l_hip, r_hip, l_eye, r_eye):
  world_z_vec = np.array([0,0,1.0])
  mid_shoulder = (l_shoulder+r_shoulder)/2
  mid_hip = (l_hip+r_hip)/2
  mid_eye = (l_eye+l_eye)/2
  z_vec = mid_shoulder-mid_hip
  z_vec = z_vec/np.linalg.norm(z_vec)
  # z_vec correction due to difference between spine-vec and midshoulder2midhip-vec
  z_vec_angle = angle_between(z_vec,world_z_vec)
  if z_vec_angle != 0:
    z_vec_angle_correct = np.power(z_vec_angle, 1.1)
    rot_axis = np.cross(z_vec,world_z_vec)
    rot_axis = rot_axis/np.linalg.norm(rot_axis)
    z_vec = rotate_vec_along_axis(world_z_vec,rot_axis,z_vec_angle_correct)
  y_vec = l_shoulder-r_shoulder
  y_vec = y_vec/np.linalg.norm(y_vec)
  x_vec = np.cross(y_vec,z_vec)
  neck_vec = mid_eye-mid_shoulder
  neck_vec = neck_vec/np.linalg.norm(neck_vec)
  # eye-to-shoulder vector has around 10 degree between spine vector by default
  angle_compensate = 10
  neck_angle = angle_between(neck_vec, z_vec)-angle_compensate
  # For back bending
  forward_angle = angle_between(neck_vec, x_vec)+angle_compensate
  if forward_angle>90:
    score = 4
  else:
    if neck_angle<10:
      score = 1
    elif neck_angle<20:
      score = 2
    else:
      score = 3
  return score
def trunk_score_from(l_hip,r_hip,mid_shoulder):
    mid_hip = (l_hip+r_hip)/2
    trunk_vec = mid_shoulder - mid_hip
    up_vec = np.array([0,0,1])
    z_vec = np.array([mid_hip[0],mid_hip[1],1])
    trunk_angle = angle_between(trunk_vec, up_vec)
    # trunk_angle = angle_between(trunk_vec, z_vec)
    if trunk_angle < 3:
        score = 1
    elif trunk_angle >= 3 and trunk_angle < 20:
        score = 2
    elif trunk_angle >= 20 and trunk_angle < 60:
        score = 3
    else: 
        score = 4
    return score

def leg_score_from(leg_support):
    if leg_support:
        score = 1
    else:
        score = 2
    return score

def tableB_posture_score(neck_score,trunk_score,leg_score):
    ns = neck_score
    ts = trunk_score
    ls = leg_score
    if ns == 1:
        if ts == 1:
            if ls == 1:
                score = 1
            elif ls == 2:
                score = 3
        elif ts == 2:
            if ls == 1:
                score = 2
            elif ls == 2:
                score = 3
        elif ts == 3:
            if ls == 1:
                score = 3
            elif ls == 2:
                score = 4
        elif ts == 4:
            if ls == 1:
                score = 5
            elif ls == 2:
                score = 5
        elif ts == 5:
            if ls == 1:
                score = 6
            elif ls == 2:
                score = 6
        elif ts == 6:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
    elif ns == 2:
        if ts == 1:
            if ls == 1:
                score = 2
            elif ls == 2:
                score = 3
        elif ts == 2:
            if ls == 1:
                score = 2
            elif ls == 2:
                score = 3
        elif ts == 3:
            if ls == 1:
                score = 4
            elif ls == 2:
                score = 5
        elif ts == 4:
            if ls == 1:
                score = 5
            elif ls == 2:
                score = 5
        elif ts == 5:
            if ls == 1:
                score = 6
            elif ls == 2:
                score = 7
        elif ts == 6:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
    elif ns == 3:
        if ts == 1:
            if ls == 1:
                score = 3
            elif ls == 2:
                score = 3
        elif ts == 2:
            if ls == 1:
                score = 3
            elif ls == 2:
                score = 4
        elif ts == 3:
            if ls == 1:
                score = 4
            elif ls == 2:
                score = 5
        elif ts == 4:
            if ls == 1:
                score = 5
            elif ls == 2:
                score = 6
        elif ts == 5:
            if ls == 1:
                score = 6
            elif ls == 2:
                score = 7
        elif ts == 6:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
    elif ns == 4:
        if ts == 1:
            if ls == 1:
                score = 5
            elif ls == 2:
                score = 5
        elif ts == 2:
            if ls == 1:
                score = 5
            elif ls == 2:
                score = 6
        elif ts == 3:
            if ls == 1:
                score = 6
            elif ls == 2:
                score = 7
        elif ts == 4:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
        elif ts == 5:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
        elif ts == 6:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
    elif ns == 5:
        if ts == 1:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
        elif ts == 2:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 7
        elif ts == 3:
            if ls == 1:
                score = 7
            elif ls == 2:
                score = 8
        elif ts == 4:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
        elif ts == 5:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
        elif ts == 6:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
    elif ns == 6:
        if ts == 1:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
        elif ts == 2:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
        elif ts == 3:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 8
        elif ts == 4:
            if ls == 1:
                score = 8
            elif ls == 2:
                score = 9
        elif ts == 5:
            if ls == 1:
                score = 9
            elif ls == 2:
                score = 9
        elif ts == 6:
            if ls == 1:
                score = 9
            elif ls == 2:
                score = 9
    return score

def leg_trunk_muscle_use_score_from():
    score = 1
    return score

def tableC_posture_score(wrist_arm_score,neck_trunk_leg_score):
    was = wrist_arm_score
    ntls = neck_trunk_leg_score
    if was == 1:
        if ntls == 1:
            score = 1
        elif ntls == 2:
            score = 2
        elif ntls == 3:
            score = 3
        elif ntls == 4:
            score = 3
        elif ntls == 5:
            score = 4
        elif ntls == 6:
            score = 5
        else :
            score = 5
    elif was == 2:
        if ntls == 1:
            score = 2
        elif ntls == 2:
            score = 2
        elif ntls == 3:
            score = 3
        elif ntls == 4:
            score = 4
        elif ntls == 5:
            score = 4
        elif ntls == 6:
            score = 5
        else :
            score = 5
    elif was == 3:
        if ntls == 1:
            score = 3
        elif ntls == 2:
            score = 3
        elif ntls == 3:
            score = 3
        elif ntls == 4:
            score = 4
        elif ntls == 5:
            score = 4
        elif ntls == 6:
            score = 5
        else :
            score = 6
    elif was == 4:
        if ntls == 1:
            score = 3
        elif ntls == 2:
            score = 3
        elif ntls == 3:
            score = 3
        elif ntls == 4:
            score = 4
        elif ntls == 5:
            score = 5
        elif ntls == 6:
            score = 6
        else :
            score = 6
    elif was == 5:
        if ntls == 1:
            score = 4
        elif ntls == 2:
            score = 4
        elif ntls == 3:
            score = 4
        elif ntls == 4:
            score = 5
        elif ntls == 5:
            score = 6
        elif ntls == 6:
            score = 7
        else :
            score = 7
    elif was == 6:
        if ntls == 1:
            score = 4
        elif ntls == 2:
            score = 4
        elif ntls == 3:
            score = 5
        elif ntls == 4:
            score = 6
        elif ntls == 5:
            score = 6
        elif ntls == 6:
            score = 7
        else :
            score = 7
    elif was == 7:
        if ntls == 1:
            score = 5
        elif ntls == 2:
            score = 5
        elif ntls == 3:
            score = 6
        elif ntls == 4:
            score = 6
        elif ntls == 5:
            score = 7
        elif ntls == 6:
            score = 7
        else :
            score = 7
    else:
        if ntls == 1:
            score = 5
        elif ntls == 2:
            score = 5
        elif ntls == 3:
            score = 6
        elif ntls == 4:
            score = 7
        elif ntls == 5:
            score = 7
        elif ntls == 6:
            score = 7
        else :
            score = 7
        
    return score    
#%% Parameters
# initial_angle = -25
initial_angle = 0
participant_id_range = range(8,16)
tool_id_range = range(1,3)
trial_id_range = range(1,4)
#%% Calculate RULA for all trials
for participant_id in participant_id_range:
    for tool_id in tool_id_range:
        for trial_id in trial_id_range:
            if os.path.isfile(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed bodypose.csv"):
                print("Participant "+str(participant_id)+" Tool "+str(tool_id)+" Trial "+str(trial_id))
                data = pd.read_csv(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed bodypose.csv",header=None)
                gonio_data = pd.read_csv(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed gonio.csv",header=None)
                file_name = r'C:\Users\anand\Desktop\RULA Labelled Data\p'+str(participant_id)+' tool'+str(tool_id)+' trial'+str(trial_id)+' w lr RULA.csv'
                    

                ##0-Nose 1,2-LREye 3,4-LRear 5,6-LRshoulder 7,8-LRelbow 9,
                #10-LRwrist 11,12-LRhip, 13,14-LRknee, 15,16-LRankle, 
                #17-Midshoulder # ForeArm,hand,FingerBase,Thumb123,
                #Index1234,Middle1234,Ring1234,Pinky1234

                # AlphaPose references
                # Nose 0-2
                # L-Eye 3-5
                # R_Eye 6-8
                # L_ear 9-11
                # R_ear 12-14
                # L_shoulder 15-17
                # R_shoulder 18-20
                # L_elbow 21-23
                # R_elbow 24-26
                # L_wrist 27-29
                # R_wrist 30-32
                # L_hip 33-35
                # R-hip 36-38
                # L_knee 39-41
                # R_knee 42-44
                # L_ankle 45-47
                # R_ankle 48-50
                # Mid_shoulder 51-53

                # Leap References
                # Hand joints : 54-173

                # Ergopak References - 174-183

                # Goniometer References - 184-188
                # Upper Arm Score
                l_eye = np.array(data.iloc[:,3:6])
                r_eye = np.array(data.iloc[:,6:9])

                mid_eye = (l_eye+r_eye)/2

                l_shoulder = np.array(data.iloc[:,15:18])
                r_shoulder = np.array(data.iloc[:,18:21])
                mid_shoulder = np.array(data.iloc[:,51:54])


                l_elbow = np.array(data.iloc[:,21:24])  
                r_elbow = np.array(data.iloc[:,24:27]) 
                
                l_wrist = np.array(data.iloc[:,27:30])
                r_wrist = np.array(data.iloc[:,30:33])

                l_hip = np.array(data.iloc[:,33:36])
                r_hip = np.array(data.iloc[:,36:39])

                l_gonio_1 = np.array(gonio_data.iloc[:,0]) 
                l_gonio_2 = np.array(gonio_data.iloc[:,1])
                r_gonio_1 = np.array(gonio_data.iloc[:,2])
                r_gonio_2 = np.array(gonio_data.iloc[:,3])

                l_RULA_scores = []
                r_RULA_scores = []
                total_load = 15
                leg_support = 1 # Legs are not supported

                data_length = np.min([len(data[0]),len(gonio_data[0])])
                if len(data[0]) >len(gonio_data[0]):
                    data = data.iloc[:data_length,:]
                else:
                    gonio_data = gonio_data.iloc[:data_length,:]
                for i in range(data_length):
                    upper_arm_score = upper_arm_score_from(l_shoulder[i,:],l_elbow[i,:])
                    # Adjustment
                    if upper_arm_score > 1:
                        upper_arm_score = upper_arm_score-1 # Arm is supported\person is leaning

                    lower_arm_score = lower_arm_score_from(l_shoulder[i,:], l_elbow[i,:], l_wrist[i,:])

                    
                    wrist_score,wrist_twist = wrist_score_from(l_gonio_1[i], l_gonio_2[i],initial_angle)

                    
                    tableA_score = tableA_posture_score_from(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
                    
                    arm_wrist_muscle_use_score = arm_wrist_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    wrist_arm_score = tableA_score + arm_wrist_muscle_use_score + force_load_score
                    
                    neck_score = neck_score_from(l_shoulder[i,:], r_shoulder[i,:],l_hip[i,:],r_hip[i,:],l_eye[i,:],r_eye[i,:])
                    
                    trunk_score  = trunk_score_from(l_hip[i,:], r_hip[i,:], mid_shoulder[i,:])
                    
                    leg_score = leg_score_from(leg_support)
                        
                    tableB_score = tableB_posture_score(neck_score, trunk_score, leg_score)
                    
                    leg_trunk_muscle_use_score = leg_trunk_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    neck_trunk_leg_score = tableB_score + leg_trunk_muscle_use_score + force_load_score
                    
                    tableC_score = tableC_posture_score(wrist_arm_score, neck_trunk_leg_score)
                    
                    l_RULA_scores.append(tableC_score)
                for i in range(data_length):
                    upper_arm_score = upper_arm_score_from(r_shoulder[i,:],r_elbow[i,:])
                    # Adjustment
                    if upper_arm_score > 1:
                        upper_arm_score = upper_arm_score-1 # Arm is supported\person is leaning

                    
                    lower_arm_score = lower_arm_score_from(r_shoulder[i,:], r_elbow[i,:], r_wrist[i,:])

                    
                    
                    wrist_score,wrist_twist = wrist_score_from(r_gonio_1[i], r_gonio_2[i],initial_angle)

                    
                    
                    tableA_score = tableA_posture_score_from(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
                    
                    arm_wrist_muscle_use_score = arm_wrist_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    wrist_arm_score = tableA_score + arm_wrist_muscle_use_score + force_load_score

                    neck_score = neck_score_from(l_shoulder[i,:], r_shoulder[i,:],l_hip[i,:],r_hip[i,:],l_eye[i,:],r_eye[i,:])

                    
                    trunk_score  = trunk_score_from(l_hip[i,:], r_hip[i,:], mid_shoulder[i,:])

                    
                    leg_score = leg_score_from(leg_support)

                    
                    
                    tableB_score = tableB_posture_score(neck_score, trunk_score, leg_score)
                    
                    leg_trunk_muscle_use_score = leg_trunk_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    neck_trunk_leg_score = tableB_score + leg_trunk_muscle_use_score + force_load_score
                    # neck_trunk_leg_scores.append(neck_trunk_leg_score)
                    
                    tableC_score = tableC_posture_score(wrist_arm_score, neck_trunk_leg_score)
                    
                    r_RULA_scores.append(tableC_score)
                data['54'] = l_gonio_1
                data['55'] = l_gonio_2
                data['56'] = r_gonio_1
                data['57'] = r_gonio_2
                data['58'] = l_RULA_scores
                data['59'] = r_RULA_scores
                data = data.round(3)
                data.to_csv(file_name,index = False, header = False)
            elif os.path.isfile(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\\tool"+str(tool_id)+" trial"+str(trial_id)+"\processed p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+" bodypose.csv"):
                print("Participant "+str(participant_id)+" Tool "+str(tool_id)+" Trial "+str(trial_id))
                data = pd.read_csv(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\\tool"+str(tool_id)+" trial"+str(trial_id)+"\processed p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+" bodypose.csv")
                gonio_data = pd.read_csv(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\\tool"+str(tool_id)+" trial"+str(trial_id)+"\processed p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+" gonio.csv")
                file_name = r'C:\Users\anand\Desktop\RULA Labelled Data\p'+str(participant_id)+' tool'+str(tool_id)+' trial'+str(trial_id)+' w lr RULA.csv'
                    
                                    

                ##0-Nose 1,2-LREye 3,4-LRear 5,6-LRshoulder 7,8-LRelbow 9,
                #10-LRwrist 11,12-LRhip, 13,14-LRknee, 15,16-LRankle, 
                #17-Midshoulder # ForeArm,hand,FingerBase,Thumb123,
                #Index1234,Middle1234,Ring1234,Pinky1234

                # AlphaPose references
                # Nose 0-2
                # L-Eye 3-5
                # R_Eye 6-8
                # L_ear 9-11
                # R_ear 12-14
                # L_shoulder 15-17
                # R_shoulder 18-20
                # L_elbow 21-23
                # R_elbow 24-26
                # L_wrist 27-29
                # R_wrist 30-32
                # L_hip 33-35
                # R-hip 36-38
                # L_knee 39-41
                # R_knee 42-44
                # L_ankle 45-47
                # R_ankle 48-50
                # Mid_shoulder 51-53

                # Leap References
                # Hand joints : 54-173

                # Ergopak References - 174-183

                # Goniometer References - 184-188
                # Upper Arm Score
                l_eye = np.array(data.iloc[:,3:6])
                r_eye = np.array(data.iloc[:,6:9])

                mid_eye = (l_eye+r_eye)/2

                l_shoulder = np.array(data.iloc[:,15:18])
                r_shoulder = np.array(data.iloc[:,18:21])
                mid_shoulder = np.array(data.iloc[:,51:54])


                l_elbow = np.array(data.iloc[:,21:24])  
                r_elbow = np.array(data.iloc[:,24:27]) 
                
                l_wrist = np.array(data.iloc[:,27:30])
                r_wrist = np.array(data.iloc[:,30:33])

                l_hip = np.array(data.iloc[:,33:36])
                r_hip = np.array(data.iloc[:,36:39])

                l_gonio_1 = np.array(gonio_data.iloc[:,0]) 
                l_gonio_2 = np.array(gonio_data.iloc[:,1])
                r_gonio_1 = np.array(gonio_data.iloc[:,2])
                r_gonio_2 = np.array(gonio_data.iloc[:,3])

                l_RULA_scores = []
                r_RULA_scores = []
                total_load = 15
                leg_support = 1 # Legs are not supported

                data_length = np.min([data.shape[0],gonio_data.shape[0]])
                data = data.iloc[:data_length,:]
                gonio_data = gonio_data.iloc[:data_length,:]
                for i in range(data_length):
                    upper_arm_score = upper_arm_score_from(l_shoulder[i,:],l_elbow[i,:])
                    # Adjustment
                    if upper_arm_score > 1:
                        upper_arm_score = upper_arm_score-1 # Arm is supported\person is leaning

                    lower_arm_score = lower_arm_score_from(l_shoulder[i,:], l_elbow[i,:], l_wrist[i,:])

                    
                    wrist_score,wrist_twist = wrist_score_from(l_gonio_1[i], l_gonio_2[i],initial_angle)

                    
                    tableA_score = tableA_posture_score_from(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
                    
                    arm_wrist_muscle_use_score = arm_wrist_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    wrist_arm_score = tableA_score + arm_wrist_muscle_use_score + force_load_score
                    
                    neck_score = neck_score_from(l_shoulder[i,:], r_shoulder[i,:],l_hip[i,:],r_hip[i,:],l_eye[i,:],r_eye[i,:])
                    
                    trunk_score  = trunk_score_from(l_hip[i,:], r_hip[i,:], mid_shoulder[i,:])
                    
                    leg_score = leg_score_from(leg_support)
                        
                    tableB_score = tableB_posture_score(neck_score, trunk_score, leg_score)
                    
                    leg_trunk_muscle_use_score = leg_trunk_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    neck_trunk_leg_score = tableB_score + leg_trunk_muscle_use_score + force_load_score
                    
                    tableC_score = tableC_posture_score(wrist_arm_score, neck_trunk_leg_score)
                    
                    l_RULA_scores.append(tableC_score)
                for i in range(data_length):
                    upper_arm_score = upper_arm_score_from(r_shoulder[i,:],r_elbow[i,:])
                    # Adjustment
                    if upper_arm_score > 1:
                        upper_arm_score = upper_arm_score-1 # Arm is supported\person is leaning

                    
                    lower_arm_score = lower_arm_score_from(r_shoulder[i,:], r_elbow[i,:], r_wrist[i,:])

                    
                    
                    wrist_score,wrist_twist = wrist_score_from(r_gonio_1[i], r_gonio_2[i],initial_angle)

                    
                    
                    tableA_score = tableA_posture_score_from(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
                    
                    arm_wrist_muscle_use_score = arm_wrist_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    wrist_arm_score = tableA_score + arm_wrist_muscle_use_score + force_load_score

                    neck_score = neck_score_from(l_shoulder[i,:], r_shoulder[i,:],l_hip[i,:],r_hip[i,:],l_eye[i,:],r_eye[i,:])

                    
                    trunk_score  = trunk_score_from(l_hip[i,:], r_hip[i,:], mid_shoulder[i,:])

                    
                    leg_score = leg_score_from(leg_support)

                    
                    
                    tableB_score = tableB_posture_score(neck_score, trunk_score, leg_score)
                    
                    leg_trunk_muscle_use_score = leg_trunk_muscle_use_score_from()
                    
                    force_load_score = force_load_score_from(total_load)
                    
                    neck_trunk_leg_score = tableB_score + leg_trunk_muscle_use_score + force_load_score
                    # neck_trunk_leg_scores.append(neck_trunk_leg_score)
                    
                    tableC_score = tableC_posture_score(wrist_arm_score, neck_trunk_leg_score)
                    
                    r_RULA_scores.append(tableC_score)
                data['54'] = l_gonio_1
                data['55'] = l_gonio_2
                data['56'] = r_gonio_1
                data['57'] = r_gonio_2
                data['58'] = l_RULA_scores
                data['59'] = r_RULA_scores
                data = data.round(3)
                data.to_csv(file_name,index = False, header = False)
               