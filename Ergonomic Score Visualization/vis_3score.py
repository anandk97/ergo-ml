import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import svd
from PIL import Image
import pickle
import re
import warnings
import json 
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import csv
import sys


class Pose():
    def __init__(self, param_json):
        print("working on:",param_json,"===================================================")
        # parameters settings
        f = open(param_json)
        params = json.load(f)

        # checkerboard 
        self.checker_params = {
            "w" : params["square_w"],
            "h" : params["square_h"],
            "x_num" : params["square_x_num"],
            "y_num" : params["square_y_num"],
            "plot_checkboard":params["plot_checkboard"]
        }

        # webcam, distortioncoefficients=(k1k2p1p2k3)
        self.cam_params = {
            "intrinsic_l":np.array(params["intrinsic_l"]),
            "intrinsic_r":np.array(params["intrinsic_r"]),
            "distort_l":np.array(params["distort_l"]),
            "distort_r":np.array(params["distort_r"])
        }

        self.camframe_info = {
            "vertical_ref_p1_l":np.array(params["vertical_ref_p1_l"]),
            "vertical_ref_p1_r":np.array(params["vertical_ref_p1_r"]),
            "vertical_ref_p2_l":np.array(params["vertical_ref_p2_l"]),
            "vertical_ref_p2_r":np.array(params["vertical_ref_p2_r"])
        }

        # file namings 
        self.file_info = {
            "vidcalib_name" : params["vidcalib_name"],
            "vidcalib_extract_interval" : params["vidcalib_extract_interval"], # extract 1/(1+interval) from calibration
            "vidcalib_extract_folder" : params["vidcalib_extract_folder"]
            }
        
        self.file_info["vidcalib_path"] = re.sub(r'(trial[0-9]+/).*$', 'calib/', param_json)
        self.file_info["sensordata_path"] = re.sub(r'param.json$', '', param_json)

        self.file_info["mouse_logs"] = {}
        for sensor in ["gonio","leap","webcam","tactile"]:
            self.file_info["mouse_logs"][sensor] = os.path.join(self.file_info["sensordata_path"], params["mouse_logs"][sensor])

        self.file_info["gonio_name"] = os.path.join(self.file_info["sensordata_path"], params["gonio_name"])
        self.file_info["leap_name"] = os.path.join(self.file_info["sensordata_path"], params["leap_name"])
        self.file_info["webcam_name"] = os.path.join(self.file_info["sensordata_path"], params["webcam_name"])
        self.file_info["tactile_name"] = os.path.join(self.file_info["sensordata_path"], params["tactile_name"])

        self.file_info["alphapose_json_l"] = os.path.join(self.file_info["sensordata_path"], params["alphapose_json_l"])
        self.file_info["alphapose_json_r"] = os.path.join(self.file_info["sensordata_path"], params["alphapose_json_r"])

        if params["alphapose_vid_l"] is not None:
            self.file_info["alphapose_vid_l"] = os.path.join(self.file_info["sensordata_path"], params["alphapose_vid_l"])
            self.file_info["alphapose_vid_r"] = os.path.join(self.file_info["sensordata_path"], params["alphapose_vid_r"])
        else:
            self.file_info["alphapose_vid_l"] = None
            self.file_info["alphapose_vid_r"] = None
        
        # set up recording info
        self.sensors_info = {
            "webcam":params["info_webcam"],
            "leap":params["info_leap"],
            "gonio":params["info_gonio"],
            "tactile":params["info_tactile"],

            # use webcam video as reference to timestamp(seconds)
            "hip_ref_l" : params["hip_ref_l"],
            "hip_ref_r" : params["hip_ref_r"],
        }

        self.sensor_data = {}

        # hand initial rotation fix(ignore for now, might be deleted)
        self.rot_bd2hand = np.array(params["rot_bd2hand"]) 

        # max finger force to visualize (unit:pound)
        self.max_force = params["max_force"]

        # code output settings
        self.output_settings = {
            "do_plot":params["do_plot"],
            "save_plot":params["save_plot"],
            "save_coord":params["save_coord"]
        }

    def read_mouselog1(self,sensor):
        start_and_end_time = []
        with open(self.file_info["mouse_logs"][sensor]) as file:
            lines = file.readlines()

        lines_filtered = lines

        for i in range(1, len(lines_filtered)):
            current_line = lines_filtered[i]
            previous_line = lines_filtered[i - 1]
            if "Button.left" in current_line and "Button.right" in previous_line:
                time_str = current_line.split()[1].rstrip(':')
                time = datetime.strptime(time_str, "%H:%M:%S,%f")
                start_and_end_time.append(time)

        return start_and_end_time

    def read_mouselog2(self,sensor):
        start_and_end_time = []
        with open(self.file_info["mouse_logs"][sensor]) as file:
            lines = file.readlines()

        lines_filtered = []
        for i in range(len(lines)):
            line = lines[i]
            if sensor == "gonio":
                if "DataLINK" in line:
                    lines_filtered.append(line)
            elif sensor == "webcam":
                if "OBS" in line:
                    lines_filtered.append(line)
            elif sensor == "leap":
                if "Brekel" in line:
                    lines_filtered.append(line)
            elif sensor == "tactile":
                if "Chameleon" in line:
                    lines_filtered.append(line)

        for i in range(1, len(lines_filtered)):
            current_line = lines_filtered[i]
            previous_line = lines_filtered[i - 1]
            if "Button.left" in current_line and "Button.right" in previous_line:
                time_str = current_line.split()[1].rstrip(':')
                time = datetime.strptime(time_str, "%H:%M:%S,%f")
                start_and_end_time.append(time)

        return start_and_end_time
    
    def find_timeoverlap(self):
        sensor_list = ["webcam","gonio","tactile","leap"]

        for sensor in sensor_list:
            if self.sensors_info[sensor]["data_exist"]:
                # read mouse-logged start and end time
                if "May" in date_n_p:
                    time_start, time_end = self.read_mouselog1(sensor)
                elif "June" in date_n_p:
                    time_start, time_end = self.read_mouselog2(sensor)

                # correct time with system and record delay
                delay = timedelta(seconds=self.sensors_info[sensor]["record_delay"]) + timedelta(seconds=self.sensors_info[sensor]["system_delay"])
                self.sensors_info[sensor]["t_start"] = time_start+delay
                self.sensors_info[sensor]["t_end"] = time_end+delay

                # if sensor == "leap":
                    # data = pd.read_csv(self.file_info["leap_name"])
                    # time_total = np.max(data["timestamp"].values)
                    # time_end_temp = self.sensors_info[sensor]["t_start"]+timedelta(seconds=time_total)
                    # if(time_end_temp<self.sensors_info[sensor]["t_end"]):
                    #     self.sensors_info[sensor]["t_end"] = time_end_temp
                if sensor == "tactile":
                    with open(self.file_info["tactile_name"], 'r') as file:
                        reader = csv.reader(file)
                        for _ in range(5):
                            next(reader)
                        # Read the data starting from the start_column and convert strings to float
                        data_tactile = np.array([[float(value) for value in row[:]] for row in reader])
                        time_total = np.max(data_tactile[:,0])-np.min(data_tactile[:,0])
                    time_end_temp = self.sensors_info[sensor]["t_start"]+timedelta(seconds=time_total)
                    if(time_end_temp<self.sensors_info[sensor]["t_end"]):
                        self.sensors_info[sensor]["t_end"] = time_end_temp

                print(sensor.ljust(10),"start_time",self.sensors_info[sensor]["t_start"].strftime('%H:%M:%S'),"  end_time",self.sensors_info[sensor]["t_end"].strftime('%H:%M:%S'))

        # find out latest start time and earliest end time
        webcam_start = self.sensors_info["webcam"]["t_start"]
        latest_start = datetime.strptime("0:0:0,0", "%H:%M:%S,%f")
        earliest_end = datetime.strptime("23:59:59,999", "%H:%M:%S,%f")
        for sensor in sensor_list:
            if self.sensors_info[sensor]["data_exist"]:

                pick_t_start = self.sensors_info[sensor]["t_start"]
                pick_t_end = self.sensors_info[sensor]["t_end"]

                if self.sensors_info[sensor]["pick_t_end"] is not None:
                    pick_t_end = pick_t_start + timedelta(seconds=self.sensors_info[sensor]["pick_t_end"])
                if self.sensors_info[sensor]["pick_t_start"] is not None:
                    pick_t_start = pick_t_start + timedelta(seconds=self.sensors_info[sensor]["pick_t_start"])
                

                if(pick_t_start > latest_start):
                    latest_start = pick_t_start
                if(pick_t_end < earliest_end):
                    earliest_end = pick_t_end

        self.sensors_info["latest_start"] = latest_start
        self.sensors_info["earliest_end"] = earliest_end
        self.sensors_info["official_start_in_video"] = (latest_start-webcam_start).total_seconds()
        print("official start time in video:",self.sensors_info["official_start_in_video"],"seconds")

        if latest_start>earliest_end:
            print("no time overlap")
            sys.exit()

def img_curve_fit():
    img = Image.open("C:/Users/69516/OneDrive/research/JCATI/wrist torque computation/flexion moment curve.png")
    img_rgb = img.load()
    width, height = img.size

    p_curve = []
    for x in range(img.width):
        y_values = []
        for y in range(img.height):
            r, g, b = img_rgb[x, y]
            if r > 200 and g < 10 and b < 10:
                y_values.append(y)
        
        if y_values:
            avg_y = sum(y_values)/len(y_values)
            p_curve.append([x, avg_y])

    p_curve = np.array(p_curve)

    # translate pixel coordinate to graph coordinate
    X = (p_curve[:,0]-787)/398*50
    Y = -(p_curve[:,1]-1093)/339*5
    S = -8

    i = np.where((X[:-1] <= S) & (X[1:] >= S))[0][0]

    # 2. Fit the straight line
    def linear(x, m, b):
        return m * x + b

    params_line, _ = curve_fit(linear, X[:i+1], Y[:i+1])

    # 3. Fit the polynomial (without constraints) to get initial parameters
    def polynomial_no_c(x, a, b):
        return a * x**2 + b * x

    params_poly_initial, _ = curve_fit(polynomial_no_c, X[i+1:], Y[i+1:], p0=(1, 1))

    # Adjust the polynomial to meet the constraint
    mid_point = (X[i] + X[i+1]) / 2
    c_initial = S - params_poly_initial[0] * mid_point**2 - params_poly_initial[1] * mid_point

    # Refit with the constraint
    def polynomial(x, a, b, c):
        return a * x**2 + b * x + c

    params_poly, _ = curve_fit(polynomial, X[i+1:], Y[i+1:], p0=(params_poly_initial[0], params_poly_initial[1], c_initial))

    # # Plotting the data and the fitted curves
    # plt.scatter(X, Y, color='orange',alpha=0.1, label='Data Points')
    # plt.plot(X[:i+2], linear(X[:i+2], *params_line), 'b--', label='Fitted Line')
    # plt.plot(X[i-2:], polynomial(X[i-2:], *params_poly), 'r--', label='Fitted Polynomial')
    # plt.axvline(x=S, color='gray', linestyle='--')
    # plt.legend()
    # plt.xlabel('flexion angle (degree)')
    # plt.ylabel('flexion moment (Nm)')
    # plt.title('Curve Fitting')
    # plt.show()
    # plt.close()

    return params_line, params_poly, S

def moment_lookup(X, params_line, params_poly, S):
    
    def linear(x, m, b):
        return m * x + b

    def polynomial(x, a, b, c):
        return a * x**2 + b * x + c
    
    if np.isscalar(X):  # Check if X is a scalar
        if X <= S:
            return linear(X, *params_line)
        else:
            return polynomial(X, *params_poly)
    else:  # If X is an array
        Y = np.zeros_like(X)*1.0
        mask = X <= S
        Y[mask] = linear(X[mask], *params_line)
        Y[~mask] = polynomial(X[~mask], *params_poly)
        return Y

params_line, params_poly, S = img_curve_fit()

def correct_gonio(gonio_wrist_angle,person,tool_n_trial):
    # correct gonio data
    if person == 'p1':
        if tool_n_trial == "tool1 trial1":
            gonio_wrist_angle = gonio_wrist_angle-25
        elif tool_n_trial == "too1 trial2":
            gonio_wrist_angle = gonio_wrist_angle-30
        elif tool_n_trial == "too1 trial3":
            gonio_wrist_angle = gonio_wrist_angle-35
        elif tool_n_trial == "tool2 trial1":
            gonio_wrist_angle = gonio_wrist_angle-30
        elif tool_n_trial == "tool2 trial2":
            gonio_wrist_angle = gonio_wrist_angle-25
        elif tool_n_trial == "tool2 trial3":
            gonio_wrist_angle = gonio_wrist_angle-30
    elif person == 'p2':
        if tool_n_trial == "tool1 trial1":
            gonio_wrist_angle = gonio_wrist_angle-30
        elif tool_n_trial == "too1 trial2":
            gonio_wrist_angle = gonio_wrist_angle-35
        elif tool_n_trial == "too1 trial3":
            gonio_wrist_angle = gonio_wrist_angle-30
        elif tool_n_trial == "tool2 trial1":
            gonio_wrist_angle = gonio_wrist_angle-25
        elif tool_n_trial == "tool2 trial2":
            gonio_wrist_angle = gonio_wrist_angle-25
        elif tool_n_trial == "tool2 trial3":
            gonio_wrist_angle = gonio_wrist_angle-35
    elif person == 'p3':
        if tool_n_trial == "tool1 trial1":
            gonio_wrist_angle = gonio_wrist_angle-10
        elif tool_n_trial == "too1 trial2":
            gonio_wrist_angle = gonio_wrist_angle-30
        elif tool_n_trial == "too1 trial3":
            gonio_wrist_angle = gonio_wrist_angle-25
        elif tool_n_trial == "tool2 trial1":
            gonio_wrist_angle = gonio_wrist_angle-15
        elif tool_n_trial == "tool2 trial2":
            gonio_wrist_angle = gonio_wrist_angle-10
    elif person == 'p4':
        if tool_n_trial == "tool1 trial1":
            gonio_wrist_angle = gonio_wrist_angle+5
        elif tool_n_trial == "too1 trial2":
            gonio_wrist_angle = gonio_wrist_angle+5
        elif tool_n_trial == "tool2 trial1":
            gonio_wrist_angle = gonio_wrist_angle
        elif tool_n_trial == "tool2 trial2":
            gonio_wrist_angle = gonio_wrist_angle
    elif person == 'p5':
        if tool_n_trial == "tool1 trial1":
            gonio_wrist_angle = gonio_wrist_angle+5
        elif tool_n_trial == "too1 trial2":
            gonio_wrist_angle = gonio_wrist_angle+5
        elif tool_n_trial == "tool2 trial1":
            gonio_wrist_angle = gonio_wrist_angle+5
        elif tool_n_trial == "tool2 trial2":
            gonio_wrist_angle = gonio_wrist_angle+5
    elif person == 'p6':
        if tool_n_trial == "tool1 trial1":
            gonio_wrist_angle = gonio_wrist_angle+5
        elif tool_n_trial == "too1 trial2":
            gonio_wrist_angle = gonio_wrist_angle-15
        elif tool_n_trial == "tool2 trial1":
            gonio_wrist_angle = gonio_wrist_angle+10
        elif tool_n_trial == "tool2 trial2":
            gonio_wrist_angle = gonio_wrist_angle-10
    elif person == 'p7':
            if tool_n_trial == "tool1 trial1":
                gonio_wrist_angle = gonio_wrist_angle
            elif tool_n_trial == "too1 trial2":
                gonio_wrist_angle = gonio_wrist_angle
            elif tool_n_trial == "tool2 trial1":
                gonio_wrist_angle = gonio_wrist_angle
            elif tool_n_trial == "tool2 trial2":
                gonio_wrist_angle = gonio_wrist_angle
    return gonio_wrist_angle

def compute_RULA_score(data_bodypose,data_gonio):
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
        upper_arm_angle = angle_between(upper_arm_vec,z_vec)
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

    def wrist_score_from(gonio_1,gonio_2):
        # Wrist score from gonio_1
        # Wrist twist from gonio_2
        if gonio_1 > -3 and gonio_1 < 3:
            wrist_score = 1
        elif gonio_1 >= -15 and gonio_1 <=15:
            wrist_score = 2
        else:
            wrist_score = 3
        
        if gonio_2 > -30 and gonio_2 < 30:
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
        trunk_angle = angle_between(trunk_vec, z_vec)
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
    l_eye = np.array(data_bodypose[:,3:6])
    r_eye = np.array(data_bodypose[:,6:9])

    mid_eye = (l_eye+r_eye)/2

    l_shoulder = np.array(data_bodypose[:,15:18])
    r_shoulder = np.array(data_bodypose[:,18:21])
    mid_shoulder = np.array(data_bodypose[:,51:54])


    l_elbow = np.array(data_bodypose[:,21:24])  
    r_elbow = np.array(data_bodypose[:,24:27]) 
    
    l_wrist = np.array(data_bodypose[:,27:30])
    r_wrist = np.array(data_bodypose[:,30:33])

    l_hip = np.array(data_bodypose[:,33:36])
    r_hip = np.array(data_bodypose[:,36:39])

    l_gonio_1 = np.array(data_gonio[:,3]) 
    l_gonio_2 = np.array(data_gonio[:,2])
    r_gonio_1 = np.array(data_gonio[:,1])
    r_gonio_2 = np.array(data_gonio[:,0])

    l_RULA_scores = []
    r_RULA_scores = []
    total_load = 15
    leg_support = 1 # Legs are not supported
    # upper_arm_scores = []
    # lower_arm_scores = []
    # wrist_scores = []
    # wrist_twists = []
    # wrist_arm_scores = []
    # neck_scores = []
    # trunk_scores = []
    # leg_scores = []
    # neck_trunk_leg_scores = []

    for i in range(len(data_bodypose)):
        upper_arm_score = upper_arm_score_from(l_shoulder[i,:],l_elbow[i,:])
        # Adjustment
        if upper_arm_score > 1:
            upper_arm_score = upper_arm_score-1 # Arm is supported/person is leaning
        # upper_arm_scores.append(upper_arm_score)
        
        lower_arm_score = lower_arm_score_from(l_shoulder[i,:], l_elbow[i,:], l_wrist[i,:])
        #Adjustment
        # lower_arm_score = lower_arm_score + 1 # arm working across midline or to side
        # lower_arm_scores.append(lower_arm_score)
        
        
        wrist_score,wrist_twist = wrist_score_from(l_gonio_1[i], l_gonio_2[i])
        # wrist_scores.append(wrist_score)
        # wrist_twists.append(wrist_twist)
        
        tableA_score = tableA_posture_score_from(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
        
        arm_wrist_muscle_use_score = arm_wrist_muscle_use_score_from()
        
        force_load_score = force_load_score_from(total_load)
        
        wrist_arm_score = tableA_score + arm_wrist_muscle_use_score + force_load_score
        # wrist_arm_scores.append(wrist_arm_score)
        
        neck_score = neck_score_from(l_shoulder[i,:], r_shoulder[i,:],l_hip[i,:],r_hip[i,:],l_eye[i,:],r_eye[i,:])
        # neck_scores.append(neck_score)
        
        trunk_score  = trunk_score_from(l_hip[i,:], r_hip[i,:], mid_shoulder[i,:])
        # trunk_scores.append(trunk_score)
        
        leg_score = leg_score_from(leg_support)
        # leg_scores.append(leg_score)
        
        
        tableB_score = tableB_posture_score(neck_score, trunk_score, leg_score)
        
        leg_trunk_muscle_use_score = leg_trunk_muscle_use_score_from()
        
        force_load_score = force_load_score_from(total_load)
        
        neck_trunk_leg_score = tableB_score + leg_trunk_muscle_use_score + force_load_score
        # neck_trunk_leg_scores.append(neck_trunk_leg_score)
        
        tableC_score = tableC_posture_score(wrist_arm_score, neck_trunk_leg_score)
        
        l_RULA_scores.append(tableC_score)

    for i in range(len(data_bodypose)):
        upper_arm_score = upper_arm_score_from(r_shoulder[i,:],r_elbow[i,:])
        # Adjustment
        if upper_arm_score > 1:
            upper_arm_score = upper_arm_score-1 # Arm is supported/person is leaning
        # upper_arm_scores.append(upper_arm_score)
        
        lower_arm_score = lower_arm_score_from(r_shoulder[i,:], r_elbow[i,:], r_wrist[i,:])
        #Adjustment
        # lower_arm_score = lower_arm_score + 1 # arm working across midline or to side
        # lower_arm_scores.append(lower_arm_score)
        
        
        wrist_score,wrist_twist = wrist_score_from(r_gonio_1[i], r_gonio_2[i])
        # wrist_scores.append(wrist_score)
        # wrist_twists.append(wrist_twist)
        
        
        tableA_score = tableA_posture_score_from(upper_arm_score, lower_arm_score, wrist_score, wrist_twist)
        
        arm_wrist_muscle_use_score = arm_wrist_muscle_use_score_from()
        
        force_load_score = force_load_score_from(total_load)
        
        wrist_arm_score = tableA_score + arm_wrist_muscle_use_score + force_load_score
        # wrist_arm_scores.append(wrist_arm_score)
        
        neck_score = neck_score_from(l_shoulder[i,:], r_shoulder[i,:],l_hip[i,:],r_hip[i,:],l_eye[i,:],r_eye[i,:])
        # neck_scores.append(neck_score)
        
        trunk_score  = trunk_score_from(l_hip[i,:], r_hip[i,:], mid_shoulder[i,:])
        # trunk_scores.append(trunk_score)
        
        leg_score = leg_score_from(leg_support)
        # leg_scores.append(leg_score)
        
        
        tableB_score = tableB_posture_score(neck_score, trunk_score, leg_score)
        
        leg_trunk_muscle_use_score = leg_trunk_muscle_use_score_from()
        
        force_load_score = force_load_score_from(total_load)
        
        neck_trunk_leg_score = tableB_score + leg_trunk_muscle_use_score + force_load_score
        # neck_trunk_leg_scores.append(neck_trunk_leg_score)
        
        tableC_score = tableC_posture_score(wrist_arm_score, neck_trunk_leg_score)
        
        r_RULA_scores.append(tableC_score)
    # end_frame = -1
    # x_axis = np.linspace(0,len(data_bodypose[0])/60,len(data_bodypose[0]))
    # plt.figure(figsize =(15,6))
    # plt.plot(x_axis[:end_frame],RULA_scores[:end_frame],markersize =2)
    # plt.title('Instantaneous RULA score')

    # plt.xlabel('Time in seconds (s)')
    # plt.ylabel('RULA Score')
    # #%% Table B scores
    # fig, axs = plt.subplots(2,2,sharex=(True),sharey=(True))
    # axs[0,0].grid()
    # axs[0,1].grid()
    # axs[1,0].grid()
    # axs[1,1].grid()
    # plt.suptitle('Table B scores')
    # axs[0,0].plot(x_axis,neck_scores)
    # axs[0,0].axhline(y=4,color = 'r',linestyle = '-')
    # axs[0,0].set_title('Neck scores')
    # axs[0,1].plot(x_axis,trunk_scores)
    # axs[0,1].axhline(y=4,color = 'r',linestyle = '-')
    # axs[0,1].set_title('Trunk scores')
    # axs[1,0].plot(x_axis,leg_scores)
    # axs[1,0].axhline(y=2,color = 'r',linestyle = '-')
    # axs[1,0].set_title('Leg scores')
    # axs[1,1].plot(x_axis,neck_trunk_leg_scores)
    # axs[1,1].axhline(y=9,color = 'r',linestyle = '-')
    # axs[1,1].set_title('Combined scores')

    # #%%
    # fig, axs = plt.subplots(2,3,sharex=(True),sharey=(True))
    # axs[0,0].grid()
    # axs[0,1].grid()
    # axs[0,2].grid()
    # axs[1,0].grid()
    # axs[1,1].grid()
    # axs[1,2].grid()

    # plt.suptitle('Table A scores')
    # axs[0,0].plot(x_axis,upper_arm_scores)
    # axs[0,0].axhline(y=4,color = 'r',linestyle = '-')
    # axs[0,0].set_title('Upper Arm scores')
    # axs[0,1].plot(x_axis,lower_arm_scores)
    # axs[0,1].axhline(y=2,color = 'r',linestyle = '-')
    # axs[0,1].set_title('Lower Arm scores')
    # axs[1,0].plot(x_axis,wrist_scores)
    # axs[1,0].axhline(y=3,color = 'r',linestyle = '-')
    # axs[1,0].set_title('Wrist scores')
    # axs[1,1].plot(x_axis,wrist_twists)
    # axs[1,1].axhline(y=2,color = 'r',linestyle = '-')
    # axs[1,1].set_title('Wrist Twist Score') 
    # axs[0,2].plot(x_axis,wrist_arm_scores)
    # axs[0,2].axhline(y=9,color = 'r',linestyle = '-')
    # axs[0,2].set_title('Combined Score') 
    # axs[1,2].plot(x_axis,RULA_scores)
    # axs[1,2].set_title('RULA Score')     
    # #%% Debugging 
    return l_RULA_scores, r_RULA_scores

def compute_HAL_score(force,list_joints):
    overall_threshold = 44.8 # 10 lbs
    fingethreshold = 15 # 3.3 lbs
    window_length = 10 # In seconds
    D = 75 # Duty Cycle

    time = np.arange(0,len(force),1)/60
    time = np.round(time,2)

    # frame_handpose: elbow0,wrist1,palm_center2,Thumb345,Index6789,Middle10/11/12/13,Ring14/15/16/17,Pinky18/19/20/21
    list_joints = ["palm1","palm2","palm3","thumb1","thumb2","thumb_tip","index1","index2","index3","index_tip","middle1","middle2","middle3","ring1","ring2","ring3","little1","little2","little3"]

    palm_total_force = force[:,0]+force[:,1]+force[:,2]
    thumb_total_force = force[:,3]+force[:,4]+force[:,5]
    index_total_force = force[:,6]+force[:,7]+force[:,8]+force[:,9]
    middle_total_force = force[:,10]+force[:,11]+force[:,12]
    ring_total_force = force[:,13]+force[:,14]+force[:,15]
    little_total_force = force[:,16]+force[:,17]+force[:,18]
    # Round to two decimal places
    palm_total_force = np.round(palm_total_force,2)
    thumb_total_force = np.round(thumb_total_force,2)
    index_total_force = np.round(index_total_force,2)
    middle_total_force = np.round(middle_total_force,2)
    ring_total_force = np.round(ring_total_force,2)
    little_total_force = np.round(little_total_force,2)
    hand_data = [palm_total_force,thumb_total_force,index_total_force,middle_total_force,ring_total_force,little_total_force]
    hand_total_force = palm_total_force + thumb_total_force + index_total_force + middle_total_force + ring_total_force + little_total_force

    def continuous_windowed_frequency(time,hand_data,window_length,fingethreshold,overall_threshold):
        F_list = []
        start_window = 0
        end_window = np.abs(np.abs(time - 10)).argmin()
        while end_window < len(time):
            Exertions = 0
            for i in range(start_window,end_window):
                force_sum = 0
                single_fingeflag = 0
                #for j in range(len(hand_data)):
                for j in range(1,len(hand_data)): # Start from 1 to ignore palm data
                    force_sum += hand_data[j][i] 
                    if hand_data[j][i] > fingethreshold:
                        single_fingeflag = 1
                if force_sum > overall_threshold or single_fingeflag == 1:
                    Exertions += 1
            F_list.append(Exertions/window_length)
            start_window += 1
            end_window += 1

        return F_list

    def non_lineaHAL(F,D):
        if F>0:
            HAL = 6.56 * np.log(D)*((F**1.31)/(1 + 3.18*F**1.31))
        else:
            HAL = 0
        return HAL

    windowed_HAL = []
    F_list = continuous_windowed_frequency(time,hand_data,window_length,fingethreshold,overall_threshold)
    for F in F_list:
        HAL = non_lineaHAL(F,D)
        windowed_HAL.append(HAL)

    # Add a buffer to the start of HALs to make them the same length as the time vector
    windowed_HAL_plot = [0]*(len(time)-len(windowed_HAL)) + windowed_HAL
    return windowed_HAL,windowed_HAL_plot

def find_3scores(date_n_p,tool_n_trial,side):
    person = date_n_p.split()[-1]
    ext = "_l" if side == "left" else "_r"

    filename_force = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/processed "+date_n_p[-2:]+" "+tool_n_trial+" force_detail"+ext+".npy"
    filename_torque = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/processed "+date_n_p[-2:]+" "+tool_n_trial+" torque_detail"+ext+".npy"
    filename_gonio = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/processed "+date_n_p[-2:]+" "+tool_n_trial+" gonio.csv"
    filename_handpose = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/processed "+date_n_p[-2:]+" "+tool_n_trial+" handpose.csv"
    filename_bodypose = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/processed "+date_n_p[-2:]+" "+tool_n_trial+" bodypose.csv"
    filename_pose_exist = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/processed "+date_n_p[-2:]+" "+tool_n_trial+" pose_exist"+ext+".npy"

    if os.path.exists(filename_force) and os.path.exists(filename_handpose):
        print("working on",person,tool_n_trial,side)
        param_json = "C:/Data/JCATI handlayup/"+date_n_p+"/"+tool_n_trial+"/param.json"
        pose = Pose(param_json)
        pose.find_timeoverlap()

        pose_exist = np.load(filename_pose_exist)
        force = np.load(filename_force)
        torque = np.load(filename_torque)
        handpose = np.loadtxt(filename_handpose, delimiter=',')
        bodypose = np.loadtxt(filename_bodypose, delimiter=',')
        gonio = np.loadtxt(filename_gonio, delimiter=',')
        
        # align the data
        if len(gonio)==len(force)+1:
            gonio = gonio[:-1,:]
        if len(handpose)==len(force)+1:
            handpose = handpose[:-1,:]
        if len(bodypose)==len(force)+1:
            bodypose = bodypose[:-1,:]
        if len(gonio)!=len(force):
            print(person,tool_n_trial,"goino:",len(gonio),"force:",len(force),"torque:",len(torque),"handpose:",len(handpose),"pose_exist:",len(pose_exist))
            return None,None,None,None

        # correct gonio
        gonio[:,3] = correct_gonio(gonio[:,3],person,tool_n_trial)
        gonio[:,1] = correct_gonio(gonio[:,1],person,tool_n_trial)
        gonio[gonio > 90] = 90
        gonio[gonio < -90] = -90

        # compute BACH score
        if ext == "_l":
            gonio_wrist_angle = gonio[:,3]
        else:
            gonio_wrist_angle = gonio[:,1]

        wrist_factor = 11.85/moment_lookup(gonio_wrist_angle, params_line, params_poly, S)

        torque_outlier_val = data_combined_1[person][side]["torque_outlier_val"]
        torque_mean = data_combined_1[person][side]["torque_mean"]
        
        torque = np.sum(torque, axis=1)
        torque[torque > torque_outlier_val] = torque_outlier_val
        torque_normalized = torque/torque_mean
        score_BACH = torque_normalized*wrist_factor

        # compure RULA score
        l_RULA_score, r_RULA_score = compute_RULA_score(bodypose, gonio)
        if side == "left":
            score_RULA = l_RULA_score
        else:
            score_RULA = r_RULA_score

        # compute HAL score
        windowed_HAL,score_HAL = compute_HAL_score(force,list_joints)
        
        start_time = pose.sensors_info["official_start_in_video"]
        time_range = np.arange(0,len(score_BACH),1)/60+start_time

        return score_RULA, score_HAL, score_BACH, time_range
    else:
        return None,None,None,None

#%% load data
with open("C:/Users/69516/OneDrive/research/JCATI/wrist torque computation/data_subject.pickle", "rb") as pickle_file:
    data_original = pickle.load(pickle_file)

list_subject = ["p1","p2","p3","p4","p5","p6","p7"]
list_joints = ["palm1","palm2","palm3","thumb1","thumb2","thumb_tip","index1","index2","index3","index_tip","middle1","middle2","middle3","ring1","ring2","ring3","little1","little2","little3"]
list_date_n_pers = ["May3 p1",
          "May3 p2",
          "May3 p3",
          "June9 p4",
          "June9 p5",
          "June9 p6",
          "June9 p7"]
list_tool_n_trail = ["tool1 trial1",
           "tool1 trial2",
           "tool1 trial3",
           "tool2 trial1",
           "tool2 trial2",
           "tool2 trial3",]

list_date_n_pers = ["May3 p2"]
list_tool_n_trail = ["tool2 trial1"]

#%% combine data
data_combined_1 = {}
for person in list_subject:
    data_combined_1[person] = {}
    for side in ["left","right"]:
        data_combined_1[person][side] = {}
        for type in ["torque","torque_mean","force","wrist_factor"]:
            data_combined_1[person][side][type] = np.array([])

#%% preprocess data
for person in list_subject:
    for side in ["left","right"]:
        for tool in ["tool1","tool2"]:
            # pose exist
            pose_exist = data_original[person][side][tool]["palm1"]["pose_exist"] 

            # wrist factor
            gonio_wrist_angle = data_original[person][side][tool]["palm1"]["gonio_wrist_angle"]
            gonio_wrist_angle[gonio_wrist_angle > 90] = 90
            gonio_wrist_angle[gonio_wrist_angle < -90] = -90
            wrist_factor = 1/moment_lookup(gonio_wrist_angle, params_line, params_poly, S)
            data_combined_1[person][side]["wrist_factor"] = np.concatenate([data_combined_1[person][side]["wrist_factor"], wrist_factor])

            # combine force and torque from all joints, eliminate zero data for data_combined_1
            for type in ["torque","force"]:
                arr = np.zeros_like(data_original[person][side][tool]["palm1"][type])
                arr_palm = np.zeros_like(data_original[person][side][tool]["palm1"][type])
                arr_finger = np.zeros_like(data_original[person][side][tool]["palm1"][type])
                for joint in list_joints:
                    arr += data_original[person][side][tool][joint][type]
                    if "palm" in joint:
                        arr_palm += data_original[person][side][tool][joint][type]
                    else:
                        arr_finger += data_original[person][side][tool][joint][type]
                
                arr = [val for val, mask_val in zip(arr, pose_exist) if mask_val == 1]
                data_combined_1[person][side][type] = np.concatenate([data_combined_1[person][side][type], arr])

            torque = data_combined_1[person][side]["torque"]
            # remove torque outlier
            num_bins = 100 
            counts, bins = np.histogram(torque,bins=num_bins)
            bin_midpoints = (bins[:-1] + bins[1:]) / 2
            counts_percent = counts/np.sum(counts)

            sum = 0
            outlier_ratio = 0.01
            for idx in range(len(counts)):
                sum += counts[num_bins-1-idx]
                if sum >= np.sum(counts)*outlier_ratio:
                    break
                else:
                    outlier_thresh = (bins[num_bins-idx]+bins[num_bins-idx-1])/2
            data_combined_1[person][side]["torque_outlier_val"] = outlier_thresh
            data_combined_1[person][side]["torque_mean"] = np.mean(torque)
    

#%% visualize data
for date_n_p in list_date_n_pers:
    
    for tool_n_trial in list_tool_n_trail:

        score_RULA_l, score_HAL_l, score_BACH_l, time_range = find_3scores(date_n_p,tool_n_trial,"left")
        score_RULA_r, score_HAL_r, score_BACH_r, time_range = find_3scores(date_n_p,tool_n_trial,"right")

        # make two subplots, one up, one down
        fig, axs = plt.subplots(2, 1, figsize=(15, 6))
        # plot BACH score, RULA score, HAL score
        axs[0].plot(time_range,score_BACH_l,'-',alpha=1)
        axs[0].plot(time_range,score_RULA_l,'-',alpha=1)
        axs[0].plot(time_range,score_HAL_l,'-',alpha=1)
        axs[0].legend(["BACH(Left)","RULA(Left)","HAL(Left)"], fontsize=13)
        axs[0].set_ylim([-1,10])

        axs[1].plot(time_range,score_BACH_r,'-')
        axs[1].plot(time_range,score_RULA_r,'-')
        axs[1].plot(time_range,score_HAL_r,'-')
        axs[1].legend(["BACH(Right)","RULA(Right)","HAL(Right)"], fontsize=13)
        axs[1].set_ylim([-1,10])

        plt.show()
        plt.close()
        _ = 0