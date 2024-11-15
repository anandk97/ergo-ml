import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from numpy.linalg import svd
from PIL import Image
import pickle

path_main = "D:/Data/JCATI handlayup/data collection"

list_joints = ["palm1","palm2","palm3","thumb1","thumb2","thumb_tip","index1","index2","index3","index_tip","middle1","middle2","middle3","ring1","ring2","ring3","little1","little2","little3"]
list_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

def handpose2wrist_angle(handpose):
    # handpose: Nx22x3
    # elbow0,wrist1,palm_center2,Thumb123,Index1234,Middle1234,Ring1234,Pinky1234 
    wrist_angle = np.zeros((handpose.shape[0]))
    for i in range(len(handpose)):
        vector_A = handpose[i,1,:]-handpose[i,0,:]
        points = np.vstack((handpose[i,1,:],handpose[i,2,:],handpose[i,3,:],handpose[i,6,:],handpose[i,10,:],handpose[i,14,:],handpose[i,18,:]))
        centered_points = points - np.mean(points, axis=0)

        _, _, vh = svd(centered_points)
        vector_B = vh[-1] # last row of vh matrix is the normal vector of the plane

        # Calculate the dot product
        dot_product = np.dot(vector_A, vector_B)

        # Calculate the magnitudes (norms) of the vectors
        norm_A = np.linalg.norm(vector_A)
        norm_B = np.linalg.norm(vector_B)

        # Calculate the angle in radians
        angle_rad = np.arccos(dot_product / (norm_A * norm_B))

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 90:
            wrist_angle[i] = angle_deg-90
        else:
            wrist_angle[i] = 90-angle_deg
    return wrist_angle

def reconstruct_handpose(handpose):
    num_cols = handpose.shape[1]

    # Calculate the column indices for left, middle, and right portions
    left_cols = num_cols // 3
    middle_cols = 2 * (num_cols // 3)

    handpose_x = handpose[:, :left_cols]
    handpose_y = handpose[:, left_cols:middle_cols]
    handpose_z = handpose[:, middle_cols:]

    handpose = np.dstack((handpose_x, handpose_y, handpose_z))
    return handpose

def img_curve_fit():
    img = Image.open("flexion moment curve.png")
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

    # Plotting the data and the fitted curves
    plt.scatter(X, Y, color='orange',alpha=1, label='Data Points')
    plt.plot(X[:i+2], linear(X[:i+2], *params_line), 'b--', label='Fitted Line')
    plt.plot(X[i-2:], polynomial(X[i-2:], *params_poly), 'r--', label='Fitted Polynomial')
    plt.axvline(x=S, color='gray', linestyle='--')
    plt.legend()
    # change the font size of the label to 20
    plt.xlabel('Flexion Angle θ (degree)', fontsize=12)
    plt.ylabel('Flexion Moment (Nm)', fontsize=12)
    plt.title('Curve Fitting', fontweight='bold')
    plt.show()

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

# assign all data
data = {}
for path_participant in os.listdir(path_main):
    person = "p"+path_participant.split(" ")[-1]
    data[person] = {}
    for side in ["left","right"]:
        data[person][side] = {}
        for tool in ["tool1","tool2"]:
            data[person][side][tool] = {}
            for joint in list_joints:
                data[person][side][tool][joint] = {}
                for type in ["torque","force","gonio_wrist_angle","pose_exist"]:
                    data[person][side][tool][joint][type] = np.array([])

# go through all the data to collect the torque info
for path_participant in os.listdir(path_main):
    person = "p"+path_participant.split(" ")[-1]
    for side in ["left","right"]:
        ext = "_l" if side == "left" else "_r"
        
        list_subfolders = os.listdir(os.path.join(path_main,path_participant))
        list_tool_n_trail = [item for item in list_subfolders if "trial" in item]
        for tool_n_trial in list_tool_n_trail:
            filename_force = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" force_detail"+ext+".npy")
            filename_torque = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" torque_detail"+ext+".npy")
            filename_gonio = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" gonio.csv")
            filename_handpose = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" handpose.csv")
            filename_pose_exist = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" pose_exist"+ext+".npy")

            if os.path.exists(filename_force) and os.path.exists(filename_handpose):
                pose_exist = np.load(filename_pose_exist)
                force = np.load(filename_force)
                torque = np.load(filename_torque)
                handpose = np.loadtxt(filename_handpose, delimiter=',')
                gonio = np.loadtxt(filename_gonio, delimiter=',')
                if len(gonio)==len(force)+1:
                    gonio = gonio[:-1,:]
                if len(handpose)==len(force)+1:
                    handpose = handpose[:-1,:]
                if len(pose_exist)==len(force)+1:
                    pose_exist = pose_exist[:-1]
                if len(gonio)!=len(force):
                    print("    ",person,tool_n_trial,"goino:",len(gonio),"force:",len(force),"torque:",len(torque),"handpose:",len(handpose),"pose_exist:",len(pose_exist))
                    continue

                if ext == "_l":
                    gonio_wrist_angle = gonio[:,3]
                else:
                    gonio_wrist_angle = gonio[:,1]
                    
                for idx_joint in range(len(list_joints)):
                    joint = list_joints[idx_joint]
                    if "tool1" in tool_n_trial:
                        data[person][side]["tool1"][joint]["torque"] = np.append(data[person][side]["tool1"][joint]["torque"], torque[:,idx_joint])
                        data[person][side]["tool1"][joint]["force"] = np.append(data[person][side]["tool1"][joint]["force"], force[:,idx_joint])
                        data[person][side]["tool1"][joint]["gonio_wrist_angle"] = np.append(data[person][side]["tool1"][joint]["gonio_wrist_angle"], gonio_wrist_angle)
                        data[person][side]["tool1"][joint]["pose_exist"] = np.append(data[person][side]["tool1"][joint]["pose_exist"], pose_exist)
                    else:
                        data[person][side]["tool2"][joint]["torque"] = np.append(data[person][side]["tool2"][joint]["torque"], torque[:,idx_joint])
                        data[person][side]["tool2"][joint]["force"] = np.append(data[person][side]["tool2"][joint]["force"], force[:,idx_joint])
                        data[person][side]["tool2"][joint]["gonio_wrist_angle"] = np.append(data[person][side]["tool2"][joint]["gonio_wrist_angle"], gonio_wrist_angle)
                        data[person][side]["tool2"][joint]["pose_exist"] = np.append(data[person][side]["tool2"][joint]["pose_exist"], pose_exist)

# use the collect torque info to compute the score
for path_participant in os.listdir(path_main):
    person = "p"+path_participant.split(" ")[-1]
    print("working on ",person," BACH score")
    
    dict_torque_upper_bound = {"left":{},"right":{}}
    dict_torque_median = {"left":{},"right":{}}
    
    # compute the median and upper bound of the torque for current side of hand
    for side in ["left","right"]:
        ext = "_l" if side == "left" else "_r"

        torque_hand_t1 = np.zeros((len(data[person][side]["tool1"]["palm1"]["torque"])))
        torque_hand_t2 = np.zeros((len(data[person][side]["tool2"]["palm1"]["torque"])))
        
        for idx_joint in range(len(list_joints)):
            joint = list_joints[idx_joint]
            torque_hand_t1 += data[person][side]["tool1"][joint]["torque"]
            torque_hand_t2 += data[person][side]["tool2"][joint]["torque"]

        torque_hand = np.concatenate((torque_hand_t1, torque_hand_t2))
        torque_hand = torque_hand[torque_hand != 0]
        
        q1 = np.percentile(torque_hand, 25)
        q3 = np.percentile(torque_hand, 75)
        iqr = q3 - q1
        lower_bound = q1 - 5 * iqr
        upper_bound = q3 + 5 * iqr
        torque_hand[torque_hand > upper_bound] = upper_bound
        
        dict_torque_upper_bound[side] = upper_bound
        dict_torque_median[side] = np.median(torque_hand)
    
    # compute the BACHscore
    list_subfolders = os.listdir(os.path.join(path_main,path_participant))
    list_tool_n_trail = [item for item in list_subfolders if "trial" in item]
    for tool_n_trial in list_tool_n_trail:
        
        score_l, score_r = None, None
                    
        for side in ["left","right"]:
            ext = "_l" if side == "left" else "_r"
            
            filename_force = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" force_detail"+ext+".npy")
            filename_torque = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" torque_detail"+ext+".npy")
            filename_gonio = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" gonio.csv")
            filename_handpose = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" handpose.csv")
            filename_pose_exist = os.path.join(path_main,path_participant,tool_n_trial,"processed "+person+" "+tool_n_trial+" pose_exist"+ext+".npy")

            if os.path.exists(filename_force) and os.path.exists(filename_handpose):
                pose_exist = np.load(filename_pose_exist)
                force = np.load(filename_force)
                torque = np.load(filename_torque)
                handpose = np.loadtxt(filename_handpose, delimiter=',')
                gonio = np.loadtxt(filename_gonio, delimiter=',')
                if len(gonio)==len(force)+1:
                    gonio = gonio[:-1,:]
                if len(handpose)==len(force)+1:
                    handpose = handpose[:-1,:]
                if len(pose_exist)==len(force)+1:
                    pose_exist = pose_exist[:-1]
                if len(gonio)!=len(force):
                    print("    ",person,tool_n_trial,"goino:",len(gonio),"force:",len(force),"torque:",len(torque),"handpose:",len(handpose),"pose_exist:",len(pose_exist))
                    continue

                if ext == "_l":
                    gonio_wrist_angle = gonio[:,3]
                else:
                    gonio_wrist_angle = gonio[:,1]
                    
                # compute the wrist factor by the max moment over current moment
                wrist_factor = 11.8/moment_lookup(gonio_wrist_angle, params_line, params_poly, S)

                # normalize the torque
                torque_hand = np.sum(torque, axis=1)
                torque_upper_bound = dict_torque_upper_bound[side]
                torque_hand[torque_hand > torque_upper_bound] = torque_upper_bound
                median = dict_torque_median[side]
                torque_normalized = torque_hand/median
                score = torque_normalized*wrist_factor
                score = score*pose_exist.astype(int)
                
                if side=="left":
                    score_l = score
                elif side=="right":
                    score_r = score

        if score_l is not None and score_r is not None:
            combined_scores = np.vstack((score_l, score_r)).T
            filename_combined_score = os.path.join(path_main, path_participant, tool_n_trial, person +" "+ tool_n_trial + " w lr BACH.csv")
            np.savetxt(filename_combined_score, combined_scores, delimiter=",")
            
    
with open("data_subject.pickle", "wb") as pickle_file:
    pickle.dump(data, pickle_file)
        





        

        



        





            


