#%% Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
#%% Functions
def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle = angle_radians*(180/np.pi)
    return angle
def create_dataset(train_data):
    l_eye = np.array(train_data.iloc[:,3:6])
    r_eye = np.array(train_data.iloc[:,6:9])

    mid_eye = (l_eye+r_eye)/2

    l_shoulder = np.array(train_data.iloc[:,15:18])
    r_shoulder = np.array(train_data.iloc[:,18:21])
    mid_shoulder = np.array(train_data.iloc[:,51:54])

    l_elbow = np.array(train_data.iloc[:,21:24])
    r_elbow = np.array(train_data.iloc[:,24:27])

    l_wrist = np.array(train_data.iloc[:,27:30])
    r_wrist = np.array(train_data.iloc[:,30:33])

    l_hip = np.array(train_data.iloc[:,33:36])
    r_hip = np.array(train_data.iloc[:,36:39])

    l_gonio_1 = np.array(train_data.iloc[:,54])
    l_gonio_2 = np.array(train_data.iloc[:,55])
    r_gonio_1 = np.array(train_data.iloc[:,56])
    r_gonio_2 = np.array(train_data.iloc[:,57])

    l_RULA_scores = np.array(train_data.iloc[:,58])
    r_RULA_scores = np.array(train_data.iloc[:,59])
    l_shoulder_elbow = l_elbow-l_shoulder
    down_vec = np.array([0,0,-1])
    upper_arm_angle = np.array([angle_between(l_shoulder_elbow[idx],down_vec) for idx in range(len(l_shoulder_elbow))])
    l_elbow_wrist = l_wrist-l_elbow
    lower_arm_angle = np.array([angle_between(l_shoulder_elbow[idx],l_elbow_wrist[idx]) for idx in range(len(l_elbow_wrist))])
    world_z_vec = np.array([0,0,1])
    mid_hip = (l_hip+r_hip)/2
    z_vec = mid_shoulder-mid_hip
    neck_vec = mid_eye-mid_shoulder
    angle_compensate = 10
    neck_angle = np.array([angle_between(z_vec[idx],neck_vec[idx])-angle_compensate for idx in range(len(z_vec))])
    trunk_vec = mid_shoulder - mid_hip
    up_vec = np.array([0,0,1])
    trunk_angle = np.array([angle_between(trunk_vec[idx],up_vec) for idx in range(len(trunk_vec))])
    # l_X_train = np.hstack((l_shoulder,r_shoulder,mid_shoulder,l_elbow,l_wrist,l_hip,r_hip,l_eye,r_eye,l_gonio_1.reshape(-1,1),l_gonio_2.reshape(-1,1)))
    # Only the angles
    l_X_train = np.hstack((upper_arm_angle.reshape(-1,1),lower_arm_angle.reshape(-1,1),neck_angle.reshape(-1,1),trunk_angle.reshape(-1,1),l_gonio_1.reshape(-1,1),l_gonio_2.reshape(-1,1)))
    l_y_train = l_RULA_scores
    # Convert to int
    l_y_train = l_y_train.astype(int)
    # Remove rows with RULA score 0,1 and 2
    l_X_train = l_X_train[l_y_train>2]
    l_y_train = l_y_train[l_y_train>2]

    # Create r_X_train and r_y_train from right hand angles
    r_shoulder_elbow = r_elbow-r_shoulder
    down_vec = np.array([0,0,-1])
    upper_arm_angle = np.array([angle_between(r_shoulder_elbow[idx],down_vec) for idx in range(len(r_shoulder_elbow))])
    r_elbow_wrist = r_wrist-r_elbow
    lower_arm_angle = np.array([angle_between(r_shoulder_elbow[idx],r_elbow_wrist[idx]) for idx in range(len(r_elbow_wrist))])
    world_z_vec = np.array([0,0,1])
    mid_hip = (l_hip+r_hip)/2
    z_vec = mid_shoulder-mid_hip
    neck_vec = mid_eye-mid_shoulder
    angle_compensate = 10
    neck_angle = np.array([angle_between(z_vec[idx],neck_vec[idx])-angle_compensate for idx in range(len(z_vec))])
    trunk_vec = mid_shoulder - mid_hip
    up_vec = np.array([0,0,1])
    trunk_angle = np.array([angle_between(trunk_vec[idx],up_vec) for idx in range(len(trunk_vec))])
    # r_X_train = np.hstack((r_shoulder,r_elbow,r_wrist,r_eye,r_gonio_1.reshape(-1,1),r_gonio_2.reshape(-1,1)))
    # Only the angles
    r_X_train = np.hstack((upper_arm_angle.reshape(-1,1),lower_arm_angle.reshape(-1,1),neck_angle.reshape(-1,1),trunk_angle.reshape(-1,1),r_gonio_1.reshape(-1,1),r_gonio_2.reshape(-1,1)))
    r_y_train = r_RULA_scores

    # Convert to int
    r_y_train = r_y_train.astype(int)
    # Remove rows with RULA score 0,1 and 2
    r_X_train = r_X_train[r_y_train>2]
    r_y_train = r_y_train[r_y_train>2]
    return l_X_train,l_y_train,r_X_train,r_y_train
# Store data for 6 training participants and 1 test participant and repeat for all 8 participants

#%% Create dataframes
right_stringer_data = np.zeros((15,8))
right_camel_hump_data = np.zeros((15,8))
left_stringer_data = np.zeros((15,8))
left_camel_hump_data = np.zeros((15,8))
for i in range(1,16):
    train_data_tool1 = pd.DataFrame()
    train_data_tool2 = pd.DataFrame()
    for j in range(1,16):
        os.chdir(r"C:\Users\anand\Desktop\RULA Labelled Data")
        extension = 'csv'
        if j==i:
            # Indices of tool split are 1-44016, 2-44442, 3-36380, 4-50676,5-50421,6-46612,7-48997
            #
            # Read all csv files in the folder C:\Users\anand\Desktop\RULA\Labelled Data\participant_i
            tool1_filenames = [k for k in glob.glob('p'+str(j)+' tool1*.{}'.format(extension))]
            tool2_filenames = [k for k in glob.glob('p'+str(j)+' tool2*.{}'.format(extension))]
            # Combine all files in the list
            test_data_tool1 = pd.concat([pd.read_csv(tool1_filenames[f_idx], header=None) for f_idx in range(len(tool1_filenames))], ignore_index=True)
            test_data_tool2 = pd.concat([pd.read_csv(tool2_filenames[f_idx], header=None) for f_idx in range(len(tool2_filenames))], ignore_index=True)
            # Drop NaN values
            test_data_tool1.dropna(inplace=True)
            continue
        
        tool1_filenames = [k for k in glob.glob('p'+str(j)+' tool1*.{}'.format(extension))]
        tool2_filenames = [k for k in glob.glob('p'+str(j)+' tool2*.{}'.format(extension))]
        # Combine all files in the list

        current_train_data_tool1 = pd.concat([pd.read_csv(tool1_filenames[f_idx], header=None) for f_idx in range(len(tool1_filenames))], ignore_index=True)
        current_train_data_tool2 = pd.concat([pd.read_csv(tool2_filenames[f_idx], header=None) for f_idx in range(len(tool2_filenames))], ignore_index=True)
        # Concatenate new_data to train_data along the rows

        train_data_tool1 = pd.concat([train_data_tool1,current_train_data_tool1],ignore_index=True)
        train_data_tool2 = pd.concat([train_data_tool2,current_train_data_tool2],ignore_index=True)
        # Drop NaN values
        train_data_tool1.dropna(inplace=True)
        train_data_tool2.dropna(inplace=True)
        

    # Create dataset for training
    l_X_train_tool1,l_y_train_tool1,r_X_train_tool1,r_y_train_tool1 = create_dataset(train_data_tool1)
    l_X_train_tool2,l_y_train_tool2,r_X_train_tool2,r_y_train_tool2 = create_dataset(train_data_tool2)
    
    l_X_test_tool1,l_y_test_tool1,r_X_test_tool1,r_y_test_tool1 = create_dataset(test_data_tool1)
    l_X_test_tool2,l_y_test_tool2,r_X_test_tool2,r_y_test_tool2 = create_dataset(test_data_tool2)


    l_sc = StandardScaler()

    l_classifier = xgb.XGBClassifier(tree_method='gpu_hist',verbosity = 2,sampling_method='gradient_based',random_state=0)

    l_y_train_tool1 = l_y_train_tool1-3
    l_y_train_tool2 = l_y_train_tool2-3
    param_dist = {'l_classifier__n_estimators' :[29],'l_classifier__max_depth':[6],'l_classifier__learning_rate':[0.18],'l_classifier__gamma':[0.12],'l_classifier__min_child_weight':[1],'l_classifier__max_leaves':[27]}

    pipe = Pipeline(steps=[('scaler',l_sc),('l_classifier', l_classifier)]) 

    grid_pipe = GridSearchCV(pipe, param_grid=param_dist, cv=5, n_jobs=-2, verbose=1,scoring='f1_weighted')
    # Combine training data for left hand stringer and left hand camel hump
    l_X_train = np.vstack((l_X_train_tool1,l_X_train_tool2))
    l_y_train = np.hstack((l_y_train_tool1,l_y_train_tool2))
    grid_pipe.fit(l_X_train, l_y_train)


    print(grid_pipe.best_params_) 
    print("Left Hand Training done for participant "+str(i))


    l_X_test_tool1 = l_X_test_tool1[~np.isnan(l_X_test_tool1).any(axis=1)]
    l_y_test_tool1 = l_y_test_tool1[~np.isnan(l_y_test_tool1)]


    l_y_pred_tool1 = grid_pipe.predict(l_X_test_tool1)
    l_y_pred_tool1 = l_y_pred_tool1+3
    # 1-3 low risk, 4-5 medium risk, 6-7 high risk

    # Report as a table for each participant left stringer and left camel hump
    # Create new variables for l_y_test and l_y_pred for each tool based on risk level
    l_y_test_risk  = np.zeros(np.shape(l_y_test_tool1))
    l_y_pred_risk = np.zeros(np.shape(l_y_pred_tool1))
    l_y_test_risk[l_y_test_tool1<=3] = 1
    l_y_test_risk[(l_y_test_tool1>3) & (l_y_test_tool1<=5)] = 2
    l_y_test_risk[l_y_test_tool1>5] = 3
    l_y_pred_risk[l_y_pred_tool1<=3] = 1
    l_y_pred_risk[(l_y_pred_tool1>3) & (l_y_pred_tool1<=5)] = 2
    l_y_pred_risk[l_y_pred_tool1>5] = 3

    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    l_accurate = 0
    l_true_low_pred_med = 0
    l_true_low_pred_high = 0
    l_true_med_pred_high = 0
    l_true_med_pred_low = 0
    l_true_high_pred_med = 0
    l_true_high_pred_low = 0
    for idx in range(len(l_y_test_risk)):
        if l_y_test_risk[idx] == l_y_pred_risk[idx]:
            l_accurate += 1
        elif l_y_test_risk[idx] == 1 and l_y_pred_risk[idx] == 2:
            l_true_low_pred_med += 1
        elif l_y_test_risk[idx] == 1 and l_y_pred_risk[idx] == 3:
            l_true_low_pred_high += 1
        elif l_y_test_risk[idx] == 2 and l_y_pred_risk[idx] == 3:
            l_true_med_pred_high += 1
        elif l_y_test_risk[idx] == 2 and l_y_pred_risk[idx] == 1:
            l_true_med_pred_low += 1
        elif l_y_test_risk[idx] == 3 and l_y_pred_risk[idx] == 2:
            l_true_high_pred_med += 1
        elif l_y_test_risk[idx] == 3 and l_y_pred_risk[idx] == 1:
            l_true_high_pred_low += 1
    left_stringer_data[i-1,:] = [i-1,l_accurate/len(l_y_test_risk),l_true_low_pred_med/len(l_y_test_risk),l_true_low_pred_high/len(l_y_test_risk),l_true_med_pred_high/len(l_y_test_risk),l_true_med_pred_low/len(l_y_test_risk),l_true_high_pred_med/len(l_y_test_risk),l_true_high_pred_low/len(l_y_test_risk)]


    l_X_test_tool2 = l_X_test_tool2[~np.isnan(l_X_test_tool2).any(axis=1)]
    l_y_test_tool2 = l_y_test_tool2[~np.isnan(l_y_test_tool2)]


    l_y_pred_tool2 = grid_pipe.predict(l_X_test_tool2)
    l_y_pred_tool2 = l_y_pred_tool2+3
    # 1-3 low risk, 4-5 medium risk, 6-7 high risk

    # Report as a table for each participant left stringer and left camel hump
    # Create new variables for l_y_test and l_y_pred for each tool based on risk level
    l_y_test_risk  = np.zeros(np.shape(l_y_test_tool2))
    l_y_pred_risk = np.zeros(np.shape(l_y_pred_tool2))
    l_y_test_risk[l_y_test_tool2<=3] = 1
    l_y_test_risk[(l_y_test_tool2>3) & (l_y_test_tool2<=5)] = 2
    l_y_test_risk[l_y_test_tool2>5] = 3
    l_y_pred_risk[l_y_pred_tool2<=3] = 1
    l_y_pred_risk[(l_y_pred_tool2>3) & (l_y_pred_tool2<=5)] = 2
    l_y_pred_risk[l_y_pred_tool2>5] = 3

    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    l_accurate = 0
    l_true_low_pred_med = 0
    l_true_low_pred_high = 0
    l_true_med_pred_high = 0
    l_true_med_pred_low = 0
    l_true_high_pred_med = 0
    l_true_high_pred_low = 0
    for idx in range(len(l_y_test_risk)):
        if l_y_test_risk[idx] == l_y_pred_risk[idx]:
            l_accurate += 1
        elif l_y_test_risk[idx] == 1 and l_y_pred_risk[idx] == 2:
            l_true_low_pred_med += 1
        elif l_y_test_risk[idx] == 1 and l_y_pred_risk[idx] == 3:
            l_true_low_pred_high += 1
        elif l_y_test_risk[idx] == 2 and l_y_pred_risk[idx] == 3:
            l_true_med_pred_high += 1
        elif l_y_test_risk[idx] == 2 and l_y_pred_risk[idx] == 1:
            l_true_med_pred_low += 1
        elif l_y_test_risk[idx] == 3 and l_y_pred_risk[idx] == 2:
            l_true_high_pred_med += 1
        elif l_y_test_risk[idx] == 3 and l_y_pred_risk[idx] == 1:
            l_true_high_pred_low += 1

    left_camel_hump_data[i-1,:] = [i-1,l_accurate/len(l_y_test_risk),l_true_low_pred_med/len(l_y_test_risk),l_true_low_pred_high/len(l_y_test_risk),l_true_med_pred_high/len(l_y_test_risk),l_true_med_pred_low/len(l_y_test_risk),l_true_high_pred_med/len(l_y_test_risk),l_true_high_pred_low/len(l_y_test_risk)]

    print("Left Hand Testing done for participant "+str(i))

    # Repeat for right hand
    r_sc = StandardScaler()

    r_classifier = xgb.XGBClassifier(tree_method='gpu_hist',verbosity = 2,sampling_method='gradient_based',random_state=0)

    r_y_train_tool1 = r_y_train_tool1-3
    r_y_train_tool2 = r_y_train_tool2-3
    param_dist = {'r_classifier__n_estimators' :[29],'r_classifier__max_depth':[6],'r_classifier__learning_rate':[0.18],'r_classifier__gamma':[0.12],'r_classifier__min_child_weight':[1],'r_classifier__max_leaves':[27]}
    pipe = Pipeline(steps=[('scaler',r_sc),('r_classifier', r_classifier)])

    grid_pipe = GridSearchCV(pipe, param_grid=param_dist, cv=5, n_jobs=-2, verbose=1,scoring='f1_weighted')
    # Combine training data for left hand stringer and left hand camel hump
    r_X_train = np.vstack((r_X_train_tool1,r_X_train_tool2))
    r_y_train = np.hstack((r_y_train_tool1,r_y_train_tool2))
    grid_pipe.fit(r_X_train, r_y_train)

    print(grid_pipe.best_params_)
    print("Right Hand Training done for participant "+str(i))

    r_X_test_tool1 = r_X_test_tool1[~np.isnan(r_X_test_tool1).any(axis=1)]
    r_y_test_tool1 = r_y_test_tool1[~np.isnan(r_y_test_tool1)]

    r_y_pred_tool1 = grid_pipe.predict(r_X_test_tool1)
    r_y_pred_tool1 = r_y_pred_tool1+3
    # 1-3 low risk, 4-5 medium risk, 6-7 high risk

    # Report as a table for each participant left stringer and left camel hump
    # Create new variables for l_y_test and l_y_pred for each tool based on risk level
    r_y_test_risk  = np.zeros(np.shape(r_y_test_tool1))
    r_y_pred_risk = np.zeros(np.shape(r_y_pred_tool1))
    r_y_test_risk[r_y_test_tool1<=3] = 1
    r_y_test_risk[(r_y_test_tool1>3) & (r_y_test_tool1<=5)] = 2
    r_y_test_risk[r_y_test_tool1>5] = 3
    r_y_pred_risk[r_y_pred_tool1<=3] = 1
    r_y_pred_risk[(r_y_pred_tool1>3) & (r_y_pred_tool1<=5)] = 2
    r_y_pred_risk[r_y_pred_tool1>5] = 3

    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    r_accurate = 0
    r_true_low_pred_med = 0
    r_true_low_pred_high = 0
    r_true_med_pred_high = 0
    r_true_med_pred_low = 0
    r_true_high_pred_med = 0
    r_true_high_pred_low = 0
    for idx in range(len(r_y_test_risk)):
        if r_y_test_risk[idx] == r_y_pred_risk[idx]:
            r_accurate += 1
        elif r_y_test_risk[idx] == 1 and r_y_pred_risk[idx] == 2:
            r_true_low_pred_med += 1
        elif r_y_test_risk[idx] == 1 and r_y_pred_risk[idx] == 3:
            r_true_low_pred_high += 1
        elif r_y_test_risk[idx] == 2 and r_y_pred_risk[idx] == 3:
            r_true_med_pred_high += 1
        elif r_y_test_risk[idx] == 2 and r_y_pred_risk[idx] == 1:
            r_true_med_pred_low += 1
        elif r_y_test_risk[idx] == 3 and r_y_pred_risk[idx] == 2:
            r_true_high_pred_med += 1
        elif r_y_test_risk[idx] == 3 and r_y_pred_risk[idx] == 1:
            r_true_high_pred_low += 1
    right_stringer_data[i-1,:] = [i-1,r_accurate/len(r_y_test_risk),r_true_low_pred_med/len(r_y_test_risk),r_true_low_pred_high/len(r_y_test_risk),r_true_med_pred_high/len(r_y_test_risk),r_true_med_pred_low/len(r_y_test_risk),r_true_high_pred_med/len(r_y_test_risk),r_true_high_pred_low/len(r_y_test_risk)]

    r_X_test_tool2 = r_X_test_tool2[~np.isnan(r_X_test_tool2).any(axis=1)]
    r_y_test_tool2 = r_y_test_tool2[~np.isnan(r_y_test_tool2)]

    r_y_pred_tool2 = grid_pipe.predict(r_X_test_tool2)
    r_y_pred_tool2 = r_y_pred_tool2+3
    # 1-3 low risk, 4-5 medium risk, 6-7 high risk

    # Report as a table for each participant left stringer and left camel hump
    # Create new variables for l_y_test and l_y_pred for each tool based on risk level
    r_y_test_risk  = np.zeros(np.shape(r_y_test_tool2))
    r_y_pred_risk = np.zeros(np.shape(r_y_pred_tool2))
    r_y_test_risk[r_y_test_tool2<=3] = 1
    r_y_test_risk[(r_y_test_tool2>3) & (r_y_test_tool2<=5)] = 2
    r_y_test_risk[r_y_test_tool2>5] = 3
    r_y_pred_risk[r_y_pred_tool2<=3] = 1
    r_y_pred_risk[(r_y_pred_tool2>3) & (r_y_pred_tool2<=5)] = 2
    r_y_pred_risk[r_y_pred_tool2>5] = 3

    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    r_accurate = 0
    r_true_low_pred_med = 0
    r_true_low_pred_high = 0
    r_true_med_pred_high = 0
    r_true_med_pred_low = 0
    r_true_high_pred_med = 0
    r_true_high_pred_low = 0
    for idx in range(len(r_y_test_risk)):
        if r_y_test_risk[idx] == r_y_pred_risk[idx]:
            r_accurate += 1
        elif r_y_test_risk[idx] == 1 and r_y_pred_risk[idx] == 2:
            r_true_low_pred_med += 1
        elif r_y_test_risk[idx] == 1 and r_y_pred_risk[idx] == 3:
            r_true_low_pred_high += 1
        elif r_y_test_risk[idx] == 2 and r_y_pred_risk[idx] == 3:
            r_true_med_pred_high += 1
        elif r_y_test_risk[idx] == 2 and r_y_pred_risk[idx] == 1:
            r_true_med_pred_low += 1
        elif r_y_test_risk[idx] == 3 and r_y_pred_risk[idx] == 2:
            r_true_high_pred_med += 1
        elif r_y_test_risk[idx] == 3 and r_y_pred_risk[idx] == 1:
            r_true_high_pred_low += 1

    right_camel_hump_data[i-1,:] = [i-1,r_accurate/len(r_y_test_risk),r_true_low_pred_med/len(r_y_test_risk),r_true_low_pred_high/len(r_y_test_risk),r_true_med_pred_high/len(r_y_test_risk),r_true_med_pred_low/len(r_y_test_risk),r_true_high_pred_med/len(r_y_test_risk),r_true_high_pred_low/len(r_y_test_risk)]

df_l_stringer = pd.DataFrame(left_stringer_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_l_stringer.to_excel(r'C:\Users\anand\Desktop\left_stringer_data.xlsx',index=False)
df_l_camel_hump = pd.DataFrame(left_camel_hump_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_l_camel_hump.to_excel(r'C:\Users\anand\Desktop\left_camel_hump_data.xlsx',index=False)

df_r_stringer = pd.DataFrame(right_stringer_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_r_stringer.to_excel(r'C:\Users\anand\Desktop\right_stringer_data.xlsx',index=False)
df_r_camel_hump = pd.DataFrame(right_camel_hump_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_r_camel_hump.to_excel(r'C:\Users\anand\Desktop\right_camel_hump_data.xlsx',index=False)




# l_stringer_accuracy = sklearn.metrics.accuracy_score(l_y_test[:tool_split_idx[i-1]], l_y_pred[:tool_split_idx[i-1]:])
# print("Accuracy for participant left hand stringer"+str(i)+" is "+str(l_stringer_accuracy))
# l_camel_hump_accuracy = sklearn.metrics.accuracy_score(l_y_test[tool_split_idx[i-1]:], l_y_pred[tool_split_idx[i-1]:])
# print("Accuracy for participant left hand camel hump"+str(i)+" is "+str(l_camel_hump_accuracy))

#%% Right hand


    


