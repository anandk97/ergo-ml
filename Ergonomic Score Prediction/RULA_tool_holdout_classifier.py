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
def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle = angle_radians*(180/np.pi)
    return angle
# Store data for 6 training participants and 1 test participant and repeat for all 8 participants
tool_split_idx = [44016,44442,36380,50676,50421,46612,48997]
for i in range(1,8):
    old_data = pd.read_csv(r"C:\Users\anand\Desktop\RULA\Labelled angle compensated data\old_data_w_lr_rula.csv", header=None)
    # Keep first 54 and columns 184,185,186,187,189,190

    train_data = old_data
    for j in range(1,8):
        if j==i:
            # Indices of tool split are 1-44016, 2-44442, 3-36380, 4-50676,5-50421,6-46612,7-48997
            #
            # Read all csv files in the folder C:\Users\anand\Desktop\RULA\Labelled Data\participant_i
            os.chdir(r"C:\Users\anand\Desktop\RULA\Labelled angle compensated data\p"+str(i))
            extension = 'csv'
            all_filenames = [k for k in glob.glob('*.{}'.format(extension))]
            # Combine all files in the list
            test_data = pd.concat([pd.read_csv(all_filenames[f_idx], header=None) for f_idx in range(len(all_filenames))], ignore_index=True)
            # Drop NaN values
            test_data.dropna(inplace=True)
            continue
        
        # Read all csv files in the folder C:\Users\anand\Desktop\RULA\Labelled Data\participant_j
        os.chdir(r"C:\Users\anand\Desktop\RULA\Labelled angle compensated data\p"+str(j))
        extension = 'csv'
        all_filenames = [filename for filename in glob.glob('*.{}'.format(extension))]
        # Combine all files in the list

        new_data = pd.concat([pd.read_csv(f, header=None) for f in all_filenames], ignore_index=True)

        # Concatenate new_data to train_data along the rows

        train_data = pd.concat([train_data,new_data], ignore_index=True)
        # Drop NaN values
        train_data.dropna(inplace=True)
        


    
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


    l_eye = np.array(test_data.iloc[:,3:6])
    r_eye = np.array(test_data.iloc[:,6:9])

    mid_eye = (l_eye+r_eye)/2

    l_shoulder = np.array(test_data.iloc[:,15:18])
    r_shoulder = np.array(test_data.iloc[:,18:21])
    mid_shoulder = np.array(test_data.iloc[:,51:54])

    l_elbow = np.array(test_data.iloc[:,21:24])
    r_elbow = np.array(test_data.iloc[:,24:27])

    l_wrist = np.array(test_data.iloc[:,27:30])
    r_wrist = np.array(test_data.iloc[:,30:33])

    l_hip = np.array(test_data.iloc[:,33:36])
    r_hip = np.array(test_data.iloc[:,36:39])

    l_gonio_1 = np.array(test_data.iloc[:,54])
    l_gonio_2 = np.array(test_data.iloc[:,55])
    r_gonio_1 = np.array(test_data.iloc[:,56])
    r_gonio_2 = np.array(test_data.iloc[:,57])

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
    l_RULA_scores = np.array(test_data.iloc[:,58])
    r_RULA_scores = np.array(test_data.iloc[:,59])
    # l_X_test = np.hstack((l_shoulder,r_shoulder,mid_shoulder,l_elbow,l_wrist,l_hip,r_hip,l_eye,r_eye,l_gonio_1.reshape(-1,1),l_gonio_2.reshape(-1,1)))
    # l_X_test = np.hstack((l_shoulder,r_shoulder,mid_shoulder,l_elbow,l_wrist,l_hip,r_hip,l_eye,r_eye,l_gonio_1.reshape(-1,1),l_gonio_2.reshape(-1,1)))
    # Only the angles
    l_X_test = np.hstack((upper_arm_angle.reshape(-1,1),lower_arm_angle.reshape(-1,1),neck_angle.reshape(-1,1),trunk_angle.reshape(-1,1),l_gonio_1.reshape(-1,1),l_gonio_2.reshape(-1,1)))
    l_y_test = l_RULA_scores
    # Convert to int
    l_y_test = l_y_test.astype(int)
# Remove rows with RULA score 0,1 and 2
    l_X_test = l_X_test[l_y_test>2]
    l_y_test = l_y_test[l_y_test>2]
    # For right hand
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
    # r_X_test = np.hstack((r_shoulder,r_elbow,r_wrist,r_eye,r_gonio_1.reshape(-1,1),r_gonio_2.reshape(-1,1)))
    # Only the angles
    r_X_test = np.hstack((upper_arm_angle.reshape(-1,1),lower_arm_angle.reshape(-1,1),neck_angle.reshape(-1,1),trunk_angle.reshape(-1,1),r_gonio_1.reshape(-1,1),r_gonio_2.reshape(-1,1)))
    r_y_test = r_RULA_scores
    # Convert to int
    r_y_test = r_y_test.astype(int)
    # Remove rows with RULA score 0,1 and 2
    r_X_test = r_X_test[r_y_test>2]
    r_y_test = r_y_test[r_y_test>2]

    # Feature Scaling

    l_sc = StandardScaler()
    r_sc = StandardScaler()
    # l_X_train = l_sc.fit_transform(l_X_train)
    # l_X_test = l_sc.transform(l_X_test)

    # Fitting classifier to the Training set
    # Random Forest Classifier

    # l_classifier = RandomForestClassifier(random_state = 0)
    # l_classifier = HistGradientBoostingClassifier(random_state=0,min_samples_leaf=5)

    # # params = {'l_classifier__n_estimators': [200],'l_classifier__max_depth':[8]}
    # params = {'l_classifier__max_iter' :[150]}
    # # pca = PCA(n_components=20)
    # pipe = Pipeline(steps=[('scaler',l_sc),('l_classifier', l_classifier)])
    # grid_pipe = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-2, verbose=4,scoring='f1_micro')
    # #l_classifier = KNeighborsClassifier(n_neighbors=1,metric='minkowski', p = 2)
    # grid_pipe.fit(l_X_train, l_y_train)
    # print("Training done for participant "+str(i))
    # # Predicting the Test set results
    # l_y_pred = grid_pipe.predict(l_X_test)

    # # Predicting the Training set results
    # l_y_pred_train = grid_pipe.predict(l_X_train)
    # Create an XGB classifier
    l_classifier = xgb.XGBClassifier(tree_method='gpu_hist',verbosity = 2,sampling_method='gradient_based',random_state=0)
    # l_classifier = HistGradientBoostingClassifier(random_state=0,min_samples_leaf=5,max_depth = 4,max_leaf_nodes=20)
    # Without GPU
    # l_classifier = xgb.XGBClassifier(verbosity = 2,random_state=0)
    l_y_train = l_y_train-3
    # Re
    # params = {'l_classifier__n_estimators' :[10,20,30,],'l_classifier__max_depth':[4],'l_classifier__learning_rate':[0.15,0.16,0.17]}
    param_dist = {'l_classifier__n_estimators' :[29],'l_classifier__max_depth':[6],'l_classifier__learning_rate':[0.18],'l_classifier__gamma':[0.12],'l_classifier__min_child_weight':[1],'l_classifier__max_leaves':[27]}
    # param_dist = {'l_classifier__n_estimators' :[10,20,50,80]}
    # pca = PCA(n_components=20)
    pipe = Pipeline(steps=[('scaler',l_sc),('l_classifier', l_classifier)]) 
    # grid_pipe = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-2, verbose=4,scoring='f1_weighted')
    # Random search
    grid_pipe = GridSearchCV(pipe, param_grid=param_dist, cv=5, n_jobs=-2, verbose=1,scoring='f1_weighted')
    #l_classifier = KNeighborsClassifier(n_neighbors=1,metric='minkowski', p = 2)
    grid_pipe.fit(l_X_train, l_y_train)

    # Print the best parameters
    print(grid_pipe.best_params_) 
    print("Training done for participant "+str(i))
    # Predicting the Test set results
    # Remove Nans from l_X_test and l_y_test
    l_X_test = l_X_test[~np.isnan(l_X_test).any(axis=1)]
    l_y_test = l_y_test[~np.isnan(l_y_test)]
    # Split predictions based on tool index
    # Predicting the Test set results
    
    l_y_pred = grid_pipe.predict(l_X_test)
    l_y_pred = l_y_pred+3
    # 1-3 low risk, 4-5 medium risk, 6-7 high risk
    
    # Report as a table for each participant left stringer and left camel hump
    # Create new variables for l_y_test and l_y_pred for each tool based on risk level
    l_y_test_risk  = np.zeros(np.shape(l_y_test))
    l_y_pred_risk = np.zeros(np.shape(l_y_pred))
    l_y_test_risk[l_y_test<=3] = 1
    l_y_test_risk[(l_y_test>3) & (l_y_test<=5)] = 2
    l_y_test_risk[l_y_test>5] = 3
    l_y_pred_risk[l_y_pred<=3] = 1
    l_y_pred_risk[(l_y_pred>3) & (l_y_pred<=5)] = 2
    l_y_pred_risk[l_y_pred>5] = 3
    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    l_accurate = 0
    l_true_low_pred_med = 0
    l_true_low_pred_high = 0
    l_true_med_pred_high = 0
    l_true_med_pred_low = 0
    l_true_high_pred_med = 0
    l_true_high_pred_low = 0
    for idx in range(len(l_y_test_risk[0:tool_split_idx[i-1]])):
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
    print("Accuracy for participant left hand stringer"+str(i)+" is "+str(l_accurate/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    print("True low pred med for participant left hand stringer"+str(i)+" is "+str(l_true_low_pred_med/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    print("True low pred high for participant left hand stringer"+str(i)+" is "+str(l_true_low_pred_high/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    print("True med pred high for participant left hand stringer"+str(i)+" is "+str(l_true_med_pred_high/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    print("True med pred low for participant left hand stringer"+str(i)+" is "+str(l_true_med_pred_low/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    print("True high pred med for participant left hand stringer"+str(i)+" is "+str(l_true_high_pred_med/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    print("True high pred low for participant left hand stringer"+str(i)+" is "+str(l_true_high_pred_low/len(l_y_test_risk[0:tool_split_idx[i-1]])))
    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    l_accurate = 0
    l_true_low_pred_med = 0
    l_true_low_pred_high = 0
    l_true_med_pred_high = 0
    l_true_med_pred_low = 0
    l_true_high_pred_med = 0
    l_true_high_pred_low = 0
    for idx in range(len(l_y_test_risk[tool_split_idx[i-1]:])):
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
    print("Accuracy for participant left hand camel hump"+str(i)+" is "+str(l_accurate/len(l_y_test_risk[tool_split_idx[i-1]:])))
    print("True low pred med for participant left hand camel hump"+str(i)+" is "+str(l_true_low_pred_med/len(l_y_test_risk[tool_split_idx[i-1]:])))
    print("True low pred high for participant left hand camel hump"+str(i)+" is "+str(l_true_low_pred_high/len(l_y_test_risk[tool_split_idx[i-1]:])))
    print("True med pred high for participant left hand camel hump"+str(i)+" is "+str(l_true_med_pred_high/len(l_y_test_risk[tool_split_idx[i-1]:])))
    print("True med pred low for participant left hand camel hump"+str(i)+" is "+str(l_true_med_pred_low/len(l_y_test_risk[tool_split_idx[i-1]:])))
    print("True high pred med for participant left hand camel hump"+str(i)+" is "+str(l_true_high_pred_med/len(l_y_test_risk[tool_split_idx[i-1]:])))
    print("True high pred low for participant left hand camel hump"+str(i)+" is "+str(l_true_high_pred_low/len(l_y_test_risk[tool_split_idx[i-1]:])))

    

    # l_stringer_accuracy = sklearn.metrics.accuracy_score(l_y_test[:tool_split_idx[i-1]], l_y_pred[:tool_split_idx[i-1]:])
    # print("Accuracy for participant left hand stringer"+str(i)+" is "+str(l_stringer_accuracy))
    # l_camel_hump_accuracy = sklearn.metrics.accuracy_score(l_y_test[tool_split_idx[i-1]:], l_y_pred[tool_split_idx[i-1]:])
    # print("Accuracy for participant left hand camel hump"+str(i)+" is "+str(l_camel_hump_accuracy))

    # # Right hand

    r_classifier = xgb.XGBClassifier(tree_method='gpu_hist',verbosity = 2,sampling_method='gradient_based',random_state=0)
    r_pipe = Pipeline(steps=[('scaler',r_sc),('r_classifier', r_classifier)])
    r_y_train = r_y_train-3
    r_param_dist = {'r_classifier__n_estimators' :[29],'r_classifier__max_depth':[6],'r_classifier__learning_rate':[0.18],'r_classifier__gamma':[0.12],'r_classifier__min_child_weight':[1],'r_classifier__max_leaves':[27]}
    # r_param_dist = {'r_classifier__n_estimators' :[10]}
    r_grid_pipe = GridSearchCV(r_pipe, param_grid=r_param_dist, cv=5, n_jobs=-2, verbose=1,scoring='f1_weighted')
    r_grid_pipe.fit(r_X_train, r_y_train)
    # Print the best parameters
    print(r_grid_pipe.best_params_)
    # Predict the test set results
    r_X_test = r_X_test[~np.isnan(r_X_test).any(axis=1)]
    r_y_test = r_y_test[~np.isnan(r_y_test)]
    r_y_pred = r_grid_pipe.predict(r_X_test)
    r_y_pred = r_y_pred+3
    # r_stringer_accuracy = sklearn.metrics.accuracy_score(r_y_test[:tool_split_idx[i-1]], r_y_pred[:tool_split_idx[i-1]:])
    # print("Accuracy for participant right hand stringer"+str(i)+" is "+str(r_stringer_accuracy))
    # r_camel_hump_accuracy = sklearn.metrics.accuracy_score(r_y_test[tool_split_idx[i-1]:], r_y_pred[tool_split_idx[i-1]:])
    # print("Accuracy for participant right hand camel hump"+str(i)+" is "+str(r_camel_hump_accuracy))
    # Create new variables for r_y_test and r_y_pred for each tool based on risk level
    r_y_test_risk  = np.zeros(np.shape(r_y_test))
    r_y_pred_risk = np.zeros(np.shape(r_y_pred))
    r_y_test_risk[r_y_test<=3] = 1
    r_y_test_risk[(r_y_test>3) & (r_y_test<=5)] = 2
    r_y_test_risk[r_y_test>5] = 3
    r_y_pred_risk[r_y_pred<=3] = 1
    r_y_pred_risk[(r_y_pred>3) & (r_y_pred<=5)] = 2
    r_y_pred_risk[r_y_pred>5] = 3
    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    r_accurate = 0
    r_true_low_pred_med = 0
    r_true_low_pred_high = 0
    r_true_med_pred_high = 0
    r_true_med_pred_low = 0
    r_true_high_pred_med = 0
    r_true_high_pred_low = 0
    for idx in range(len(r_y_test_risk[0:tool_split_idx[i-1]])):
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
    print("Accuracy for participant right hand stringer"+str(i)+" is "+str(r_accurate/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    print("True low pred med for participant right hand stringer"+str(i)+" is "+str(r_true_low_pred_med/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    print("True low pred high for participant right hand stringer"+str(i)+" is "+str(r_true_low_pred_high/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    print("True med pred high for participant right hand stringer"+str(i)+" is "+str(r_true_med_pred_high/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    print("True med pred low for participant right hand stringer"+str(i)+" is "+str(r_true_med_pred_low/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    print("True high pred med for participant right hand stringer"+str(i)+" is "+str(r_true_high_pred_med/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    print("True high pred low for participant right hand stringer"+str(i)+" is "+str(r_true_high_pred_low/len(r_y_test_risk[0:tool_split_idx[i-1]])))
    # Report percentage accuracy based on correct risk level, true = low predicted = medium, true = low predicted = high, true = medium predicted = high, true = medium predicted = low, true = high predicted = medium, true = high predicted = low
    r_accurate = 0
    r_true_low_pred_med = 0
    r_true_low_pred_high = 0
    r_true_med_pred_high = 0
    r_true_med_pred_low = 0
    r_true_high_pred_med = 0
    r_true_high_pred_low = 0

    for idx in range(len(r_y_test_risk[tool_split_idx[i-1]:])):
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
    print("Accuracy for participant right hand camel hump"+str(i)+" is "+str(r_accurate/len(r_y_test_risk[tool_split_idx[i-1]:])))
    print("True low pred med for participant right hand camel hump"+str(i)+" is "+str(r_true_low_pred_med/len(r_y_test_risk[tool_split_idx[i-1]:])))
    print("True low pred high for participant right hand camel hump"+str(i)+" is "+str(r_true_low_pred_high/len(r_y_test_risk[tool_split_idx[i-1]:])))
    print("True med pred high for participant right hand camel hump"+str(i)+" is "+str(r_true_med_pred_high/len(r_y_test_risk[tool_split_idx[i-1]:])))
    print("True med pred low for participant right hand camel hump"+str(i)+" is "+str(r_true_med_pred_low/len(r_y_test_risk[tool_split_idx[i-1]:])))
    print("True high pred med for participant right hand camel hump"+str(i)+" is "+str(r_true_high_pred_med/len(r_y_test_risk[tool_split_idx[i-1]:])))
    print("True high pred low for participant right hand camel hump"+str(i)+" is "+str(r_true_high_pred_low/len(r_y_test_risk[tool_split_idx[i-1]:])))
    # # Report as a table for each participant right stringer and right camel hump
    
    



    # l_cm = confusion_matrix(l_y_test, l_y_pred)
    # # Get number of unique labels in l_y_test
    # num_labels = len(np.unique(l_y_test))
    # # Convert to float
    # l_cm = l_cm.astype('float')
    # for label in range(num_labels):
    #     if np.sum(l_cm[label,:]) != 0:
    #         l_cm[label,:] = l_cm[label,:]/np.sum(l_cm[label,:])
    # l_cm_normalized = l_cm

    # # Visualize the confusion matrix using sklearn


    # disp = ConfusionMatrixDisplay(confusion_matrix=l_cm_normalized,display_labels=np.arange(3,8))
    # disp.plot()
    # plt.title('Test Confusion Matrix')
    # plt.show()

    # Save the figure to the current directory
    # save_file_name = r'C:\Users\anand\Desktop\RULA\l_xgb_random_search_best_params'+str(i)+'.png'
    # plt.savefig(save_file_name, dpi=300)
    # print('Saved figure number '+str(i))
    # Visualize the training set confusion matrix using sklearn


