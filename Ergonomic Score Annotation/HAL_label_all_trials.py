#%% Import Section
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
#%% Define Functions
def continuous_windowed_frequency(time,hand_data,window_length,finger_threshold,overall_threshold):
    F_list = []
    start_window = 0
    end_window = np.abs(np.abs(time - 10)).argmin()
    force_peak = 10*4.44822 # 10 lbs
    # Clip all values above the force peak

    while end_window < len(time):
        Exertions = 0
        for i in range(start_window,end_window):
            force_sum = 0
            single_finger_flag = 0
            force_sum = np.sum(hand_data[i,:])
            for j in range(1,hand_data.shape[1]): # Start from 1 to ignore palm data
                if hand_data[i,j] > finger_threshold:
                    single_finger_flag = 1
            if force_sum > overall_threshold or single_finger_flag == 1:
                Exertions += 1
        F_list.append(Exertions/window_length)
        start_window += 1
        end_window += 1

    return F_list




def non_linear_HAL(F,D):
    if F>0:
        HAL = 6.56 * np.log(D)*((F**1.31)/(1 + 3.18*F**1.31))
    else:
        HAL = 0
    return HAL
#%% Adjustable parameters for HAL TLV calculation
overall_threshold = 44.8 # 10 lbs
finger_threshold = 15 # 3.3 lbs
window_length = 10 # In seconds
peak_finger_force = 44.8 # 10 lbs
peak_palm_force = 44.8*4 # 40 lbs
D = 75 # Duty Cycle
#%% Import Data for a specific participant, tool, and trial
data_fps = 60 # The data was interpolated to 60 fps
participant_id = 1
tool_id = 1
trial_id = 1

participant_id_range = range(1,8)
tool_id_range = range(1,3)
trial_id_range = range(1,4)
#%% Calculate HAL for all trials
for participant_id in participant_id_range:
    for tool_id in tool_id_range:
        for trial_id in trial_id_range:
            # Check if the file exists
            if os.path.isfile(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed force_detail_l.npy"):
                print('Participant: ',participant_id,' Tool: ',tool_id,' Trial: ',trial_id)
                left_hand_force = np.load(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed force_detail_l.npy")
                right_hand_force = np.load(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\Participant "+str(participant_id)+"\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed force_detail_r.npy")
                file_name = r'C:\Users\anand\Desktop\HAL Labelled Data\p'+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+' w lr HAL.csv'
                # The 19 hand regions are listed in order as follows: 
                # palm123, thumb12, thumb_tip, index123, index_tip, middle123, ring123, little123
                time = np.linspace(0, left_hand_force.shape[0]//data_fps, left_hand_force.shape[0])
                # Calculate total force for each region of the left hand, convert to N, and round to 2 decimal places
                l_palm_total_force = (np.round(np.sum(left_hand_force[:,0:3], axis=1)*4.44822, 2))
                l_palm_total_force = np.clip(l_palm_total_force,0,peak_palm_force) 
                l_thumb_total_force = (np.round(np.sum(left_hand_force[:,3:6], axis=1)*4.44822, 2))
                l_thumb_total_force = np.clip(l_thumb_total_force,0,peak_finger_force)
                l_index_total_force = (np.round(np.sum(left_hand_force[:,6:10], axis=1)*4.44822, 2))
                l_index_total_force = np.clip(l_index_total_force,0,peak_finger_force)
                l_middle_total_force = (np.round(np.sum(left_hand_force[:,10:13], axis=1)*4.44822, 2))
                l_middle_total_force = np.clip(l_middle_total_force,0,peak_finger_force)
                l_ring_total_force = (np.round(np.sum(left_hand_force[:,13:16], axis=1)*4.44822, 2))
                l_ring_total_force = np.clip(l_ring_total_force,0,peak_finger_force)
                l_little_total_force = (np.round(np.sum(left_hand_force[:,16:19], axis=1)*4.44822, 2))

                # Calculate total force for each region of the right hand, convert to N, and round to 2 decimal places
                r_palm_total_force = (np.round(np.sum(right_hand_force[:,0:3], axis=1)*4.44822, 2))
                r_palm_total_force = np.clip(r_palm_total_force,0,peak_palm_force)
                r_thumb_total_force = (np.round(np.sum(right_hand_force[:,3:6], axis=1)*4.44822, 2))
                r_thumb_total_force = np.clip(r_thumb_total_force,0,peak_finger_force)
                r_index_total_force = (np.round(np.sum(right_hand_force[:,6:10], axis=1)*4.44822, 2))
                r_index_total_force = np.clip(r_index_total_force,0,peak_finger_force)
                r_middle_total_force = (np.round(np.sum(right_hand_force[:,10:13], axis=1)*4.44822, 2))
                r_middle_total_force = np.clip(r_middle_total_force,0,peak_finger_force)
                r_ring_total_force = (np.round(np.sum(right_hand_force[:,13:16], axis=1)*4.44822, 2))
                r_ring_total_force = np.clip(r_ring_total_force,0,peak_finger_force)
                r_little_total_force = (np.round(np.sum(right_hand_force[:,16:19], axis=1)*4.44822, 2))
                r_little_total_force = np.clip(r_little_total_force,0,peak_finger_force)

                l_hand_data = np.array([l_palm_total_force,l_thumb_total_force,l_index_total_force,l_middle_total_force,l_ring_total_force,l_little_total_force]).T
                l_hand_total_force = l_palm_total_force + l_thumb_total_force + l_index_total_force + l_middle_total_force + l_ring_total_force + l_little_total_force
                r_hand_data = np.array([r_palm_total_force,r_thumb_total_force,r_index_total_force,r_middle_total_force,r_ring_total_force,r_little_total_force]).T
                r_hand_total_force = r_palm_total_force + r_thumb_total_force + r_index_total_force + r_middle_total_force + r_ring_total_force + r_little_total_force

                # Left Hand
                l_windowed_HAL = []
                l_F_list = continuous_windowed_frequency(time,l_hand_data,window_length,finger_threshold,overall_threshold)
                for F in l_F_list:
                    HAL = non_linear_HAL(F,D)
                    l_windowed_HAL.append(HAL)
                # Right Hand
                r_windowed_HAL = []
                r_F_list = continuous_windowed_frequency(time,r_hand_data,window_length,finger_threshold,overall_threshold)
                for F in r_F_list:
                    HAL = non_linear_HAL(F,D)
                    r_windowed_HAL.append(HAL)

                # Add a buffer to the start of HALs to make them the same length as the time vector
                l_windowed_HAL_plot = [0]*(len(time)-len(l_windowed_HAL)) + l_windowed_HAL
                r_windowed_HAL_plot = [0]*(len(time)-len(r_windowed_HAL)) + r_windowed_HAL


                df = pd.DataFrame({'Time (s)':time,'Left Palm Force (N)':l_palm_total_force,'Left Thumb Force (N)':l_thumb_total_force,'Left Index Force (N)':l_index_total_force,'Left Middle Force (N)':l_middle_total_force,'Left Ring Force (N)':l_ring_total_force,'Left Little Force (N)':l_little_total_force,'Right Palm Force (N)':r_palm_total_force,'Right Thumb Force (N)':r_thumb_total_force,'Right Index Force (N)':r_index_total_force,'Right Middle Force (N)':r_middle_total_force,'Right Ring Force (N)':r_ring_total_force,'Right Little Force (N)':r_little_total_force,'Left HAL':l_windowed_HAL_plot,'Right HAL':r_windowed_HAL_plot})
                # df['Time (s)'] = np.linspace(time[0],time[-1],len(df))
                # df.interpolate(method='linear',inplace=True)
                df = df.round(3)
                df.to_csv(file_name,index=False)