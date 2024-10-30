#%% Import Section
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Adjustable parameters for HAL TLV calculation
overall_threshold = 44.8 # 10 lbs
finger_threshold = 15 # 3.3 lbs
window_length = 10 # In seconds
D = 75 # Duty Cycle
#%% Import Data for a specific participant, tool, and trial
data_fps = 60 # The data was interpolated to 60 fps
participant_id = 1
tool_id = 1
trial_id = 1
left_hand_force = np.load(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed force_detail_l.npy")
right_hand_force = np.load(r"C:\Users\anand\Desktop\Hand-intensive Manufacturing Processes Dataset\p"+str(participant_id)+" tool"+str(tool_id)+" trial"+str(trial_id)+"\processed force_detail_r.npy")
# The 19 hand regions are listed in order as follows: 
# palm123, thumb12, thumb_tip, index123, index_tip, middle123, ring123, little123
time = np.arange(0, left_hand_force.shape[0]/data_fps-1/data_fps, 1/data_fps)
# Calculate total force for each region of the left hand, convert to N, and round to 2 decimal places
l_palm_total_force = (np.round(np.sum(left_hand_force[:,0:3], axis=1)*4.44822, 2))
l_thumb_total_force = (np.round(np.sum(left_hand_force[:,3:6], axis=1)*4.44822, 2))
l_index_total_force = (np.round(np.sum(left_hand_force[:,6:10], axis=1)*4.44822, 2))
l_middle_total_force = (np.round(np.sum(left_hand_force[:,10:13], axis=1)*4.44822, 2))
l_ring_total_force = (np.round(np.sum(left_hand_force[:,13:16], axis=1)*4.44822, 2))
l_little_total_force = (np.round(np.sum(left_hand_force[:,16:19], axis=1)*4.44822, 2))

# Calculate total force for each region of the right hand, convert to N, and round to 2 decimal places
r_palm_total_force = (np.round(np.sum(right_hand_force[:,0:3], axis=1)*4.44822, 2))
r_thumb_total_force = (np.round(np.sum(right_hand_force[:,3:6], axis=1)*4.44822, 2))
r_index_total_force = (np.round(np.sum(right_hand_force[:,6:10], axis=1)*4.44822, 2))
r_middle_total_force = (np.round(np.sum(right_hand_force[:,10:13], axis=1)*4.44822, 2))
r_ring_total_force = (np.round(np.sum(right_hand_force[:,13:16], axis=1)*4.44822, 2))
r_little_total_force = (np.round(np.sum(right_hand_force[:,16:19], axis=1)*4.44822, 2))

l_hand_data = [l_palm_total_force,l_thumb_total_force,l_index_total_force,l_middle_total_force,l_ring_total_force,l_little_total_force]
l_hand_total_force = l_palm_total_force + l_thumb_total_force + l_index_total_force + l_middle_total_force + l_ring_total_force + l_little_total_force
r_hand_data = [r_palm_total_force,r_thumb_total_force,r_index_total_force,r_middle_total_force,r_ring_total_force,r_little_total_force]
r_hand_total_force = r_palm_total_force + r_thumb_total_force + r_index_total_force + r_middle_total_force + r_ring_total_force + r_little_total_force
#%% Define Functions
def continuous_windowed_frequency(time,hand_data,window_length,finger_threshold,overall_threshold):
    F_list = []
    start_window = 0
    end_window = np.abs(np.abs(time - 10)).argmin()

    while end_window < len(time):
        Exertions = 0
        for i in range(start_window,end_window):
            force_sum = 0
            single_finger_flag = 0
            #for j in range(len(hand_data)):
            for j in range(1,len(hand_data)): # Start from 1 to ignore palm data
                if hand_data[j][i] > 0:
                    force_sum += hand_data[j][i] 
                if hand_data[j][i] > finger_threshold:
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



# #%% Plot Results
# Create 2*2 plot
# Top 2 plots are all the forces for the left and right hand
# Bottom 2 plots are left and right hand HAL

# fig, axs = plt.subplots(2, 2)
# # make the plot bigger
# fig.set_size_inches(17, 8.5)
# fig.suptitle('Left and Right Hand Forces and HALs for file:' + filename)
# axs[0, 0].plot(time,l_palm_total_force)
# axs[0, 0].plot(time,l_thumb_total_force)
# axs[0, 0].plot(time,l_index_total_force)
# axs[0, 0].plot(time,l_middle_total_force)
# axs[0, 0].plot(time,l_ring_total_force)
# axs[0, 0].plot(time,l_little_total_force)
# axs[0, 0].set_title('Left Hand Forces')
# axs[0, 0].set(ylabel='Force (N)')
# axs[0, 0].legend(['Palm','Thumb','Index','Middle','Ring','Little'])
# axs[0, 1].plot(time,r_palm_total_force)
# axs[0, 1].plot(time,r_thumb_total_force)
# axs[0, 1].plot(time,r_index_total_force)
# axs[0, 1].plot(time,r_middle_total_force)
# axs[0, 1].plot(time,r_ring_total_force)
# axs[0, 1].plot(time,r_little_total_force)
# axs[0, 1].set_title('Right Hand Forces')
# axs[0, 1].set(ylabel='Force (N)')
# axs[0, 1].legend(['Palm','Thumb','Index','Middle','Ring','Little'])
# axs[1, 0].plot(time,l_windowed_HAL_plot)
# axs[1, 0].set_title('Left Hand HAL')
# axs[1, 0].set(xlabel='Time (s)', ylabel='HAL')
# axs[1, 1].plot(time,r_windowed_HAL_plot)
# axs[1, 1].set_title('Right Hand HAL')
# axs[1, 1].set(xlabel='Time (s)', ylabel='HAL')
# plt.show()

# Save time, hand forces, and HALs to a csv file with two decimal places
df = pd.DataFrame({'Time (s)':time,'Left Palm Force (N)':l_palm_total_force,'Left Thumb Force (N)':l_thumb_total_force,'Left Index Force (N)':l_index_total_force,'Left Middle Force (N)':l_middle_total_force,'Left Ring Force (N)':l_ring_total_force,'Left Little Force (N)':l_little_total_force,'Right Palm Force (N)':r_palm_total_force,'Right Thumb Force (N)':r_thumb_total_force,'Right Index Force (N)':r_index_total_force,'Right Middle Force (N)':r_middle_total_force,'Right Ring Force (N)':r_ring_total_force,'Right Little Force (N)':r_little_total_force,'Left HAL':l_windowed_HAL_plot,'Right HAL':r_windowed_HAL_plot})
# Store the variables in the dataset as numpy arrays

# Convert time to datetime format
# date_time = pd.to_datetime(time,unit='s')
# df['Time (s)'] = date_time
plt.plot(df['Time (s)'],df['Left HAL'])
# plt.show()

df['Time (s)'] = np.linspace(time[0],time[-1],len(df))
plt.figure()
plt.plot(df['Time (s)'],df['Left HAL'])
# plt.show()
df.interpolate(method='linear',inplace=True)
# Round data to 2 decimal places
df = df.round(3)
# df.to_csv(filename + '_w_HAL_25Hz.csv',index=False)
df.to_csv(r'C:\Users\anand\Desktop\test.csv',index=False)
