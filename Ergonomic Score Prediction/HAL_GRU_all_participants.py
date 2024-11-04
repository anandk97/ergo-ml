#%% Import libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import time
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)
class HAL_GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, midlayer_size = 10, dropout=0.2):
        super(HAL_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, midlayer_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(midlayer_size, output_size)
        self.criterion = nn.HuberLoss(delta=1.0, reduction='mean')

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear1(out[:, -1, :])
        out = self.dropout(out)
        out = self.linear2(out).flatten()
        return out
    def training_step(self, batch, batch_idx):
        data, targets = batch
        data = data.float()
        targets = targets.float()
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

def collate_fn(batch):
    data, targets = zip(*batch)
    data = [x.to(torch.float32) for x in data]
    targets = [y for y in targets]  # Adjust dtype as needed
    data = torch.stack(data)
    targets = torch.stack(targets)
    return data, targets  

#%% Load data
input_size = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
drop_out = 0
mid_layer_size = 90
hidden_layer_size = 10
number_layers = 3
torch.manual_seed(0)
data_directory = r'C:\Users\anand\Desktop\HAL Pytorch Datasets'
participant_id_range = range(1,16)
tool_id_range = range(1,3)
right_stringer_data = np.zeros((15,8))
right_camel_hump_data = np.zeros((15,8))
left_stringer_data = np.zeros((15,8))
left_camel_hump_data = np.zeros((15,8))
for test_participant_id in participant_id_range:
    train_participant_id_range = [i for i in participant_id_range if i != test_participant_id]


    # Read all datasets beginning with p+test_participant_id and tool+tool_id and ending with .pt
    tool1_left_test_data = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.startswith('p'+str(test_participant_id)+'_tool1_left') and f.endswith('.pt')]
    tool1_right_test_data = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.startswith('p'+str(test_participant_id)+'_tool1_right') and f.endswith('.pt')]
    tool2_left_test_data = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.startswith('p'+str(test_participant_id)+'_tool2_left') and f.endswith('.pt')]
    tool2_right_test_data = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.startswith('p'+str(test_participant_id)+'_tool2_right') and f.endswith('.pt')]

    # Combine all datasets into one
    tool1_left_test_data = ConcatDataset([torch.load(f) for f in tool1_left_test_data])
    tool1_right_test_data = ConcatDataset([torch.load(f) for f in tool1_right_test_data])
    tool2_left_test_data = ConcatDataset([torch.load(f) for f in tool2_left_test_data])
    tool2_right_test_data = ConcatDataset([torch.load(f) for f in tool2_right_test_data])

    # Read all datasets beginning with p+train_participant_id and tool+tool_id and ending with .pt
    left_train_data = [os.path.join(data_directory, f) for f in os.listdir(data_directory) for i in train_participant_id_range if f.startswith('p'+str(i)+'_tool1_left') or f.startswith('p'+str(i)+'_tool2_left') and f.endswith('.pt')]
    right_train_data = [os.path.join(data_directory, f) for f in os.listdir(data_directory) for i in train_participant_id_range if f.startswith('p'+str(i)+'_tool1_right') or f.startswith('p'+str(i)+'_tool2_right') and f.endswith('.pt')]

    # Combine datasets
    left_train_data = ConcatDataset([torch.load(f) for f in left_train_data])
    # Convert to float 32
    right_train_data = ConcatDataset([torch.load(f) for f in right_train_data])

    # Load data into DataLoader
    batch_size = 1024
    left_train_loader = DataLoader(left_train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    left_test_data = ConcatDataset([tool1_left_test_data, tool2_left_test_data])
    left_test_loader = DataLoader(left_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    right_train_loader = DataLoader(right_train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    right_test_data = ConcatDataset([tool1_right_test_data, tool2_right_test_data])
    right_test_loader = DataLoader(right_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience = 3,
        verbose=True,
        mode='min'
    )

    trainer_left = pl.Trainer(
        max_epochs=25,
        callbacks=[early_stopping_callback],
        accelerator = 'gpu',
        devices =1
    )

    # Train for left hand
    model = HAL_GRU(input_size=6, hidden_size=hidden_layer_size, num_layers=number_layers, output_size=1,midlayer_size=mid_layer_size, dropout=drop_out)
    # Try more layers, 3 onwards
    start_time = time.time()
    trainer_left.fit(model, left_train_loader, left_test_loader)
    print(f"Time taken for left hand: {time.time() - start_time}\n")
    # Save the trained model
    torch.save(model.state_dict(),r'C:\Users\anand\Desktop\Saved HAL Models\HAL_GRU_test_'+str(test_participant_id)+'_left.pt')
    # Evaluate the model
    l_accurate = 0
    l_true_low_pred_med = 0
    l_true_low_pred_high = 0
    l_true_med_pred_high = 0
    l_true_med_pred_low = 0
    l_true_high_pred_med = 0
    l_true_high_pred_low = 0
    tool1_left_dataloader = DataLoader(tool1_left_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for (X, y) in tool1_left_dataloader:

        y_pred = model(X)
        for i in range(len(y)):
            if y[i] < 4:
                if y_pred[i] < 4:
                    l_accurate += 1
                elif y_pred[i] < 7:
                    l_true_low_pred_med += 1
                else:
                    l_true_low_pred_high += 1
            elif y[i] < 7:
                if y_pred[i] < 4:
                    l_true_med_pred_low += 1
                elif y_pred[i] < 7:
                    l_accurate += 1
                else:
                    l_true_med_pred_high += 1
            else:
                if y_pred[i] < 4:
                    l_true_high_pred_low += 1
                elif y_pred[i] < 7:
                    l_true_high_pred_med += 1
                else:
                    l_accurate += 1
    l_accurate = l_accurate/len(tool1_left_test_data)
    l_true_low_pred_med = l_true_low_pred_med/len(tool1_left_test_data)
    l_true_low_pred_high = l_true_low_pred_high/len(tool1_left_test_data)
    l_true_med_pred_high = l_true_med_pred_high/len(tool1_left_test_data)
    l_true_med_pred_low = l_true_med_pred_low/len(tool1_left_test_data)
    l_true_high_pred_med = l_true_high_pred_med/len(tool1_left_test_data)
    l_true_high_pred_low = l_true_high_pred_low/len(tool1_left_test_data)
    left_stringer_data[test_participant_id-1,:] = [test_participant_id, l_accurate, l_true_low_pred_med, l_true_low_pred_high, l_true_med_pred_high, l_true_med_pred_low, l_true_high_pred_med, l_true_high_pred_low]

    l_accurate = 0
    l_true_low_pred_med = 0
    l_true_low_pred_high = 0
    l_true_med_pred_high = 0
    l_true_med_pred_low = 0
    l_true_high_pred_med = 0
    l_true_high_pred_low = 0
    tool2_left_dataloader = DataLoader(tool2_left_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for (X, y) in tool2_left_dataloader:
  
        y_pred = model(X)
        for i in range(len(y)):
            if y[i] < 4:
                if y_pred[i] < 4:
                    l_accurate += 1
                elif y_pred[i] < 7:
                    l_true_low_pred_med += 1
                else:
                    l_true_low_pred_high += 1
            elif y[i] < 7:
                if y_pred[i] < 4:
                    l_true_med_pred_low += 1
                elif y_pred[i] < 7:
                    l_accurate += 1
                else:
                    l_true_med_pred_high += 1
            else:
                if y_pred[i] < 4:
                    l_true_high_pred_low += 1
                elif y_pred[i] < 7:
                    l_true_high_pred_med += 1
                else:
                    l_accurate += 1

    len_tool2_left = len(tool2_left_test_data)
    left_camel_hump_data[test_participant_id-1,:] = [test_participant_id, l_accurate/len_tool2_left, l_true_low_pred_med/len_tool2_left, l_true_low_pred_high/len_tool2_left, l_true_med_pred_high/len_tool2_left, l_true_med_pred_low/len_tool2_left, l_true_high_pred_med/len_tool2_left, l_true_high_pred_low/len_tool2_left]

    # Train for right hand
    trainer_right = pl.Trainer(
        max_epochs=25,
        callbacks=[early_stopping_callback],
        accelerator = 'gpu',
        devices =1
    )
    model = HAL_GRU(input_size=6, hidden_size=hidden_layer_size, num_layers=number_layers, output_size=1,midlayer_size=mid_layer_size, dropout=drop_out)
    start_time = time.time()
    trainer_right.fit(model, right_train_loader, right_test_loader)
    print(f"Time taken for right hand: {time.time() - start_time}\n")
    torch.save(model.state_dict(),r'C:\Users\anand\Desktop\Saved HAL Models\HAL_GRU_test_'+str(test_participant_id)+'_right.pt')

    r_accurate = 0
    r_true_low_pred_med = 0
    r_true_low_pred_high = 0
    r_true_med_pred_high = 0
    r_true_med_pred_low = 0
    r_true_high_pred_med = 0
    r_true_high_pred_low = 0
    tool1_right_dataloader = DataLoader(tool1_right_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for (X, y) in tool1_right_dataloader:

        y_pred = model(X)
        for i in range(len(y)):
            if y[i] < 4:
                if y_pred[i] < 4:
                    r_accurate += 1
                elif y_pred[i] < 7:
                    r_true_low_pred_med += 1
                else:
                    r_true_low_pred_high += 1
            elif y[i] < 7:
                if y_pred[i] < 4:
                    r_true_med_pred_low += 1
                elif y_pred[i] < 7:
                    r_accurate += 1
                else:
                    r_true_med_pred_high += 1
            else:
                if y_pred[i] < 4:
                    r_true_high_pred_low += 1
                elif y_pred[i] < 7:
                    r_true_high_pred_med += 1
                else:
                    r_accurate += 1
    r_accurate = r_accurate/len(tool1_right_test_data)
    r_true_low_pred_med = r_true_low_pred_med/len(tool1_right_test_data)
    r_true_low_pred_high = r_true_low_pred_high/len(tool1_right_test_data)
    r_true_med_pred_high = r_true_med_pred_high/len(tool1_right_test_data)
    r_true_med_pred_low = r_true_med_pred_low/len(tool1_right_test_data)
    r_true_high_pred_med = r_true_high_pred_med/len(tool1_right_test_data)
    r_true_high_pred_low = r_true_high_pred_low/len(tool1_right_test_data)
    right_stringer_data[test_participant_id-1,:] = [test_participant_id, r_accurate, r_true_low_pred_med, r_true_low_pred_high, r_true_med_pred_high, r_true_med_pred_low, r_true_high_pred_med,r_true_high_pred_low]

    r_accurate = 0
    r_true_low_pred_med = 0
    r_true_low_pred_high = 0
    r_true_med_pred_high = 0
    r_true_med_pred_low = 0
    r_true_high_pred_med = 0
    r_true_high_pred_low = 0
    tool2_right_dataloader = DataLoader(tool2_right_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for (X, y) in tool2_right_dataloader:
            
            y_pred = model(X)
            for i in range(len(y)):
                if y[i] < 4:
                    if y_pred[i] < 4:
                        r_accurate += 1
                    elif y_pred[i] < 7:
                        r_true_low_pred_med += 1
                    else:
                        r_true_low_pred_high += 1
                elif y[i] < 7:
                    if y_pred[i] < 4:
                        r_true_med_pred_low += 1
                    elif y_pred[i] < 7:
                        r_accurate += 1
                    else:
                        r_true_med_pred_high += 1
                else:
                    if y_pred[i] < 4:
                        r_true_high_pred_low += 1
                    elif y_pred[i] < 7:
                        r_true_high_pred_med += 1
                    else:
                        r_accurate += 1
    len_tool2_right = len(tool2_right_test_data)

    right_camel_hump_data[test_participant_id-1,:] = [test_participant_id, r_accurate/len_tool2_right, r_true_low_pred_med/len_tool2_right, r_true_low_pred_high/len_tool2_right, r_true_med_pred_high/len_tool2_right, r_true_med_pred_low/len_tool2_right, r_true_high_pred_med/len_tool2_right, r_true_high_pred_low/len_tool2_right]
    print(f"Participant {test_participant_id} done\n")
#%% Save the results
df_l_stringer = pd.DataFrame(left_stringer_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_l_stringer.to_excel(r'C:\Users\anand\Desktop\left_stringer_data.xlsx',index=False)
df_l_camel_hump = pd.DataFrame(left_camel_hump_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_l_camel_hump.to_excel(r'C:\Users\anand\Desktop\left_camel_hump_data.xlsx',index=False)

df_r_stringer = pd.DataFrame(right_stringer_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_r_stringer.to_excel(r'C:\Users\anand\Desktop\right_stringer_data.xlsx',index=False)
df_r_camel_hump = pd.DataFrame(right_camel_hump_data,columns=['Participant ID','Accuracy','True Low Predicted Medium','True Low Predicted High','True Medium Predicted High','True Medium Predicted Low','True High Predicted Medium','True High Predicted Low'])
df_r_camel_hump.to_excel(r'C:\Users\anand\Desktop\right_camel_hump_data.xlsx',index=False)



# %%
