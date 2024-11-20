#%% Import libraries
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
#%% Load data
torch.set_default_dtype(torch.float32)
data_directory = r'C:\Users\anand\Desktop\HAL Pytorch Datasets'
test_participant_id = 1
train_participant_id_range = range(2,16)
tool_id_range = range(1,3)

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
# Shuffle the data
# left_train_loader = DataLoader(left_train_data, batch_size=batch_size, shuffle=True)
# tool1_left_test_loader = DataLoader(tool1_left_test_data, batch_size=batch_size, shuffle=True)
# tool2_left_test_loader = DataLoader(tool2_left_test_data, batch_size=batch_size, shuffle=True)
# Don't shuffle the data
def collate_fn(batch):
    data, targets = zip(*batch)
    data = [x.to(torch.float32) for x in data]
    targets = [y for y in targets]  # Adjust dtype as needed
    data = torch.stack(data)
    targets = torch.stack(targets)
    return data, targets
left_train_loader = DataLoader(left_train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
left_test_data = ConcatDataset([tool1_left_test_data, tool2_left_test_data])
left_test_loader = DataLoader(left_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#%% Define the model
torch.set_float32_matmul_precision('medium')
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



#%% Initialize the model
# from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience = 3,
    verbose=True,
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=25,
    callbacks=[early_stopping_callback],
    accelerator = 'gpu',
    devices =1
)

input_size = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create a model instance
drop_out = 0
mid_layer_size = 90
hidden_layer_size = 10
number_layers = 3
# Random seed = 0
torch.manual_seed(0)
model = HAL_GRU(input_size=6, hidden_size=hidden_layer_size, num_layers=number_layers, output_size=1,midlayer_size=mid_layer_size, dropout=drop_out)
# Try more layers, 3 onwards
# start_time = time.time()
# trainer.fit(model, left_train_loader, left_test_loader)
# print(f"Time taken: {time.time() - start_time}\n")
# print all the details of the model
# model structure, batch size, optimizer, loss function, learning rate, number of epochs, early stopping, etc.
print(model)

# Save the trained model
# torch.save(model.state_dict(), 'C:\Users\anand\Desktop\Saved HAL Models\HAL_GRU_test_'+str(test_participant_id)+'.pt')
#%% Risk scores
left_stringer = np.zeros((1,8))
l_accurate = 0
l_true_low_pred_med = 0
l_true_low_pred_high = 0
l_true_med_pred_high = 0
l_true_med_pred_low = 0
l_true_high_pred_med = 0
l_true_high_pred_low = 0
tool1_left_dataloader = DataLoader(tool1_left_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
for (X, y) in tool1_left_dataloader:
    X = X.to(torch.float32)
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
left_stringer = [test_participant_id, l_accurate, l_true_low_pred_med, l_true_low_pred_high, l_true_med_pred_high, l_true_med_pred_low, l_true_high_pred_med]
print("Left Stringer: ", left_stringer)


left_camel_hump = np.zeros((1,8))
l_accurate = 0
l_true_low_pred_med = 0
l_true_low_pred_high = 0
l_true_med_pred_high = 0
l_true_med_pred_low = 0
l_true_high_pred_med = 0
l_true_high_pred_low = 0
tool2_left_dataloader = DataLoader(tool2_left_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

for (X, y) in tool2_left_dataloader:
    X = X.to(torch.float32)
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
left_camel_hump = [test_participant_id, l_accurate/len_tool2_left, l_true_low_pred_med/len_tool2_left, l_true_low_pred_high/len_tool2_left, l_true_med_pred_high/len_tool2_left, l_true_med_pred_low/len_tool2_left, l_true_high_pred_med/len_tool2_left]

print("Left Camel Hump: ", left_camel_hump)

