# Import statements
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
torch.manual_seed(0)
# Load the dataset from Pytorch datasets folder, train on ID 0,1,2,3,4,6,7 and test on ID 5
# Load pytorch dataset from pt file for stringer and camel hump right hand data


for j in range(3,8):
    train_data = torch.load(r"C:\Users\anand\Desktop\HAL TLV\Pytorch datasets\id0_left_hist_bin_w_force_sum_dataset.pt")
    for i in range(1,8):
        if i == j:
            test_data = torch.load(r"C:\Users\anand\Desktop\HAL TLV\Pytorch datasets\id"+str(i)+"_left_hist_bin_w_force_sum_dataset.pt")
            continue
        # Concatenate the new tensors into the existing tensor
        train_data = torch.utils.data.ConcatDataset([train_data, torch.load(r"C:\Users\anand\Desktop\HAL TLV\Pytorch datasets\id"+str(i)+"_left_hist_bin_w_force_sum_dataset.pt")])





    print("Train data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    batch_size = 64
    epochs = 20
    # Bin all X values into 25 bins and use the bin number as the input




    train_dataloader = DataLoader(train_data, batch_size=batch_size,drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,drop_last=True)

    # # Define model class with 2D CNN structure with 250x5 input and 1 output
    # class HAL_NN(nn.Module):
    #     def __init__(self, midlayer_size = 60, dropout=0.2,batch_size=64,last_layer_size=30):
    #         super().__init__()

    #         self.linear1 = nn.Linear(1250, midlayer_size)
    #         self.dropout = nn.Dropout(dropout)
    #         self.linear3 = nn.Linear(midlayer_size, last_layer_size)
    #         self.linear4 = nn.Linear(last_layer_size, 1)
            
    #     def forward(self, x):
    #         out = self.linear1(x.reshape(batch_size,1250))
    #         out = self.dropout(out)
    #         out = self.linear3(out)
    #         out = self.dropout(out)
    #         out = self.linear4(out).flatten()
    #         return out

    # # Initialize the model
    # model = HAL_NN().to(device)
    # print(model)

    # # Define model class with an RNN structure
    class HAL_GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, midlayer_size = 10, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.linear1 = nn.Linear(hidden_size, midlayer_size)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(midlayer_size, output_size)
            
        def forward(self, x):
            out, _ = self.gru(x)
            out = self.linear1(out[:, -1, :])
            out = self.dropout(out)
            out = self.linear2(out).flatten()
            return out
        
    # Create a model instance
    drop_out = 0.5
    mid_layer_size = 90
    hidden_layer_size = 10
    number_layers = 3
    model = HAL_GRU(input_size=6, hidden_size=hidden_layer_size, num_layers=number_layers, output_size=1,midlayer_size=mid_layer_size, dropout=drop_out)
    # Try more layers, 3 onwards
    model.to(device)
    print(model)
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define training loop
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X = X.float().to(device)
            y = y.float().to(device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 45000 == 0:
                loss, current = loss.item(), batch*len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    # Define test loop
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in dataloader:
                X = X.float().to(device)
                y = y.float().to(device)
                
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        return test_loss
    # Train the model
    start_time = time.time()
    test_loss_history = []
    for t in range(epochs):

        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss_history.append(test(test_dataloader, model, loss_fn))

    print(f"Time taken: {time.time() - start_time}\n")
    plot_epochs = np.arange(1,epochs+1,1)
    plt.figure()
    plt.plot(plot_epochs,test_loss_history)
    plt.xlabel('Epoch')

    # highlight lowest test loss
    min_loss = min(test_loss_history)
    min_loss_index = test_loss_history.index(min_loss)
    plt.plot(min_loss_index+1,min_loss,'ro')
    # plt.annotate('Lowest test loss', xy=(min_loss_index+1,min_loss), xytext=(min_loss_index+1,min_loss+0.1))
    # Add loss value as well
    plt.annotate(str(round(1 - min_loss/len(test_data),2)), xy=(min_loss_index+1,min_loss), xytext=(min_loss_index+1,min_loss+0.1))
    # only integer x ticks
    plt.xticks(np.arange(1, epochs+1, 1.0))
    plt.ylabel('Test loss')
    plt.title('Test loss vs Epoch')
    filename = r'GRU_left_id_'+str(j)+'_test'
    plt.savefig(filename+'.png', dpi=300)
    # Save the model
    torch.save(model.state_dict(), r"C:\Users\anand\Desktop\HAL TLV\\"+filename+".pt")