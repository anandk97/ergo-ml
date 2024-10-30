import torch
from torch import nn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
# Load Pytorch dataset from pt file
# Avg test loss with 1 second lookback : 9.616, Time taken : 34 seconds, 2 hidden layers, midlayer size 5 - RNN
# Avg loss : 7.041364, Time taken : 838 seconds, 2 hidden layers, midlayer size 5 - GRU - 1 second lookback
data = torch.load('/Users/anandkrishnan/Documents/HAL TLV Data/Pytorch datasets/combined_dataset_00_to_03_lookback_125.pt')

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

device = "cpu"


# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Define model class with an RNN structure
class HAL_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, midlayer_size = 5):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, midlayer_size)
        self.linear2 = nn.Linear(midlayer_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear1(out[:, -1, :])
        out = self.linear2(out).flatten()
        return out
    
# Create a model instance
model = HAL_GRU(input_size=5, hidden_size=10, num_layers=2, output_size=1,midlayer_size=5)
# Try more layers, 3 onwards
model.to(device)
print(model)
# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.float()
        y = y.float()
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2500 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            y = y.float()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

epochs = 50
start_time = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
end_time = time.time()
print("Time taken: ", end_time - start_time)
print("Done!")

# Save the model weights
torch.save(model.state_dict(), '/Users/anandkrishnan/Documents/HAL TLV Data/Pytorch datasets/GRU_lookback_50.pt')






