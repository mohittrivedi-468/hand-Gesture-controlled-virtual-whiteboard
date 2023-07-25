import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader

# Define the custom dataset class
class Data(Dataset):
    def __init__(self, path):
        self.dataset = np.array(pd.read_csv(path))
        self.map = {'draw': 0, 'erase': 1, 'none': 2}

    def __getitem__(self, index):
        yvals = torch.tensor(self.map[self.dataset[index][0]], dtype=torch.long)
        xvals = torch.from_numpy(self.dataset[index][1:].astype(np.float32))
        return xvals, yvals

    def __len__(self):
        return self.dataset.shape[0]

# Create an instance of the dataset
dataset = Data(path='dataset/dataset.csv')

# Create data loaders for training
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the model
model = Model()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"models{epoch+1}.pt")

# Test the model
model.eval()
with torch.no_grad():
    noise = torch.randn((20, 63)).to(device)
    outputs = model.forward(noise)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
