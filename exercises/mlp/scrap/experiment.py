import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch import nn
from torch.nn import functional as nnf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import itertools
from collections import defaultdict
import numpy as np




############### data
# data gen params
N = 128
# N = 256 # first experiment
var = 1
start = 0.1
end = 20
# end = 25 # first experiment
split = 0.5
seed = 42
np.random.seed(seed)
def f_x(x):
    return np.sin(x) + np.log(x)

# data
x = np.linspace(start, end, N)
y = f_x(x)
noise = np.random.normal(0, var,N)
y = y + noise
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.5, random_state=seed)




###################### training objects
class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_ratio=0.1):
        super().__init__()

        modules = []
        for i in range(len(layer_sizes)-1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            if not i == len(layer_sizes)-2:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout_ratio))

        self.layers = nn.Sequential(*modules)


    def forward(self, x):
        return self.layers(x)

def train(model, trainloader, valloader, optimizer, epochs, device, val_interval, savepath_best_statedict=None):
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    model.train()

    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(model, trainloader, optimizer, device)
        train_losses.append(train_loss)

        if epoch % val_interval == 0:
            val_loss = validate(model, valloader, device)
            val_losses.append(val_loss)

        if savepath_best_statedict is not None:
            if val_loss < min_val_loss:
                torch.save(model.state_dict(), savepath_best_statedict)
                min_val_loss = val_loss
    return train_losses, val_losses

def train_one_epoch(model, trainloader, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in trainloader:
        x_batch = x_batch[:, None].float().to(device)
        y_batch = y_batch[:, None].float().to(device)
        y_pred = model(x_batch)

        optimizer.zero_grad()
        loss = nnf.mse_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(trainloader)

def validate(model, valloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in valloader:
            x_batch = x_batch[:, None].float().to(device)
            y_batch = y_batch[:, None].float().to(device)
            y_pred = model(x_batch)
            loss = nnf.mse_loss(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(valloader)


####################### experiment

# fixed training hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The model is running on {device}.")
epochs = 5000
lr = 0.01
val_interval = 1
batch_size = 64

# hvaried training hyperparameters
dropout_ratios = [0, 0.1, 0.3, 0.6]
weight_decays = [0, 0.1, 0.01, 0.001]
# hidden_layer_sizes = [[64, 128, 256, 128, 64],
#                [28, 56, 112, 56, 28],
#                [14, 28, 56, 28, 14],]
parameter_combinations = itertools.product(dropout_ratios, weight_decays)

# fixed training objects
trainset = ToyDataset(x_train, y_train)
valset = ToyDataset(x_val, y_val)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# experiment
experimental_results = defaultdict(list)
for combination in tqdm(parameter_combinations):
    dropout_ratio = combination[0]
    weight_decay = combination[1]
    # layer_sizes = combination[2]
    experimental_results['dropout_ratio'].append(dropout_ratio)
    experimental_results['weight_decay'].append(weight_decay)
    # experimental_results['layer_sizes'].append(layer_sizes)

    # layer_sizes = [1] + layer_sizes + [1]
    layer_sizes = [1, 64, 128, 256, 512, 256, 128, 64, 1]
    model = MLP(layer_sizes=layer_sizes, dropout_ratio=dropout_ratio).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = train(model, trainloader, valloader, optimizer, epochs, device, val_interval)
    experimental_results['min_val_loss'].append(np.min(val_losses))

    experimental_results_df = pd.DataFrame.from_dict(experimental_results)
    experimental_results_df.to_pickle('/usr/users/henrich1/repos/exercises_summer_school/exercises/mlp/experimental_results_2.pkl')
    