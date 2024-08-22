import torchvision
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np
import itertools
from collections import defaultdict
from torch.optim import AdamW
import pandas as pd


###################### training objects
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_ratio=0.1):
        super().__init__()

        modules = []
        modules.append(nn.Flatten(start_dim=1))
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
    train_accuracies = []
    val_accuracies = []
    min_val_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_one_epoch(model, trainloader, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if epoch % val_interval == 0:
            val_loss, val_accuracy = validate(model, valloader, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        if savepath_best_statedict is not None:
            if val_loss < min_val_loss:
                torch.save(model.state_dict(), savepath_best_statedict)
                min_val_loss = val_loss
    return train_losses, val_losses, train_accuracies, val_accuracies

def train_one_epoch(model, trainloader, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for x_batch, y_batch in trainloader:
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.squeeze().long().to(device)
        y_pred = model(x_batch)
        pred_class = y_pred.argmax(dim=1)
        accuracy = (pred_class == y_batch).sum() / len(y_batch)

        optimizer.zero_grad()
        loss = nnf.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy.item()
    return total_loss / len(trainloader), total_accuracy / len(trainloader)

def validate(model, valloader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for x_batch, y_batch in valloader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.squeeze().long().to(device)
            y_pred = model(x_batch)
            pred_class = y_pred.argmax(dim=1)
            accuracy = (pred_class == y_batch).sum() / len(y_batch)
            
            loss = nnf.cross_entropy(y_pred, y_batch)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
    return total_loss / len(valloader), total_accuracy / len(valloader)


####################### experiment

# fixed training hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The model is running on {device}.")
epochs = 30
lr = 0.001
val_interval = 1

# varied training hyperparameters
dropout_ratios = [0, 0.1, 0.3, 0.6]
weight_decays = [0.1, 0.01]
hidden_layer_sizes = [[3*32*32, 32, 32, 10],
               [3*32*32, 64, 64, 10],
               [3*32*32, 128, 128, 10],
               [3*32*32, 256, 256, 10],
               [3*32*32, 384, 384, 10],
               [3*32*32, 32, 32, 32, 10],
               [3*32*32, 64, 64, 64, 10],
               [3*32*32, 128, 128, 128, 10],
               [3*32*32, 256, 256, 256, 10],
               [3*32*32, 384, 384, 384, 10],
               [3*32*32, 32, 32, 32, 32, 10],
               [3*32*32, 64, 64, 64, 64, 10],
               [3*32*32, 128, 128, 128, 128, 10],
               [3*32*32, 256, 256, 256, 256, 10],
               [3*32*32, 384, 384, 384, 384, 10]]
parameter_combinations = itertools.product(dropout_ratios, weight_decays, hidden_layer_sizes)

# fixed training objects
trainset = torchvision.datasets.CIFAR10(root='/usr/users/henrich1/repos/exercises_summer_school/data/convnet', train=False, transform=torchvision.transforms.ToTensor())
every_fifth_idx = list(range(0, len(trainset), 5))
trainset = torch.utils.data.Subset(trainset, every_fifth_idx)

num_samples = trainset.dataset.data.shape[0]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_samples, 
                                            num_workers=2)
imgs, _ = next(iter(trainloader))
dataset_mean = torch.mean(imgs, dim=(0,2,3))
dataset_std = torch.std(imgs, dim=(0,2,3))

normalized_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(dataset_mean, dataset_std)
])

trainset = torchvision.datasets.CIFAR10(root='/usr/users/henrich1/repos/exercises_summer_school/data/convnet', train=True, transform=normalized_transform)
every_fifth_idx = list(range(0, len(trainset), 5))
trainset = torch.utils.data.Subset(trainset, every_fifth_idx)
valset = torchvision.datasets.CIFAR10(root='/usr/users/henrich1/repos/exercises_summer_school/data/convnet', train=False, transform=normalized_transform)
every_fifth_idx = list(range(0, len(valset), 5))
valset = torch.utils.data.Subset(valset, every_fifth_idx)

trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=1000, shuffle=False, num_workers=2)

# experiment
experimental_results = defaultdict(list)
for combination in tqdm(parameter_combinations):
    dropout_ratio = combination[0]
    weight_decay = combination[1]
    layer_sizes = combination[2]
    experimental_results['dropout_ratio'].append(dropout_ratio)
    experimental_results['weight_decay'].append(weight_decay)
    experimental_results['layer_sizes'].append(layer_sizes)

    model = MLP(layer_sizes=layer_sizes, dropout_ratio=dropout_ratio).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses, train_accuracies, val_accuracies = train(model, trainloader, valloader, optimizer, epochs, device, val_interval)
    experimental_results['min_val_loss'].append(np.min(val_losses))
    experimental_results['max_val_acc'].append(np.max(val_accuracies))
    experimental_results['train_losses'].append(train_losses)
    experimental_results['val_losses'].append(val_losses)
    experimental_results['train_accuracies'].append(train_accuracies)
    experimental_results['val_accuracies'].append(val_accuracies)

    experimental_results_df = pd.DataFrame.from_dict(experimental_results)
    experimental_results_df.to_pickle('/usr/users/henrich1/repos/exercises_summer_school/exercises/convnet/experimental_results.pkl')
    