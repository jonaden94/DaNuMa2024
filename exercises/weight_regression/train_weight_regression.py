import os.path as osp
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch

from exlib.model import WeightResNet34
from exlib.dataset import WeightDataset
from exlib.logger import init_train_logger, print_log, close_logger




def train(model, optimizer, trainloader):
    losses_epoch = []
    model.train()
    for input, target in trainloader:
        input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(input).squeeze()
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses_epoch.append(loss.item())
    
    return np.mean(losses_epoch)


def val(model, valloader):
    losses_val = []
    model.eval()
    with torch.no_grad():
        for input, target in valloader:
            input, target = input.cuda(), target.cuda()

            output = model(input).squeeze()
            loss = criterion(output, target)
            losses_val.append(loss.item())
    return np.mean(losses_val)


# def init_train_logger(args)
# def print_log(msg, logger=None)
# def close_logger(logger)


if __name__ == '__main__':

    # directories
    data_dir = 'data/weight_regression'
    depth_dir = osp.join(data_dir, 'depth_subsampled')
    weights_train_df_path = osp.join(data_dir, 'weights_train.csv')
    weights_val_df_path = osp.join(data_dir, 'weights_val.csv')

    # parameters
    initial_lr = 0.01


    logger = init_train_logger(save_dir), print_log, close_logger
    
    
    # Initialize the model
    model = WeightResNet34().cuda()

    # datasets and dataloaders
    trainset = WeightDataset(weights_train_df_path, depth_dir)
    valset = WeightDataset(weights_val_df_path, depth_dir)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    valloader = DataLoader(valset, batch_size=16, shuffle=False)    

    # loss criterion. optimizer and scheduler
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

    # train loop
    epochs = 300
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, optimizer, trainloader)
        val_loss = val(model, valloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'trainloss: {train_loss}')
        print(f'valloss: {val_loss}')

        metrics = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        metrics.to_pickle('/usr/users/henrich1/exercises_summer_school/data/train_weight/metrics.pkl')

        