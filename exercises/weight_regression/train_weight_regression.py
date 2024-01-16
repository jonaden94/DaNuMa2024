import os.path as osp
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

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




if __name__ == '__main__':

    # directories
    # data_dir = 'data/weight_regression'
    data_dir = '/scratch/users/henrich1/data_summer_school/weight_regression'
    images_base_dir = osp.join(data_dir, 'images')
    train_df_path = osp.join(data_dir, 'train.csv')
    val_df_path = osp.join(data_dir, 'val.csv')

    exercise_dir = 'exercises/weight_regression'
    results_dir = osp.join(exercise_dir, 'results')

    # parameters
    initial_lr = 0.001
    batch_size = 8
    decay_factor = 0.1
    patience = 20
    epochs = 150

    # logger
    logger = init_train_logger(results_dir)
    
    # Initialize the model
    print_log('Create training objects...', logger=logger)
    model = WeightResNet34().cuda()

    # datasets and dataloaders
    trainset = WeightDataset(train_df_path, images_base_dir)
    valset = WeightDataset(val_df_path, images_base_dir)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)    

    # loss criterion. optimizer and scheduler
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, patience=patience)

    # train loop
    print_log('Start training...', logger=logger)
    train_losses = []
    val_losses = []
    min_loss = 10000
    for epoch in range(epochs):
        train_loss = train(model, optimizer, trainloader)
        val_loss = val(model, valloader)
        scheduler.step(val_loss)
        print_log(f'epoch: {epoch}/{epochs}, lr: {optimizer.param_groups[0]["lr"]}, train_loss: {train_loss}, val_loss: {val_loss}', logger=logger)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save data
        if val_loss < min_loss:
            torch.save(model.state_dict(), osp.join(results_dir, f'best_ckpt_epoch.pt'))
        metrics = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'lr': optimizer.param_groups[0]['lr']
        })
        metrics.to_pickle(osp.join(results_dir, 'metrics.pkl'))

    close_logger(logger)
