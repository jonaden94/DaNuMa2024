import os
import os.path as osp
import pandas as pd
import random
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch.utils.data import Dataset


class WeightDataset(Dataset):
    def __init__(self, weights_df_path, images_base_dir):
        self.weights_df = pd.read_csv(weights_df_path)
        self.images_base_dir = images_base_dir
        self.transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                ])

    def __getitem__(self, i):
        # select row from dataframe and get corresponding group, weighting number, transponder number and weight
        info = self.weights_df.loc[i, ['group', 'weighting', 'transponder', 'weight']]
        transponder = info['transponder']
        weighting = info['weighting']
        group = info['group']
        weight = info['weight']
        weight = torch.tensor(weight).float()

        # load one random image corresponding to the weighting of the selected row
        images_folder = f'Gr_{group}_WG_{weighting}_{transponder}_depth'
        images_dir = osp.join(self.images_base_dir, images_folder)
        image_name = random.choice(os.listdir(images_dir))
        image_path = osp.join(images_dir, image_name)
        image = Image.open(image_path)

        # transform and return image
        image = self.transform(image)
        return image, weight

    def __len__(self):
        return len(self.weights_df)