import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A

from skimage.io import imread
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from csbdeep.utils import normalize


# def get_training_augmentation(dim=224):
    
#     train_transform = [
#         A.Resize(dim, dim),
#         A.Flip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.RandomResizedCrop(dim, dim, scale=(0.8, 1.0), ratio=(1., 1.), p=0.5),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
    
#     return A.Compose(train_transform)


# def get_validation_augmentation(dim):
    
#     test_transform = [
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
    
#     return A.Compose(test_transform)


class MyDataset(Dataset):
    def __init__(self, csv_path, data_type):
        df = pd.read_csv(csv_path)
        df = df[df['data_type'] == data_type]
        # if data_type == 'train':
        #     df = df[df['data_type'] != 'test']
        # else:
        #     df = df[df['data_type'] == 'test'] 
        
        # Original implementation: directly use the absolute paths stored in CSV.
        # self.x = df['x'].tolist()

        # Rebuild local feature paths without changing the CSV file. For example:
        # data_c16_ctranspath.csv -> <csv_dir>/c16_ctranspath/pt_files/*.pt
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        dataset_name = csv_name.removeprefix('data_')
        feature_folder = 'pts' if dataset_name == 'tcga' else 'pt_files'
        feature_dir = os.path.join(csv_dir, dataset_name, feature_folder)

        self.x = [
            os.path.join(feature_dir, os.path.basename(str(old_path)))
            for old_path in df['x'].tolist()
        ]
        self.y = df['y'].tolist()
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, i):
        # print('Load data {}.'.format(i))
        x = torch.load(self.x[i])
        y = float(self.y[i])
        mark = x.shape[0]
        # Original implementation only handled paths separated with '/'.
        # name = self.x[i].split('/')[-1].split('.')[0]
        name = os.path.splitext(os.path.basename(self.x[i]))[0]
        
        return x, y, mark, name
    
