import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class AsteroidDataset(Dataset):
    def __init__(self, csv_file, transform=None, input_param=None):
        self.asteroid_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.input_param = input_param

    def __len__(self):
        return len(self.asteroid_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = np.array([], dtype=np.float32)
        for param in self.input_param:
            param_index = self.asteroid_frame.columns.get_loc(param)
            param_value = self.asteroid_frame.iloc[idx, param_index].astype(np.float32)
            inputs = np.append(inputs, param_value)

        albedo = self.asteroid_frame.iloc[idx, -2].astype(np.float32)
        diameter = self.asteroid_frame.iloc[idx, -1].astype(np.float32)
        targets = np.array([albedo, diameter], dtype=np.float32)

        if self.transform:
            inputs = self.transform(inputs)

        return inputs, targets


def get_dataloaders(train_data_file, val_data_file, batch_size=32, input_param=None):

    train_dataset = AsteroidDataset(train_data_file, input_param=input_param)
    val_dataset = AsteroidDataset(val_data_file, input_param=input_param)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
