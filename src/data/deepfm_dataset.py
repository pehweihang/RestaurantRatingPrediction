import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RestaurantDataset(Dataset):
    def __init__(self, data_path, train=True):
        print(data_path)
        df = pd.read_csv(data_path)
        self.data = df.iloc[:, :-1].values
        self.target = df.iloc[:, -1].values
        self.features = self.data.shape[1]
        self.feature_sizes = [
            len(np.unique(self.data[:, i])) for i in range(self.data.shape[1])
        ]
        if train:
            assert np.all(
                [
                    self.feature_sizes[i] == self.data[:, i].max() + 1
                    for i in range(self.data.shape[1])
                ]
            )

    def __getitem__(self, idx):
        data, target = self.data[idx, :], self.target[idx]
        Xi = torch.from_numpy(data.astype(np.int32)).unsqueeze(-1)

        Xv = torch.from_numpy(np.ones_like(data).astype(np.int32))
        return Xi, Xv, target

    def __len__(self):
        return len(self.data)
