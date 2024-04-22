import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import datetime
import pdb

class NiftyReturnsDataset(Dataset):
    def __init__(self, returns_data, labels, t=None):
        assert len(returns_data) == len(labels)
        self.returns_data = returns_data
        self.labels = labels

        self.T = t
        self.returns_data=torch.tensor(
            self.returns_data, dtype=torch.float32)
        self.labels=torch.tensor(
            self.labels, dtype=torch.float32)

    def __getitem__(self, index):
        target, sample= self.returns_data[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.returns_data)

def create_dataset(df, start_date, end_date, mean=None, std=None):
    data=df
    dates = pd.read_pickle('/Users/chinu/Downloads/adarnn/dates.pkl')
    # pdb.set_trace()
    # print(data[0])
    # print(data[1])
    label = data[1]
    feat=data[0]
    referece_start_time=datetime.datetime(2015, 2, 2, 9, 15,0)
    referece_end_time=datetime.datetime(2024, 1, 23, 15, 29, 0)
    # pdb.set_trace()
    # print(start_date)
    # print(end_date)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    #index_start=(pd.to_datetime(start_date) - referece_start_time).days
    #index_end=(pd.to_datetime(end_date) - referece_start_time).days
    index_start = dates.index(pd.to_datetime(start_date).date())
    index_end = dates.index(pd.to_datetime(end_date).date())
    # pdb.set_trace()
    # print(index_end)
    # print(index_start)
    # print(f'*Length of Features {len(feat)}')
    # print(f'*{index_end-index_start}')
    # print(f'*{len(feat)-index_start}')
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]

    # pdb.set_trace()
    # print(label)
    # print(feat)

    return NiftyReturnsDataset(label, feat)

def load_nifty_returns_data(file_path, start_date, end_date, batch_size=32, shuffle=True):
    df = pd.read_pickle (file_path)  # Assuming the data is stored in a CSV file
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Filter data based on start and end dates
    df = df.loc[start_date:end_date]
    
    returns_data = df['Returns'].values.astype(np.float32)
    labels = df['Label'].values.astype(np.float32)
    
    dataset = NiftyReturnsDataset(returns_data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def get_nifty_data(data_file, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    df=pd.read_pickle(data_file)
    dataset=create_dataset(df, start_time,
                             end_time, mean=mean, std=std)
    # pdb.set_trace()
    train_loader=DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def compute_nifty_returns_statistic(file_path, start_date, end_date):
    df = pd.read_pickle(file_path) 
    feat, label= df[0], df[1]
    referece_start_time=datetime.datetime(2015, 2, 2, 9, 15,0)
    referece_end_time=datetime.datetime(2022, 2, 2, 15, 29,0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start=(pd.to_datetime(start_date) - referece_start_time).days
    index_end=(pd.to_datetime(end_date) - referece_start_time).days
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]
    feat=feat.reshape(-1, feat.shape[2])
    mu_train=np.mean(feat, axis=0)
    sigma_train=np.std(feat, axis=0)

    return mu_train, sigma_train
