import os
import dataset.data_act as data_act
import pandas as pd
import dataset.nifty_data as nifty_data
from datetime import time, datetime
import datetime
from base.loss_transfer import TransferLoss
import torch
import math
import pdb
from dataset import nifty_process

def load_act_data(data_folder, batch_size=64, domain="1_20"):
    x_train, y_train, x_test, y_test = data_act.load_data(data_folder, domain)
    x_train, x_test = x_train.reshape(
        (-1, x_train.shape[2], 1, x_train.shape[1])), x_test.reshape((-1, x_train.shape[2], 1, x_train.shape[1]))
    transform = None
    train_set = data_act.data_loader(x_train, y_train, transform)
    test_set = data_act.data_loader(x_test, y_test, transform)
    train_loader = data_act.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data_act.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, train_loader, test_loader

def get_split_time(num_domain = 3, mode='pre_process', data_file=None, dis_type='coral'):
    spilt_time = {
        '4': [('2015-03-06 09:15:00', '2016-12-31 15:29:00'), ('2017-01-03 09:15:00', '2018-12-28 15:29:00'),('2019-01-03 09:15:00', '2020-12-28 15:29:00'),('2021-01-03 09:15:00', '2021-12-28 15:29:00')]
    }
    if mode == 'pre_process':
        return spilt_time[str(num_domain)]
    if mode == 'tdc':
        return TDC(num_domain, data_file, dis_type=dis_type)
    else:
        print("error in mode")

def TDC(num_domain, data_file, dis_type='coral'):
    start_time_1 = datetime.datetime.strptime(
        '2015-02-02 09:15:00', '%Y-%m-%d %H:%M:%S')
    end_time_1 = datetime.datetime.strptime(
        '2021-01-12 15:29:00', '%Y-%m-%d %H:%M:%S')
    start_time = datetime.datetime.strptime(
        '2015-02-02 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(
        '2021-01-12 00:00:00', '%Y-%m-%d %H:%M:%S')
    dates = "/Users/chinu/Downloads/adarnn/dates.pkl"
    df = pd.read_pickle(dates)
    index = []
    counter = 0
    for date in df:
        if start_time_1.date() == date or end_time_1.date() == date:
            index.append(counter)
        counter+=1
    split_N = 10
    num_day = index[1] + 1
    data = pd.read_pickle(data_file)
    feat = data[0][0:num_day]
    feat = torch.tensor(feat, dtype=torch.float32)
    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[2])
    if torch.cuda.is_available():
        feat = feat
    else:
        print("CUDA is not available. Running on CPU.")

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0

    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_day) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_day) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_day) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start = df[int(num_day / split_N * selected[i - 1])]
                day_time = time(9,15,0)
                final_time_start = datetime.datetime.combine(sel_start,day_time)
            else:
                sel_start = df[int(num_day / split_N * selected[i - 1] + 1) + 1]
                day_time = time(9,15,0)
                final_time_start = datetime.datetime.combine(sel_start,day_time)
            sel_end = df[int(num_day / split_N * selected[i])]
            day_time_end = time(15,29,0)
            final_time_end = datetime.datetime.combine(sel_end, day_time_end)
            sel_start_time = datetime.datetime.strftime(final_time_start, '%Y-%m-%d %H:%M:%S')
            sel_end_time = datetime.datetime.strftime(final_time_end, '%Y-%m-%d %H:%M:%S')
            res.append((sel_start_time, sel_end_time))
        # pdb.set_trace()
        # print(res)
        return res
    else:
        print("error in number of domain")


#Resolve Issue in test loader
def load_nifty_data_multi_domain(file_path, batch_size=6, number_domain=2, mode='pre_process', dis_type='coral'):
    data_file = os.path.join(file_path, "nifty_1.pkl")
    mean_train, std_train = nifty_data.compute_nifty_returns_statistic(data_file, start_date='2015-02-02 09:15:00',
                                                                    end_date='2021-01-12 15:29:00')
    split_time_list = get_split_time(number_domain, mode=mode, data_file=data_file, dis_type=dis_type)

    train_list = []
    for i in range(len(split_time_list)):
        time_temp = split_time_list[i]
        train_loader = nifty_data.get_nifty_data(data_file, start_time=time_temp[0],
                                                     end_time=time_temp[1], batch_size=batch_size, mean=mean_train, std=std_train)
        
        train_list.append(train_loader)

    valid_vld_loader = nifty_data.get_nifty_data(data_file, start_time='2021-01-13 09:15:00',
                                                     end_time='2022-02-07 15:29:00', batch_size=batch_size, mean=mean_train, std=std_train)
    test_loader = nifty_data.get_nifty_data(data_file, start_time='2022-02-08 09:15:00',
       
                                         end_time='2024-01-20 15:29:00', batch_size=batch_size, mean=mean_train, std=std_train, shuffle=False)
    return train_list, valid_vld_loader, test_loader
