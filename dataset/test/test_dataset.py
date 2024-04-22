import pandas as pd
import datetime
import math
import pandas as pd
import torch
from base.loss import TransferLoss  # Assuming TransferLoss is defined elsewhere

def TDC(num_domain, data_file, station, dis_type='coral'):
    start_time = datetime.datetime.strptime('2008-01-01 09:45:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime('2020-11-27 15:30:00', '%Y-%m-%d %H:%M:%S')
    num_day = (end_time - start_time).days
    split_N = 10
    
    # Load OHLC data
    data = pd.read_csv(data_file)
    ohlc_data = data[['open', 'high', 'low', 'close']][:num_day]  # Assuming OHLC data is available for the specified time period

    # Convert OHLC data to PyTorch tensor
    feat = torch.tensor(ohlc_data, dtype=torch.float32)
    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[2])
    feat = feat.cuda()  # Assuming CUDA is available and you want to use GPU
    
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
                        # Extract OHLC data for selected time periods
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_day) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]

                        index_part2_start = start + math.floor(selected[j] / split_N * num_day) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_day) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]

                        # Compute distance using TransferLoss
                        criterion_transfer = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transfer.compute(feat_part1, feat_part2)

                distance_list.append(dis_temp)
                selected.remove(can)

            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])

        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]),
                                                                  hours=0)
            else:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]) + 1,
                                                                  hours=0)
            sel_end_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i]), hours=23)
            sel_start_time = datetime.datetime.strftime(sel_start_time, '%Y-%m-%d %H:%M')
            sel_end_time = datetime.datetime.strftime(sel_end_time, '%Y-%m-%d %H:%M')
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("Error: Invalid number of domains")