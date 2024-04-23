import torch.nn as nn
import torch
import torch.optim as optim

import os
import argparse
import datetime
import numpy as np

from tqdm import tqdm
from utils import utils
from base.AdaRNN import AdaRNN

import pretty_errors
import dataset.nifty_process as nifty_process
import matplotlib.pyplot as plt
import pdb
import pickle


def pprint(*text):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow() +
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)

# Changed functionality to remove the cuda so that it can be used on mac 
def get_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return AdaRNN(use_bottleneck=True, bottleneck_width=64, n_input=args.d_feat, n_hiddens=n_hiddens,  n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq, trans_loss=args.loss_type)


def train_AdaRNN(args, model, optimizer, train_loader_list, epoch, dist_old=None, weight_mat=None):
    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    dist_mat = torch.zeros(args.num_layers, args.len_seq) #changed for cuda functionality 
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)
    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].float(
            ), data[1].float()

            list_feat.append(feature)
            list_label.append(label_reg)
        
        # pdb.set_trace()
        # print(list_feat)
        # print(list_label)

        flag = False
        index = get_index(len(data_all) - 1)
        # print(index)
        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            # print(s1)
            # print(s2)
            # print(list_feat[s1].shape[0])
            # print(list_feat[s2].shape[0])
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1)
        for i in range(len(index)):
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]
            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            #pdb.set_trace()
            #print(label_reg_s.shape)
            #print(label_reg_t.shape)
            feature_all = torch.cat((feature_s, feature_t), 0)

             #debugger block
            # pdb.set_trace()
            # print(feature_s)
            # pdb.set_trace()
            # print(feature_t)
            # pdb.set_trace()
            # print(feature_all)

            if epoch < args.pre_epoch:
                pred_all, loss_transfer, out_weight_list = model.forward_pre_train(
                    feature_all, len_win=args.len_win)
                # pdb.set_trace()
                # print(out_weight_list)
            else:
                pred_all, loss_transfer, dist, weight_mat = model.forward_Boosting(
                    feature_all, weight_mat)
                dist_mat = dist_mat + dist
            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]

            loss_s = criterion(pred_s, label_reg_s)
            loss_t = criterion(pred_t, label_reg_t)
            #pdb.set_trace()
            #print(pred_s.shape)
            #print(label_reg_s.shape)
            loss_l1 = criterion_1(pred_s, label_reg_s)
            # loss_s = criterion(pred_s)
            # loss_t = criterion(pred_t)
            # loss_l1 = criterion_1(pred_s)


            total_loss = total_loss + loss_s + loss_t + args.dw * loss_transfer
        loss_all.append(
            [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    if epoch >= args.pre_epoch:
        if epoch > args.pre_epoch:
            weight_mat = model.update_weight_Boosting(
                weight_mat, dist_old, dist_mat)
        return loss, loss_l1, weight_mat, dist_mat
    else:
        weight_mat = transform_type(out_weight_list)
        return loss, loss_l1, weight_mat, None


def train_epoch_transfer_Boosting(model, optimizer, train_loader_list, epoch, dist_old=None, weight_mat=None):
    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    dist_mat = torch.zeros(args.num_layers, args.len_seq)
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)
    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].float(
            ),data[1].float()
            list_feat.append(feature)
            list_label.append(label_reg)
        flag = False
        index = get_index(len(data_all) - 1)
        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1)
        for i in range(len(index)):
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]
            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            feature_all = torch.cat((feature_s, feature_t), 0)

            pred_all, loss_transfer, dist, weight_mat = model.forward_Boosting(
                feature_all, weight_mat)
            dist_mat = dist_mat + dist
            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]

            loss_s = criterion(pred_s, label_reg_s)
            loss_t = criterion(pred_t, label_reg_t)
            loss_l1 = criterion_1(pred_s, label_reg_s)

            total_loss = total_loss + loss_s + loss_t + args.dw * loss_transfer

        loss_all.append(
            [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    if epoch > 0: #args.pre_epoch:
        weight_mat = model.update_weight_Boosting(
            weight_mat, dist_old, dist_mat)
    return loss, loss_l1, weight_mat, dist_mat


def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i+1, num_domain+1):
            index.append((i, j))
    return index


def train_epoch_transfer(args, model, optimizer, train_loader_list):
    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)

    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].float(
            ), data[1].float()
            list_feat.append(feature)
            list_label.append(label_reg)
        # pdb.set_trace()
        # print(list_label)
        flag = False
        index = get_index(len(data_all) - 1)
        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        ###############
        total_loss = torch.zeros(1)
        for i in range(len(index)):
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]
            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            # pdb.set_trace()
            # print(label_reg_s)
            # print(label_reg_t)
            feature_all = torch.cat((feature_s, feature_t), 0)

            pred_all, loss_transfer, out_weight_list = model.forward_pre_train(
                feature_all, len_win=args.len_win)
            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]

            loss_s = criterion(pred_s, label_reg_s)
            loss_t = criterion(pred_t, label_reg_t)
            loss_l1 = criterion_1(pred_s, label_reg_s)

            total_loss = total_loss + loss_s + loss_t + args.dw * loss_transfer
        loss_all.append(
            [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    return loss, loss_l1, out_weight_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_epoch(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    # for feature, label, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
    for feature,label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        # feature, label_reg = feature.float(), label_reg.float()
        feature, label_reg = feature.float(), label_reg.float()
        with torch.no_grad():
            pred = model.predict(feature)
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
    # pdb.set_trace()
    # print(test_loader)
    # print(len(test_loader))
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = loss_r / len(test_loader)
    return loss, loss_1, loss_r


def test_epoch_inference(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    i = 0
    #pdb.set_trace()

    for feature, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.float(), label_reg.float()
        with torch.no_grad():
            pred = model.predict(feature)

        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
        if i == 0:
            label_list = label_reg.cpu().numpy()
            predict_list = pred.cpu().numpy()
        else:
            # print(label_list.shape)
            # print(label_reg.size())
            # pdb.set_trace()
            label_list = np.hstack((label_list, label_reg.cpu().numpy()))
            predict_list = np.hstack((predict_list, pred.cpu().numpy()))

        i = i + 1
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = total_loss_r / len(test_loader)
    return loss, loss_1, loss_r, label_list, predict_list


def inference(model, data_loader):
    loss, loss_1, loss_r, label_list, predict_list = test_epoch_inference(
        model, data_loader, prefix='Inference')
    return loss, loss_1, loss_r, label_list, predict_list


def inference_all(output_path, model, model_path, loaders):
    pprint('inference...')
    loss_list = []
    loss_l1_list = []
    loss_r_list = []
    model.load_state_dict(torch.load(model_path))
    i = 0
    list_name = ['train', 'valid', 'test']
    for loader in loaders:
        # pdb.set_trace()
        loss, loss_1, loss_r, label_list, predict_list = inference(
            model, loader)
        loss_list.append(loss)
        loss_l1_list.append(loss_1)
        loss_r_list.append(loss_r)
        i = i + 1
    return loss_list, loss_l1_list, loss_r_list



def transform_type(init_weight):
    weight = torch.ones(args.num_layers, args.len_seq)
    for i in range(args.num_layers):
        for j in range(args.len_seq):
            weight[i, j] = init_weight[i][j].item()
    return weight


def main_transfer(args):
    print(args)

    output_path = args.outdir + '_' + '_' + args.model_name + \
        args.loss_type + '_' + str(args.pre_epoch) + \
        '_' + str(args.dw) + '_' + str(args.lr)
    save_model_name = args.model_name + '_' + args.loss_type + \
        '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)
    pprint('create loaders...')

    train_loader_list, valid_loader, test_loader = nifty_process.load_nifty_data_multi_domain(
        '/Users/chinu/Downloads/adarnn', args.batch_size, args.num_domain, args.data_mode)

    args.log_file = os.path.join(output_path, 'run.log')
    pprint('create model...')
    model = get_model(args.model_name)
    num_model = count_parameters(model)
    print('#model params:', num_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
   
    best_score = np.inf
    best_epoch, stop_round = 0, 0
    weight_mat, dist_mat = None, None
    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)
        pprint('training...')
        if args.model_name in ['Boosting']:
            loss, loss1, weight_mat, dist_mat = train_epoch_transfer_Boosting(
                model, optimizer, train_loader_list,  epoch, dist_mat, weight_mat)
        elif args.model_name in ['AdaRNN']:
            loss, loss1, weight_mat, dist_mat = train_AdaRNN(
                args, model, optimizer, train_loader_list, epoch, dist_mat, weight_mat)
        else:
            print("error in model_name!")
        pprint(loss, loss1)

        pprint('evaluating...')
        train_loss, train_loss_l1, train_loss_r = test_epoch(
            model, train_loader_list[0], prefix='Train')
        val_loss, val_loss_l1, val_loss_r = test_epoch(
            model, valid_loader, prefix='Valid')
        test_loss, test_loss_l1, test_loss_r = test_epoch(
            model, test_loader, prefix='Test')

        pprint('valid %.6f, test %.6f' %
               (val_loss_l1, test_loss_l1))
        pprint('valid %.6f' % test_loss_l1)

        if val_loss < best_score:
            best_score = val_loss
            stop_round = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_path, save_model_name))
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best val score:', best_score, '@', best_epoch)

    loaders = train_loader_list[0], valid_loader, test_loader
    # pdb.set_trace()
    # print(test_loader)
    # loaders = train_loader_list[0], test_loader
    loss_list, loss_l1_list, loss_r_list = inference_all(output_path, model, os.path.join(
        output_path, save_model_name), loaders)
    pprint('MSE: train %.6f, valid %.6f, test %.6f' %
           (loss_list[0], loss_list[1], loss_list[2]))
    pprint('L1:  train %.6f, valid %.6f, test %.6f' %
           (loss_l1_list[0], loss_l1_list[1], loss_l1_list[2]))
    pprint('RMSE: train %.6f, valid %.6f, test %.6f' %
           (loss_r_list[0], loss_r_list[1], loss_r_list[2]))
    pprint('Finished.')
    pprint('MSE: train %.6f, test %.6f' %
           (loss_list[0],loss_list[1]))
    pprint('L1:  train %.6f, test %.6f' %
           (loss_l1_list[0], loss_l1_list[1]))
    pprint('RMSE: train %.6f,test %.6f' %
           (loss_r_list[0], loss_r_list[1]))
    pprint('Finished.')


def get_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='AdaRNN')
    parser.add_argument('--d_feat', type=int, default=9)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=40)  # 20, 30, 50

    # training
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--smooth_steps', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--dw', type=float, default=0.5) # 0.01, 0.05, 5.0
    parser.add_argument('--loss_type', type=str, default='adv')
    parser.add_argument('--data_mode', type=str,
                        default='tdc')
    parser.add_argument('--num_domain', type=int, default=3)
    parser.add_argument('--len_seq', type=int, default=375)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--outdir', default='/Users/chinu/Downloads/adarnn/outputs/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main_transfer(args)