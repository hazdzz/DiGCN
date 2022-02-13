import os
import argparse

import random
import time

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim

from script import dataloader, utility, earlystopping
from model import models

import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='DiGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable or disable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
    parser.add_argument('--mode', type=str, default='test', choices=['tuning', 'test'], \
                        help='running mode, default as test, tuning as alternative')
    parser.add_argument('--dataset', type=str, default='corar')
    parser.add_argument('--model', type=str, default='digcn', help='graph neural network model')
    parser.add_argument('--gso_type', type=str, default='rw_renorm_adj', \
                        help='graph shift operator, default as rw_renorm_adj')
    parser.add_argument('--pr_type', type=str, default='pr', choices=['pr', 'ppr'], \
                        help='default as pr')
    parser.add_argument('--alpha', type=float, default=0.2, help='transport probability alpha')
    parser.add_argument('--K', type=int, default=2, help='K layer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, defaut as 0.01')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (L2 penalty)')
    parser.add_argument('--n_hid', type=int, default=64, help='the channel size of hidden layer feature, default as 64')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5, help='dropout rate, default as 0.5')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    SEED = args.seed
    set_env(SEED)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.mode != 'test' and args.mode != 'tuning':
        raise ValueError(f'ERROR: Wrong running mode')
    else:
        mode = args.mode

    dataset = args.dataset

    if args.model != 'digcn':
        raise ValueError(f'ERROR: This model is undefined.')
    else:
        model_name = args.model

    if args.gso_type != 'rw_renorm_adj' and args.gso_type != 'rw_renorm_lap':
        raise ValueError(f'ERROR: This graph shift operator is not adopted in this model.')
    else:
        gso_type = args.gso_type
    
    if args.pr_type != 'pr' and args.pr_type != 'ppr':
        raise ValueError(f'ERROR: Wrong PageRank type!')
    else:
        pr_type = args.pr_type

    if mode == 'tuning':
        param = nni.get_next_parameter()
        alpha, K, lr, weight_decay, droprate = [*param.values()]
    else:
        if args.alpha <= 0 or args.alpha > 1:
            raise ValueError(f'ERROR: The transport probability alpha should be in (0, 1]')
        else:
            alpha = args.alpha
        if args.K <= 1:
            raise ValueError(f'ERROR: The layer number K is smaller than 2!')
        else:
            K = args.K
        lr = args.lr
        weight_decay = args.weight_decay
        droprate = args.droprate
    
    n_hid = args.n_hid
    enable_bias = args.enable_bias
    epochs = args.epochs
    opt = args.opt
    patience = args.patience

    model_save_dir = os.path.join('./model/save', dataset)
    os.makedirs(name=model_save_dir, exist_ok=True)
    model_save_path = model_name + '_' + gso_type + '_' + str(pr_type) + '_' + str(alpha) + '_' + str(K) + '_layer' + '.pth'
    model_save_path = os.path.join(model_save_dir, model_save_path)

    return device, dataset, model_name, gso_type, pr_type, alpha, lr, weight_decay, droprate, n_hid, enable_bias, K, epochs, opt, patience, model_save_path
    
def process_data(device, dataset, gso_type, pr_type, alpha):
    if dataset == 'corar' or dataset == 'citeseerr' or dataset == 'pubmed' or dataset == 'ogbn-arxiv':
        feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_citation_data(dataset)
    elif dataset == 'cornell' or dataset == 'texas' or dataset == 'washington' or dataset == 'wisconsin':
        feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_webkb_data(dataset)

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    if pr_type == 'pr':
        filter = utility.calc_chung_pr_gso(adj, gso_type, alpha)
    else:
        filter = utility.calc_chung_ppr_gso(adj, gso_type, alpha)

    # convert matrix to tensor
    # move tensor to device
    if sp.issparse(feature):
        feature = utility.cnv_sparse_mat_to_coo_tensor(feature, device)
    else:
        feature = torch.from_numpy(feature).to(device)
    if sp.issparse(filter):
        filter = utility.cnv_sparse_mat_to_coo_tensor(filter, device)
    else:
        filter = torch.from_numpy(filter).to(device)

    label = torch.LongTensor(label).to(device)

    return feature, filter, label, idx_train, idx_val, idx_test, n_feat, n_class

def prepare_model(n_feat, n_hid, n_class, enable_bias, K, droprate, patience, model_save_path, opt, lr, weight_decay):
    model = models.DiGCN(n_feat, n_hid, n_class, enable_bias, K, droprate).to(device)
    
    loss = nn.NLLLoss()
    early_stopping = earlystopping.EarlyStopping(patience=patience, path=model_save_path, verbose=True)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    else:
        raise ValueError(f'ERROR: The {opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    return model, loss, early_stopping, optimizer, scheduler

def train(epochs, model, optimizer, scheduler, early_stopping, feature, filter, label, loss, idx_train, idx_val):
    train_time_list = []
    for epoch in range(epochs):
        train_epoch_begin_time = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        output = model(feature, filter)
        loss_train = loss(output[idx_train], label[idx_train])
        acc_train = utility.calc_accuracy(output[idx_train], label[idx_train])
        loss_train.backward()
        optimizer.step()
        #scheduler.step()
        train_epoch_end_time = time.perf_counter()
        train_epoch_time_duration = train_epoch_end_time - train_epoch_begin_time
        train_time_list.append(train_epoch_time_duration)

        loss_val, acc_val = val(model, label, output, loss, idx_val)
        print('Epoch: {:03d} | Learning rate: {:.8f} | Train loss: {:.6f} | Train acc: {:.6f} | Val loss: {:.6f} | Val acc: {:.6f} | Training duration: {:.6f}'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), train_epoch_time_duration))
        #nni.report_intermediate_result(acc_val.item())

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print('Early stopping.')
            break
    
    mean_train_epoch_time_duration = np.mean(train_time_list)
    print('\nTraining finished.\n')

    return mean_train_epoch_time_duration

def val(model, label, output, loss, idx_val):
    model.eval()
    with torch.no_grad():
        loss_val = loss(output[idx_val], label[idx_val])
        acc_val = utility.calc_accuracy(output[idx_val], label[idx_val])

    return loss_val, acc_val

def test(model, model_save_path, feature, filter, label, loss, idx_test, model_name, dataset, mean_train_epoch_time_duration):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        output = model(feature, filter)
        loss_test = loss(output[idx_test], label[idx_test])
        acc_test = utility.calc_accuracy(output[idx_test], label[idx_test])
        print('Model: {} | Dataset: {} | Test loss: {:.6f} | Test acc: {:.6f} | Training duration: {:.6f}'.format(model_name, dataset, loss_test.item(), acc_test.item(), mean_train_epoch_time_duration))
    #nni.report_final_result(acc_test.item())

if __name__ == "__main__":
    device, dataset, model_name, gso_type, pr_type, alpha, lr, weight_decay, droprate, n_hid, enable_bias, K, epochs, opt, patience, model_save_path = get_parameters()
    feature, filter, label, idx_train, idx_val, idx_test, n_feat, n_class = process_data(device, dataset, gso_type, pr_type, alpha)
    model, loss, early_stopping, optimizer, scheduler = prepare_model(n_feat, n_hid, n_class, enable_bias, K, droprate, patience, model_save_path, opt, lr, weight_decay)
    mean_train_epoch_time_duration = train(epochs, model, optimizer, scheduler, early_stopping, feature, filter, label, loss, idx_train, idx_val)
    test(model, model_save_path, feature, filter, label, loss, idx_test, model_name, dataset, mean_train_epoch_time_duration)