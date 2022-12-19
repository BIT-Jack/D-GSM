from __future__ import print_function
import torch
from utils import ngsimDataset, laplacianTopKvec, laplacianFixEigNum
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
plt.switch_backend('agg')
plt.rcParams['agg.path.chunksize'] = 10000
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

## Arguments
args = {}
args['hist_length'] =8
args['fut_length'] = 20
args['batch_size'] = 512
args['sample_size_1'] = args['batch_size']*8
args['sample_size_2'] = args['batch_size']*16
args['max_eig_num'] = 3
args['max_vehicle_num'] = 5
args['use_cuda'] = True
args['decay_rate'] = 0.90
args['out_range_dist'] = 10.0

batch_to_sample = [85, 51, 201, 61, 86]
# max_vehicle_num need to be decided after reviewed one by one for both datasets
# Initialize network
dataset_name_list = ['data/Scenario-MA','data/Scenario-FT','data/Scenario-ZS','data/Scenario-EP','data/Scenario-SR']
distance_arr = torch.zeros(len(dataset_name_list),len(dataset_name_list))
for u in range(0, len(dataset_name_list)):
        dataset_name_1 = dataset_name_list[u]+'.mat'

        dataSet_1 = ngsimDataset(mat_file=dataset_name_1)
        dataDataloader_1 = DataLoader(dataSet_1,batch_size=args['batch_size'],shuffle=True,num_workers=4,
                                      drop_last = True, collate_fn=dataSet_1.collate_fn)
        print(dataSet_1)
        

        # print(len(dataDataloader_1),len(dataDataloader_2))
        args['sample_size_1'] = args['batch_size']*len(dataDataloader_1)
        print('len:', len(dataDataloader_1), dataset_name_1)
        print('total sample num:', args['sample_size_1'])
        ## Variables holding train and validation loss values:
        # we implement WGAN-GP instead
        hist_arr_1 = np.zeros((args['sample_size_1'],2*args['hist_length']))
        fut_arr_1 = np.zeros((args['sample_size_1'],2*args['fut_length']))
        decay_arr = pow(args['decay_rate'], np.linspace(0, args['hist_length']-1,args['hist_length']))

        # 特征向量的维度等于取的最大特征值的数目乘以每个特征向量的维度
        # 每个特征向量的维度等于laplacian matrix的维度，即args['max_vehicle_num']
        nei_arr_1 = np.zeros((args['sample_size_1'],args['max_eig_num']*args['max_vehicle_num']))
        la_arr = np.zeros((args['max_vehicle_num'],args['max_vehicle_num']))
        sampled_num_1 = 0

        start_time = time.time()
        while sampled_num_1 < args['sample_size_1']:
            # time*batch*dim
            hist, nbrs, _, _, _, fut, _, _, index_division = next(iter(dataDataloader_1))
            if torch.isnan(hist).any():
                print('hist_isnan')
            

            hist = hist.permute(1,0,2).contiguous().view(args['batch_size'],2*args['hist_length'])
            fut = fut.permute(1,0,2).contiguous().view(args['batch_size'],2*args['fut_length'])
            hist_arr_1[sampled_num_1:sampled_num_1+args['batch_size'], :] = hist.numpy()
            fut_arr_1[sampled_num_1:sampled_num_1+args['batch_size'], :] = fut.numpy()
            nei_arr_1[sampled_num_1:sampled_num_1+args['batch_size'], :] = \
                laplacianFixEigNum(hist, nbrs, index_division,
                                   decay_arr, args['max_vehicle_num'],
                                   args['out_range_dist'], args['max_eig_num'])
            #print('size_nei_arr_1:', nei_arr_1.shape)
           # print('size_hist_arr_1:', hist_arr_1.shape)
            if sampled_num_1> args['batch_size']*batch_to_sample[u]:
                break
            sampled_num_1 += args['batch_size']
        print('dataset ',dataset_name_1, ' fetch complete.')
        feature_1 = np.concatenate((hist_arr_1,nei_arr_1),axis=1)
        print('feature1:', feature_1.shape)
        save_filename = dataset_name_list[u]+'-v'+str(args['max_vehicle_num'])+'-'+str(args['max_eig_num'])+'.mat'
        sio.savemat(save_filename, {'hist': feature_1,
                                    'fut': fut_arr_1})
        end_time = time.time()
        print('time cost ', end_time-start_time)

