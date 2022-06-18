from __future__ import print_function
import torch
from utils_lap_kld import trajDataset, RMSE, gmm_kld, toggle_grad
from model_lap_kld import TFunc, GFunc, MdnModelTraj
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import itertools
import pandas as pd
plt.switch_backend('agg')
plt.rcParams['agg.path.chunksize'] = 10000
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

## Arguments
args = {}
# args['sample_size'] = 0
args['sample_size_1'] = 0
args['sample_size_2'] = 0

# parameters for MDN
args_mdn = {}
args_mdn['use_cuda'] = True
args_mdn['encoder_size'] = 256
args_mdn['hist_length'] =9
args_mdn['out_length'] = 20
args_mdn['dyn_embedding_size'] = 64
args_mdn['input_embedding_size'] = 64
args_mdn['num_gauss'] = 20
args_mdn['batch_size'] = 4096
# L1 or L2 regularization
args_mdn['weight_regularization'] = 2
# lambda coef for pi, sigma, mu penalty
args_mdn['lambda_coef'] = (0.1, 0.1, 0.1)
#defaut1000
trainMdnEpochs = 10
learning_rate_m = 0.0004
b1 = 0.90
b2 = 0.999
vehicle_num = 5
eig_num = 3
prefix = './data/'
postfix = '-v'+str(vehicle_num)+'-'+str(eig_num)+'.mat'
dataset_name_list = ['ZS-3','MA-3', 'FT-2']
args_mdn['hist_length'] = args_mdn['hist_length']*2+eig_num*vehicle_num
args_mdn['out_length'] = args_mdn['out_length']*2
mdn_model = MdnModelTraj(args_mdn)
mdn_model_1 = MdnModelTraj(args_mdn)
mdn_model_2 = MdnModelTraj(args_mdn)
if args_mdn['use_cuda']:
    mdn_model = mdn_model.cuda()
    mdn_model_1 = mdn_model_1.cuda()
    mdn_model_2 = mdn_model_2.cuda()
optimizer_m = torch.optim.Adam(mdn_model.parameters(), lr=learning_rate_m, betas=(b1,b2))

distance_arr = torch.zeros(len(dataset_name_list),len(dataset_name_list))
start_mdn_epoch = 0
start_uv = 0
# before all begin, we need to save initial state for another dataset pair
checkpoint_initial = {
    'mdn_model': mdn_model.state_dict(),
    'optimizer_m': optimizer_m.state_dict(),
}
kld_initial_check_name = 'checkpoint_lap_kld_initial_g'+str(args_mdn['num_gauss'])+'.pkl'
torch.save(checkpoint_initial, kld_initial_check_name)
resume = True
kld_calculation_check_name = 'checkpoint_lap_kld_calculation_g'+str(args_mdn['num_gauss'])+'.pkl'
if resume:
    if os.path.isfile(kld_calculation_check_name):
        checkpoint = torch.load(kld_calculation_check_name)
        start_uv = checkpoint['uv'] + 1
        tqdm.write("=> loaded kld calculation checkpoint (restart from {})".format(checkpoint['uv']))
    else:
        tqdm.write("=> no kld calculation checkpoint found")
# sometimes we did not reach stable convergence, so we need to re-run wasted ones
# if os.path.exists(best_param_file):
#     best_param_arr = load_best_param(best_param_file, best_param_arr)
uv_list = np.arange(0, len(dataset_name_list)**2).tolist()
uv_list = uv_list[start_uv:]

for u in range(len(dataset_name_list)):
    fig_path = 'lap_kld_record-v'+str(vehicle_num)+'-eig'+str(eig_num) + \
               '/gaussian'+str(args_mdn['num_gauss'])+'/dataset'+str(u)+'/figs'
    model_path = 'lap_kld_record-v'+str(vehicle_num)+'-eig'+str(eig_num) + \
                 '/gaussian'+str(args_mdn['num_gauss'])+'/dataset'+str(u)+'/trained_models'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dataset_name_1 = prefix + dataset_name_list[u] + postfix

    tqdm.write('fitting GMM model of {}'.format(dataset_name_1))
    dataSet_1 = trajDataset(mat_file=dataset_name_1)
    dataloader_1 = DataLoader(dataSet_1, batch_size=args_mdn['batch_size'], shuffle=True, num_workers=12,
                              drop_last=True, collate_fn=dataSet_1.collate_fn)
    args['sample_size_1'] = dataSet_1.__len__() // args_mdn['batch_size'] * args_mdn['batch_size']
    info_interval = (args['sample_size_1'] // args_mdn['batch_size']) // 2
    iterations = args['sample_size_1'] // args_mdn['batch_size']
    m_loss_record = torch.zeros(trainMdnEpochs * iterations, 1)
    if resume:
        if os.path.isfile(model_path+'/checkpoint_lap_kld.pkl'):
            checkpoint = torch.load(model_path+'/checkpoint_lap_kld.pkl')
            start_mdn_epoch = checkpoint['epoch'] + 1
            mdn_model.load_state_dict(checkpoint['mdn_model'])
            optimizer_m.load_state_dict(checkpoint['optimizer_m'])
            m_loss_record = checkpoint['m_loss_record']
            tqdm.write("=> loaded mdn checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            tqdm.write("=> no mdn checkpoint found")

    with trange(start_mdn_epoch, trainMdnEpochs) as t_epoch:
        for epoch in t_epoch:
            t_epoch.set_description("mdn epoch %i for dataset %i" % (epoch, u))
            epoch_start_t = time.time()
            avg_m_loss = 0
            m_loss = 0
            avg_tr_time = 0
            # dataset_iter_1 = iter(dataloader_1)
            for i, (hist, fut) in tqdm(enumerate(dataloader_1)):
                #print('hist:', hist)
                #print('fut:',fut)
                start_time = time.time()
                if i == 0 and epoch == 0:
                    gaussian_center_init = fut[0:args_mdn['num_gauss'], :].view(-1, )
                    mdn_model.bias_initialization(gaussian_center_init)
                if args_mdn['use_cuda']:
                    hist = hist.cuda()
                    fut = fut.cuda()
                    #hist = F.normalize(hist,p=1)
                    #fut=F.normalize(fut,p=1)
                #print('is_any_nan_hist', torch.isnan(hist).any())
                #print('is_any_nan_fut', torch.isnan(fut).any())
                toggle_grad(mdn_model, True)
                mdn_model.zero_grad()
                optimizer_m.zero_grad()

                pi, sigma, mu = mdn_model(hist)
                #print('pi',pi,'sigma',sigma,'mu',mu)
                m_loss = mdn_model.mdn_loss(pi, sigma, mu, fut)
                # if torch.isnan(m_loss):
                #     print(pi, sigma, mu, m_loss)
                #     assert 2 > 3
                m_loss.backward()
                optimizer_m.step()
                m_loss_record[epoch*iterations+i] = m_loss.item()

                avg_m_loss += m_loss.item()
                batch_time = time.time()-start_time
                avg_tr_time += batch_time
                if i % info_interval == (info_interval-1):
                    tqdm.write('Training Epoch no:{},iter: {}/{},'
                               'm_loss:{},avg m loss:{} over {} iterations,'
                               'avg train time:{}'.format(epoch,i,iterations,
                                                          m_loss.item(),avg_m_loss / info_interval,
                                                          info_interval,avg_tr_time))
                    # tqdm.write('pi:{}'.format(pi[0,:]))
                    avg_m_loss = 0
                    avg_tr_time = 0
            epoch_end_t = time.time()
            tqdm.write('mdn epoch no:{},spent total time is:{}'.format(epoch, epoch_end_t - epoch_start_t))
            plot_epoch_num = 10
            if epoch > plot_epoch_num:
                plt.plot(np.arange(iterations*(epoch - plot_epoch_num), iterations*epoch),
                         m_loss_record[iterations*(epoch - plot_epoch_num):iterations*epoch].detach().cpu().numpy())
            else:
                plt.plot(np.arange((epoch+1)*iterations), m_loss_record[:(epoch+1)*iterations].detach().cpu().numpy())
            path = fig_path + '/m_loss_epoch' + str(epoch)
            plt.savefig(path)
            #plt.show()
            plt.close()

            plt.plot(np.arange((epoch+1)*iterations), m_loss_record[:(epoch+1)*iterations].detach().cpu().numpy())
            path = fig_path + '/full_m_loss'
            plt.savefig(path)
            #plt.show()
            plt.close()

            # checkpoint,save every 3 epochs
            if epoch % 3 == 0 or epoch == trainMdnEpochs-1:
                checkpoint = {
                    'epoch': epoch,
                    'mdn_model': mdn_model.state_dict(),
                    'optimizer_m': optimizer_m.state_dict(),
                    'm_loss_record': m_loss_record
                }
                # torch.cuda.empty_cache()
                torch.save(checkpoint, model_path+'/checkpoint_lap_kld.pkl')
                torch.save(mdn_model.state_dict(),
                           model_path + '/mdn_model_epoch' + str(epoch) + '.tar')

                sio.savemat(model_path + '/m_loss.mat',
                            {'m_loss': m_loss_record.detach().cpu().numpy()})
                tqdm.write('mdn checkpoint saved')
    checkpoint_initial = torch.load(kld_initial_check_name)
    mdn_model.load_state_dict(checkpoint_initial['mdn_model'])
    optimizer_m.load_state_dict(checkpoint_initial['optimizer_m'])
    start_mdn_epoch = 0
    tqdm.write('mdn model and optimizer are REINITIALIZED!')
# all models are saved, now start calculating kl divergence through sampling
for uv in uv_list:
    u = uv // len(dataset_name_list)
    v = uv % len(dataset_name_list)
    data_path = 'lap_kld_record-v' + str(vehicle_num) + '-eig' + str(eig_num) + \
                '/gaussian'+str(args_mdn['num_gauss'])+'/dataset' + str(u) + '/kldis2dataset' + str(v)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset_name_1 = prefix+dataset_name_list[u]+postfix
    dataset_name_2 = prefix+dataset_name_list[v]+postfix
    tqdm.write('calculate distance between {} and {}'.format(dataset_name_1, dataset_name_2))

    dataSet_1 = trajDataset(mat_file=dataset_name_1)
    dataSet_2 = trajDataset(mat_file=dataset_name_2)
    dataloader_1 = DataLoader(dataSet_1, batch_size=args_mdn['batch_size'], shuffle=True, num_workers=16,
                              drop_last=True, collate_fn=dataSet_1.collate_fn)
    dataloader_2 = DataLoader(dataSet_2, batch_size=args_mdn['batch_size'], shuffle=True, num_workers=16,
                              drop_last=True, collate_fn=dataSet_2.collate_fn)

    args['sample_size_1'] = dataSet_1.__len__()//args_mdn['batch_size']*args_mdn['batch_size']
    args['sample_size_2'] = dataSet_2.__len__()//args_mdn['batch_size']*args_mdn['batch_size']
    # args['sample_size'] = min(args['sample_size_1'], args['sample_size_2'])
    info_interval = (args['sample_size_1'] // args_mdn['batch_size']) // 2

    # we calculate conditional kl divergence for each condition x in dataset1 and dataset2
    # ckl_distance_record = torch.zeros(args['sample_size_1']+args['sample_size_2'],1)
    ckl_distance_record = torch.zeros(args['sample_size_1'],1)

    model_path_1 = 'lap_kld_record-v'+str(vehicle_num)+'-eig'+str(eig_num) + \
                   '/gaussian'+str(args_mdn['num_gauss'])+'/dataset'+str(u)+'/trained_models'
    model_path_2 = 'lap_kld_record-v'+str(vehicle_num)+'-eig'+str(eig_num) + \
                   '/gaussian'+str(args_mdn['num_gauss'])+'/dataset'+str(v)+'/trained_models'
    mdn_model_1.load_state_dict(torch.load(model_path_1+'/mdn_model_epoch' + str(trainMdnEpochs-1) + '.tar'))
    mdn_model_2.load_state_dict(torch.load(model_path_2+'/mdn_model_epoch' + str(trainMdnEpochs-1) + '.tar'))
    # set weight_regularization to 0 to avoid calculation penalty loss
    mdn_model_1.set_weight_reg(0)
    mdn_model_2.set_weight_reg(0)
    sample_times = 5
    sample_size = args_mdn['batch_size']*sample_times
    tqdm.write('calculating GMM KLD on dataset {}'.format(dataset_name_1))
    for i, (hist, fut) in enumerate(tqdm(dataloader_1)):
        if args_mdn['use_cuda']:
            hist = hist.cuda()
            fut = fut.cuda()
        toggle_grad(mdn_model_1, False)
        toggle_grad(mdn_model_2, False)
        with torch.no_grad():
            pi_1, sigma_1, mu_1 = mdn_model_1(hist)
            pi_2, sigma_2, mu_2 = mdn_model_2(hist)
        # calculate for each condition on dataset 1
        for k in trange(args_mdn['batch_size']):
            param_1 = {'pi':pi_1[k,:].unsqueeze(0).expand(sample_size, args_mdn['num_gauss']),
                       'sigma':sigma_1[k,:,:].unsqueeze(0).expand(
                           sample_size,args_mdn['num_gauss'],args_mdn['out_length']),
                       'mu':mu_1[k,:,:].unsqueeze(0).expand(
                           sample_size, args_mdn['num_gauss'], args_mdn['out_length']
                       )}
            param_2 = {'pi': pi_2[k, :].unsqueeze(0).expand(sample_size, args_mdn['num_gauss']),
                       'sigma': sigma_2[k, :, :].unsqueeze(0).expand(
                           sample_size, args_mdn['num_gauss'], args_mdn['out_length']
                       ),
                       'mu': mu_2[k, :, :].unsqueeze(0).expand(
                           sample_size, args_mdn['num_gauss'], args_mdn['out_length']
                       )}
            index = i*args_mdn['batch_size']+k
            ckl_distance_record[index] = gmm_kld(mdn_model_1, param_1, mdn_model_2, param_2).item()
    # tqdm.write('calculating GMM KLD on dataset {}'.format(dataset_name_2))
    # for i, (hist, fut) in enumerate(tqdm(dataloader_2)):
    #     if args_mdn['use_cuda']:
    #         hist = hist.cuda()
    #         fut = fut.cuda()
    #     toggle_grad(mdn_model_1, False)
    #     toggle_grad(mdn_model_2, False)
    #     with torch.no_grad():
    #         pi_1, sigma_1, mu_1 = mdn_model_1(hist)
    #         pi_2, sigma_2, mu_2 = mdn_model_2(hist)
    #     # calculate for each condition on dataset 2
    #     for k in trange(args_mdn['batch_size']):
    #         param_1 = {'pi': pi_1[k, :].unsqueeze(0).expand(sample_size, args_mdn['num_gauss']),
    #                    'sigma': sigma_1[k, :, :].unsqueeze(0).expand(
    #                        sample_size, args_mdn['num_gauss'], args_mdn['out_length']),
    #                    'mu': mu_1[k, :, :].unsqueeze(0).expand(
    #                        sample_size, args_mdn['num_gauss'], args_mdn['out_length']
    #                    )}
    #         param_2 = {'pi': pi_2[k, :].unsqueeze(0).expand(sample_size, args_mdn['num_gauss']),
    #                    'sigma': sigma_2[k, :, :].unsqueeze(0).expand(
    #                        sample_size, args_mdn['num_gauss'], args_mdn['out_length']
    #                    ),
    #                    'mu': mu_2[k, :, :].unsqueeze(0).expand(
    #                        sample_size, args_mdn['num_gauss'], args_mdn['out_length']
    #                    )}
    #         index = args['sample_size_1'] + i * args_mdn['batch_size'] + k
    #         ckl_distance_record[index] = gmm_kld(mdn_model_1, param_1, mdn_model_2, param_2).item()

    ckl_distance = ckl_distance_record.mean()
    sio.savemat(data_path+'/ckl_distance.mat', {'distance': ckl_distance.numpy(),
                                                'distance_record': ckl_distance_record.numpy()})
    checkpoint = {
        'uv': uv
    }
    torch.save(checkpoint, kld_calculation_check_name)

