

import os

import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd, tensor
import torch.optim.lr_scheduler as lr_scheduler
from model import *

import quadprog
import time
parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=20)
parser.add_argument('--pred_seq_len', type=int, default=40)
parser.add_argument('--dataset', default='MA',
                    help='MA, FT, SR, EP, ZS')    

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')

#GEMLearning specific parameters
parser.add_argument('--tasks', type=int, default=1,
                    help='the number of continual scenarios, from 1 to 5')
parser.add_argument('--mem_size', type=int, default=128,
                    help='allocated data number of each episodic memory, how many batches in each memory w.r.t each task')
parser.add_argument('--margin', type=float, default=0.5,
                    help='for quadprog computing')
parser.add_argument('--eps', type=float, default=0.001,
                    help='for quadprog computing too')
parser.add_argument('--cur_task', type=int, default=0,
                    help='current task index, from 0 to 4')

args = parser.parse_args()
##
avg_mem =int( args.mem_size/args.cur_task) # total memory / num of past scenarios





print('*'*30)
print("Training initiating....")
print(args)


def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = './datasets/'+args.dataset+'/'

dset_train = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)

loader_train = DataLoader(
        dset_train,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=16,
        drop_last = True)


dset_val = TrajectoryDataset(
        data_set+'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)

loader_val = DataLoader(
        dset_val,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=16,
        drop_last = True)

checkpoint_dir = './checkpoint/'+args.tag+'/'

#Defining the model 

model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
#Continual learning, load the last learned model
#cur_task: 0, 1, 2, 3, 4
#loaded the last learned model (CL)
if args.cur_task == 1:
    model.load_state_dict(torch.load('./checkpoint/social-stgcnn-MA/val_best.pth'))
    #print("last model(MA) is loaded")
elif args.cur_task == 2:
    model.load_state_dict(torch.load('./checkpoint/social-stgcnn-FT/val_best_{:.0f}.pth'.format(int(args.mem_size/(args.cur_task-1)))))
    #print("last model(FT) is loaded")
elif args.cur_task == 3:
    model.load_state_dict(torch.load('./checkpoint/social-stgcnn-ZS/val_best_{:.0f}.pth'.format(int(args.mem_size/(args.cur_task-1)))))
    #print("last model(ZS) is loaded")
elif args.cur_task == 4:
    model.load_state_dict(torch.load('./checkpoint/social-stgcnn-EP/val_best_{:.0f}.pth'.format(int(args.mem_size/(args.cur_task-1)))))
    #print("last model(EP) is loaded")

#Training settings 

optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    




if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args_{:.0f}.pkl'.format(avg_mem), 'wb') as fp:
    pickle.dump(args, fp)
    


#print('Data and model loaded')
#print('Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

#Initialize the episodic memory

memory_data = [[], [], [], [], []] #load memory data into this list


#Calculate each parameters' number of elements
grad_numels = []
grad_data_num = []
ct_params = 1
for params in model.parameters():
    if ct_params == 23 or ct_params==24 or ct_params==31:
        continue
    grad_numels.append(params.data.numel())
    ct_params +=1
#print("grad_numels:", grad_numels)

#Matrix for gradient w.r.t past tasks
G = torch.zeros((sum(grad_numels), args.tasks))
G = G.cuda()


#grad_list_temp = []#used for computing the gradients of equation (5) in the paper

def train(epoch):
    global metrics,loader_train, G, memory_data, avg_mem
    model.train()
    loss_batch = 0 
    batch_count = 0
    batch_count_ =0
    is_fst_loss = True
    is_fst_loss_ = True
    loader_len = len(loader_train)
    #print(loader_len)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1


    if args.cur_task > 0:

        for tid in range(1, args.tasks):
            #load the episodic memory
            memory_data[tid-1].append((np.load('./mem_data/task{:.0f}_mem_3500.npy'.format(tid), allow_pickle=True)).tolist())
        #print("memory data loaded.")

        for mem_task in range(0, args.cur_task):
            for mem_batch in range(0, avg_mem):
                memory_data[mem_task][0][mem_batch] = [tensor.cuda() for tensor in memory_data[mem_task][0][mem_batch]]
    if epoch==0:
        time_record = []
    for cnt,batch in enumerate(loader_train): 
        if cnt > 7000:
            break
        batch_count+=1


        
        #Initialize the G matrix
        G.data.fill_(0.0)
        #Compute gradient w.r.t past tasks with episodic memory
        #at every opitimizing step
        if args.cur_task > 0 and not(batch_count%args.batch_size !=0 and cnt != turn_point) :
            #print("here 1")
            for mem_task in range(0, args.cur_task):
                #print("cur_task:", mem_task)
                if epoch == 0: # just need to record one epoch
                    time_st = time.time()
                for mem_batch in range(0, avg_mem):
                    #print('mem_batch', mem_batch)
                    batch_count_+=1
        
                    

                    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
                    loss_mask,V_obs,A_obs,V_tr,A_tr = memory_data[mem_task][0][mem_batch]

                    optimizer.zero_grad()
                    #Forward
                    #V_obs = batch,seq,node,feat
                    #V_obs_tmp = batch,feat,seq,node
                    V_obs_tmp =V_obs.permute(0,3,1,2)
                    #-----------------input--to--model----------------
                    V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
                    
               
                    V_pred = V_pred.permute(0,2,3,1)
                    
                    

                    V_tr = V_tr.squeeze()
                    A_tr = A_tr.squeeze()
                    V_pred = V_pred.squeeze()

                    
                        #print("if sentence 1")
                    l = graph_loss(V_pred,V_tr)
                    if is_fst_loss_:
                        loss_ = l
                        is_fst_loss_ = False
                    else:
                        loss_ = loss_ + l


                    if mem_batch == (avg_mem-1):
                        loss_ = loss_/avg_mem
                        is_fst_loss_ = True
                        loss_.backward()
                            


                        if args.clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
                    #Compute the pre-defined gradient of episodic memory loss, euqation (5) in the paper
                        j = 0
                        #ii = 0
                        ct_params=1
                        for params in model.parameters():
                            #ii +=1
                            #print(ii)
                            #print(type(params),'grad,',type(params.grad), params.is_leaf, params.requires_grad)
                            if ct_params==23 or ct_params==24 or ct_params==31:
                                continue
                            ct_params +=1

                            if params is not None:
                                if j == 0:
                                    stpt = 0
                                else:
                                    stpt = sum(grad_numels[:j])

                                endpt = sum(grad_numels[:j+1])
                                
                                G[stpt:endpt, mem_task].data.copy_(params.grad.data.view(-1))

                                j += 1
                if epoch == 0:
                    time_ed = time.time()
                    time_duration = time_ed - time_st
                    time_record.append(time_duration)
              #  time_record = open('time{}')
                #print("past task gradient is computed")

        

        #Compute gradient w.r.t current task
        #Get data
        batch = [tensor.cuda() for tensor in batch]

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        
        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)
        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())        
        V_pred = V_pred.permute(0,2,3,1)
        
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        
        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            if args.cur_task > 0:
                grad = []
                j = 0

                ct_params=1
                for params in model.parameters():
                    if ct_params==23 or ct_params==24 or ct_params==31:
                        continue
                    ct_params +=1

                    if params is not None:
                        if j == 0:
                            stpt = 0
                        else:
                            stpt = sum(grad_numels[:j])
                        
                        endpt = sum(grad_numels[:j+1])
                        G[stpt:endpt, args.cur_task].data.copy_(params.grad.view(-1))
                        j += 1
                #print("current gradient is computed")
                #Solve Quadratic Problem
                #print("G1:\n",G[:, args.cur_task].unsqueeze(0) )
                #print("G2:\n",  G[:, :args.cur_task+1])
                dotprod = torch.mm(G[:, args.cur_task].unsqueeze(0), G[:, :args.cur_task+1])
               # print("dotprod:", dotprod)

                if (dotprod < 0).sum() > 0:
                    #if cnt % 100 == 99:
                        #print("projection")
                    G_ = G
                    mem_grad_np = G_[:, :args.cur_task+1].cpu().t().double().numpy()
                    curtask_grad_np = G_[:, args.cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()

                    t = mem_grad_np.shape[0]
                    P = np.dot(mem_grad_np, mem_grad_np.transpose())
                    P = 0.5 * (P + P.transpose()) + np.eye(t) * args.eps
                    q = np.dot(mem_grad_np, curtask_grad_np)*(-1)
                    G_ = np.eye(t)
                    h = np.zeros(t) + args.margin
                    v = quadprog.solve_qp(P, q, G_, h)[0]
                    x = np.dot(v, mem_grad_np) + curtask_grad_np
                    newgrad = torch.Tensor(x).view(-1, )


                    #Copy gradients into params
                    j = 0
                    ct_params = 1
                    for params in model.parameters():
                        if ct_params == 23 or ct_params == 24 or ct_params == 31:
                            continue
                        ct_params += 1

                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(grad_numels[:j])
                            
                            endpt = sum(grad_numels[:j+1])
                            params.grad.data.copy_(newgrad[stpt:endpt].contiguous().view(params.grad.data.size()))
                            j += 1
                #print("GEM learning done, go on!")
                
            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            #print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
            
    metrics['train_loss'].append(loss_batch/batch_count)
    if epoch==0:
        avg_time_a_task_in_each_epoch =  np.mean(time_record)
        timefile = open('./time/avg_time_a_task_{:.0f}mem.txt'.format(avg_mem), 'w')
        timefile.writelines(str(avg_time_a_task_in_each_epoch))
        timefile.close()
    



def vald(epoch):
    global metrics,loader_val,constant_metrics
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    
    for cnt,batch in enumerate(loader_val): 
        if cnt > 1000:
            break
        batch_count+=1

        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        

        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        
        V_pred = V_pred.permute(0,2,3,1)
        
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            #print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)
    
    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_best_{:.0f}.pth'.format(avg_mem))  # OK


#print('Training started ...')
#print("obs",args.obs_seq_len)
#print("obs",args.pred_seq_len)

time_epochs_rcd = []
for epoch in range(args.num_epochs):
    epoch_tr_start = time.time()
    train(epoch)
    epoch_tr_end = time.time()
    epoch_timeduration = epoch_tr_end-epoch_tr_start
    time_epochs_rcd.append(epoch_timeduration)
    time_each_epoch = open("./time/epoch{:.0f}_time.txt".format(epoch), 'w')
    time_each_epoch.writelines(str(epoch_timeduration))
    time_each_epoch.close()
    if epoch==9 or epoch==19 or epoch==29:
        avg_epochs_time = open("./time/avg/avgtime_of{:.0f}epochs.txt".format(epoch), 'w')
        avg_epochs_time.writelines(str(np.mean(time_epochs_rcd)))
        avg_epochs_time.close()
    print("epoch:", epoch)
















    
    vald(epoch)
    if args.use_lrschd:
        scheduler.step()


    #print('*'*30)
    #print('Epoch:',args.tag,":", epoch)
    #for k,v in metrics.items():
     #   if len(v)>0:
      #      print(k,v[-1])


    #print(constant_metrics)
    #print('*'*30)
    
    with open(checkpoint_dir+'metrics_{:.0f}.pkl'.format(avg_mem), 'wb') as fp:
        pickle.dump(metrics, fp)
    
    with open(checkpoint_dir+'constant_metrics_{:.0f}.pkl'.format(avg_mem), 'wb') as fp:
        pickle.dump(constant_metrics, fp)  



