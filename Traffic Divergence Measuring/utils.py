from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import math
from scipy.spatial.distance import pdist, squareform


#___________________________________________________________________________________________________________________________
### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):
    def __init__(self, mat_file, t_h=15, t_f=40, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        self.mat_file = mat_file

        # self.pre_processing()


    def __len__(self):
        return len(self.D)

    def pre_processing(self):
        invalid_idx = []
        for i in range(len(self.D)):
            print('processing %d / %d' % (i, len(self.D)))
            # first deal with neighbor
            grid = self.D[i, 8:]
            # then we should check whether invalid neighbor exists
            t = self.D[i, 2]
            hero_id = self.D[i, 1].astype(int)
            dsId = self.D[i, 0].astype(int)
            fut = self.getHistory(hero_id, t, hero_id, dsId)
            if fut.shape[0] == 0:
                invalid_idx.append(i)
                continue

            fut_max = abs(fut).max(axis=0)
            if np.sum(np.isnan(fut / fut_max)):
                invalid_idx.append(i)
                continue

            for idx, neighbor_id in enumerate(grid):
                if neighbor_id == 0:
                    continue
                if neighbor_id > self.T.shape[1]:
                    grid[idx] = 0
                    continue
                fut = self.getHistory(neighbor_id.astype(int), t, hero_id, dsId)
                if fut.shape[0] == 0:
                    invalid_idx.append(i)
                    continue

                fut_max = abs(fut).max(axis=0)
                if np.sum(np.isnan(fut / fut_max)):
                    invalid_idx.append(i)
                    continue
            self.D[i, 8:] = grid
            if not grid.nonzero()[0].shape[0]:
                invalid_idx.append(i)

        print('before processing, D shape is:', self.D.shape)
        self.D = np.delete(self.D, invalid_idx, axis=0)
        print('after processing, D shape is:', self.D.shape)
        scp.savemat(self.mat_file, {'traj': self.D, 'tracks': self.T})

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        # t = self.D[idx, 2]
        # remember we use different now_time in train and test
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)
       # isnan_hist = np.isnan(hist)
        #print('hist_isnan?:',  True in isnan_hist)
        fut = self.getFuture(vehId,t,dsId)
        #isnan_fut = np.isnan(fut)
        #print('fut_isnan?:', True in isnan_fut)
        hero = hist

        #print('hist', hist)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        # the coordinate is relative to this idx vehicle
        for i in grid:
            other = self.getHistory(i.astype(int), t,vehId,dsId)
            neighbors.append(other)
            if other.shape[0]>0:
                hero = np.concatenate((hero,other),axis=1)
        # 为了计算邻接矩阵，必须在neighbors这里也把hist加入
        neighbors.append(hist)
        assert hero.shape[1]>0
        hero_max = abs(hero).max(axis=0)
        #print('hero_max:', hero_max)
        neigh_num = hero.shape[1]//2-1
       ## x_index = np.linspace(0, 2 * neigh_num, neigh_num + 1).astype(int)
        #print("x_index", x_index)
       ## y_index = np.linspace(1, 2 * neigh_num + 1, neigh_num + 1).astype(int)
       ## x_max = hero_max[x_index].max()
       ## y_max = hero_max[y_index].max()
       ## xy_max = np.array([[x_max, y_max]])
        ##hero_max = np.repeat(xy_max, neigh_num + 1, axis=0).reshape(1, -1)
       # print("hero_max:", hero_max)

        #hero = (hero - hero_min) / (hero_max - hero_min) * 2 - 1
        ##hero = hero / hero_max
        ##is_nan_hero = np.isnan(hero)
        ##if True in is_nan_hero:
           ## print(hero_max)
        ##hist = hist/hero_max[:,0:2]
        #print('[:,0:2]-------', hero_max[:,0:2])
        #isnan_hist = np.isnan(hist)
        #if True in isnan_hist:
         #   print('hist_isnan')
        ##fut = fut/hero_max[:,0:2]
        #isnan_fut = np.isnan(fut)
        #if True in isnan_fut:
        #    print('fut_isnan')
        grid_loc = self.grid_calculate(hero)
        for id,nei in enumerate(neighbors):
            if nei.shape[0]>0:
                neighbors[id] = nei
                ##neighbors[id] = nei/hero_max[:,0:2]

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist,fut,neighbors,lat_enc,lon_enc,grid_loc, hero_max

    # traj1(x),traj1(y),traj2(x),traj2(y)
    def grid_calculate(self,trajs):
        neigh_num = trajs.shape[1]//2-1
        if neigh_num==0:
            return np.zeros((self.grid_size[1],self.grid_size[0],self.enc_size))
        trajs = (trajs+1)/2
        trajs_cp = np.ones_like(trajs)
        trajs_cp[:] = trajs[:]
        x_index = np.linspace(0, 2 * neigh_num, neigh_num+1).astype(int)
        y_index = np.linspace(1, 2 * neigh_num + 1, neigh_num+1).astype(int)
        x_min = trajs[:,x_index].min()
        x_max = trajs[:,x_index].max()
        y_min = trajs[:,y_index].min()
        y_max = trajs[:,y_index].max()

        x_index = np.linspace(2, 2 * neigh_num, neigh_num).astype(int)
        y_index = np.linspace(3, 2 * neigh_num + 1, neigh_num).astype(int)
        mask = np.zeros((self.grid_size[1],self.grid_size[0],self.enc_size))
        x_grid = (trajs[-1,x_index]-x_min)/(x_max-x_min)*self.grid_size[1]
        y_grid = (trajs[-1,y_index]-y_min)/(y_max-y_min)*self.grid_size[0]
        x_grid = np.minimum(x_grid.astype(int),self.grid_size[1]-1)
        y_grid = np.minimum(y_grid.astype(int),self.grid_size[0]-1)

        grid_arr = np.array([x_grid,y_grid])
        grid_arr_uni = np.unique(grid_arr,axis=1)
        # if not grid_arr_uni.shape[1]==grid_arr.shape[1]:
        #     print(grid_arr)
        #     print(trajs[-1,:].reshape(-1,2))
        # assert grid_arr_uni.shape[1]==grid_arr.shape[1]

        if not np.all(x_grid>=0):
            print(trajs_cp,x_min,x_max)
        if not np.all(y_grid>=0):
            print(trajs_cp,y_min,y_max)
        mask[x_grid,y_grid,:] = np.ones(self.enc_size)
        # print('---------------------------------------------')
        # print(trajs[-1,:],x_grid,y_grid)
        return mask

    def grid_calculate_feet(self,trajs,x_scale):
        y_scale = 15/((self.grid_size[0]+1)/12)
        neigh_num = trajs.shape[1]//2-1
        if neigh_num==0:
            return np.zeros((self.grid_size[1],self.grid_size[0],self.enc_size))
        trajs = (trajs+1)/2
        trajs_cp = np.ones_like(trajs)
        trajs_cp[:] = trajs[:]
        x_index = np.linspace(0, 2 * neigh_num, neigh_num+1).astype(int)
        y_index = np.linspace(1, 2 * neigh_num + 1, neigh_num+1).astype(int)
        x_hero = trajs[-1,0]
        y_hero = trajs[-1,1]

        x_index = np.linspace(2, 2 * neigh_num, neigh_num).astype(int)
        y_index = np.linspace(3, 2 * neigh_num + 1, neigh_num).astype(int)
        mask = np.zeros((self.grid_size[1],self.grid_size[0],self.enc_size))
        x_grid = np.round((trajs[-1,x_index]-x_hero)/x_scale)+self.grid_size[1]//2
        y_grid = np.round((trajs[-1,y_index]-y_hero)/y_scale)+self.grid_size[0]//2
        x_grid = np.minimum(x_grid.astype(int),self.grid_size[1]-1)
        y_grid = np.minimum(y_grid.astype(int),self.grid_size[0]-1)

        grid_arr = np.array([x_grid,y_grid])
        grid_arr_uni = np.unique(grid_arr,axis=1)
        if not grid_arr_uni.shape[1]==grid_arr.shape[1]:
            print(grid_arr,'\n')
            print(trajs[-1,:],trajs[-1,x_index],x_hero,x_scale,'\n')
            print(trajs[-1, :], trajs[-1, y_index], y_hero, y_scale,'\n')
        assert grid_arr_uni.shape[1]==grid_arr.shape[1]

        if not np.all(x_grid>=0):
            print(trajs[-1,:],trajs[0,x_index],x_hero,x_grid,y_grid,x_scale,'\n')
        if not np.all(y_grid>=0):
            print(trajs[-1,:],trajs[0,y_index],y_hero,x_grid,y_grid,y_scale,'\n')
        mask[x_grid,y_grid,:] = np.ones(self.enc_size)
        # print('---------------------------------------------')
        # print(trajs[-1,:],x_grid,y_grid)
        return mask

    ## Helper function to get track history
    # get 3s history trajectory from time t and convert to relative pos to refVehicle
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()

            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
           
            #theta = refTrack[np.where(refTrack[:,0]==t)][0,5]-0.5*np.pi
            #assert(theta!=0)
            #rotate_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
                
               # hist = np.dot(hist,rotate_matrix)
            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist



    ## Helper function to get track future
    # get all future trajectory from time t and convert to relative pos to self
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]

        theta = vehTrack[np.where(vehTrack[:,0]==t)][0,5]-0.5*np.pi
        assert(theta!=0)
        rotate_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        
        fut = np.dot(fut,rotate_matrix)
        return fut



    ## Collate function for dataloader
    # for each time read, form a 25*batchsize*(x,y) dimension array
    # represent batchsize number of item, each item form a future 25 sample time
    # 简单来说，构成一个三维数组，每一行是第0维时候的未来位置，所以第0维是25
    # 第一维是batchsize，是把多条记录放在了一起
    # 所以这是批处理而不是实时计算。
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_,_,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.bool()
        index_division = [None] * len(samples)

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        scale_batch = []

        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc,grid,scale) in enumerate(samples):
            cur_count = 0
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            index = []
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    # pos[0] = id % self.grid_size[0]
                    # pos[1] = id // self.grid_size[0]
                    # mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    index.append(count)
                    count+=1
                    cur_count+=1

            index_division[sampleId] = index
            # mask_batch[sampleId,:,:,:] = torch.from_numpy(grid)
            scale_batch.append(torch.from_numpy(scale))
        scale_batch = torch.cat(scale_batch,dim=0)

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch,scale_batch,index_division


def laplacianTopKvec(nbrs,index_division,decay_arr,max_vehicle_num,out_range_dist,max_eig_num):
    batch_size = len(index_division)
    hist_length = nbrs.shape[0]
    nei_arr = np.zeros((batch_size, max_eig_num*max_vehicle_num))
    for j in range(0, batch_size):
        nei_id = index_division[j]
        nei = nbrs[:, nei_id, :]
        dist_arr = np.zeros((len(nei_id), len(nei_id)))
        for k in range(0, hist_length):
            dist_arr_tmp = squareform(pdist(nei[k,:,:]))
            dist_arr += decay_arr[k]*dist_arr_tmp
        dist_arr /= sum(decay_arr)
        dist_arr_full = np.ones((max_vehicle_num,max_vehicle_num))*out_range_dist
        dist_arr_full[0:len(nei_id),0:len(nei_id)] = dist_arr
        dist_arr_full = np.exp(-dist_arr_full)
        la_arr = np.diag(np.sum(dist_arr_full,1))-dist_arr_full
        value, vec = np.linalg.eig(la_arr)
        sorted_indices = np.argsort(value)
        topk_evecs = vec[:, sorted_indices[:-max_eig_num - 1:-1]]
        nei_arr[j, :] = topk_evecs.flatten()

    return nei_arr


def laplacianFixEigNum(hist, nbrs,index_division,decay_arr,max_vehicle_num,out_range_dist,max_eig_num):
    batch_size = len(index_division)
    hist_length = nbrs.shape[0]
    nei_arr = np.zeros((batch_size, max_eig_num*max_vehicle_num))
    for j in range(0, batch_size):
        nei_id = index_division[j]
        nei = nbrs[:, nei_id, :]
        if len(nei_id) > max_vehicle_num:
            nei_vec = nbrs[:, nei_id, :].permute(1,0,2).contiguous().view(len(nei_id),2*nbrs.shape[0])
            nei_hero_dist = np.sum((nei_vec.numpy()-hist[j,:].numpy())**2,axis=1)
            sorted_indices = np.argsort(nei_hero_dist)
            nei = nei[:,sorted_indices[0:max_vehicle_num],:]
            nei_id = [nei_id[k] for k in sorted_indices[0:max_vehicle_num]]
        dist_arr = np.zeros((len(nei_id), len(nei_id)))
        for k in range(0, hist_length):
            dist_arr_tmp = squareform(pdist(nei[k,:,:]))
            # dist_arr += decay_arr[k]*dist_arr_tmp
            dist_arr += decay_arr[hist_length-1-k]*dist_arr_tmp
        dist_arr /= sum(decay_arr)
        dist_arr_full = np.ones((max_vehicle_num,max_vehicle_num))*out_range_dist
        dist_arr_full[0:len(nei_id),0:len(nei_id)] = dist_arr
        dist_arr_full = np.exp(-dist_arr_full)
        la_arr = np.diag(np.sum(dist_arr_full,1))-dist_arr_full
        value, vec = np.linalg.eig(la_arr)
        sorted_indices = np.argsort(value)
        topk_evecs = vec[:, sorted_indices[:-max_eig_num - 1:-1]]
        nei_arr[j, :] = topk_evecs.flatten()

    return nei_arr

class testDataset(Dataset):
    def __init__(self, mat_file):
        self.D = scp.loadmat(mat_file)['data']
        self.mat_file = mat_file
    def __len__(self):
        return len(self.D)
    def __getitem__(self, idx):
        return self.D[idx]
