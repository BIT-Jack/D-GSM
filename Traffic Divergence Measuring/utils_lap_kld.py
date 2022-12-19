from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import torch
from tqdm import tqdm
from cde.density_estimator import KernelMixtureNetwork, MixtureDensityNetwork
import numpy as np
from tqdm import trange, tqdm


class trajDataset(Dataset):
    def __init__(self, mat_file):
        self.h = sio.loadmat(mat_file)['hist']
        self.f = sio.loadmat(mat_file)['fut']
        self.hist_length = self.h.shape[1]
        self.fut_length = self.f.shape[1]
        self.mat_file = mat_file

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        return self.h[idx],self.f[idx]

    def vec_len(self):
        return self.hist_length,self.fut_length

    def collate_fn(self,samples):
        batch_size = len(samples)
        hist_batch = torch.zeros(batch_size,self.hist_length)
        fut_batch = torch.zeros(batch_size,self.fut_length)
        for sampleId, (hist,fut) in enumerate(samples):
            hist_batch[sampleId, :] = torch.from_numpy(hist)
            fut_batch[sampleId, :] = torch.from_numpy(fut)
        return hist_batch, fut_batch


def RMSE(pred,real):
    assert pred.shape == real.shape
    batch_size = pred.shape[0]
    time_length = pred.shape[1]//2
    pred = pred.view(batch_size,time_length,2)
    real = real.view(batch_size,time_length,2)
    error = torch.sqrt(torch.sum((pred-real)**2,dim=2))
    return error.mean(0).view(1,time_length)


# mean(log(p))-mean(log(q))
# to facilitate large sampling num, first sum in batch then average
def gmm_kld(gmm_p, param_p, gmm_q, param_q):
    sample_p = gmm_p.mdn_sample(param_p['pi'].cuda(),param_p['sigma'].cuda(),param_p['mu'].cuda())
    logprob_p = -gmm_p.mdn_loss(param_p['pi'].cuda(),param_p['sigma'].cuda(),param_p['mu'].cuda(),sample_p.cuda())
    logprob_q = -gmm_q.mdn_loss(param_q['pi'].cuda(),param_q['sigma'].cuda(),param_q['mu'].cuda(),sample_p.cuda())
    kld = logprob_p-logprob_q
    return kld


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

