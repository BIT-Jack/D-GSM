from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import math
import sys
sys.path.append('./mdn_model')
import mdn


class GFunc(nn.Module):
    def __init__(self,args):
        super(GFunc,self).__init__()
        self.args = args
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        # self.soc_embedding_size = self.encoder_size

        ## Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)
        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)
        self.nbr_emb = torch.nn.Linear(2 * self.in_length, self.encoder_size)
        # Decoder
        self.out_linear = torch.nn.Linear(self.dyn_embedding_size * 3, self.decoder_size)
        # Output layers:
        # self.op = torch.nn.Linear(self.decoder_size,2*self.out_length)
        self.op_linear = torch.nn.Linear(self.decoder_size, 1)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, hist, nbrs, fut, index_division,index_len):
        hero_index = np.arange(hist.shape[0]).tolist()
        # index_len = [len(i) for i in index_division]
        hero_repeated = np.repeat(hero_index, index_len)
        relative = hist[hero_repeated, :] - nbrs
        hist = hist.view(-1, self.in_length, 2).permute(1, 0, 2)
        fut = fut.view(-1, self.out_length, 2).permute(1, 0, 2)

        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = hist_enc.squeeze(0)

        _, (fut_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(fut)))
        fut_enc = fut_enc.squeeze(0)

        # relative = hist[:, hero_repeated, :] - nbrs
        rela_enc = self.leaky_relu(self.nbr_emb(relative))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))
        fut_enc = self.leaky_relu(self.dyn_emb(fut_enc))

        ## Forward pass nbrs
        # index_1212,index_1122,index_repeated = self.make_index(index_division)
        # if not self.train_flag:
        #     print(index_division)
        scene_pooled = torch.cat([rela_enc[index, :].max(0)[0].unsqueeze(0) for index in index_division], dim=0)
        scene_pooled = self.leaky_relu(self.dyn_emb(scene_pooled))

        enc = torch.cat((hist_enc, scene_pooled, fut_enc), 1)
        logit = self.decode_mlp(enc)
        return logit

    def decode_mlp(self, enc):
        h_dec = self.leaky_relu(self.out_linear(enc))
        fut_pred = self.op_linear(h_dec)
        return fut_pred


class TFunc(nn.Module):
    ## Initialization
    def __init__(self,args):
        super(TFunc, self).__init__()
        ## Unpack arguments
        self.args = args
        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.linear_decoder = args['use_linear']
        # self.soc_embedding_size = self.encoder_size

        ## Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)
        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)
        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
        self.nbr_emb = torch.nn.Linear(2*self.in_length,self.encoder_size)
        # Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(self.dyn_embedding_size*3, self.decoder_size)
        self.out_linear = torch.nn.Linear(self.dyn_embedding_size*3, self.decoder_size)
        # Output layers:
        # self.op = torch.nn.Linear(self.decoder_size,2*self.out_length)
        self.op_linear = torch.nn.Linear(self.decoder_size,2*self.out_length)
        self.op_lstm = torch.nn.Linear(self.decoder_size,2)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, hist, nbrs, fut, index_division,index_len):
        hero_index = np.arange(hist.shape[0]).tolist()
        # index_len = [len(i) for i in index_division]
        hero_repeated = np.repeat(hero_index, index_len)
        relative = hist[hero_repeated, :] - nbrs
        hist = hist.view(-1,self.in_length,2).permute(1,0,2)
        fut = fut.view(-1,self.out_length,2).permute(1,0,2)

        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = hist_enc.squeeze(0)

        _, (fut_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(fut)))
        fut_enc = fut_enc.squeeze(0)

        rela_enc = self.leaky_relu(self.nbr_emb(relative))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))
        fut_enc = self.leaky_relu(self.dyn_emb(fut_enc))

        ## Forward pass nbrs
        # index_1212,index_1122,index_repeated = self.make_index(index_division)
        # if not self.train_flag:
        #     print(index_division)
        scene_pooled = torch.cat([rela_enc[index, :].max(0)[0].unsqueeze(0) for index in index_division], dim=0)
        scene_pooled = self.leaky_relu(self.dyn_emb(scene_pooled))
        ## Masked scatter
        # print(hist_enc.shape,scene_pooled.shape)
        enc = torch.cat((hist_enc,scene_pooled,fut_enc),1)
        if self.linear_decoder:
            fut_pred = self.decode_mlp(enc)
        else:
            fut_pred = self.decode_lstm(enc)
        return fut_pred

    def decode_mlp(self, enc):
        h_dec = self.leaky_relu(self.out_linear(enc))
        fut_pred = self.op_linear(h_dec)
        # fut_pred = fut_pred.view(-1, 2, self.out_length).permute(2, 0, 1)
        return fut_pred

    def decode_lstm(self, enc):
        enc = enc.unsqueeze(0).repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        fut_pred = self.op_lstm(h_dec)
        fut_pred = fut_pred.permute(1,0,2).contiguous().view(-1,2*self.out_length)
        return fut_pred


class MdnModelTraj(nn.Module):
    def __init__(self, args):
        super(MdnModelTraj, self).__init__()
        self.args = args
        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.hist_length = args['hist_length']
        self.out_length = args['out_length']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_gauss = args['num_gauss']
        self.weight_regularization = args['weight_regularization']
        self.lambda_pi = args['lambda_coef'][0]
        self.lambda_sigma = args['lambda_coef'][1]
        self.lambda_mu = args['lambda_coef'][2]
        # self.nonlinearity = torch.nn.LeakyReLU(0.1)
        self.nonlinearity = torch.nn.Tanh()

        self.ip_emb = torch.nn.Linear(self.hist_length, self.input_embedding_size)
        self.dyn_emb = torch.nn.Linear(self.input_embedding_size, self.dyn_embedding_size)
        # Decoder LSTM
        self.mdn_layer = mdn.MDN(self.dyn_embedding_size, self.out_length, self.num_gauss)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

    def bias_initialization(self, bias_init):
        for i in range(len(bias_init)):
            torch.nn.init.constant_(self.mdn_layer.mu.bias[i], bias_init[i])

    def forward(self, hist):
        hist_enc = self.nonlinearity(self.ip_emb(hist))
        enc = self.nonlinearity(self.dyn_emb(hist_enc))

        pi, sigma, mu = self.mdn_layer(enc)
        return pi, sigma, mu

    def mdn_loss(self, pi, sigma, mu, target):
        nll_loss = mdn.mdn_loss(pi, sigma, mu, target)
        if self.weight_regularization:
            pi_params = torch.cat([x.view(-1) for x in self.mdn_layer.pi.parameters()])
            pi_l1_reg = self.lambda_pi * torch.norm(pi_params, self.weight_regularization)

            sigma_params = torch.cat([x.view(-1) for x in self.mdn_layer.sigma.parameters()])
            sigma_l1_reg = self.lambda_sigma * torch.norm(sigma_params, self.weight_regularization)

            mu_params = torch.cat([x.view(-1) for x in self.mdn_layer.mu.parameters()])
            mu_l1_reg = self.lambda_mu * torch.norm(mu_params, self.weight_regularization)
            nll_loss = nll_loss + pi_l1_reg + sigma_l1_reg + mu_l1_reg
        return nll_loss

    def mdn_mean(self, pi, sigma, mu):
        return mdn.mean(pi, sigma, mu)

    def mdn_sample(self, pi, sigma, mu):
        return mdn.sample(pi, sigma, mu)

    def set_weight_reg(self, reg):
        self.weight_regularization = reg
        return

