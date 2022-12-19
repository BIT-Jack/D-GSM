"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import scipy.io as sio
from gpytorch.kernels import RBFKernel, MaternKernel

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MGP(nn.Module):
    """A mixture gaussian process layer
    It can be regarded as multidimensional mixture density network with dependent variable
    Compared with MDN, we need to modify loss function computation
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MGP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        # self.pi = nn.Sequential(
        #     nn.Linear(in_features, num_gaussians),
        #     nn.Softmax(dim=1)
        # )
        self.pi = nn.Linear(in_features, num_gaussians)
        torch.nn.init.xavier_uniform_(self.pi.weight, gain=1)
        self.sigma_coef = nn.Linear(in_features, num_gaussians)
        torch.nn.init.xavier_uniform_(self.sigma_coef.weight, gain=1)
        self.lengthscale = nn.Linear(in_features, num_gaussians)
        torch.nn.init.xavier_uniform_(self.lengthscale.weight, gain=1)
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        torch.nn.init.xavier_uniform_(self.sigma.weight, gain=1)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)
        torch.nn.init.xavier_uniform_(self.mu.weight, gain=1)

    def forward(self, minibatch):
        pi = nn.Softmax(dim=1)(self.pi(minibatch))
        sigma_coef = self.sigma_coef(minibatch)
        sigma_coef = torch.nn.Softplus()(sigma_coef)
        lengthscale = self.lengthscale(minibatch)
        lengthscale = torch.nn.Softplus()(lengthscale)
        # sigma = torch.nn.Softplus()(self.sigma(minibatch))
        # this sigma is not for gaussian distribution, but form a rbf kernel to make a covariance matrix
        sigma = self.sigma(minibatch)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, sigma_coef, lengthscale, mu


# def mdn_loss(pi, sigma, mu, target):
#     """Calculates the error, given the MoG parameters and the target
#
#     The loss is the negative log likelihood of the data given the MoG
#     parameters.
#     """
#     prob = pi * gaussian_probability(sigma, mu, target)
#     torch.clamp(prob,min=1e-8,max=1-1e-8)
#     nll = -torch.log(torch.sum(prob, dim=1))
#     return torch.mean(nll)

def mdn_loss(pi, sigma, mu, sigma_coef, lengthscale, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.

    Use Logsumexp to stabalize numeric computation. See
    https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
    """
    rbf_kernel = RBFKernel(batchshape=torch.Size([pi.shape[0]]))
    # rbf_kernel = MaternKernel()
    rbf_kernel.lengthscale = 1
    identity = torch.eye(sigma.shape[2])
    sigma_coef = sigma_coef.cpu()
    for i in range(sigma.shape[1]):
        covariance = rbf_kernel(sigma[:,i,:].unsqueeze(2).cpu())
        # convert lazy tensor into tensor (cannot find a built-in function)
        covariance = covariance.matmul(identity)
        covariance += identity * 1e-4
        covariance = torch.log(covariance)/lengthscale[:,i].unsqueeze(1).unsqueeze(2).cpu()

        covariance = torch.exp(covariance)*(sigma_coef[:,i].unsqueeze(1).unsqueeze(2))
        # sio.savemat('cov.mat',{'sigma':sigma[:,i,:].detach().cpu().numpy(),
        #                        'covariance':covariance.detach().cpu().numpy()})
        m = MultivariateNormal(mu[:,i,:].cuda(),covariance_matrix=covariance.cuda())
        if i == 0:
            prob = m.log_prob(target)
            prob = prob.unsqueeze(1)
        else:
            prob = torch.cat((prob,m.log_prob(target).unsqueeze(1)),dim=1)
    # here prob equals log(p(x))
    nll = -torch.logsumexp(torch.log(pi)+prob, dim=1)
    # sio.savemat('variable.mat',{'pi':pi.detach().cpu().numpy(),
    #                             'sigma':sigma.detach().cpu().numpy(),
    #                             'mu':mu.detach().cpu().numpy(),
    #                             'fut':target.detach().cpu().numpy(),
    #                             'prob':prob.detach().cpu().numpy(),
    #                             'nll':nll.detach().cpu().numpy()})
    return torch.mean(nll)


def sample(pi, sigma, sigma_coef, lengthscale, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis_2d = Categorical(pi).sample().view(pi.size(0), 1, 1).detach().cpu()
    pis = pis_2d.expand(pis_2d.size(0),1,sigma.size(2)).detach().cpu()
    # gaussian_noise = torch.randn(
    #     (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.detach().cpu().gather(1, pis).squeeze()
    mean_samples = mu.detach().cpu().gather(1, pis).squeeze()
    sigma_coef_samples = sigma_coef.unsqueeze(2).detach().cpu().gather(1,pis_2d).squeeze()
    lengthscale_samples = lengthscale.unsqueeze(2).detach().cpu().gather(1,pis_2d).squeeze()
    rbf_kernel = RBFKernel()
    rbf_kernel.lengthscale = 1
    covariance = rbf_kernel(variance_samples.unsqueeze(2))
    identity = torch.eye(sigma.shape[2])
    covariance = covariance.matmul(torch.eye(covariance.shape[1])).detach().cpu()
    covariance += identity * 1e-4
    covariance = torch.log(covariance) / lengthscale_samples.unsqueeze(1).unsqueeze(2).cpu()

    covariance = torch.exp(covariance) * (sigma_coef_samples.unsqueeze(1).unsqueeze(2))

    sio.savemat('sample_info.mat',{'mean_sample':mean_samples.numpy(),
                                   'cov_sample':covariance.numpy(),
                                   'sigma_coef_sample':sigma_coef_samples.numpy(),
                                   'lengthscale_sample':lengthscale_samples.numpy()})
    m = MultivariateNormal(mean_samples, covariance_matrix=covariance)
    return m.sample()


def mean(pi, sigma, mu):
    return torch.sum(pi.unsqueeze(2)*mu, dim=1)
