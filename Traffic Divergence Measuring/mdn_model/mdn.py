"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
import scipy.io as sio

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer

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
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        # self.pi = nn.Sequential(
        #     nn.Linear(in_features, num_gaussians),
        #     nn.Softmax(dim=1)
        # )
        self.pi = nn.Linear(in_features, num_gaussians)
        # nn.init.normal_(self.pi.weight)
        torch.nn.init.xavier_uniform_(self.pi.weight, gain=1)
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        # nn.init.normal_(self.sigma.weight)
        torch.nn.init.xavier_uniform_(self.sigma.weight, gain=1)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)
        # nn.init.normal_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight, gain=1)

    def forward(self, minibatch):
        pi = nn.Softmax(dim=1)(self.pi(minibatch))
        # sigma = torch.exp(self.sigma(minibatch))
        # sigma = torch.nn.ELU()(self.sigma(minibatch))+1+1e-15
        sigma = torch.nn.Softplus()(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


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

def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.

    Use Logsumexp to stabalize numeric computation. See
    https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
    """
    target = target.unsqueeze(1).expand_as(sigma)
    prob = torch.log(sigma)+0.5*math.log(2*math.pi)+0.5*((target - mu) / sigma)**2
    prob = torch.sum(-prob, dim=2)
    # here prob equals log(p(x))
    nll = -torch.logsumexp(torch.log(pi)+prob, dim=1)
    # sio.savemat('variable.mat',{'pi':pi.detach().cpu().numpy(),
    #                             'sigma':sigma.detach().cpu().numpy(),
    #                             'mu':mu.detach().cpu().numpy(),
    #                             'fut':target.detach().cpu().numpy(),
    #                             'prob':prob.detach().cpu().numpy(),
    #                             'nll':nll.detach().cpu().numpy()})
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1).expand(pi.size(0),1,sigma.size(2))
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(0), sigma.size(2)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().cpu().squeeze()
    mean_samples = mu.detach().gather(1, pis).cpu().squeeze()
    return gaussian_noise * variance_samples + mean_samples


def mean(pi, sigma, mu):
    return torch.sum(pi.unsqueeze(2)*mu, dim=1)
