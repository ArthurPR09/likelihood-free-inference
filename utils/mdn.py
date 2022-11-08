import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Tanh, Softmax


class MDN(nn.Module):
    def __init__(self, input_size, n_hidden, n_gaussians, gaussian_dim):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.gaussian_dim = gaussian_dim

        self.hidden_layer = Sequential(
            Linear(input_size, n_hidden),
            Tanh(),
        )
        self.pi_layer = Sequential(
            Linear(n_hidden, n_gaussians),
            Softmax(dim=0)
        )
        self.mu_layer = Linear(n_hidden, n_gaussians * gaussian_dim)
        self.sigma_layer = Linear(n_hidden, n_gaussians * gaussian_dim * (gaussian_dim + 1) / 2)


    def mog_cdf(self, y, pi, mu, sigma):
        y = y.expand_as(mu)
        normalization = 1 / torch.sqrt(torch.linalg.det(sigma) * torch.pi ** self.gaussian_dim)
        prob = - 1/2 * torch.dot(torch.matmul(y-mu, torch.linalg.inv(sigma)), y-mu)
        prob = torch.exp(prob) * normalization
        prob = prob * pi
        prob = torch.sum(prob, dim=1)
        return prob


    def format_sigma(self, sigma_values):
        sigma = torch.zeros((self.gaussian_dim, self.gaussian_dim))
        tri_ids = torch.triu_indices(*sigma.shape, offset=0).tolist()
        sigma[tri_ids] = sigma_values

        variances = torch.diag(torch.exp(torch.diag(sigma)))
        mask = torch.diag(torch.ones_like(variances))
        sigma = sigma - mask * sigma + variances

        upper_ids = torch.triu_indices(*sigma.shape, offset=1).tolist()
        lower_ids = torch.tril_indices(*sigma.shape, offset=-1).tolist()
        covariances = torch.tanh(sigma[upper_ids])
        for k, (i, j) in enumerate(upper_ids):
            covariances[k] = covariances[k] * torch.sqrt(sigma[i, i] * sigma[j, j])
        sigma[upper_ids] = covariances
        sigma[lower_ids] = sigma[upper_ids]

        return sigma


    def forward(self, x):
        features = self.hidden_layer(x)
        pi = self.pi_layer(features)
        mu = self.mu_layer(features)
        sigma = self.sigma_layer(features)

        if self.gaussian_dim == 1:
            sigma = torch.exp(sigma)
        else:
            mu = mu.reshape((self.n_gaussians, self.gaussian_dim))
            sigma = sigma.reshape((self.n_gaussians, self.gaussian_dim * (self.gaussian_dim + 1) / 2))
            sigma = self.format_sigma(sigma)

            assert all(torch.linalg.det(sigma) != 0), "Not all sigmas are invertible"

        return pi, mu, sigma


def mdn_loss(model, y, pi, mu, sigma):
    probs = model.mog_cdf(y, pi, mu, sigma)
    nll = -torch.log(torch.sum(probs, dim=1))
    return torch.mean(nll)