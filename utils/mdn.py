import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Tanh, Softmax
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class MDN(nn.Module):
    def __init__(self, input_size, n_hidden, n_gaussians, gaussian_dim, sigma=None):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.gaussian_dim = gaussian_dim
        self.sigma = sigma

        self.hidden_layer = Sequential(
            Linear(input_size, n_hidden),
            Tanh(),
        )
        self.pi_layer = Sequential(
            Linear(n_hidden, n_gaussians),
            Softmax(dim=1)
        )
        self.mu_layer = Linear(n_hidden, n_gaussians * gaussian_dim)

        if sigma == None:
            self.sigma_layer = Linear(n_hidden, n_gaussians * gaussian_dim * (gaussian_dim + 1) // 2)
        else:
            assert sigma.size()[0] == gaussian_dim


    def mog_cdf(self, y, pi, mu, sigma):
        y = y.expand_as(mu).reshape(-1, self.n_gaussians, 1, self.gaussian_dim)
        mu = mu.reshape(-1, self.n_gaussians, 1, self.gaussian_dim)
        normalization = torch.sqrt(torch.abs(torch.linalg.det(sigma)) * torch.pi ** self.gaussian_dim)
        prob = - 1/2 * torch.matmul(torch.matmul((y-mu), torch.linalg.inv(sigma)), (y-mu).mT)
        prob = torch.exp(prob.squeeze()) / normalization
        prob = prob * pi
        prob = torch.sum(prob, dim=1)
        return prob


    def format_sigma(self, sigma_values):
        sigma = torch.zeros((sigma_values.shape[0], self.n_gaussians, self.gaussian_dim, self.gaussian_dim))

        for i in range(sigma.shape[0]): # loop over the batch
            for k in range(self.n_gaussians): # loop over the gaussians
                tri_ids = torch.triu_indices(*sigma[i, k].shape, offset=0).tolist()
                sigma[i, k][tri_ids] = sigma_values[i, k]

                variances = torch.diag(torch.exp(torch.diag(sigma[i, k])))
                sigma[i, k] = sigma[i, k] - torch.eye(self.gaussian_dim) * torch.diag(sigma[i, k]) + variances

                upper_ids = torch.triu_indices(*sigma[i, k].shape, offset=1).tolist()
                lower_ids = torch.tril_indices(*sigma[i, k].shape, offset=-1).tolist()
                covariances = torch.tanh(sigma[i, k][upper_ids])
                #for l, (m, n) in enumerate(zip(upper_ids[0], upper_ids[1])):
                    #covariances[l] = covariances[l] * torch.sqrt(sigma[i, k, m, m] * sigma[i, k, n, n])
                sigma[i, k][upper_ids] = covariances
                sigma[i, k][lower_ids] = sigma[i, k][upper_ids]

        return sigma


    def forward(self, x):
        features = self.hidden_layer(x)
        pi = self.pi_layer(features)
        mu = self.mu_layer(features)

        if self.sigma == None:
            sigma = self.sigma_layer(features)
            if self.gaussian_dim == 1:
                sigma = torch.exp(sigma)
            else:
                mu = mu.reshape((-1, self.n_gaussians, self.gaussian_dim))
                sigma = sigma.reshape((-1, self.n_gaussians, self.gaussian_dim * (self.gaussian_dim + 1) // 2))
                sigma = self.format_sigma(sigma)

        else:
            pass

        return pi, mu, sigma


def mdn_loss(model, y, pi, mu, sigma):
    probs = model.mog_cdf(y, pi, mu, sigma)
    nll = -torch.log(probs)
    return torch.mean(nll)


def get_loader(X, theta, batch_size):
        X, theta = torch.Tensor(X), torch.Tensor(theta)
        dataset = TensorDataset(X, theta)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader


def train_mdn(model, X, theta, config):
    train_loader = get_loader(X, theta, config['batch_size'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(config['n_epochs']):
            for data, theta in train_loader:
                optimizer.zero_grad()
                pi, mu, sigma = model(data)
                loss = mdn_loss(model, theta, pi, mu, sigma)
                print("Loss: %f" %loss)
                loss.backward()
                optimizer.step()