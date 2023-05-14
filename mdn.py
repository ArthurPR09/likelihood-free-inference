import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Tanh, Softmax


class BaseMDN(nn.Module):
    def __init__(self, n_gaussians, gaussian_dim, n_hidden, input_size=None, encoder=None):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.gaussian_dim = gaussian_dim
        self.n_hidden = n_hidden
        self.input_size = input_size

        self.encoder = encoder
        """if not isinstance(self.encoder, type(None)):
            self.input_size = self.encoder.encoding_size
        elif input_size == None:
            print("If no encoder is used input_size must be specified")
            return(0)"""

        self.hidden_layer = Sequential(
            Linear(input_size, n_hidden),
            Tanh(),
        )

        if n_gaussians > 1:
            self.pi_layer = Sequential(
                Linear(n_hidden, n_gaussians),
                Softmax(dim=1)
            )
        self.mu_layer = Linear(n_hidden, n_gaussians * gaussian_dim)

        self.fitted = False


    def forward(self, x):
        # if self.encoder != None:
            # x = self.encoder.forward(x)
        features = self.hidden_layer(x)
        pi = self.pi_layer(features) if self.n_gaussians > 1 else torch.Tensor([[1.]])
        mu = self.mu_layer(features)
        mu = mu.reshape((-1, self.n_gaussians, self.gaussian_dim)) # mu.reshape((self.n_gaussians, self.gaussian_dim))
        return features, pi, mu


    def mog_pdf(self, y, pi, mu, sigma):
        det = torch.prod(sigma, axis=1)
        normalization = torch.sqrt(torch.abs(det) * (2 * torch.pi) ** self.gaussian_dim)
        prob = - (((y-mu) * (y-mu)) / sigma).sum(axis=1) / 2
        prob = torch.exp(prob) / normalization
        prob = (prob * pi).sum()
        return prob


    def replicate(self, n_gaussians):
        if self.n_gaussians == 1:
            new_mdn = self.__class__(n_gaussians, self.gaussian_dim, self.n_hidden, self.input_size)
            with torch.no_grad():
                for layer_name in self.__dict__['_modules'].keys():
                    if layer_name == "hidden_layer":
                        new_mdn.hidden_layer[0].weight.copy_(self.hidden_layer[0].weight)
                    else:
                        for k in range(self.n_gaussians):
                            weights = getattr(self, layer_name).weight[k]
                            getattr(new_mdn, layer_name).weight[k].copy_(weights)
            return new_mdn
        else:
            raise ValueError("Only MDN with single component can be replicated")



class DiagonalMDN(BaseMDN):
    def __init__(self, input_size, n_gaussians, gaussian_dim, n_hidden, encoder=None):
        super().__init__(input_size, n_gaussians, gaussian_dim, n_hidden, encoder)
        self.sigma_layer = Linear(n_hidden, n_gaussians * gaussian_dim)
        self.type = "diagonal"


    def forward(self, x):
        features, pi, mu = super().forward(x)
        sigma = self.sigma_layer(features)
        sigma = torch.exp(sigma)
        sigma = sigma.reshape((self.n_gaussians, self.gaussian_dim))
        return pi, mu, sigma



class FullMDN(BaseMDN):
    def __init__(self, n_gaussians, gaussian_dim, n_hidden, input_size=None, encoder=None):
        super().__init__(n_gaussians, gaussian_dim, n_hidden, input_size, encoder)
        self.sigma_diag_layer = Linear(n_hidden, n_gaussians * gaussian_dim)
        self.sigma_utri_layer = Linear(n_hidden, n_gaussians * (gaussian_dim - 1) * gaussian_dim // 2)
        self.type = "full"


    def get_inv_sigma(self, sigma_diag, sigma_utri):
        sigma_diag = torch.exp(sigma_diag).reshape((-1, self.n_gaussians, self.gaussian_dim))
        U = torch.diag_embed(sigma_diag)

        sigma_utri = sigma_utri.reshape((-1, self.n_gaussians, self.gaussian_dim * (self.gaussian_dim - 1) // 2))
        utri_idx = torch.triu_indices(self.gaussian_dim, self.gaussian_dim, offset=1)
        U_utri = torch.zeros((sigma_utri.shape[0], self.n_gaussians, self.gaussian_dim, self.gaussian_dim))
        U_utri[:, :, utri_idx[0], utri_idx[1]] = sigma_utri
        U += U_utri

        inv_sigma = torch.einsum('bgij,bgjk->bgik', U.mT, U)
        return inv_sigma


    def forward(self, x):
        features, pi, mu = super().forward(x)
        sigma_diag = self.sigma_diag_layer(features)
        sigma_utri = self.sigma_utri_layer(features)
        inv_sigma = self.get_inv_sigma(sigma_diag, sigma_utri)
        return pi, mu, inv_sigma


    def mog_pdf(self, y, pi, mu, inv_sigma):
        y = y.reshape(-1, 1, self.gaussian_dim)
        normalization = torch.sqrt((2 * torch.pi) ** self.gaussian_dim / torch.abs(torch.linalg.det(inv_sigma)))
        prob = torch.einsum('bgi,bgij->bgj', y-mu, inv_sigma)
        prob = torch.einsum('bgi,bgi->bg', prob, y-mu)
        prob = torch.exp(- prob / 2) / normalization
        prob = prob * pi
        prob = torch.sum(prob, dim=1)
        return prob


def set_up_mdn(mdn_type, n_gaussians, gaussian_dim, n_hidden, input_size=None, encoder=None):
    mdn_setup = {'n_gaussians': n_gaussians,
                 'gaussian_dim': gaussian_dim,
                 'n_hidden': n_hidden,
                 'input_size': input_size,
                 'encoder': encoder}
    if mdn_type == "diagonal":
        mdn = DiagonalMDN(**mdn_setup)
    elif mdn_type == "full":
        mdn = FullMDN(**mdn_setup)
    return mdn