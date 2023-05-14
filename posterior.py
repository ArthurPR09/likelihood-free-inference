import numpy as np
import torch

from .mdn import set_up_mdn
from .train_mdn import train_mdn
from distributions.gaussian import Gaussian, MultivariateGaussian
from distributions.mixture import MoG, MultivariateMoG



def get_posterior(data, prior, generating_process, proposal=None, sumstats=None, n_samples=1000,
                  mdn_type="full", encoder=None, prop_model=None, n_gaussians=1, n_hidden=20, sim=None,
                  batch_size=32, lr=1e-4, n_epoch=50,
                  verbose=1):

    # target data
    if sumstats != None:
        data = sumstats(data)
    elif not isinstance(encoder, type(None)):
        data = encoder(data)
    data = torch.Tensor(np.expand_dims(data, axis=0))

    # train data
    np.random.seed(42)
    if isinstance(sim, type(None)):
        theta_train = prior.rvs(size=n_samples) if proposal == None else proposal.rvs(size=n_samples)
        X_train = [generating_process(theta) for theta in theta_train]
    else:
        theta_train, X_train = sim[0][:n_samples], sim[1][:n_samples]

    if sumstats != None:
        X_train = np.array([sumstats(x) for x in X_train])
    elif not isinstance(encoder, type(None)):
        X_train = np.array([encoder(x) for x in X_train])


    # set up mdn
    torch.manual_seed(42)
    if prop_model == None:
        model = set_up_mdn(mdn_type, n_gaussians, theta_train.shape[1], n_hidden, X_train.shape[1], encoder=encoder)
    else:
        if n_gaussians == 1:
            model = prop_model
        else:
            model = prop_model.replicate(n_gaussians)

    # train
    model = train_mdn(model, X_train.astype('float64'), theta_train, batch_size, lr, n_epoch, verbose=verbose)

    # get posterior
    params = model.forward(data)
    pi, mu, var = [param.squeeze().detach().numpy() for param in params]

    if model.type != "full":
        if model.n_gaussians == 1:
            sigma = np.diag(var)
        else:
            sigma = [np.diag(v) for v in var]
    else:
        sigma = np.linalg.inv(var)

    if model.n_gaussians == 1:
        if model.gaussian_dim == 1:
            posterior = Gaussian(mu, sigma) * prior
        else:
            posterior = MultivariateGaussian(mu, sigma) * prior
    else:
        if model.gaussian_dim == 1:
            posterior = MoG(pi, mu, sigma) * prior
        else:
            posterior = MultivariateMoG(pi, mu, sigma) * prior

    return posterior