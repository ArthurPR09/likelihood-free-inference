import numpy as np
from mdn import MDN, train_mdn
from ma import MAProcess, MA2Prior


prior = MA2Prior()

def mdn_with_prior(target, N, n_hidden, n_gaussians, config):
    # original data
    data = MAProcess(*target).rvs()

    # training data
    theta_train = prior.rvs(size=N)
    X_train = np.array([MAProcess(*tuple(theta)).rvs() for theta in theta_train])

    # set up mdn
    model = MDN(input_size=data.size, n_hidden=n_hidden, n_gaussians=n_gaussians, gaussian_dim=len(target))

    # train
    train_mdn(model, X_train, theta_train, config)

    # update posterior
    #posterior = model.forward(data) # prior same as proposal

    #return posterior