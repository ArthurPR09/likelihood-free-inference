import numpy as np
import scipy.stats as stats


def samples_dist(sample1, sample2, sumstat=None):
    if sumstat != None:
        return np.linalg.norm(sumstat(sample1) - sumstat(sample2))
    else:
        return np.linalg.norm(sample1 - sample2)


def rejection_sampler(data, prior, likelihood, N, eps, sumstat=None):
    theta_sample = np.empty(N)
    for i in range(N):
        dist = eps + 1
        while dist > eps:
            theta_sim = prior.rvs(size=None)
            data_sim = likelihood(*theta_sim).rvs(size=len(data))
            dist = samples_dist(data, data_sim, sumstat=sumstat)
        theta_sample[i] = theta_sim
    return theta_sample


def rejection_sampler_qt(data, prior, likelihood, N, qt, dim_prior=1, sumstat=None):
    theta_sample = np.empty(N) if dim_prior == 1 else np.empty((N, dim_prior))
    dist = np.empty(N)
    for i in range(N):
        theta_sample[i] = prior.rvs(size=None)
        data_sim = likelihood(*theta_sample[i]).rvs(size=len(data))
        dist[i] = samples_dist(data, data_sim, sumstat)
    theta_sample = np.array([theta for i, theta in enumerate(theta_sample) if dist[i] < np.quantile(dist, qt)])
    return theta_sample


def mcmc_sampler(data, prior, likelihood, markov_kernel, mkvar, N, eps):
    theta_init, data_init = rejection_sampler(data, prior, likelihood, 1, eps)
    theta_chain = np.empty(N)
    theta_chain[0] = theta_init

    for i in range(1, N):
        theta_prop = markov_kernel(theta_chain[i-1], mkvar).rvs(size=None)
        data_sim = likelihood(*theta_prop).rvs(size=len(data))
        dist = samples_dist(data, data_sim)

        u = stats.uniform.rvs(0, 1, size=None)
        r = prior.cdf(theta_prop) * markov_kernel(theta_prop, mkvar).cdf(theta_chain[i-1]) / \
            prior.cdf(theta_chain[i-1]) * markov_kernel(theta_chain[i-1], mkvar).cdf(theta_prop)
        if u <= r and dist <= eps:
            theta_chain[i] = theta_prop
        else:
            theta_chain[i] = theta_chain[i-1]

    return theta_chain