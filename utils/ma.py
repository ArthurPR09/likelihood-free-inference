import numpy as np
from scipy import stats


class MAProcess:
    def __init__(self, *theta, length=100):
        self.q = len(theta)
        self.theta = list(theta)
        self.length = length

    def rvs(self, size=None):
        if size == None:
            u = np.random.normal(0, 1, self.length+self.q)
            z = np.array([u[i+self.q] + sum(u[i:i+self.q-1] * self.theta) for i in range(self.length)])
        else:
            u = np.random.normal(0, 1, (size, self.length+self.q))
            z = np.array([[u[j, i+self.q] + sum(u[j, i:i+self.q-1] * self.theta) for i in range(self.length)]
                          for j in range(size)])
        return z


class Theta1Prior(stats.rv_continuous):
    def _pdf(self, theta1):
        return (2 - np.abs(theta1)) / 4


class MA2Prior:
    def __init__(self):
        self.theta1prior = Theta1Prior(a=-2, b=2, name="theta1prior")

    def rvs(self, size=None):
        theta1 = self.theta1prior.rvs(size=size)
        theta2 = stats.uniform.rvs(np.abs(theta1)-1, 2-np.abs(theta1), size=size)
        return np.c_[theta1, theta2] if size != None else theta1, theta2

    def cdf(self, theta1, theta2):
        return 1/4


def autocovariance(ts, k):
    ts_mean = np.mean(ts)
    autocov = sum([(ts[i+k] - ts_mean) * (ts[i] - ts_mean) for i in range(len(ts)-k)])
    autocov /= len(ts) - 1
    return autocov