import numpy as np


def brownianMotion(tend, dt, x0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + np.random.normal() * np.sqrt(dt)
    return x


def geometricBrownianMotion(tend, dt, x0, mu, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] * np.exp(
            (mu - sigma**2 / 2) * dt + sigma * np.random.normal() * np.sqrt(dt)
        )
    return x


def diffsim1d(drift, diffusion, x0, tend, dt, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    W = np.zeros(t.shape)
    for i in range(1, len(t)):
        dW = np.random.normal(0, np.sqrt(dt))
        W[i] = W[i - 1] + dW
        x[i] = x[i - 1] + drift(x[i - 1]) * dt + diffusion(x[i - 1]) * dW
    return x
