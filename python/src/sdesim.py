import numpy as np


def brownianMotion(tend, dt, x0):
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + np.random.normal() * np.sqrt(dt)
    return x


def geometricBrownianMotion(tend, dt, x0, mu, sigma):
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] * np.exp(
            (mu - sigma**2 / 2) * dt + sigma * np.random.normal() * np.sqrt(dt)
        )
    return x


def brownianBridge(tend, dt, x0, x1):
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    W = brownianMotion(tend, dt, x0)
    for i in range(1, len(t) - 1):
        x[i] = x0 + W[i] - (i / len(t)) * (W[i] - x1 + x0)
    x[-1] = x1
    return x
