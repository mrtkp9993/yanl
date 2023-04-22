import numpy as np


def brownianMotion(x0, tend, dt, seed=None):
    """Simulates a Brownian motion starting at x0 and ending at time tend.

    Parameters
    ----------
    x0 : float
        The starting position of the Brownian motion.
    tend : float
        The time at which the Brownian motion should end.
    dt : float
        The time step for the Brownian motion.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    x : numpy.ndarray
        The simulated Brownian motion.
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + np.random.normal() * np.sqrt(dt)
    return x


def geometricBrownianMotion(mu, sigma, x0, tend, dt, seed=None):
    """Simulate geometric brownian motion using Euler-Maruyama method
    Parameters
    ----------
    mu: float
        Mean of diffusion process
    sigma: float
        Standard deviation of diffusion process
    x0: float
        Initial condition
    tend: float
        End time of simulation
    dt: float
        Time step size
    seed: int, optional
        Seed for random number generator
    Returns
    -------
    x: np.ndarray
        Array of simulated values of geometric brownian motion
    """
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


def diffsim1dem(drift, diffusion, x0, tend, dt, seed=None):
    """A function that simulates a one-dimensional diffusion process.

    Args:
        drift: A function that takes a position and a time as arguments and returns the drift at that position and time.
        diffusion: A function that takes a position and a time as arguments and returns the diffusion at that position and time.
        x0: The initial position of the particle.
        tend: The time at which to stop the simulation.
        dt: The time step for the simulation.
        seed: The seed for the random number generator. Defaults to None.
    Returns:
        x: An array of positions of the particle at each time step.
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    W = np.zeros(t.shape)
    for i in range(1, len(t)):
        dW = np.random.normal() * np.sqrt(dt)
        W[i] = W[i - 1] + dW
        x[i] = (
            x[i - 1]
            + drift(x[i - 1], t[i - 1]) * dt
            + diffusion(x[i - 1], t[i - 1]) * dW
        )
    return x


def diffsim1dmil(drift, diffusion, diffusionx, x0, tend, dt, seed=None):
    """simulates a one-dimensional diffusion process

    Parameters
    ----------
    drift : callable
        Drift function.
    diffusion : callable
        Diffusion function.
    diffusionx : callable
        Derivative of diffusion function.
    x0 : float
        Initial value.
    tend : float
        End time.
    dt : float
        Time step size.
    seed : int, optional
        Random seed.

    Returns
    -------
    x : ndarray
        Diffusion process.
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, tend, dt)
    x = np.zeros(t.shape)
    x[0] = x0
    W = np.zeros(t.shape)
    for i in range(1, len(t)):
        dW = np.random.normal() * np.sqrt(dt)
        W[i] = W[i - 1] + dW
        x[i] = (
            x[i - 1]
            + drift(x[i - 1], t[i - 1]) * dt
            + diffusion(x[i - 1, t[i - 1]]) * dW
            + 0.5
            * diffusion(x[i - 1], t[i - 1])
            * diffusionx(x[i - 1], t[i - 1])
            * (dW**2 - dt)
        )
    return x
