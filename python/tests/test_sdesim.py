import sys
from unittest import TestCase

sys.path.insert(0, "..")

import numpy as np
from src.sdesim import *


class BrownianMotion(TestCase):
    def testQuadraticVar(self):
        tend = 1
        dt = 0.01
        x0 = 0
        qvs = []
        for i in range(1000):
            x = brownianMotion(tend, dt, x0)
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(np.mean(qvs), tend, delta=0.1)

        tend = 2
        qvs = []
        for i in range(1000):
            x = brownianMotion(tend, dt, x0)
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(np.mean(qvs), tend, delta=0.1)


class GeometricBrownianMotion(TestCase):
    def testQuadraticVar(self):
        tend = 1
        dt = 0.01
        x0 = 1
        mu = 0.1
        sigma = 0.1
        qvs = []
        for i in range(1000):
            x = geometricBrownianMotion(tend, dt, x0, mu, sigma)
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(
            np.mean(qvs), sigma**2 * tend * (np.exp(2 * mu * tend) - 1), delta=0.1
        )

        tend = 2
        qvs = []
        for i in range(1000):
            x = geometricBrownianMotion(tend, dt, x0, mu, sigma)
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(
            np.mean(qvs), sigma**2 * tend * (np.exp(2 * mu * tend) - 1), delta=0.1
        )


class EulerMaruyama(TestCase):
    def test_ouprocess(self):
        theta = 1
        sigma = 1
        mu = 0

        def drift(x):
            return theta * (mu - x)

        def diffusion(x):
            return sigma

        tend = 1
        dt = 0.01
        x0 = 0
        means = []
        vars = []
        for i in range(1000):
            x = diffsim1d(drift, diffusion, x0, tend, dt, seed=12345)
            means.append(np.mean(x))
            vars.append(np.var(x))

        self.assertAlmostEqual(
            np.mean(means),
            x0 * np.exp(-theta * tend) + mu * (1 - np.exp(-theta * tend)),
            delta=0.01,
        )
