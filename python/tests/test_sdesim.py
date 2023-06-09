import sys
from unittest import TestCase

sys.path.insert(0, "..")

import numpy as np
from src.sdesim import *


class BrownianMotion(TestCase):
    def setUp(self):
        self.tend1 = 1
        self.tend2 = 2
        self.dt = 0.01
        self.x0 = 0

    def testQuadraticVarte1(self):
        qvs = []
        for i in range(1000):
            x = brownianMotion(self.x0, self.tend1, self.dt)
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(np.mean(qvs), self.tend1, delta=0.1)

    def testQuadraticVarte2(self):
        qvs = []
        for i in range(1000):
            x = brownianMotion(self.x0, self.tend2, self.dt)
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(np.mean(qvs), self.tend2, delta=0.1)


class GeometricBrownianMotion(TestCase):
    def setUp(self):
        self.tend1 = 1
        self.tend2 = 2
        self.dt = 0.01
        self.x0 = 1
        self.mu = 0.1
        self.sigma = 0.1

    def testQuadraticVarte1(self):
        qvs = []
        for i in range(1000):
            x = geometricBrownianMotion(
                self.mu, self.sigma, self.x0, self.tend1, self.dt
            )
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(
            np.mean(qvs),
            self.sigma**2 * self.tend1 * (np.exp(2 * self.mu * self.tend1) - 1),
            delta=0.1,
        )

    def testQuadraticVarte2(self):
        qvs = []
        for i in range(1000):
            x = geometricBrownianMotion(
                self.mu, self.sigma, self.x0, self.tend2, self.dt
            )
            qvs.append(np.sum(np.diff(x) ** 2))
        self.assertAlmostEqual(
            np.mean(qvs),
            self.sigma**2 * self.tend2 * (np.exp(2 * self.mu * self.tend2) - 1),
            delta=0.1,
        )


class EulerMaruyama(TestCase):
    def test_ouprocess(self):
        theta = 1
        sigma = 1
        mu = 0

        def drift(x, t):
            return theta * (mu - x)

        def diffusion(x, t):
            return sigma

        tend = 1
        dt = 0.01
        x0 = 0
        means = []
        vars = []
        for i in range(1000):
            x = diffsim1dem(drift, diffusion, x0, tend, dt, seed=12345)
            means.append(np.mean(x))
            vars.append(np.var(x))

        self.assertAlmostEqual(
            np.mean(means),
            x0 * np.exp(-theta * tend) + mu * (1 - np.exp(-theta * tend)),
            delta=0.01,
        )

    def test_bsm(self):
        r = 0.05
        sigma = 0.2

        def drift(x, t):
            return r * x

        def diffusion(x, t):
            return sigma * x

        x0 = 100
        tend = 1
        dt = 0.01
        means = []
        vars = []
        for i in range(1000):
            x = diffsim1dem(drift, diffusion, x0, tend, dt, seed=123)
            means.append(np.mean(x))
            vars.append(np.var(x))

        expected_mean = x0 * np.exp(r * tend)
        expected_var = x0**2 * np.exp(2 * 0.05 * tend) * (np.exp(0.2**2 * tend) - 1)
        self.assertAlmostEqual(
            np.mean(means),
            expected_mean,
            delta=0.1 * expected_mean,
        )
        # self.assertAlmostEqual(
        #     np.mean(vars),
        #     expected_var,
        #     delta=0.1 * expected_var,
        # )

    # def test_cir(self):
    #     # Read test data
    #     with open("tests/data/cir.txt") as f:
    #         x = []
    #         for line in f.readlines():
    #             x.append(float(line))

    #     # Set simulation parameters
    #     def drift(x, t):
    #         return 6 - 3 * x

    #     def diffusion(x, t):
    #         return 2 * np.sqrt(x)

    #     x0 = 10
    #     tend = 1
    #     dt = 0.01

    #     # Simulate
    #     x_sim = diffsim1dem(drift, diffusion, x0, tend, dt, seed=123)

    #     # Compare
    #     for i in range(100):
    #         print(x[i], x_sim[i])
    #     import matplotlib.pyplot as plt

    #     plt.scatter(x, x_sim)
    #     plt.show()
    #     self.assertTrue(np.allclose(x, x_sim, atol=0.1))


class Milstein(TestCase):
    def test_ouprocess(self):
        theta = 1
        sigma = 1
        mu = 0

        def drift(x, t):
            return theta * (mu - x)

        def diffusion(x, t):
            return sigma

        def diffusionx(x, t):
            return 0

        tend = 1
        dt = 0.01
        x0 = 0
        means = []
        vars = []
        for i in range(1000):
            x = diffsim1dmil(drift, diffusion, diffusionx, x0, tend, dt, seed=12345)
            means.append(np.mean(x))
            vars.append(np.var(x))

        self.assertAlmostEqual(
            np.mean(means),
            x0 * np.exp(-theta * tend) + mu * (1 - np.exp(-theta * tend)),
            delta=0.01,
        )

    def test_bsm(self):
        r = 0.05
        sigma = 0.2

        def drift(x, t):
            return r * x

        def diffusion(x, t):
            return sigma * x

        def diffusionx(x, t):
            return sigma

        x0 = 100
        tend = 1
        dt = 0.01
        means = []
        vars = []
        for i in range(1000):
            x = diffsim1dmil(drift, diffusion, diffusionx, x0, tend, dt, seed=123)
            means.append(np.mean(x))
            vars.append(np.var(x))

        expected_mean = x0 * np.exp(r * tend)
        expected_var = x0**2 * np.exp(2 * 0.05 * tend) * (np.exp(0.2**2 * tend) - 1)
        self.assertAlmostEqual(
            np.mean(means),
            expected_mean,
            delta=0.1 * expected_mean,
        )
        # self.assertAlmostEqual(
        #     np.mean(vars),
        #     expected_var,
        #     delta=0.1 * expected_var,
        # )


class SO(TestCase):
    def test_ouprocess(self):
        theta = 1
        sigma = 1
        mu = 0

        def drift(x, t):
            return theta * (mu - x)

        def driftx(x, t):
            return -theta

        def driftxx(x, t):
            return 0

        def driftt(x, t):
            return 0

        def diffusion(x, t):
            return sigma

        tend = 1
        dt = 0.01
        x0 = 0
        means = []
        vars = []
        for i in range(1000):
            x = diffsim1dso(
                drift, driftx, driftxx, driftt, diffusion, x0, tend, dt, seed=12345
            )
            means.append(np.mean(x))
            vars.append(np.var(x))

        self.assertAlmostEqual(
            np.mean(means),
            x0 * np.exp(-theta * tend) + mu * (1 - np.exp(-theta * tend)),
            delta=0.01,
        )
