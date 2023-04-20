import sys
from unittest import TestCase

sys.path.insert(0, "..")

import numpy as np
from src.optim import nelderMead, secant


class Secant(TestCase):
    def testfun1(self):
        tf = lambda x: x**5 + x**3 + 3
        x0 = -1
        x1 = 1
        res = secant(tf, x0, x1)
        self.assertTrue(res[2])
        self.assertAlmostEqual(res[0], -1.1053, 4)


class NelderMead(TestCase):
    def test_rosenbrock(self):
        def rosenbrock(x):
            return (x[0] - 1) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        res = nelderMead(rosenbrock, np.array([-1.2, 1.0]), step=0.2)
        self.assertAlmostEqual(res[0][0], 1, 6)
        self.assertAlmostEqual(res[0][1], 1, 6)
        self.assertAlmostEqual(res[1], 0, 6)

    def test_ackley(self):
        def ackley2(x):
            return -200 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2))

        res = nelderMead(ackley2, np.array([-1, 0.9]))
        self.assertAlmostEqual(res[0][0], 0, 6)
        self.assertAlmostEqual(res[0][1], 0, 6)
        self.assertAlmostEqual(res[1], -200, 6)
