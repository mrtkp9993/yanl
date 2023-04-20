import sys
from unittest import TestCase

sys.path.insert(0, "..")

from src.optim import secant


class Secant(TestCase):
    def testfun1(self):
        tf = lambda x: x**5 + x**3 + 3
        x0 = -1
        x1 = 1
        res = secant(tf, x0, x1)
        self.assertTrue(res[2])
        self.assertAlmostEqual(res[0], -1.1053, 4)
