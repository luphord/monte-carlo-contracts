import unittest
import numpy as np
from mcc.pricing_models.stochastic_processes import (
    BrownianMotion,
    GeometricBrownianMotion,
)


class TestStochasticProcesses(unittest.TestCase):
    def test_brownian_motions(self) -> None:
        mu = 0.123
        sigma = 0.456
        bm = BrownianMotion(mu_t=lambda t: mu * t, sigma=sigma)
        gbm = GeometricBrownianMotion(mu_t=lambda t: mu * t, sigma=sigma)
        rnd = np.random.RandomState(1234)
        t = np.linspace(0, 20, 20)
        n = 200
        x = bm.simulate(t, n, rnd)
        bm.expected(t), bm.stddev(t)
        bm.moment_match(t, x)
        self.assertEqual((n, t.size), x.shape)
        self.assertEqual(n, x[:, -1].size)
        x = gbm.simulate(t, n, rnd)
        self.assertEqual((n, t.size), x.shape)
        self.assertEqual(n, x[:, -1].size)
        gbm.expected(t), gbm.stddev(t)
        gbm.moment_match(t, x)

    def test_moment_matched_brownian_motions(self) -> None:
        bm = BrownianMotion(lambda t: t)
        rnd = np.random.RandomState(1234)
        t = np.linspace(0, 20, 20)
        n = 200
        x = bm.simulate_with_moment_matching(t, n, rnd)
        self.assertTrue(np.allclose(x.mean(axis=0), t))
        self.assertTrue(np.allclose(x.std(axis=0), np.sqrt(t)))
        gbm = GeometricBrownianMotion()
        x = gbm.simulate_with_moment_matching(t, n, rnd)
        self.assertTrue(np.allclose(x.mean(axis=0), np.ones(t.shape)))
        self.assertTrue(np.allclose(x.var(axis=0), np.exp(t) - 1))
