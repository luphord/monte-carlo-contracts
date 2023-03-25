import unittest
import numpy as np
from mcc import When, Stock, At, One, evaluate
from mcc.pricing_models.black_scholes import simulate_equity_black_scholes_model


class TestBlackScholesModel(unittest.TestCase):
    def test_equity_black_scholes(self) -> None:
        dategrid = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-10"),
            dtype="datetime64[D]",
        )
        rnd = np.random.RandomState(seed=123)
        ndates = dategrid.size
        n = 100
        m = simulate_equity_black_scholes_model(
            "ABC", "EUR", 123, dategrid, 0.2, 0.2, n, rnd, use_moment_matching=True
        )
        self.assertIn("ABC", m.simulated_stocks)
        self.assertEqual(m.simulated_stocks["ABC"].shape, (n, ndates))
        self.assertEqual(m.numeraire.shape, (n, ndates))
        for t in dategrid:
            c = When(At(t), Stock("ABC") * One("EUR"))
            self.assertTrue(np.isclose(evaluate(m, c), 123))
