import unittest
from typing import Callable
import numpy as np
from mcc.pricing_models.ho_lee import HoLeeModel


class TestHoLeeModel(unittest.TestCase):
    def test_ho_lee_model(self) -> None:
        dategrid = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-12-01"),
            30,
            dtype="datetime64[D]",
        )
        rnd = np.random.RandomState(seed=123)
        n = 100
        sigma = 0.2
        rate = 0.08
        discount_curve: Callable[[np.ndarray], np.ndarray] = lambda t: np.exp(-rate * t)
        hl = HoLeeModel(dategrid, discount_curve, sigma, n, rnd, True)
        self.assertEqual(hl.shortrates.shape, (n, dategrid.size))
        self.assertTrue(np.allclose(hl.mu_t(hl.yearfractions), hl._mu_t))
        self.assertTrue(
            (
                np.abs(
                    discount_curve(hl.yearfractions)
                    - hl.discount_factors().mean(axis=0)
                )
                < 0.001
            ).all()
        )
        bp = hl.bond_prices(hl.yearfractions[-1])
        self.assertTrue(np.allclose(bp[:, -1], 1))
