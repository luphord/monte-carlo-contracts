import unittest

import numpy as np
from mcc import parser, SimulatedCashflows, Zero, One, Give, And, Or


class TestMonteCarloContracts(unittest.TestCase):
    def test_argument_parsing(self):
        args = parser.parse_args([])
        self.assertEqual(args.version, False)
        args = parser.parse_args(["--version"])
        self.assertEqual(args.version, True)

    def test_cashflows(self):
        n = 10
        k = 2
        cf1 = SimulatedCashflows(np.zeros((n, k), dtype=SimulatedCashflows.dtype))
        self.assertEqual(cf1.nsim, n)
        self.assertEqual(cf1.ncashflows, k)
        cf2 = SimulatedCashflows(
            np.array(
                [
                    [(np.datetime64("2030-07-14"), "USD", 123.45)],
                    [(np.datetime64("2031-07-14"), "USD", 123.45)],
                ],
                SimulatedCashflows.dtype,
            )
        )
        self.assertEqual(cf2.nsim, 2)
        self.assertEqual(cf2.ncashflows, 1)

    def test_contract_creation(self):
        And(Or(Zero(), One("EUR")), Give(One("USD")))
