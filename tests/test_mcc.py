import unittest

import numpy as np
from mcc import parser, SimulatedCashflows, Model, Zero, One, Give, And, Or


class TestMonteCarloContracts(unittest.TestCase):
    def test_argument_parsing(self):
        args = parser.parse_args([])
        self.assertEqual(args.version, False)
        args = parser.parse_args(["--version"])
        self.assertEqual(args.version, True)

    def test_cashflows(self):
        n = 10
        k = 2
        cf1 = SimulatedCashflows(
            np.zeros((n, k), dtype=SimulatedCashflows.dtype),
            np.array(["USD"] * k, dtype=(np.string_, 3)),
        )
        self.assertEqual(cf1.nsim, n)
        self.assertEqual(cf1.ncashflows, k)
        self.assertEqual(cf1.currencies.size, k)
        cf2 = SimulatedCashflows(
            np.array(
                [
                    [(np.datetime64("2030-07-14"), 123.45)],
                    [(np.datetime64("2031-07-14"), 123.45)],
                ],
                SimulatedCashflows.dtype,
            ),
            np.array(["USD"], dtype=(np.string_, 3)),
        )
        self.assertEqual(cf2.nsim, 2)
        self.assertEqual(cf2.ncashflows, 1)
        self.assertEqual(cf2.currencies.size, 1)

    def test_contract_creation(self):
        And(Or(Zero(), One("EUR")), Give(One("USD")))

    def test_model_creation(self):
        nsim = 100
        dategrid = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-10"),
            dtype="datetime64[D]",
        )
        numeraire = np.ones((nsim, dategrid.size), dtype=np.float)
        model = Model(dategrid, {}, numeraire, "EUR")
        self.assertEqual(model.shape, (nsim, dategrid.size))
