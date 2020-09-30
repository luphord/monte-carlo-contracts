import unittest

import numpy as np
from mcc import parser, IndexedCashflows, DateIndex, Model, Zero, One, Give, And, Or


def _make_model(nsim=100) -> Model:
    dategrid = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-10"),
        dtype="datetime64[D]",
    )
    numeraire = np.ones((nsim, dategrid.size), dtype=np.float)
    return Model(dategrid, {}, numeraire, "EUR")


class TestMonteCarloContracts(unittest.TestCase):
    def test_argument_parsing(self):
        args = parser.parse_args([])
        self.assertEqual(args.version, False)
        args = parser.parse_args(["--version"])
        self.assertEqual(args.version, True)

    def test_cashflows(self):
        n = 10
        k = 2
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf1 = IndexedCashflows(
            np.zeros((n, k), dtype=IndexedCashflows.dtype),
            np.array(["USD"] * k, dtype=(np.string_, 3)),
            dategrid,
        )
        self.assertEqual(cf1.nsim, n)
        self.assertEqual(cf1.ncashflows, k)
        self.assertEqual(cf1.currencies.size, k)
        cf2 = IndexedCashflows(
            np.array(
                [
                    [(0, 123.45)],
                    [(1, 123.45)],
                ],
                IndexedCashflows.dtype,
            ),
            np.array(["USD"], dtype=(np.string_, 3)),
            dategrid,
        )
        self.assertEqual(cf2.nsim, 2)
        self.assertEqual(cf2.ncashflows, 1)
        self.assertEqual(cf2.currencies.size, 1)
        cf3 = cf1 + cf1
        self.assertTrue((cf3.cashflows[:, :k] == cf1.cashflows).all())
        self.assertTrue((cf3.cashflows[:, k:] == cf1.cashflows).all())
        cf3.apply_index()

    def test_contract_creation(self):
        And(Or(Zero(), One("EUR")), Give(One("USD")))

    def test_model_creation(self):
        nsim = 100
        model = _make_model(nsim=nsim)
        self.assertEqual(model.shape, (nsim, model.dategrid.size))

    def test_simple_cashflow_generation(self):
        ccy = "EUR"
        c = One(ccy)
        model = _make_model()
        eval_idx = DateIndex(np.zeros((model.nsim,), dtype=np.int))
        cf = c.generate_cashflows(eval_idx, model)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        self.assertTrue(cf.currencies[0], ccy)
        eval_idx2 = DateIndex(np.zeros((model.nsim + 1,), dtype=np.int))
        self.assertRaises(
            AssertionError, lambda: c.generate_cashflows(eval_idx2, model)
        )

        c2 = And(c, One("USD"))
        cf2 = c2.generate_cashflows(eval_idx, model)
        self.assertEqual(cf2.currencies.shape, (2,))
        self.assertTrue(cf2.currencies[0], ccy)
        self.assertTrue(cf2.currencies[1], "USD")
        self.assertEqual(cf2.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf2.cashflows["value"] == 1).all())
        self.assertTrue((cf2.cashflows["index"] == 0).all())
