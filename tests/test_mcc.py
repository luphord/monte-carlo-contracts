import unittest

import numpy as np
from mcc import (
    parser,
    IndexedCashflows,
    DateIndex,
    Model,
    KonstFloat,
    At,
    Zero,
    One,
    Give,
    Scale,
    And,
    Or,
)


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

    def test_date_index(self):
        model = _make_model()
        at0 = At(model.dategrid[0])
        idx0 = model.eval_date_index.next_after(at0.simulate(model))
        self.assertTrue((idx0 == 0).all())
        at1 = At(model.dategrid[1])
        idx1 = model.eval_date_index.next_after(at1.simulate(model))
        self.assertTrue((idx1 == 1).all())

    def test_boolean_obs_at(self):
        model = _make_model()
        at0 = At(model.dategrid[0])
        at0sim = at0.simulate(model)
        self.assertEqual(at0sim.dtype, np.bool_)
        self.assertEqual(at0sim.shape, model.shape)
        self.assertTrue(at0sim[:, 0].all())
        self.assertFalse(at0sim[:, 1].any())
        at1 = At(model.dategrid[1])
        at1sim = at1.simulate(model)
        self.assertEqual(at1sim.dtype, np.bool_)
        self.assertEqual(at1sim.shape, model.shape)
        self.assertFalse(at1sim[:, 0].any())
        self.assertTrue(at1sim[:, 1].all())

    def test_zero_cashflow_generation(self):
        model = _make_model()
        cf = model.generate_cashflows(Zero())
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertTrue(cf.currencies[0], "NNN")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 0).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_one_cashflow_generation(self):
        ccy = "EUR"
        c = One(ccy)
        model = _make_model()
        cf = c.generate_cashflows(model.eval_date_index, model)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        self.assertTrue(cf.currencies[0], ccy)
        bad_eval_idx = DateIndex(np.zeros((model.nsim + 1,), dtype=np.int))
        self.assertRaises(
            AssertionError, lambda: c.generate_cashflows(bad_eval_idx, model)
        )

    def test_give_cashflow_generation(self):
        model = _make_model()
        cf = model.generate_cashflows(Give(One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertTrue(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == -1).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_scale_cashflow_generation(self):
        model = _make_model()
        cf = model.generate_cashflows(Scale(KonstFloat(1.23), One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertTrue(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1.23).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_and_cashflow_generation(self):
        c = And(One("EUR"), One("USD"))
        model = _make_model()
        cf = c.generate_cashflows(model.eval_date_index, model)
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertTrue(cf.currencies[0], "EUR")
        self.assertTrue(cf.currencies[1], "USD")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        cf_alt = model.generate_cashflows(c)
        self.assertTrue(cf_alt, cf.apply_index())
