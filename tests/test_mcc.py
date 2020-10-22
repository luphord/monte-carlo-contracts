import unittest

import numpy as np
from dataclasses import dataclass
from mcc import (
    parser,
    IndexedCashflows,
    DateIndex,
    Model,
    ObservableBool,
    KonstFloat,
    Stock,
    FX,
    At,
    Zero,
    One,
    Give,
    Scale,
    And,
    Or,
    When,
    Cond,
    Until,
    Contract,
    ResolvableContract,
    ZeroCouponBond,
    EuropeanOption,
    BrownianMotion,
    GeometricBrownianMotion,
)


def _make_model(nsim: int = 100) -> Model:
    dategrid = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-10"),
        dtype="datetime64[D]",
    )
    numeraire = np.ones((nsim, dategrid.size), dtype=np.float)
    rnd = np.random.RandomState(123)
    eurusd = rnd.lognormal(size=(nsim, dategrid.size))
    abc = rnd.lognormal(size=(nsim, dategrid.size))
    return Model(dategrid, {("EUR", "USD"): eurusd}, {"ABC": abc}, numeraire, "EUR")


@dataclass
class MyContract(ResolvableContract):
    maturity: np.datetime64
    notional: float

    def resolve(self) -> Contract:
        return When(At(self.maturity), Scale(KonstFloat(self.notional), One("EUR")))


class AlternatingBool(ObservableBool):
    def simulate(self, model: Model) -> np.ndarray:
        mask = np.array(np.arange(model.nsim) % 2, dtype=np.bool_).reshape(
            (model.nsim, 1)
        )
        return np.repeat(mask, model.ndates, axis=1)


class TestMonteCarloContracts(unittest.TestCase):
    def test_argument_parsing(self) -> None:
        args = parser.parse_args([])
        self.assertEqual(args.version, False)
        args = parser.parse_args(["--version"])
        self.assertEqual(args.version, True)

    def test_cashflows(self) -> None:
        n = 10
        k = 2
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf1 = IndexedCashflows(
            np.zeros((n, k), dtype=IndexedCashflows.dtype),
            np.array(["USD"] * k, dtype=(np.unicode_, 3)),
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
            np.array(["USD"], dtype=(np.unicode_, 3)),
            dategrid,
        )
        self.assertEqual(cf2.nsim, 2)
        self.assertEqual(cf2.ncashflows, 1)
        self.assertEqual(cf2.currencies.size, 1)
        cf3 = cf1 + cf1
        self.assertTrue((cf3.cashflows[:, :k] == cf1.cashflows).all())
        self.assertTrue((cf3.cashflows[:, k:] == cf1.cashflows).all())
        cf3.apply_index()

    def test_negative_date_index(self) -> None:
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf = IndexedCashflows(
            np.array(
                [
                    [(0, 123.45)],
                    [(-1, 123.45)],
                ],
                IndexedCashflows.dtype,
            ),
            np.array(["USD"], dtype=(np.unicode_, 3)),
            dategrid,
        ).apply_index()
        self.assertEqual(cf.cashflows["date"][0], dategrid[0])
        self.assertTrue(np.isnat(cf.cashflows["date"][1]))

    def test_contract_creation(self) -> None:
        And(
            Or(
                Cond(AlternatingBool(), Zero(), One("USD")),
                When(At(np.datetime64("2030-07-14")), One("EUR")),
            ),
            Give(Scale(KonstFloat(1.23), One("USD"))),
        )
        KonstFloat(1.23)
        KonstFloat(123)
        KonstFloat(np.float_(1.23))
        KonstFloat(np.int_(123))
        KonstFloat(np.float64(123))
        KonstFloat(np.float32(123))
        KonstFloat(np.int64(123))
        KonstFloat(np.int32(123))
        KonstFloat(np.int16(123))
        KonstFloat(np.int8(123))

    def test_resolvable_contract_creation(self) -> None:
        model = _make_model()
        c = MyContract(model.dategrid[-1], 1234)
        model.generate_cashflows(c)
        self.assertRaises(TypeError, lambda: ResolvableContract())  # type: ignore

    def test_model_creation(self) -> None:
        nsim = 100
        model = _make_model(nsim=nsim)
        self.assertEqual(model.shape, (nsim, model.dategrid.size))

    def test_date_index(self) -> None:
        model = _make_model()
        at0 = At(model.dategrid[0])
        idx0 = model.eval_date_index.next_after(at0.simulate(model))
        self.assertTrue((idx0.index == 0).all())
        at1 = At(model.dategrid[1])
        idx1 = model.eval_date_index.next_after(at1.simulate(model))
        self.assertTrue((idx1.index == 1).all())

    def test_stock(self) -> None:
        model = _make_model()
        c = When(At(model.dategrid[-1]), Scale(Stock("ABC"), One("EUR")))
        cf = model.generate_cashflows(c)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf.cashflows["value"][:, 0] == model.simulated_stocks["ABC"][:, -1]).all()
        )
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-1]).all())
        self.assertEqual(cf.currencies[0], "EUR")

    def test_fx(self) -> None:
        model = _make_model()
        c = When(At(model.dategrid[-1]), Scale(FX("EUR", "USD"), One("EUR")))
        cf = model.generate_cashflows(c)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (
                cf.cashflows["value"][:, 0] == model.simulated_fx[("EUR", "USD")][:, -1]
            ).all()
        )
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-1]).all())
        self.assertEqual(cf.currencies[0], "EUR")
        c = When(At(model.dategrid[-1]), Scale(FX("USD", "EUR"), One("EUR")))
        cf = model.generate_cashflows(c)
        self.assertTrue(
            (
                cf.cashflows["value"][:, 0]
                == 1 / model.simulated_fx[("EUR", "USD")][:, -1]
            ).all()
        )

    def test_cashflow_currency_conversion(self) -> None:
        model = _make_model()
        self.assertEqual(model.currencies, {"EUR", "USD"})
        c = When(At(model.dategrid[-1]), Scale(Stock("ABC"), One("EUR")))
        cf_eur = c.generate_cashflows(model.eval_date_index, model)
        self.assertRaises(AssertionError, lambda: model.in_currency(cf_eur, "GBP"))
        cf_usd = model.in_currency(cf_eur, "USD")
        self.assertEqual(cf_eur.cashflows.shape, cf_usd.cashflows.shape)
        self.assertEqual(cf_usd.currencies[0], "USD")
        cf_eur_conv = model.in_numeraire_currency(cf_usd)
        self.assertEqual(cf_eur_conv.currencies[0], "EUR")
        self.assertTrue(
            np.allclose(cf_eur.cashflows["value"], cf_eur_conv.cashflows["value"])
        )
        # test with deterministic spots
        numeraire = np.ones((1, model.dategrid.size), dtype=np.float_)
        ccyspot = np.arange(1, model.dategrid.size + 1, dtype=np.float_).reshape(
            numeraire.shape
        )
        model2 = Model(model.dategrid, {("UND", "ACC"): ccyspot}, {}, numeraire, "UND")
        self.assertEqual(model2.currencies, {"UND", "ACC"})
        for i, dt in enumerate(model2.dategrid):
            c = When(At(dt), One("UND"))
            cf_und = c.generate_cashflows(model2.eval_date_index, model2)
            cf_acc = model2.in_currency(cf_und, "ACC")
            self.assertEqual(cf_acc.cashflows["value"][0, 0], i + 1)
        for i, dt in enumerate(model2.dategrid):
            c = When(At(dt), One("ACC"))
            cf_acc = c.generate_cashflows(model2.eval_date_index, model2)
            cf_und = model2.in_numeraire_currency(cf_acc)
            self.assertEqual(cf_und.cashflows["value"][0, 0], 1 / (i + 1))

    def test_cashflow_currency_conversion_nnn(self) -> None:
        model = _make_model()
        self.assertEqual(model.currencies, {"EUR", "USD"})
        c = Zero()
        cf = c.generate_cashflows(model.eval_date_index, model)
        converted = model.in_currency(cf, "EUR")
        self.assertEqual(converted.currencies[0], "NNN")
        c2 = One("EUR")
        cf2 = c2.generate_cashflows(model.eval_date_index, model)
        self.assertRaises(AssertionError, lambda: model.in_currency(cf2, "NNN"))
        self.assertRaises(AssertionError, lambda: model.get_simulated_fx("NNN", "EUR"))
        self.assertRaises(AssertionError, lambda: model.get_simulated_fx("EUR", "NNN"))

    def test_boolean_obs_at(self) -> None:
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
        alt = AlternatingBool()
        altsim = alt.simulate(model)
        self.assertFalse(altsim[np.arange(0, model.nsim, 2), :].any())
        self.assertTrue(altsim[np.arange(1, model.nsim, 2), :].all())

    def test_zero_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(Zero())
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "NNN")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 0).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_one_cashflow_generation(self) -> None:
        ccy = "EUR"
        c = One(ccy)
        model = _make_model()
        cf = c.generate_cashflows(model.eval_date_index, model)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        self.assertEqual(cf.currencies[0], ccy)
        bad_eval_idx = DateIndex(np.zeros((model.nsim + 1,), dtype=np.int))
        self.assertRaises(
            AssertionError, lambda: c.generate_cashflows(bad_eval_idx, model)
        )

    def test_give_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(Give(One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == -1).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_scale_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(Scale(KonstFloat(1.23), One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1.23).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_and_cashflow_generation(self) -> None:
        c = And(One("EUR"), One("USD"))
        model = _make_model()
        cf = c.generate_cashflows(model.eval_date_index, model)
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.currencies[1], "USD")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        cf_alt = model.generate_cashflows(c)
        self.assertTrue(cf_alt, cf.apply_index())

    def test_or_cashflow_generation(self) -> None:
        model = _make_model()
        c2 = Or(One("EUR"), When(At(model.dategrid[-1]), One("EUR")))
        self.assertRaises(NotImplementedError, lambda: model.generate_cashflows(c2))
        c3 = Or(One("EUR"), Scale(KonstFloat(2), One("EUR")))
        cf = model.generate_cashflows(c3)
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.currencies[1], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf.cashflows["value"][:, 0] == 0).all())
        self.assertTrue((np.isnat(cf.cashflows["date"][:, 0])).all())
        self.assertTrue((cf.cashflows["value"][:, 1] == 2).all())
        self.assertTrue((cf.cashflows["date"][:, 1] == model.eval_date).all())
        c4 = Or(One("EUR"), One("USD"))
        cf4 = model.generate_cashflows(c4)
        self.assertEqual(cf4.currencies.shape, (2,))
        self.assertEqual(cf4.currencies[0], "EUR")
        self.assertEqual(cf4.currencies[1], "USD")

    def test_when_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(When(At(model.dategrid[0]), One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["date"] == model.dategrid[0]).all())
        cf1 = model.generate_cashflows(When(At(model.dategrid[1]), One("EUR")))
        self.assertEqual(cf1.currencies.shape, (1,))
        self.assertTrue(cf1.currencies[0], "EUR")
        self.assertEqual(cf1.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf1.cashflows["value"] == 1).all())
        self.assertTrue((cf1.cashflows["date"] == model.dategrid[1]).all())
        cf2 = model.generate_cashflows(When(AlternatingBool(), One("EUR")))
        self.assertEqual(cf2.currencies.shape, (1,))
        self.assertTrue(cf2.currencies[0], "EUR")
        self.assertEqual(cf2.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf2.cashflows["value"][np.arange(0, model.nsim, 2), :] == 0).all()
        )
        self.assertTrue(
            (cf2.cashflows["value"][np.arange(1, model.nsim, 2), :] == 1).all()
        )

    def test_cond_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(Cond(AlternatingBool(), One("EUR"), One("USD")))
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.currencies[1], "USD")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue(
            (cf.cashflows["value"][np.arange(0, model.nsim, 2), 0] == 0).all()
        )
        self.assertTrue(
            (cf.cashflows["value"][np.arange(0, model.nsim, 2), 1] == 1).all()
        )
        self.assertTrue(
            (cf.cashflows["value"][np.arange(1, model.nsim, 2), 0] == 1).all()
        )
        self.assertTrue(
            (cf.cashflows["value"][np.arange(1, model.nsim, 2), 1] == 0).all()
        )

    def test_until_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(Until(AlternatingBool(), One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf.cashflows["value"][np.arange(0, model.nsim, 2), 0] == 0).all()
        )
        self.assertTrue(
            (cf.cashflows["value"][np.arange(1, model.nsim, 2), 0] == 1).all()
        )

    def test_zero_coupon_bond(self) -> None:
        model = _make_model()
        notional = 1234
        currency = "USD"
        zcb = ZeroCouponBond(model.dategrid[-2], notional, currency)
        cf = model.generate_cashflows(zcb)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], currency)
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == notional).all())
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-2]).all())

    def test_european_option_on_zcb(self) -> None:
        model = _make_model()
        notional = 1234
        currency = "USD"
        strike = 1000
        zcb = ZeroCouponBond(model.dategrid[-2], notional, currency)
        opt = EuropeanOption(
            model.dategrid[-2], And(zcb, Give(Scale(KonstFloat(strike), One(currency))))
        )
        cf = model.generate_cashflows(opt)
        self.assertEqual(cf.currencies.shape, (3,))
        self.assertEqual(cf.currencies[0], currency)
        self.assertEqual(cf.currencies[1], currency)
        self.assertEqual(cf.currencies[2], "NNN")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 3))
        self.assertTrue((cf.cashflows["date"][:, 0] == model.dategrid[-2]).all())
        self.assertTrue((cf.cashflows["value"][:, 0] == notional).all())
        self.assertTrue((cf.cashflows["date"][:, 1] == model.dategrid[-2]).all())
        self.assertTrue((cf.cashflows["value"][:, 1] == -strike).all())
        self.assertTrue((np.isnat(cf.cashflows["date"][:, 2])).all())
        self.assertTrue((cf.cashflows["value"][:, 2] == 0).all())

    def test_brownian_motions(self) -> None:
        mu = 0.123
        sigma = 0.456
        bm = BrownianMotion(mu=mu, sigma=sigma)
        gbm = GeometricBrownianMotion(mu=mu, sigma=sigma)
        rnd = np.random.RandomState(1234)
        t = np.linspace(0, 20, 20)
        n = 200
        x = bm.simulate(t, n, rnd)
        self.assertEqual((n, t.size), x.shape)
        self.assertEqual(n, x[:, -1].size)
        x = gbm.simulate(t, n, rnd)
        self.assertEqual((n, t.size), x.shape)
        self.assertEqual(n, x[:, -1].size)
