import unittest
import numpy as np
from mcc import Model, Zero, One, When, At, Stock, evaluate
from mcc.pricing_models import simulate_equity_black_scholes_model
from .test_utils import make_model


class TestModel(unittest.TestCase):
    def test_model_without_fx(self) -> None:
        model = make_model()
        model2 = Model(
            model.dategrid,
            {},
            {},
            model.simulated_stocks,
            model.numeraire,
            model.numeraire_currency,
        )
        indexedcf = One(model2.numeraire_currency).generate_cashflows(
            model2.eval_date_index, model
        )
        cf = model2.in_numeraire_currency(indexedcf)
        self.assertTrue((cf.cashflows["value"][:, 0] == 1).all())
        self.assertEqual(model2.currencies, {model2.numeraire_currency})

    def test_model_creation(self) -> None:
        nsim = 100
        model = make_model(nsim=nsim)
        self.assertEqual(model.shape, (nsim, model.dategrid.size))

    def test_cashflow_currency_conversion(self) -> None:
        model = make_model()
        self.assertEqual(model.currencies, {"EUR", "USD"})
        c = When(At(model.dategrid[-1]), Stock("ABC") * One("EUR"))
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
        model2 = Model(
            model.dategrid, {}, {("UND", "ACC"): ccyspot}, {}, numeraire, "UND"
        )
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
        model = make_model()
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

    def test_discounting(self) -> None:
        dategrid = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-06-01"),
            30,
            dtype="datetime64[D]",
        )
        rnd = np.random.RandomState(seed=123)
        n = 100
        rate = 0.03
        model = simulate_equity_black_scholes_model(
            "ABC", "USD", 123, dategrid, 0.2, rate, n, rnd, use_moment_matching=True
        )
        for t in dategrid:
            c = When(At(t), One("USD"))
            cf = c.generate_cashflows(model.eval_date_index, model)
            npv = model.discount(cf)[:, 0].mean()
            self.assertEqual(evaluate(model, c), npv)
            self.assertEqual(model.evaluate(cf), npv)
            dt = (t - dategrid[0]).astype(np.float64) / 365
            self.assertTrue(np.isclose(npv, np.exp(-rate * dt)))
        evaluate(model, When(Stock("ABC") > 130, One("USD")))

    def test_evaluation(self) -> None:
        model = make_model()
        c = When(At(model.dategrid[-1]), One("EUR"))
        npv = evaluate(model, c)
        self.assertEqual(npv, 1)
