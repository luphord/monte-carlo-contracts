import unittest

import numpy as np
from mcc import (
    DateIndex,
    generate_cashflows,
    At,
    KonstFloat,
    Stock,
    FX,
    Zero,
    One,
    Give,
    Scale,
    And,
    Or,
    When,
    Delay,
    Cond,
    Until,
    Anytime,
)

from .test_utils import make_model, AlternatingBool


class TestContracts(unittest.TestCase):
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

    def test_contract_str(self) -> None:
        c = (
            Cond((Stock("ABC") > 28) & ~(Stock("DEF") > 28), Zero(), One("USD"))
            | When(At(np.datetime64("2030-07-14")), One("EUR"))
        ) + (
            Until(FX("EUR", "USD") < 1.0, -(1.23 * One("USD")))
            + Anytime(
                (Stock("DEF") >= 50) | (Stock("DEF") < 20),
                Stock("ABC") * One("EUR"),
            )
        )
        expected = (
            "And(Or(Cond((Stock(ABC) > 28) & (~(Stock(DEF) > 28)), Zero, One(USD)), "
            "When(2030-07-14, One(EUR))), Until(~(FX(EUR/USD) >= 1.0), "
            "Give(Scale(1.23, One(USD)))), Anytime((Stock(DEF) >= 50) "
            "| (~(Stock(DEF) >= 20)), Scale(Stock(ABC), One(EUR))))"
        )
        self.assertEqual(str(c), expected)
        c2 = (Stock("ABC") ** 2 / (Stock("DEF") - 1.7) + 42) * One("EUR")
        self.assertEqual(
            str(c2),
            "Scale((((Stock(ABC)) ** (2)) / ((Stock(DEF)) + (-1.7))) + (42), One(EUR))",
        )

    def test_zero_cashflow_generation(self) -> None:
        model = make_model()
        cf = generate_cashflows(model, Zero())
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "NNN")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 0).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_one_cashflow_generation(self) -> None:
        ccy = "EUR"
        c = One(ccy)
        model = make_model()
        cf = c.generate_cashflows(model.eval_date_index, model)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        self.assertEqual(cf.currencies[0], ccy)
        bad_eval_idx = DateIndex(np.zeros((model.nsim + 1,), dtype=np.int64))
        self.assertRaises(
            AssertionError, lambda: c.generate_cashflows(bad_eval_idx, model)
        )

    def test_give_cashflow_generation(self) -> None:
        model = make_model()
        cf = generate_cashflows(model, -One("EUR"))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == -1).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_scale_cashflow_generation(self) -> None:
        model = make_model()
        cf = generate_cashflows(model, 1.23 * One("EUR"))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1.23).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_and_cashflow_generation(self) -> None:
        c = One("EUR") + One("USD")
        model = make_model()
        cf = c.generate_cashflows(model.eval_date_index, model)
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.currencies[1], "USD")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["index"] == 0).all())
        cf_alt = generate_cashflows(model, c)
        self.assertTrue(cf_alt, cf.apply_index())

    def test_or_cashflow_generation(self) -> None:
        model = make_model()
        c2 = One("EUR") | When(At(model.dategrid[-1]), One("EUR"))
        self.assertRaises(NotImplementedError, lambda: generate_cashflows(model, c2))
        c3 = One("EUR") | 2 * One("EUR")
        cf = generate_cashflows(model, c3)
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.currencies[1], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf.cashflows["value"][:, 0] == 0).all())
        self.assertTrue((np.isnat(cf.cashflows["date"][:, 0])).all())
        self.assertTrue((cf.cashflows["value"][:, 1] == 2).all())
        self.assertTrue((cf.cashflows["date"][:, 1] == model.eval_date).all())
        c4 = One("EUR") | One("USD")
        cf4 = generate_cashflows(model, c4)
        self.assertEqual(cf4.currencies.shape, (2,))
        self.assertEqual(cf4.currencies[0], "EUR")
        self.assertEqual(cf4.currencies[1], "USD")
        c5 = One("EUR") | One("USD") | 2 * One("EUR")
        assert isinstance(c5, Or)
        self.assertEqual(3, len(c5.contracts))
        cf5 = generate_cashflows(model, c5)
        self.assertEqual(cf5.currencies.shape, (3,))
        self.assertEqual(cf5.currencies[0], "EUR")
        self.assertEqual(cf5.currencies[1], "USD")
        self.assertEqual(cf5.currencies[2], "EUR")
        self.assertTrue((np.isnat(cf.cashflows["date"][:, 0])).all())
        c6 = One("EUR") | (Zero() + (One("USD") | 2 * One("EUR")))
        cf6 = generate_cashflows(model, c6)
        self.assertEqual(cf6.currencies.shape, (4,))

    def test_when_cashflow_generation(self) -> None:
        model = make_model()
        cf = generate_cashflows(model, When(At(model.dategrid[0]), One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["date"] == model.dategrid[0]).all())
        cf1 = generate_cashflows(model, When(At(model.dategrid[1]), One("EUR")))
        self.assertEqual(cf1.currencies.shape, (1,))
        self.assertTrue(cf1.currencies[0], "EUR")
        self.assertEqual(cf1.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf1.cashflows["value"] == 1).all())
        self.assertTrue((cf1.cashflows["date"] == model.dategrid[1]).all())
        cf2 = generate_cashflows(model, When(AlternatingBool(), One("EUR")))
        self.assertEqual(cf2.currencies.shape, (1,))
        self.assertTrue(cf2.currencies[0], "EUR")
        self.assertEqual(cf2.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf2.cashflows["value"][np.arange(0, model.nsim, 2), :] == 0).all()
        )
        self.assertTrue(
            (cf2.cashflows["value"][np.arange(1, model.nsim, 2), :] == 1).all()
        )

    def test_delay_cashflow_generation(self) -> None:
        model = make_model()
        cf = generate_cashflows(
            model, Delay(At(model.dategrid[0]), When(At(model.dategrid[0]), One("EUR")))
        )
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1).all())
        self.assertTrue((cf.cashflows["date"] == model.dategrid[0]).all())
        cf1 = generate_cashflows(
            model, Delay(At(model.dategrid[1]), When(At(model.dategrid[0]), One("EUR")))
        )
        self.assertEqual(cf1.currencies.shape, (1,))
        self.assertTrue(cf1.currencies[0], "EUR")
        self.assertEqual(cf1.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf1.cashflows["value"] == 1).all())
        self.assertTrue((cf1.cashflows["date"] == model.dategrid[1]).all())
        cf2 = generate_cashflows(
            model, Delay(At(model.dategrid[1]), When(AlternatingBool(), One("EUR")))
        )
        self.assertEqual(cf2.currencies.shape, (1,))
        self.assertTrue(cf2.currencies[0], "EUR")
        self.assertEqual(cf2.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf2.cashflows["value"][np.arange(0, model.nsim, 2), :] == 0).all()
        )
        self.assertTrue(
            (np.isnat(cf2.cashflows["date"][np.arange(0, model.nsim, 2), :])).all()
        )
        self.assertTrue(
            (cf2.cashflows["value"][np.arange(1, model.nsim, 2), :] == 1).all()
        )
        self.assertTrue(
            (
                cf2.cashflows["date"][np.arange(1, model.nsim, 2), :]
                == model.dategrid[1]
            ).all()
        )

    def test_cond_cashflow_generation(self) -> None:
        model = make_model()
        cf = generate_cashflows(model, Cond(AlternatingBool(), One("EUR"), One("USD")))
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
        model = make_model()
        cf = generate_cashflows(model, Until(AlternatingBool(), One("EUR")))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf.cashflows["value"][np.arange(0, model.nsim, 2), 0] == 1).all()
        )
        self.assertTrue(
            (cf.cashflows["value"][np.arange(1, model.nsim, 2), 0] == 0).all()
        )

    def test_and_contract_flattening(self) -> None:
        c = And(Zero(), Zero())
        self.assertEqual(2, len(c.contracts))
        c2 = And(c, Zero())
        self.assertEqual(3, len(c2.contracts))
        c3 = And(Zero(), c)
        self.assertEqual(3, len(c3.contracts))
        c4 = sum([Zero() for i in range(9)], Zero())
        assert isinstance(c4, And)
        self.assertEqual(10, len(c4.contracts))
        c5 = And(Zero(), And(Zero(), And(Zero(), And(Zero()))))
        self.assertEqual(4, len(c5.contracts))
