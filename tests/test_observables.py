import unittest

import numpy as np
from mcc import (
    DateIndex,
    Model,
    generate_cashflows,
    ObservableBool,
    LinearRate,
    FixedAfter,
    RunningMax,
    RunningMin,
    Maximum,
    Minimum,
    Stock,
    FX,
    At,
    One,
    When,
)

from .test_utils import make_model, AlternatingBool


class TestObservables(unittest.TestCase):
    def test_observable_float_calculations(self) -> None:
        model = make_model()
        stock_twice = Stock("ABC") + Stock("ABC")
        once = Stock("ABC").simulate(model.eval_date_index, model)
        twice = stock_twice.simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(2 * once, twice))
        addconst = (Stock("ABC") + 123).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(once + 123, addconst))
        constadd = (123 + Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(123 + once, constadd))
        minus = (-Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(minus, -once))
        constsub = (Stock("ABC") - 123).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(constsub, once - 123))
        subconst = (123 - Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(subconst, 123 - once))
        zero = (Stock("ABC") - Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(zero, np.zeros(shape=zero.shape)))
        stockmulti = (Stock("ABC") * 123).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(stockmulti, 123 * once))
        multistock = (123 * Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(multistock, once * 123))
        stocksquared = (Stock("ABC") * Stock("ABC")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.allclose(stocksquared, once**2))
        stockdiv = (Stock("ABC") / 123).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(stockdiv, once / 123))
        divstock = (123 / Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(divstock, 123 / once))
        one = (Stock("ABC") / Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(one, np.ones(shape=zero.shape)))
        org = (1 / (1 / Stock("ABC"))).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(org, once))
        stockpower = (Stock("ABC") ** 123).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(stockpower, once**123))
        powerstock = (123 ** Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(powerstock, 123**once))
        stockstock = (Stock("ABC") ** Stock("ABC")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.allclose(stockstock, once**once))
        org2 = ((Stock("ABC") ** 3) ** (1 / 3)).simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(org2, once))

    def test_observable_float_comparisons(self) -> None:
        model = make_model()
        # greater or equal
        barrier_breach = Stock("ABC") >= 1
        bbsim = barrier_breach.simulate(model.eval_date_index, model)
        self.assertTrue((bbsim == (model.simulated_stocks["ABC"] >= 1)).all())
        shouldbeall = Stock("ABC") >= Stock("ABC")
        self.assertTrue(shouldbeall.simulate(model.eval_date_index, model).all())
        # strictly greater
        barrier_breach = Stock("ABC") > 1
        bbsim = barrier_breach.simulate(model.eval_date_index, model)
        self.assertTrue((bbsim == (model.simulated_stocks["ABC"] > 1)).all())
        shouldbenone = Stock("ABC") > Stock("ABC")
        self.assertFalse(shouldbenone.simulate(model.eval_date_index, model).any())
        # less or equal
        barrier_breach = Stock("ABC") <= 1
        bbsim = barrier_breach.simulate(model.eval_date_index, model)
        self.assertTrue((bbsim == (model.simulated_stocks["ABC"] <= 1)).all())
        shouldbeall = Stock("ABC") <= Stock("ABC")
        self.assertTrue(shouldbeall.simulate(model.eval_date_index, model).all())
        # strictly less
        barrier_breach = Stock("ABC") < 1
        bbsim = barrier_breach.simulate(model.eval_date_index, model)
        self.assertTrue((bbsim == (model.simulated_stocks["ABC"] < 1)).all())
        shouldbenone = Stock("ABC") < Stock("ABC")
        self.assertFalse(shouldbenone.simulate(model.eval_date_index, model).any())
        # test right comparison operator application
        self.assertIsInstance(1 <= Stock("ABC"), ObservableBool)
        self.assertIsInstance(1 < Stock("ABC"), ObservableBool)
        self.assertIsInstance(1 >= Stock("ABC"), ObservableBool)
        self.assertIsInstance(1 > Stock("ABC"), ObservableBool)

    def test_maximum(self) -> None:
        model = make_model()
        abcsim = Stock("ABC").simulate(model.eval_date_index, model)
        defgsim = Stock("DEFG").simulate(model.eval_date_index, model)
        maxsim = Maximum(Stock("ABC"), Stock("DEFG")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.all(maxsim >= abcsim))
        self.assertTrue(np.any(maxsim > abcsim))
        self.assertTrue(np.all(maxsim >= defgsim))
        self.assertTrue(np.any(maxsim > defgsim))
        maxreversim = Maximum(Stock("DEFG"), Stock("ABC")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.allclose(maxsim, maxreversim))

    def test_maximum_specific_example(self) -> None:
        dategrid = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-02"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-04"),
                np.datetime64("2020-01-05"),
                np.datetime64("2020-01-06"),
                np.datetime64("2020-01-07"),
                np.datetime64("2020-01-09"),
            ],
            dtype="datetime64[D]",
        )
        abc = np.reshape(
            np.array([[1, 2, 1, 1.5, 3, -1, 4, 3], [1, 2, 3, 4, 5, 6, 7, 8]]),
            newshape=(2, dategrid.size),
        )
        defg = np.reshape(
            np.array([[-1, -2, 10, 1.5, 2, 0, 5, 5], [8, 7, 6, 5, 4, 3, 2, 1]]),
            newshape=(2, dategrid.size),
        )
        targets = np.reshape(
            np.array([[1, 2, 10, 1.5, 3, 0, 5, 5], [8, 7, 6, 5, 5, 6, 7, 8]]),
            newshape=(2, dategrid.size),
        )
        model = Model(
            dategrid,
            {},
            {},
            {"ABC": abc, "DEFG": defg},
            np.ones((2, dategrid.size), dtype=np.float64),
            "EUR",
        )
        maxsim = Maximum(Stock("ABC"), Stock("DEFG")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.allclose(targets, maxsim))

    def test_minimum(self) -> None:
        model = make_model()
        abcsim = Stock("ABC").simulate(model.eval_date_index, model)
        defgsim = Stock("DEFG").simulate(model.eval_date_index, model)
        minsim = Minimum(Stock("ABC"), Stock("DEFG")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.all(minsim <= abcsim))
        self.assertTrue(np.any(minsim < abcsim))
        self.assertTrue(np.all(minsim <= defgsim))
        self.assertTrue(np.any(minsim < defgsim))
        minreversesim = Minimum(Stock("DEFG"), Stock("ABC")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.allclose(minsim, minreversesim))

    def test_minimum_specific_example(self) -> None:
        dategrid = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-02"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-04"),
                np.datetime64("2020-01-05"),
                np.datetime64("2020-01-06"),
                np.datetime64("2020-01-07"),
                np.datetime64("2020-01-09"),
            ],
            dtype="datetime64[D]",
        )
        abc = np.reshape(
            np.array([[1, 2, 1, 1.5, 3, -1, 4, 3], [1, 2, 3, 4, 5, 6, 7, 8]]),
            newshape=(2, dategrid.size),
        )
        defg = np.reshape(
            np.array([[-1, -2, 10, 1.5, 2, 0, 5, 5], [8, 7, 6, 5, 4, 3, 2, 1]]),
            newshape=(2, dategrid.size),
        )
        targets = np.reshape(
            np.array([[-1, -2, 1, 1.5, 2, -1, 4, 3], [1, 2, 3, 4, 4, 3, 2, 1]]),
            newshape=(2, dategrid.size),
        )
        model = Model(
            dategrid,
            {},
            {},
            {"ABC": abc, "DEFG": defg},
            np.ones((2, dategrid.size), dtype=np.float64),
            "EUR",
        )
        minsim = Minimum(Stock("ABC"), Stock("DEFG")).simulate(
            model.eval_date_index, model
        )
        self.assertTrue(np.allclose(targets, minsim))

    def test_fixed_after(self) -> None:
        model = make_model()
        fixed1 = FixedAfter(AlternatingBool(), Stock("ABC"))
        fixed1sim = fixed1.simulate(model.eval_date_index, model)
        altsim = AlternatingBool().simulate(model.eval_date_index, model)
        self.assertTrue(
            np.allclose(fixed1sim[altsim[:, 0], 0], fixed1sim[altsim[:, 0], -1])
        )
        self.assertFalse(
            np.isclose(fixed1sim[~altsim[:, 0], 0], fixed1sim[~altsim[:, 0], -1]).any()
        )
        fixed2 = FixedAfter(At(model.dategrid[-2, 0]), Stock("ABC"))
        fixed2sim = fixed2.simulate(model.eval_date_index, model)
        self.assertTrue(np.allclose(fixed2sim[:, -2], fixed2sim[:, -1]))
        self.assertFalse(np.isclose(fixed2sim[:, -3], fixed2sim[:, -1]).any())
        # ensure fixing condition is checked after first_observation_idx
        always = Stock("ABC") > 0
        fixed3 = FixedAfter(always, Stock("ABC"))
        fixed3sim1 = fixed3.simulate(model.eval_date_index, model)
        self.assertTrue(
            np.allclose(
                fixed3sim1,
                np.repeat(model.simulated_stocks["ABC"][:, :1], model.ndates, axis=1),
            )  # all columns equal to first one as immediately fixed
        )
        fixed3sim2 = fixed3.simulate(
            DateIndex(model.eval_date_index.index + model.ndates - 1), model
        )
        self.assertTrue(
            np.allclose(fixed3sim2, Stock("ABC").simulate(model.eval_date_index, model))
        )

    def test_running_maximum(self) -> None:
        model = make_model()
        stocksim = Stock("ABC").simulate(model.eval_date_index, model)
        maxsim = RunningMax(Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.all(maxsim >= stocksim))
        self.assertTrue(np.any(maxsim > stocksim))
        increments = np.diff(maxsim, axis=1)
        self.assertTrue(np.all(increments >= 0))
        self.assertTrue(np.any(increments > 0))

    def test_running_maximum_specific_example(self) -> None:
        dategrid = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-02"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-04"),
                np.datetime64("2020-01-05"),
                np.datetime64("2020-01-06"),
                np.datetime64("2020-01-07"),
                np.datetime64("2020-01-09"),
            ],
            dtype="datetime64[D]",
        )
        abc = np.reshape(
            np.array([1, 2, 1, 1.5, 3, -1, 4, 3]), newshape=(1, dategrid.size)
        )
        targets = np.array(
            [
                [1, 2, 2, 2, 3, 3, 4, 4],
                [1, 2, 2, 2, 3, 3, 4, 4],
                [1, 2, 1, 1.5, 3, 3, 4, 4],
                [1, 2, 1, 1.5, 3, 3, 4, 4],
                [1, 2, 1, 1.5, 3, 3, 4, 4],
                [1, 2, 1, 1.5, 3, -1, 4, 4],
                [1, 2, 1, 1.5, 3, -1, 4, 4],
                [1, 2, 1, 1.5, 3, -1, 4, 3],
            ]
        )
        model = Model(
            dategrid,
            {},
            {},
            {"ABC": abc},
            np.ones((1, dategrid.size), dtype=np.float64),
            "EUR",
        )
        for i in range(targets.shape[0]):
            idx = DateIndex(np.array([i]))
            maxsim = RunningMax(Stock("ABC")).simulate(idx, model)
            self.assertTrue(np.allclose(targets[i, :], maxsim))

    def test_running_minimum(self) -> None:
        model = make_model()
        stocksim = Stock("ABC").simulate(model.eval_date_index, model)
        maxsim = RunningMin(Stock("ABC")).simulate(model.eval_date_index, model)
        self.assertTrue(np.all(maxsim <= stocksim))
        self.assertTrue(np.any(maxsim < stocksim))
        increments = np.diff(maxsim, axis=1)
        self.assertTrue(np.all(increments <= 0))
        self.assertTrue(np.any(increments < 0))

    def test_running_minimum_specific_example(self) -> None:
        dategrid = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-02"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-04"),
                np.datetime64("2020-01-05"),
                np.datetime64("2020-01-06"),
                np.datetime64("2020-01-07"),
                np.datetime64("2020-01-09"),
            ],
            dtype="datetime64[D]",
        )
        abc = np.reshape(
            np.array([1, 2, 1, 1.5, 3, -1, 4, 3]), newshape=(1, dategrid.size)
        )
        targets = np.array(
            [
                [1, 1, 1, 1, 1, -1, -1, -1],
                [1, 2, 1, 1, 1, -1, -1, -1],
                [1, 2, 1, 1, 1, -1, -1, -1],
                [1, 2, 1, 1.5, 1.5, -1, -1, -1],
                [1, 2, 1, 1.5, 3, -1, -1, -1],
                [1, 2, 1, 1.5, 3, -1, -1, -1],
                [1, 2, 1, 1.5, 3, -1, 4, 3],
                [1, 2, 1, 1.5, 3, -1, 4, 3],
            ]
        )
        model = Model(
            dategrid,
            {},
            {},
            {"ABC": abc},
            np.ones((1, dategrid.size), dtype=np.float64),
            "EUR",
        )
        for i in range(targets.shape[0]):
            idx = DateIndex(np.array([i]))
            maxsim = RunningMin(Stock("ABC")).simulate(idx, model)
            self.assertTrue(np.allclose(targets[i, :], maxsim))

    def test_boolean_operators(self) -> None:
        model = make_model()
        alt = AlternatingBool()
        altsim = alt.simulate(model.eval_date_index, model)
        altinvertsim = (~alt).simulate(model.eval_date_index, model)
        self.assertTrue((altsim == ~altinvertsim).all())
        alt2 = AlternatingBool(False)
        self.assertFalse((alt & alt2).simulate(model.eval_date_index, model).any())
        self.assertTrue((alt | alt2).simulate(model.eval_date_index, model).all())

    def test_rates(self) -> None:
        model = make_model()
        dategrid = model.dategrid.flatten()
        yearfraction = (dategrid[-1] - dategrid[-3]).astype(np.float64) / 365
        c = When(
            At(model.dategrid[-1]),
            FixedAfter(At(dategrid[-3]), LinearRate("EUR", "3M"))
            * yearfraction
            * One("EUR"),
        )
        cf = generate_cashflows(model, c)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-1]).all())
        self.assertEqual(cf.currencies[0], "EUR")

    def test_stock(self) -> None:
        model = make_model()
        c = When(At(model.dategrid[-1]), Stock("ABC") * One("EUR"))
        cf = generate_cashflows(model, c)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (cf.cashflows["value"][:, 0] == model.simulated_stocks["ABC"][:, -1]).all()
        )
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-1]).all())
        self.assertEqual(cf.currencies[0], "EUR")

    def test_fx(self) -> None:
        model = make_model()
        c = When(At(model.dategrid[-1]), FX("EUR", "USD") * One("EUR"))
        cf = generate_cashflows(model, c)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue(
            (
                cf.cashflows["value"][:, 0] == model.simulated_fx[("EUR", "USD")][:, -1]
            ).all()
        )
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-1]).all())
        self.assertEqual(cf.currencies[0], "EUR")
        c = When(At(model.dategrid[-1]), FX("USD", "EUR") * One("EUR"))
        cf = generate_cashflows(model, c)
        self.assertTrue(
            (
                cf.cashflows["value"][:, 0]
                == 1 / model.simulated_fx[("EUR", "USD")][:, -1]
            ).all()
        )

    def test_boolean_obs_at(self) -> None:
        model = make_model()
        at0 = At(model.dategrid[0])
        at0sim = at0.simulate(model.eval_date_index, model)
        self.assertEqual(at0sim.dtype, np.bool_)
        self.assertEqual(at0sim.shape, model.shape)
        self.assertTrue(at0sim[:, 0].all())
        self.assertFalse(at0sim[:, 1].any())
        at1 = At(model.dategrid[1])
        at1sim = at1.simulate(model.eval_date_index, model)
        self.assertEqual(at1sim.dtype, np.bool_)
        self.assertEqual(at1sim.shape, model.shape)
        self.assertFalse(at1sim[:, 0].any())
        self.assertTrue(at1sim[:, 1].all())
        alt = AlternatingBool()
        altsim = alt.simulate(model.eval_date_index, model)
        self.assertFalse(altsim[np.arange(0, model.nsim, 2), :].any())
        self.assertTrue(altsim[np.arange(1, model.nsim, 2), :].all())
