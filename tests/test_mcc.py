import unittest

import numpy as np
from pandas.testing import assert_frame_equal
from dataclasses import dataclass
from typing import Callable
from mcc import (
    parser,
    IndexedCashflows,
    DateIndex,
    ModelRequirements,
    Model,
    TermStructuresModel,
    ObservableBool,
    KonstFloat,
    LinearRate,
    FixedAfter,
    RunningMax,
    RunningMin,
    Maximum,
    Minimum,
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
    Anytime,
    Contract,
    ResolvableContract,
    ZeroCouponBond,
    EuropeanOption,
    BrownianMotion,
    GeometricBrownianMotion,
    simulate_equity_black_scholes_model,
    HoLeeModel,
)


@dataclass
class DummyTermStructureModel(TermStructuresModel):
    rate: np.ndarray

    def linear_rate(self, frequency: str) -> np.ndarray:
        return self.rate


def _make_model(nsim: int = 100) -> Model:
    dategrid = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-10"),
        dtype="datetime64[D]",
    )
    numeraire = np.ones((nsim, dategrid.size), dtype=np.float64)
    rnd = np.random.RandomState(123)
    rate = rnd.normal(size=(nsim, dategrid.size))
    eurusd = rnd.lognormal(size=(nsim, dategrid.size))
    abc = rnd.lognormal(size=(nsim, dategrid.size))
    defg = rnd.lognormal(size=(nsim, dategrid.size))
    return Model(
        dategrid,
        {"EUR": DummyTermStructureModel(rate)},
        {("EUR", "USD"): eurusd},
        {"ABC": abc, "DEFG": defg},
        numeraire,
        "EUR",
    )


@dataclass
class MyContract(ResolvableContract):
    maturity: np.datetime64
    notional: float

    def resolve(self) -> Contract:
        return When(At(self.maturity), self.notional * One("EUR"))


class AlternatingBool(ObservableBool):
    def __init__(self, start_with_false: bool = True):
        self.offset = 0 if start_with_false else 1

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        mask = np.array(
            (np.arange(model.nsim) + self.offset) % 2, dtype=np.bool_
        ).reshape((model.nsim, 1))
        return np.repeat(mask, model.ndates, axis=1)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()


class TestMonteCarloContracts(unittest.TestCase):
    def test_argument_parsing(self) -> None:
        args = parser.parse_args([])
        self.assertEqual(args.version, False)
        args = parser.parse_args(["--version"])
        self.assertEqual(args.version, True)

    def test_simple_cashflows(self) -> None:
        model = _make_model()
        c = Cond(AlternatingBool(), One("EUR"), One("USD"))
        cf = model.generate_cashflows(c)
        simplecf = cf.to_simple_cashflows()
        self.assertEqual(simplecf.shape[1], model.nsim)
        self.assertEqual(simplecf.shape[0], 2)
        assert_frame_equal(simplecf, model.generate_simple_cashflows(c))
        simplecf2 = model.generate_simple_cashflows_in_currency(c, "USD")
        self.assertEqual(simplecf2.shape[1], model.nsim)
        self.assertEqual(simplecf2.shape[0], 1)
        self.assertEqual(simplecf2.index[0][1], "USD")
        simplecf3 = model.generate_simple_cashflows_in_numeraire_currency(c)
        self.assertEqual(simplecf3.shape[1], model.nsim)
        self.assertEqual(simplecf3.shape[0], 1)
        self.assertEqual(simplecf3.index[0][1], model.numeraire_currency)

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

    def test_cashflow_shifting(self) -> None:
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf = IndexedCashflows(
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
        # shift to same index as cashflows are anyway
        shifted1 = cf.shift_to(DateIndex(np.array([0, 1])))
        self.assertEqual(cf, shifted1)
        # shift to index 0, meaning effectively no change
        shifted2 = cf.shift_to(DateIndex(np.array([0, 0])))
        self.assertEqual(cf, shifted2)
        # shift to index 1, meaning all cashflows at 1
        shifted3 = cf.shift_to(DateIndex(np.array([1, 1])))
        self.assertTrue((shifted3.cashflows["index"] == 1).all())

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

    def test_model_without_fx(self) -> None:
        model = _make_model()
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

    def test_resolvable_contract_creation(self) -> None:
        model = _make_model()
        c = MyContract(model.dategrid[-1], 1234)
        model.generate_cashflows(c)
        self.assertRaises(TypeError, lambda: ResolvableContract())  # type: ignore

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
            "When(2030-07-14, One(EUR))), And(Until(~(FX(EUR/USD) >= 1.0), "
            "Give(Scale(1.23, One(USD)))), Anytime((Stock(DEF) >= 50) "
            "| (~(Stock(DEF) >= 20)), Scale(Stock(ABC), One(EUR)))))"
        )
        self.assertEqual(str(c), expected)
        c2 = (Stock("ABC") ** 2 / (Stock("DEF") - 1.7) + 42) * One("EUR")
        self.assertEqual(
            str(c2),
            "Scale((((Stock(ABC)) ** (2)) / ((Stock(DEF)) + (-1.7))) + (42), One(EUR))",
        )

    def test_model_creation(self) -> None:
        nsim = 100
        model = _make_model(nsim=nsim)
        self.assertEqual(model.shape, (nsim, model.dategrid.size))

    def test_date_index_specific_example(self) -> None:
        x = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.1, 1.1, 2.1, 3.1],
                [0.2, 1.2, 2.2, 3.2],
                [0.9, 0.8, 0.7, 0.6],
                [0.0, -1, -2, -3],
            ]
        )
        idx = DateIndex(np.array([1, 1, 2, 0, 3]))
        self.assertTrue(np.allclose([1.0, 1.1, 2.2, 0.9, -3], idx.index_column(x)))
        self.assertTrue(np.allclose([2, 1, 2, -1, -1], idx.next_after(x > 1).index))
        self.assertTrue(
            np.all(
                np.array(
                    [
                        [False, True, True, True],
                        [False, True, True, True],
                        [False, False, True, True],
                        [True, True, True, True],
                        [False, False, False, True],
                    ]
                )
                == idx.after_mask(4),
            )
        )
        self.assertTrue(
            np.all(
                np.array(
                    [
                        [False, False, True, True],
                        [False, True, True, True],
                        [False, False, True, True],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                )
                == idx.next_after(x > 1).after_mask(4),
            )
        )
        self.assertTrue(
            np.all(
                idx.next_after(x > 1).after_mask(10)
                == ~idx.next_after(x > 1).before_mask(10),
            )
        )

    def test_date_index(self) -> None:
        model = _make_model()
        at0 = At(model.dategrid[0])
        idx0 = model.eval_date_index.next_after(
            at0.simulate(model.eval_date_index, model)
        )
        self.assertTrue((idx0.index == 0).all())
        at1 = At(model.dategrid[1])
        idx1 = model.eval_date_index.next_after(
            at1.simulate(model.eval_date_index, model)
        )
        self.assertTrue((idx1.index == 1).all())

    def test_observable_float_calculations(self) -> None:
        model = _make_model()
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
        model = _make_model()
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
        model = _make_model()
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
        model = _make_model()
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
        model = _make_model()
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
        model = _make_model()
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
        model = _make_model()
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
        model = _make_model()
        alt = AlternatingBool()
        altsim = alt.simulate(model.eval_date_index, model)
        altinvertsim = (~alt).simulate(model.eval_date_index, model)
        self.assertTrue((altsim == ~altinvertsim).all())
        alt2 = AlternatingBool(False)
        self.assertFalse((alt & alt2).simulate(model.eval_date_index, model).any())
        self.assertTrue((alt | alt2).simulate(model.eval_date_index, model).all())

    def test_rates(self) -> None:
        model = _make_model()
        dategrid = model.dategrid.flatten()
        yearfraction = (dategrid[-1] - dategrid[-3]).astype(np.float64) / 365
        c = When(
            At(model.dategrid[-1]),
            FixedAfter(At(dategrid[-3]), LinearRate("EUR", "3M"))
            * yearfraction
            * One("EUR"),
        )
        cf = model.generate_cashflows(c)
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-1]).all())
        self.assertEqual(cf.currencies[0], "EUR")

    def test_stock(self) -> None:
        model = _make_model()
        c = When(At(model.dategrid[-1]), Stock("ABC") * One("EUR"))
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
        c = When(At(model.dategrid[-1]), FX("EUR", "USD") * One("EUR"))
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
        c = When(At(model.dategrid[-1]), FX("USD", "EUR") * One("EUR"))
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
        bad_eval_idx = DateIndex(np.zeros((model.nsim + 1,), dtype=np.int64))
        self.assertRaises(
            AssertionError, lambda: c.generate_cashflows(bad_eval_idx, model)
        )

    def test_give_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(-One("EUR"))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == -1).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_scale_cashflow_generation(self) -> None:
        model = _make_model()
        cf = model.generate_cashflows(1.23 * One("EUR"))
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == 1.23).all())
        self.assertTrue((cf.cashflows["date"] == model.eval_date).all())

    def test_and_cashflow_generation(self) -> None:
        c = One("EUR") + One("USD")
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
        c2 = One("EUR") | When(At(model.dategrid[-1]), One("EUR"))
        self.assertRaises(NotImplementedError, lambda: model.generate_cashflows(c2))
        c3 = One("EUR") | 2 * One("EUR")
        cf = model.generate_cashflows(c3)
        self.assertEqual(cf.currencies.shape, (2,))
        self.assertEqual(cf.currencies[0], "EUR")
        self.assertEqual(cf.currencies[1], "EUR")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 2))
        self.assertTrue((cf.cashflows["value"][:, 0] == 0).all())
        self.assertTrue((np.isnat(cf.cashflows["date"][:, 0])).all())
        self.assertTrue((cf.cashflows["value"][:, 1] == 2).all())
        self.assertTrue((cf.cashflows["date"][:, 1] == model.eval_date).all())
        c4 = One("EUR") | One("USD")
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
            (cf.cashflows["value"][np.arange(0, model.nsim, 2), 0] == 1).all()
        )
        self.assertTrue(
            (cf.cashflows["value"][np.arange(1, model.nsim, 2), 0] == 0).all()
        )

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
            self.assertEqual(model.evaluate(c), npv)
            self.assertEqual(model.evaluate(cf), npv)
            dt = (t - dategrid[0]).astype(np.float64) / 365
            self.assertTrue(np.isclose(npv, np.exp(-rate * dt)))
        model.evaluate(When(Stock("ABC") > 130, One("USD")))

    def test_evaluation(self) -> None:
        model = _make_model()
        c = When(At(model.dategrid[-1]), One("EUR"))
        npv = model.evaluate(c)
        self.assertEqual(npv, 1)

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
        opt = EuropeanOption(model.dategrid[-2], zcb - strike * One(currency))
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
        bm = BrownianMotion(mu_t=lambda t: mu * t, sigma=sigma)
        gbm = GeometricBrownianMotion(mu_t=lambda t: mu * t, sigma=sigma)
        rnd = np.random.RandomState(1234)
        t = np.linspace(0, 20, 20)
        n = 200
        x = bm.simulate(t, n, rnd)
        bm.expected(t), bm.stddev(t)
        bm.moment_match(t, x)
        self.assertEqual((n, t.size), x.shape)
        self.assertEqual(n, x[:, -1].size)
        x = gbm.simulate(t, n, rnd)
        self.assertEqual((n, t.size), x.shape)
        self.assertEqual(n, x[:, -1].size)
        gbm.expected(t), gbm.stddev(t)
        gbm.moment_match(t, x)

    def test_moment_matched_brownian_motions(self) -> None:
        bm = BrownianMotion(lambda t: t)
        rnd = np.random.RandomState(1234)
        t = np.linspace(0, 20, 20)
        n = 200
        x = bm.simulate_with_moment_matching(t, n, rnd)
        self.assertTrue(np.allclose(x.mean(axis=0), t))
        self.assertTrue(np.allclose(x.std(axis=0), np.sqrt(t)))
        gbm = GeometricBrownianMotion()
        x = gbm.simulate_with_moment_matching(t, n, rnd)
        self.assertTrue(np.allclose(x.mean(axis=0), np.ones(t.shape)))
        self.assertTrue(np.allclose(x.var(axis=0), np.exp(t) - 1))

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
            self.assertTrue(np.isclose(m.evaluate(c), 123))

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
