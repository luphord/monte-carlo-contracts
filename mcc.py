#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.3.0"""


from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from typing import Final, Union, Mapping, Tuple, Set
from numbers import Real
import numpy as np
from dataclasses import dataclass


_ccy_letters = 3
_null_ccy = "NNN"
ArrayLike = Union[np.ndarray, float]


class SimulatedCashflows:
    dtype: Final = np.dtype([("date", "datetime64[D]"), ("value", np.float64)])
    cashflows: Final[np.ndarray]
    currencies: Final[np.ndarray]
    nsim: Final[int]
    ncashflows: Final[int]

    def __init__(self, cashflows: np.ndarray, currencies: np.ndarray):
        assert (
            cashflows.dtype == self.dtype
        ), f"Got cashflow array with dtype {cashflows.dtype}, expecting {self.dtype}"
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        assert currencies.dtype == (np.unicode_, _ccy_letters)
        assert currencies.shape == (cashflows.shape[1],)
        self.cashflows = cashflows
        self.currencies = currencies
        self.nsim = cashflows.shape[0]
        self.ncashflows = cashflows.shape[1]


class IndexedCashflows:
    dtype: Final = np.dtype([("index", np.int), ("value", np.float64)])
    cashflows: Final[np.ndarray]
    currencies: Final[np.ndarray]
    dategrid: Final[np.ndarray]
    nsim: Final[int]
    ncashflows: Final[int]

    def __init__(
        self, cashflows: np.ndarray, currencies: np.ndarray, dategrid: np.ndarray
    ):
        assert (
            cashflows.dtype == self.dtype
        ), f"Got cashflow array with dtype {cashflows.dtype}, expecting {self.dtype}"
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        assert currencies.dtype == (np.unicode_, _ccy_letters)
        assert currencies.shape == (cashflows.shape[1],)
        self.cashflows = cashflows
        self.currencies = currencies
        assert dategrid.dtype == "datetime64[D]"
        self.dategrid = dategrid
        self.nsim = cashflows.shape[0]
        self.ncashflows = cashflows.shape[1]

    def __add__(self, other: "IndexedCashflows") -> "IndexedCashflows":
        assert (
            self.nsim == other.nsim
        ), f"Cannot add cashflows with {self.nsim} and {other.nsim} simulations"
        assert (self.dategrid == other.dategrid).all()
        return IndexedCashflows(
            np.concatenate((self.cashflows, other.cashflows), axis=1),
            np.concatenate((self.currencies, other.currencies)),
            self.dategrid,
        )

    def __mul__(self, factor: ArrayLike) -> "IndexedCashflows":
        cf = self.cashflows.copy()
        if isinstance(factor, np.ndarray) and factor.ndim == 1:
            factor = factor.reshape((self.nsim, 1))
        cf["value"] *= factor
        return IndexedCashflows(cf, self.currencies, self.dategrid)

    def __neg__(self) -> "IndexedCashflows":
        return self * -1

    def zero_after(self, date_idx: "DateIndex") -> "IndexedCashflows":
        assert self.nsim == date_idx.nsim
        zeroedcf = self.cashflows.copy()
        for i, cf in enumerate(self.cashflows.T):
            ko_mask = (cf["index"] >= date_idx.index) & (date_idx.index >= 0)
            zeroedcf["value"][ko_mask, i] = 0
        return IndexedCashflows(zeroedcf, self.currencies, self.dategrid)

    def apply_index(self) -> SimulatedCashflows:
        dategrid_rep = np.reshape(self.dategrid, (1, self.dategrid.size))
        dategrid_rep = np.repeat(dategrid_rep, self.nsim, axis=0)
        assert dategrid_rep.shape[0] == self.nsim
        datecfs = np.zeros(self.cashflows.shape, dtype=SimulatedCashflows.dtype)
        for i, cf in enumerate(self.cashflows.T):
            datecfs["date"][:, i] = dategrid_rep[np.arange(self.nsim), cf["index"]]
            datecfs["date"][
                cf["index"] < 0, i
            ] = np.datetime64()  # index < 0 means never
            datecfs["value"][:, i] = cf["value"]
        return SimulatedCashflows(np.array(datecfs), self.currencies)


class DateIndex:
    index: Final[np.ndarray]
    nsim: Final[int]

    def __init__(self, index: np.ndarray):
        assert index.dtype == np.int
        assert index.ndim == 1
        self.index = index
        self.nsim = index.size

    def index_column(self, observable: np.ndarray) -> np.ndarray:
        assert observable.ndim == 2
        assert observable.shape[0] == self.nsim
        assert (self.index < observable.shape[1]).all()
        obs = observable[np.arange(self.nsim), self.index.clip(0)]
        obs[self.index < 0] = 0
        assert obs.shape == (self.nsim,)
        return obs

    def next_after(self, obs: np.ndarray) -> "DateIndex":
        assert obs.dtype == np.bool_
        assert obs.ndim == 2
        assert obs.shape[0] == self.nsim
        ndates = obs.shape[1]
        assert (self.index < ndates).all()
        idx = np.repeat(np.arange(ndates).reshape((1, ndates)), self.nsim, axis=0)
        idx[~obs] = ndates  # larger than any index
        idx = np.maximum(idx.min(axis=1), self.index)
        idx[np.logical_or(idx == ndates, self.index == -1)] = -1
        assert idx.shape == (self.nsim,)
        assert (idx < ndates).all()
        return DateIndex(idx)


class Model:
    dategrid: Final[np.ndarray]
    simulated_fx: Final[Mapping[Tuple[str, str], np.ndarray]]
    simulated_stocks: Final[Mapping[str, np.ndarray]]
    numeraire: Final[np.ndarray]
    numeraire_currency: Final[str]
    ndates: Final[int]
    nsim: Final[int]
    shape: Final[Tuple[int, int]]

    def __init__(
        self,
        dategrid: np.ndarray,
        simulated_fx: Mapping[Tuple[str, str], np.ndarray],
        simulated_stocks: Mapping[str, np.ndarray],
        numeraire: np.ndarray,
        numeraire_currency: str,
    ):
        assert dategrid.dtype == "datetime64[D]"
        self.ndates = dategrid.size
        assert self.ndates > 0
        self.dategrid = np.reshape(dategrid, (self.ndates, 1))
        assert numeraire.dtype == np.float
        assert numeraire.ndim == 2
        self.numeraire = numeraire
        self.nsim = numeraire.shape[0]
        self.shape = (self.nsim, self.ndates)
        assert numeraire.shape == self.shape
        self.numeraire_currency = numeraire_currency
        for fxkey, val in simulated_fx.items():
            assert (
                val.dtype == np.float
            ), f"FX spot '{fxkey}' is of dtype {val.dtype}, expecting float"
            assert (
                val.shape == self.shape
            ), f"FX Spot '{fxkey}' has shape {val.shape}, expecting {self.shape}"
        self.simulated_fx = simulated_fx
        for key, val in simulated_stocks.items():
            assert (
                val.dtype == np.float
            ), f"Stock '{key}' is of dtype {val.dtype}, expecting float"
            assert (
                val.shape == self.shape
            ), f"Stock '{key}' has shape {val.shape}, expecting {self.shape}"
        self.simulated_stocks = simulated_stocks

    @property
    def eval_date(self) -> np.datetime64:
        return self.dategrid[0]

    @property
    def eval_date_index(self) -> DateIndex:
        return DateIndex(np.zeros((self.nsim,), dtype=np.int))

    @property
    def currencies(self) -> Set[str]:
        return set([ccy for fxspot in self.simulated_fx for ccy in fxspot]) or {
            self.numeraire_currency
        }

    def get_simulated_fx(self, base_currency: str, counter_currency: str) -> np.ndarray:
        assert (
            base_currency != _null_ccy and counter_currency != _null_ccy
        ), "Cannot retrieve spot containing null currency NNN"
        assert (
            base_currency in self.currencies
        ), f"{base_currency} not contained in model currencies {self.currencies}"
        assert (
            counter_currency in self.currencies
        ), f"{counter_currency} not contained in model currencies {self.currencies}"
        if (base_currency, counter_currency) in self.simulated_fx:
            return self.simulated_fx[(base_currency, counter_currency)]
        elif (counter_currency, base_currency) in self.simulated_fx:
            return 1 / self.simulated_fx[(counter_currency, base_currency)]
        else:
            raise NotImplementedError("Cross calculation of FX spots not implemented")

    def validate_date_index(self, date_index: DateIndex) -> None:
        assert date_index.index.size == self.nsim
        assert (date_index.index < self.ndates).all()

    def generate_cashflows(self, contract: "Contract") -> SimulatedCashflows:
        return contract.generate_cashflows(self.eval_date_index, self).apply_index()

    def in_currency(
        self, cashflows: IndexedCashflows, currency: str
    ) -> IndexedCashflows:
        assert currency != _null_ccy, "Cannot convert to null currency NNN"
        currencies = np.zeros(
            cashflows.currencies.shape, dtype=(np.unicode_, _ccy_letters)
        )
        currencies[:] = currency
        converted = np.zeros(cashflows.cashflows.shape, dtype=IndexedCashflows.dtype)
        for i, cf in enumerate(cashflows.cashflows.T):
            converted["index"][:, i] = cf["index"]
            if cashflows.currencies[i] == _null_ccy:
                currencies[i] = _null_ccy
            if cashflows.currencies[i] == currencies[i]:
                converted["value"][:, i] = cf["value"]
            else:
                base_ccy = str(cashflows.currencies[i])
                counter_ccy = str(currencies[i])
                fx = self.get_simulated_fx(base_ccy, counter_ccy)
                fx_indexed = DateIndex(cf["index"]).index_column(fx)
                converted["value"][:, i] = cf["value"] * fx_indexed
        return IndexedCashflows(converted, currencies, cashflows.dategrid)

    def in_numeraire_currency(self, cashflows: IndexedCashflows) -> IndexedCashflows:
        return self.in_currency(cashflows, self.numeraire_currency)


class ObservableFloat(ABC):
    """Abstract base class for all observables of underlying type float,
    essentially a real-valued stochastic process"""

    @abstractmethod
    def simulate(self, model: Model) -> np.ndarray:
        pass

    def __ge__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        if isinstance(other, ObservableFloat):
            return GreaterOrEqualThan(self, other)
        elif isinstance(other, Real):
            return GreaterOrEqualThan(self, KonstFloat(other))
        else:
            raise TypeError(f"Expecting real number, got {other} of type {type(other)}")

    def __gt__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        if isinstance(other, ObservableFloat):
            return GreaterThan(self, other)
        elif isinstance(other, Real):
            return GreaterThan(self, KonstFloat(other))
        else:
            raise TypeError(f"Expecting real number, got {other} of type {type(other)}")

    def __le__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        return ~(self > other)

    def __lt__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        return ~(self >= other)


@dataclass
class Stock(ObservableFloat):
    """Value of the stock identified by identifier"""

    identifier: str

    def simulate(self, model: Model) -> np.ndarray:
        return model.simulated_stocks[self.identifier]


@dataclass
class FX(ObservableFloat):
    """Value of the currency spot between base_currency and counter_currency,
    i.e. 'one unit counter_currency' / 'one unit of base_currency'"""

    base_currency: str
    counter_currency: str

    def simulate(self, model: Model) -> np.ndarray:
        return model.get_simulated_fx(self.base_currency, self.counter_currency)


@dataclass
class KonstFloat(ObservableFloat):
    """Always equal to constant"""

    constant: float

    def simulate(self, model: Model) -> np.ndarray:
        return self.constant * np.ones(model.shape, dtype=np.float)


class ObservableBool(ABC):
    """Abstract base class for all observables of underlying type bool"""

    @abstractmethod
    def simulate(self, model: Model) -> np.ndarray:
        pass

    def __invert__(self) -> "ObservableBool":
        return Not(self)

    def __and__(self, other: "ObservableBool") -> "ObservableBool":
        return AndObservable(self, other)

    def __or__(self, other: "ObservableBool") -> "ObservableBool":
        return OrObservable(self, other)


@dataclass
class Not(ObservableBool):
    """True if observable is False and vice versa"""

    observable: ObservableBool

    def simulate(self, model: Model) -> np.ndarray:
        return ~self.observable.simulate(model)


@dataclass
class AndObservable(ObservableBool):
    """True if and only if both observables are True"""

    observable1: ObservableBool
    observable2: ObservableBool

    def simulate(self, model: Model) -> np.ndarray:
        return self.observable1.simulate(model) & self.observable2.simulate(model)


@dataclass
class OrObservable(ObservableBool):
    """True if either or both observable are True"""

    observable1: ObservableBool
    observable2: ObservableBool

    def simulate(self, model: Model) -> np.ndarray:
        return self.observable1.simulate(model) | self.observable2.simulate(model)


@dataclass
class GreaterOrEqualThan(ObservableBool):
    """True if and only if observable1 is greater or equal than observable2"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, model: Model) -> np.ndarray:
        return self.observable1.simulate(model) >= self.observable2.simulate(model)


@dataclass
class GreaterThan(ObservableBool):
    """True if and only if observable1 is strictly greater than observable2"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, model: Model) -> np.ndarray:
        return self.observable1.simulate(model) > self.observable2.simulate(model)


@dataclass
class At(ObservableBool):
    """True only at date"""

    date: np.datetime64

    def simulate(self, model: Model) -> np.ndarray:
        mask = (model.dategrid == self.date).reshape((1, model.ndates))
        assert mask.any(), f"{self.date} not contained in dategrid"
        return np.repeat(mask, model.nsim, axis=0)


class Contract(ABC):
    """Abstract base class for all contracts"""

    @abstractmethod
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        pass


@dataclass
class Zero(Contract):
    """Neither receive nor pay anything"""

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        model.validate_date_index(acquisition_idx)
        cf = np.zeros((model.nsim, 1), dtype=IndexedCashflows.dtype)
        cf["index"][:, 0] = acquisition_idx.index
        ccys = np.array([_null_ccy], dtype=(np.unicode_, _ccy_letters))
        return IndexedCashflows(cf, ccys, model.dategrid)


@dataclass
class One(Contract):
    """Receive one unit of currency at acquisition"""

    currency: str

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        model.validate_date_index(acquisition_idx)
        cf = np.ones((model.nsim, 1), dtype=IndexedCashflows.dtype)
        cf["index"][:, 0] = acquisition_idx.index
        cf["index"][acquisition_idx.index < 0, 0] = -1  # index < 0 means never
        cf["value"][acquisition_idx.index < 0, 0] = 0  # index < 0 means never
        ccys = np.array([self.currency], dtype=(np.unicode_, _ccy_letters))
        return IndexedCashflows(cf, ccys, model.dategrid)


@dataclass
class Give(Contract):
    """Receive all obligations of the underlying contract and pay all rights,
    i.e. invert the underlying contract"""

    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        return -self.contract.generate_cashflows(acquisition_idx, model)


@dataclass
class And(Contract):
    """Obtain rights and obligations of both underlying contracts"""

    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cf1 = self.contract1.generate_cashflows(acquisition_idx, model)
        cf2 = self.contract2.generate_cashflows(acquisition_idx, model)
        return cf1 + cf2


@dataclass
class Or(Contract):
    """Choose at acquisition between the underlying contracts"""

    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cf1 = self.contract1.generate_cashflows(acquisition_idx, model)
        cf2 = self.contract2.generate_cashflows(acquisition_idx, model)
        ccys = np.unique(
            np.concatenate((cf1.currencies.flatten(), cf2.currencies.flatten()))
        )
        ccys = ccys[ccys != "NNN"].flatten()
        for cf in (cf1 + cf2).cashflows.T:
            if (acquisition_idx.index != cf["index"]).any():
                raise NotImplementedError(
                    "Cashflow generation for OR contract at any moment"
                    " other than cashflow date is not implemented"
                )
        cf1sum = model.in_numeraire_currency(cf1).cashflows["value"].sum(axis=1)
        cf2sum = model.in_numeraire_currency(cf2).cashflows["value"].sum(axis=1)
        choose1 = cf1sum > cf2sum
        cf1.cashflows["index"][~choose1] = -1
        cf1.cashflows["value"][~choose1] = 0
        cf2.cashflows["index"][choose1] = -1
        cf2.cashflows["value"][choose1] = 0
        return cf1 + cf2


@dataclass
class Cond(Contract):
    """If observable is True at acquisition, obtain contract1, otherwise contract2"""

    observable: ObservableBool
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        obs = acquisition_idx.index_column(self.observable.simulate(model))
        never = acquisition_idx.index < 0
        cf1 = self.contract1.generate_cashflows(acquisition_idx, model)
        cf1.cashflows["value"][~obs, :] = 0
        cf1.cashflows["index"][never, :] = 0
        cf1.cashflows["value"][never, :] = 0
        cf2 = self.contract2.generate_cashflows(acquisition_idx, model)
        cf2.cashflows["value"][obs, :] = 0
        cf2.cashflows["index"][never, :] = 0
        cf2.cashflows["value"][never, :] = 0
        return cf1 + cf2


@dataclass
class Scale(Contract):
    """Same as the underling contract, but all payments scaled by the value of
    observable at acquisition"""

    observable: ObservableFloat
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cf = self.contract.generate_cashflows(acquisition_idx, model)
        obs = acquisition_idx.index_column(self.observable.simulate(model))
        assert cf.cashflows.ndim == 2
        assert cf.cashflows.shape[0] == model.nsim
        assert obs.ndim == 1
        assert obs.shape[0] == model.nsim
        return cf * obs


@dataclass
class When(Contract):
    """Obtain the underlying contract as soon as observable
    becomes True after acquisition"""

    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        idx = acquisition_idx.next_after(self.observable.simulate(model))
        return self.contract.generate_cashflows(idx, model)


@dataclass
class Anytime(Contract):
    """At any point in time after acquisition when observable is True, choose whether
    to obtain the underlying contract or not; can be exercised only once"""

    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        raise NotImplementedError()


@dataclass
class Until(Contract):
    """Obtain the underlying contract, but as soon as observable
    becomes True after acquisition all following payments are nullified"""

    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cf = self.contract.generate_cashflows(acquisition_idx, model)
        ko_idx = acquisition_idx.next_after(self.observable.simulate(model))
        return cf.zero_after(ko_idx)


class ResolvableContract(Contract):
    @abstractmethod
    def resolve(self) -> Contract:
        pass

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        return self.resolve().generate_cashflows(acquisition_idx, model)


@dataclass
class ZeroCouponBond(ResolvableContract):
    maturity: np.datetime64
    notional: float
    currency: str

    def resolve(self) -> Contract:
        return When(
            At(self.maturity), Scale(KonstFloat(self.notional), One(self.currency))
        )


@dataclass
class EuropeanOption(ResolvableContract):
    maturity: np.datetime64
    contract: Contract

    def resolve(self) -> Contract:
        return When(At(self.maturity), Or(self.contract, Zero()))


class BrownianMotion:
    """Brownian Motion (Wiener Process) with optional drift."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) -> np.array:
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        W = rnd.normal(size=(n, t.size))
        W_drift = W * np.sqrt(dt) * self.sigma + self.mu * dt
        return np.cumsum(W_drift, axis=1)


class GeometricBrownianMotion:
    """Geometric Brownian Motion.(with optional drift)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) -> np.array:
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + (self.mu - self.sigma ** 2 / 2) * t)


parser = ArgumentParser(description=__doc__)
parser.add_argument(
    "--version", help="Print version number", default=False, action="store_true"
)

subparsers = parser.add_subparsers(
    title="subcommands", dest="subcommand", help="Available subcommands"
)

mycmd_parser = subparsers.add_parser("mycmd", help="An example subcommand")
mycmd_parser.add_argument("-n", "--number", help="some number", default=17, type=int)


def _mycmd(args: Namespace) -> None:
    print("Running mycmd subcommand with n={}...".format(args.number))
    print("mycmd completed")


mycmd_parser.set_defaults(func=_mycmd)


def main() -> None:
    args = parser.parse_args()
    if args.version:
        print("""Monte Carlo Contracts """ + __version__)
    elif hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
