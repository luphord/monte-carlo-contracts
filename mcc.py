#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.1.0"""


from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from typing import Final, Union, Mapping, Tuple
from numbers import Number
import numpy as np
from dataclasses import dataclass


_ccy_letters = 3
_null_ccy = "NNN"
ArrayLike = Union[np.array, Number]


class SimulatedCashflows:
    dtype: Final = np.dtype([("date", "datetime64[D]"), ("value", np.float64)])
    cashflows: Final[np.array]
    currencies: Final[np.array]
    nsim: Final[int]
    ncashflows: Final[int]

    def __init__(self, cashflows: np.array, currencies: np.array):
        assert (
            cashflows.dtype == self.dtype
        ), f"Got cashflow array with dtype {cashflows.dtype}, expecting {self.dtype}"
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        assert currencies.dtype == (np.string_, _ccy_letters)
        assert currencies.shape == (cashflows.shape[1],)
        self.cashflows = cashflows
        self.currencies = currencies
        self.nsim = cashflows.shape[0]
        self.ncashflows = cashflows.shape[1]


class IndexedCashflows:
    dtype: Final = np.dtype([("index", np.int), ("value", np.float64)])
    cashflows: Final[np.array]
    currencies: Final[np.array]
    dategrid: Final[np.array]
    nsim: Final[int]
    ncashflows: Final[int]

    def __init__(self, cashflows: np.array, currencies: np.array, dategrid: np.array):
        assert (
            cashflows.dtype == self.dtype
        ), f"Got cashflow array with dtype {cashflows.dtype}, expecting {self.dtype}"
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        assert currencies.dtype == (np.string_, _ccy_letters)
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

    def apply_index(self) -> SimulatedCashflows:
        dategrid_rep = np.reshape(self.dategrid, (1, self.dategrid.size))
        dategrid_rep = np.repeat(dategrid_rep, self.nsim, axis=0)
        assert dategrid_rep.shape[0] == self.nsim
        datecfs = np.zeros(self.cashflows.shape, dtype=SimulatedCashflows.dtype)
        for i, cf in enumerate(self.cashflows.T):
            datecfs["date"][:, i] = dategrid_rep[np.arange(self.nsim), cf["index"]]
            datecfs["value"][:, i] = cf["value"]
        return SimulatedCashflows(np.array(datecfs), self.currencies)


class DateIndex:
    index: Final[np.array]
    nsim: Final[int]

    def __init__(self, index: np.array):
        assert index.dtype == np.int
        assert index.ndim == 1
        self.index = index
        self.nsim = index.size

    def index_column(self, observable: np.array) -> np.array:
        assert observable.ndim == 2
        assert observable.shape[0] == self.nsim
        assert (self.index < observable.shape[1]).all()
        obs = observable[np.arange(self.nsim), self.index.clip(0)]
        obs[self.index < 0] = 0
        assert obs.shape == (self.nsim,)
        return obs


class Model:
    dategrid: Final[np.array]
    simulated_stocks: Final[Mapping[str, np.array]]
    numeraire: Final[np.array]
    numeraire_currency: Final[str]
    ndates: Final[int]
    nsim: Final[int]
    shape: Final[Tuple[int, int]]

    def __init__(
        self,
        dategrid: np.array,
        simulated_stocks: Mapping[str, np.array],
        numeraire: np.array,
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

    def validate_date_index(self, date_index: DateIndex) -> None:
        assert date_index.index.size == self.nsim
        assert (date_index.index < self.ndates).all()

    def generate_cashflows(self, contract: "Contract") -> SimulatedCashflows:
        return contract.generate_cashflows(self.eval_date_index, self).apply_index()


class ObservableFloat(ABC):
    @abstractmethod
    def simulate(self, model: Model) -> np.array:
        pass


@dataclass
class KonstFloat(ObservableFloat):
    constant: Number

    def simulate(self, model: Model) -> np.array:
        return self.constant * np.ones(model.shape, dtype=np.float)


class ObservableBool(ABC):
    @abstractmethod
    def simulate(self, model: Model) -> np.array:
        pass


class Contract(ABC):
    @abstractmethod
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        pass


@dataclass
class Zero(Contract):
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        model.validate_date_index(acquisition_idx)
        cf = np.zeros((model.nsim, 1), dtype=IndexedCashflows.dtype)
        cf["index"][:, 0] = acquisition_idx.index
        ccys = np.array([_null_ccy], dtype=(np.string_, _ccy_letters))
        return IndexedCashflows(cf, ccys, model.dategrid)


@dataclass
class One(Contract):
    currency: str

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        model.validate_date_index(acquisition_idx)
        cf = np.ones((model.nsim, 1), dtype=IndexedCashflows.dtype)
        cf["index"][:, 0] = acquisition_idx.index
        ccys = np.array([self.currency], dtype=(np.string_, _ccy_letters))
        return IndexedCashflows(cf, ccys, model.dategrid)


@dataclass
class Give(Contract):
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        return -self.contract.generate_cashflows(acquisition_idx, model)


@dataclass
class And(Contract):
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
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        raise NotImplementedError()


@dataclass
class Cond(Contract):
    observable: ObservableBool
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        raise NotImplementedError()


@dataclass
class Scale(Contract):
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
    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        raise NotImplementedError()


@dataclass
class Anytime(Contract):
    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        raise NotImplementedError()


@dataclass
class Until(Contract):
    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        raise NotImplementedError()


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
