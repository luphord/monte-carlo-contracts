#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.1.0"""


from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from typing import Final, Mapping, Tuple
import numpy as np
from dataclasses import dataclass


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
        assert currencies.dtype == (np.string_, 3)
        assert currencies.shape == (cashflows.shape[1],)
        self.cashflows = cashflows
        self.currencies = currencies
        self.nsim = cashflows.shape[0]
        self.ncashflows = cashflows.shape[1]


class IndexedCashflows:
    dtype: Final = np.dtype([("index", np.int), ("value", np.float64)])
    cashflows: Final[np.array]
    currencies: Final[np.array]
    nsim: Final[int]
    ncashflows: Final[int]

    def __init__(self, cashflows: np.array, currencies: np.array):
        assert (
            cashflows.dtype == self.dtype
        ), f"Got cashflow array with dtype {cashflows.dtype}, expecting {self.dtype}"
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        assert currencies.dtype == (np.string_, 3)
        assert currencies.shape == (cashflows.shape[1],)
        self.cashflows = cashflows
        self.currencies = currencies
        self.nsim = cashflows.shape[0]
        self.ncashflows = cashflows.shape[1]

    def __add__(self, other: "IndexedCashflows") -> "IndexedCashflows":
        assert (
            self.nsim == other.nsim
        ), f"Cannot add cashflows with {self.nsim} and {other.nsim} simulations"
        return IndexedCashflows(
            np.concatenate((self.cashflows, other.cashflows), axis=1),
            np.concatenate((self.currencies, other.currencies)),
        )

    def apply_index(self, dategrid: np.array) -> SimulatedCashflows:
        assert dategrid.dtype == "datetime64[D]"
        dategrid = np.reshape(dategrid, (1, dategrid.size))
        dategrid = np.repeat(dategrid, self.nsim, axis=0)
        assert dategrid.shape[0] == self.nsim
        datecfs = np.zeros(self.cashflows.shape, dtype=SimulatedCashflows.dtype)
        for i, cf in enumerate(self.cashflows.T):
            datecfs["date"][:, i] = dategrid[np.arange(self.nsim), cf["index"]]
            datecfs["value"][:, i] = cf["value"]
        return SimulatedCashflows(np.array(datecfs), self.currencies)


class DateIndex:
    pass


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
        self.dategrid = np.reshape(dategrid, (1, self.ndates))
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


class ObservableFloat(ABC):
    pass


class ObservableBool(ABC):
    pass


class Contract(ABC):
    @abstractmethod
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        pass


@dataclass
class Zero(Contract):
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class One(Contract):
    currency: str

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class Give(Contract):
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class And(Contract):
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class Or(Contract):
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class Cond(Contract):
    observable: ObservableBool
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class Scale(Contract):
    observable: ObservableFloat
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class When(Contract):
    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class Anytime(Contract):
    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
        raise NotImplementedError()


@dataclass
class Until(Contract):
    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> SimulatedCashflows:
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
