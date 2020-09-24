#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.1.0"""


from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
from dataclasses import dataclass

T = TypeVar("T")


class SimulatedCashflows:
    dtype = np.dtype(
        [("date", "datetime64[D]"), ("currency", np.string_, 3), ("value", np.float64)]
    )
    cashflows: np.array

    def __init__(self, cashflows: np.array):
        assert cashflows.dtype == self.dtype
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        self.cashflows = cashflows

    @property
    def nsim(self) -> int:
        return self.cashflows.shape[0]

    @property
    def ncashflows(self) -> int:
        return self.cashflows.shape[1]


class DateIndex:
    pass


class Model:
    pass


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
