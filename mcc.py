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

T = TypeVar("T")


class SimulatedCashflows:
    dtype = np.dtype(
        [("date", "datetime64[D]"), ("currency", np.string_, 3), ("value", np.float64)]
    )
    cashflows: np.array

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
