#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.9.0"""


from typing import Union, List, Iterable

from .cashflows import (
    SimpleCashflows,
    IndexedCashflows,
    SimulatedCashflows,
    DateIndex,
    CURRENCY_LETTER_COUNT,
    NULL_CURRENCY,
)

from .model import Model, ModelRequirements, TermStructuresModel

from .observables import (
    ObservableBool,
    ObservableFloat,
    AndObservable,
    At,
    FixedAfter,
    FX,
    GreaterOrEqualThan,
    GreaterThan,
    KonstFloat,
    LinearRate,
    Maximum,
    Minimum,
    Minus,
    Not,
    OrObservable,
    Power,
    Product,
    Quotient,
    RunningMax,
    RunningMin,
    Stock,
    Sum,
)

from .contracts import (
    And,
    Anytime,
    Cond,
    Contract,
    Delay,
    Give,
    One,
    Or,
    ResolvableContract,
    Scale,
    Until,
    When,
    Zero,
)

from .reusable_contracts import ZeroCouponBond, EuropeanOption

from .cashflow_generation import (
    generate_cashflows,
    generate_simple_cashflows,
    generate_simple_cashflows_in_currency,
    generate_simple_cashflows_in_numeraire_currency,
    evaluate,
)


__all__ = [
    "AndObservable",
    "And",
    "Anytime",
    "At",
    "Cond",
    "Contract",
    "DateIndex",
    "Delay",
    "EuropeanOption",
    "FX",
    "FixedAfter",
    "Give",
    "GreaterOrEqualThan",
    "GreaterThan",
    "IndexedCashflows",
    "Iterable",
    "KonstFloat",
    "LinearRate",
    "List",
    "Maximum",
    "Minimum",
    "Minus",
    "Model",
    "ModelRequirements",
    "Not",
    "ObservableBool",
    "ObservableFloat",
    "One",
    "Or",
    "OrObservable",
    "Power",
    "Product",
    "Quotient",
    "ResolvableContract",
    "RunningMax",
    "RunningMin",
    "Scale",
    "SimpleCashflows",
    "SimulatedCashflows",
    "Stock",
    "Sum",
    "TermStructuresModel",
    "Union",
    "Until",
    "When",
    "Zero",
    "ZeroCouponBond",
    "CURRENCY_LETTER_COUNT",
    "NULL_CURRENCY",
    "evaluate",
    "generate_cashflows",
    "generate_simple_cashflows",
    "generate_simple_cashflows_in_currency",
    "generate_simple_cashflows_in_numeraire_currency",
]

from . import contracts


def _contracts_table(first_col_width: int = 9, second_col_width: int = 80) -> str:
    """Build a Markdown table for contracts"""

    def _lines() -> Iterable[str]:
        yield "| {contract:{left}} | {description:{right}} |".format(
            contract="Contract",
            left=first_col_width,
            description="Description",
            right=second_col_width,
        )
        yield f"|-{first_col_width * '-'} |-{second_col_width * '-'}-|"
        for _, obj in contracts.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, contracts.Contract):
                yield (
                    "| {name:{first_col_width}} ".format(
                        name=obj.__name__, first_col_width=first_col_width
                    )
                    + "| {description:{second_col_width}} |".format(
                        description=" ".join((obj.__doc__ or "").split())[
                            :second_col_width
                        ],
                        second_col_width=second_col_width,
                    )
                )

    return "\n".join(_lines())


__doc__ += "\n\n" + _contracts_table()
