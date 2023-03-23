#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.8.0"""


from typing import (
    Union,
    List,
)

from .cashflows import (
    SimpleCashflows,
    IndexedCashflows,
    SimulatedCashflows,
    DateIndex,
    _ccy_letters,
    _null_ccy,
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


def generate_cashflows(model: Model, contract: Contract) -> SimulatedCashflows:
    return contract.generate_cashflows(model.eval_date_index, model).apply_index()


def generate_simple_cashflows(model: Model, contract: Contract) -> SimpleCashflows:
    return generate_cashflows(model, contract).to_simple_cashflows()


def generate_simple_cashflows_in_currency(
    model: Model, contract: Contract, currency: str
) -> SimpleCashflows:
    return (
        model.in_currency(
            contract.generate_cashflows(model.eval_date_index, model), currency
        )
        .apply_index()
        .to_simple_cashflows()
    )


def generate_simple_cashflows_in_numeraire_currency(
    model: Model, contract: Contract
) -> SimpleCashflows:
    return generate_simple_cashflows_in_currency(
        model, contract, model.numeraire_currency
    )


def evaluate(model: Model, contract: Contract) -> float:
    cf = contract.generate_cashflows(model.eval_date_index, model)
    return float(model.discount(cf).sum(axis=1).mean(axis=0))


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
    "_ccy_letters",
    "_null_ccy",
    "evaluateAnd",
    "generate_cashflows",
    "generate_simple_cashflows",
    "generate_simple_cashflows_in_currency",
    "generate_simple_cashflows_in_numeraire_currency",
]
