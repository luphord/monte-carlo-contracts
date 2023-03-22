#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.8.0"""


from abc import ABC, abstractmethod
from typing import (
    Final,
    Optional,
    Union,
    List,
    Callable,
)
import numpy as np
from dataclasses import dataclass

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
    "_ccy_letters",
    "_null_ccy",
    "evaluateAnd",
    "generate_cashflows",
    "generate_simple_cashflows",
    "generate_simple_cashflows_in_currency",
    "generate_simple_cashflows_in_numeraire_currency",
]


@dataclass
class ZeroCouponBond(ResolvableContract):
    maturity: np.datetime64
    notional: float
    currency: str

    def resolve(self) -> Contract:
        return When(At(self.maturity), self.notional * One(self.currency))


@dataclass
class EuropeanOption(ResolvableContract):
    maturity: np.datetime64
    contract: Contract

    def resolve(self) -> Contract:
        return When(At(self.maturity), self.contract | Zero())


class StochasticProcess(ABC):
    """Base class for stochastic processes."""

    @abstractmethod
    def simulate(self, t: np.ndarray, n: int, rnd: np.random.RandomState) -> np.ndarray:
        pass

    @abstractmethod
    def expected(self, t: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def stddev(self, t: np.ndarray) -> np.ndarray:
        pass

    def moment_match(self, t: np.ndarray, paths: np.ndarray) -> np.ndarray:
        assert t.ndim == 1, "One dimensional time vector required"
        ndates = t.size
        assert ndates > 0, "At least one time point is required"
        assert paths.ndim == 2
        assert paths.shape[1] == ndates
        target_mean = self.expected(t)
        target_stddev = self.stddev(t)
        actual_mean = paths.mean(axis=0)
        actual_stddev = paths.std(axis=0)
        assert isinstance(actual_stddev, np.ndarray)
        assert (
            target_stddev[actual_stddev == 0] == 0
        ).all(), "Cannot scale actual stddev of zero to any other value than zero"
        actual_stddev[actual_stddev == 0] = 1  # prevent division by zero
        return (paths - actual_mean) / actual_stddev * target_stddev + target_mean

    def simulate_with_moment_matching(
        self, t: np.ndarray, n: int, rnd: np.random.RandomState
    ) -> np.ndarray:
        return self.moment_match(t, self.simulate(t, n, rnd))


class BrownianMotion(StochasticProcess):
    """Brownian Motion (Wiener Process) with optional drift."""

    def __init__(
        self,
        mu_t: Callable[[np.ndarray], np.ndarray] = np.zeros_like,
        sigma: float = 1.0,
    ):
        self.mu_t = mu_t
        self.sigma = sigma

    def simulate(self, t: np.ndarray, n: int, rnd: np.random.RandomState) -> np.ndarray:
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        W = rnd.normal(size=(n, t.size))
        W_drift = W * np.sqrt(dt) * self.sigma + self.mu_t(t)
        return np.cumsum(W_drift, axis=1)

    def expected(self, t: np.ndarray) -> np.ndarray:
        return self.mu_t(t)

    def stddev(self, t: np.ndarray) -> np.ndarray:
        return np.sqrt(self.sigma**2 * t)


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion.(with optional drift)."""

    def __init__(
        self,
        mu_t: Callable[[np.ndarray], np.ndarray] = np.zeros_like,
        sigma: float = 1.0,
    ):
        self.mu_t = mu_t
        self.sigma = sigma

    def simulate(self, t: np.ndarray, n: int, rnd: np.random.RandomState) -> np.ndarray:
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + self.mu_t(t) - self.sigma**2 / 2 * t)

    def expected(self, t: np.ndarray) -> np.ndarray:
        return np.exp(self.mu_t(t))

    def stddev(self, t: np.ndarray) -> np.ndarray:
        return np.sqrt(np.exp(2 * self.mu_t(t)) * (np.exp(self.sigma**2 * t) - 1))


def _get_year_fractions(dategrid: np.ndarray) -> np.ndarray:
    assert dategrid.dtype == "datetime64[D]"
    return (dategrid - dategrid[0]).astype(np.float64) / 365


def simulate_equity_black_scholes_model(
    stock: str,
    currency: str,
    S0: float,
    dategrid: np.ndarray,
    sigma: float,
    r: float,
    n: int,
    rnd: np.random.RandomState,
    use_moment_matching: bool = False,
) -> Model:
    assert dategrid.dtype == "datetime64[D]"
    ndates = dategrid.size
    yearfractions = _get_year_fractions(dategrid)
    gbm = GeometricBrownianMotion(lambda t: r * t, sigma)
    s = S0 * (
        gbm.simulate_with_moment_matching(yearfractions, n, rnd)
        if use_moment_matching
        else gbm.simulate(yearfractions, n, rnd)
    )
    numeraire = np.repeat(np.exp(r * yearfractions).reshape((1, ndates)), n, axis=0)
    return Model(dategrid, {}, {}, {stock: s}, numeraire, currency)


class HoLeeModel(TermStructuresModel):

    h: Final[float] = np.sqrt(np.finfo(float).eps)

    dategrid: Final[np.ndarray]
    yearfractions: Final[np.ndarray]
    shortrates: Final[np.ndarray]
    discount_curve: Optional[
        Callable[[np.ndarray], np.ndarray]
    ]  # mypy requires hack, see https://ogy.de/mypy-callable-members
    sigma: Final[float]
    _mu_t: Final[np.ndarray]

    def __init__(
        self,
        dategrid: np.ndarray,
        discount_curve: Callable[[np.ndarray], np.ndarray],
        sigma: float,
        n: int,
        rnd: np.random.RandomState,
        use_moment_matching: bool = False,
    ):
        assert dategrid.dtype == "datetime64[D]"
        self.dategrid = dategrid
        self.yearfractions = _get_year_fractions(dategrid)
        self.discount_curve = discount_curve
        self.sigma = sigma
        self._mu_t = self._integral_theta()
        self.shortrates = self._simulate(n, rnd, use_moment_matching)
        assert self.shortrates.shape == (n, dategrid.size)

    def _integral_theta(self) -> np.ndarray:
        assert self.discount_curve
        dlogBond = (
            np.log(self.discount_curve(self.yearfractions + self.h))
            - np.log(self.discount_curve(self.yearfractions))
        ) / self.h
        return self.sigma**2 * self.yearfractions**2 / 2 - dlogBond

    def _integral_mu_t(self, T: np.ndarray) -> np.ndarray:
        """Integral over mu_t (i.e. double integral over theta)
        from model grid to T"""
        assert T >= self.yearfractions[0]
        assert T <= self.yearfractions[-1]
        t = np.linspace(self.yearfractions[0], self.yearfractions[-1], num=100)
        int_mu = np.cumsum(self.mu_t(t) * np.diff(t, prepend=self.yearfractions[0] - 1))
        int_mu_fn: Callable[[np.ndarray], np.ndarray] = lambda tau: np.interp(
            tau, t, int_mu
        )
        return int_mu_fn(T) - int_mu_fn(self.yearfractions)

    def _simulate(
        self, n: int, rnd: np.random.RandomState, use_moment_matching: bool
    ) -> np.ndarray:
        bm = BrownianMotion(mu_t=self.mu_t, sigma=self.sigma)
        return (
            bm.simulate_with_moment_matching(self.yearfractions, n, rnd)
            if use_moment_matching
            else bm.simulate(self.yearfractions, n, rnd)
        )

    def mu_t(self, yearfractions: np.ndarray) -> np.ndarray:
        return np.interp(yearfractions, self.yearfractions, self._mu_t)

    def linear_rate(self, frequency: str) -> np.ndarray:
        raise NotImplementedError()

    def discount_factors(self) -> np.ndarray:
        """Stochastic discount factors implied by the model"""
        dt = np.diff(self.yearfractions, prepend=0).reshape(1, self.dategrid.size)
        r_before_0 = np.zeros((self.shortrates.shape[0], 1))
        r = np.hstack((r_before_0, self.shortrates[:, :-1]))
        return np.exp(-np.cumsum(r * dt, axis=1))

    def bond_prices(self, T: np.ndarray) -> np.ndarray:
        """Stochastic T-bond prices implied by the model,
        quoted for all times even after T"""
        t = self.yearfractions
        lb = (
            -(T - t) * self.shortrates
            - self._integral_mu_t(T)
            + self.sigma**2 / 6 * (T - t) ** 3
        )
        return np.exp(lb)

    def pathwise_terminal_bond(self) -> np.ndarray:
        """Pathwise value of the zero coupon bond maturing at model horizon"""
        dt_reversed = np.diff(self.yearfractions, append=self.yearfractions[-1])[
            ::-1
        ].reshape(1, self.dategrid.size)
        r_reversed = self.shortrates[:, ::-1]
        return np.exp(-np.cumsum(r_reversed * dt_reversed, axis=1))[:, ::-1]
