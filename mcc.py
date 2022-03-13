#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composable financial contracts with Monte Carlo valuation
"""

__author__ = """luphord"""
__email__ = """luphord@protonmail.com"""
__version__ = """0.7.0"""


from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from itertools import groupby
from typing import Final, Optional, Union, Mapping, Tuple, Set, Iterable, Callable
from numbers import Real
import numpy as np
import pandas as pd
from dataclasses import dataclass


_ccy_letters = 3
_null_ccy = "NNN"
ArrayLike = Union[np.ndarray, float]


class SimpleCashflows(pd.DataFrame):
    """Simple data structure to represent simulated cashflows.
    Designed for external usage where an accumulated, simplified
    view of the cashflow structure is sufficient.
    Essentially a mapping from date and currency (row index)
    to cashflow value per path (column index).
    For convenience, this class is derived from `pandas.DataFrame`
    to simplify further processing (such as plotting).
    Can be created from `SimulatedCashflows`, but the reverse
    direction is not possible.
    """

    @property
    def _constructor(self) -> Callable:
        return SimpleCashflows

    @classmethod
    def from_arrays(
        cls, cashflows: np.ndarray, currencies: np.ndarray, dates: np.ndarray
    ) -> "SimpleCashflows":
        assert cashflows.dtype == np.float64
        assert cashflows.ndim == 2, f"Array must have ndim 2, got {cashflows.ndim}"
        assert currencies.dtype == (np.unicode_, _ccy_letters)
        assert currencies.shape == (cashflows.shape[1],)
        assert dates.dtype == "datetime64[D]"
        assert dates.shape == (cashflows.shape[1],)
        index = pd.MultiIndex.from_arrays([dates, currencies])
        return cls(pd.DataFrame(cashflows.T, index=index))


class SimulatedCashflows:
    """Complex data structure to represent simulated cashflows.
    Designed for external usage where a detailed view
    of the cashflow structure is desired.
    Contains full information about the cashflow structure
    indexed by datetime64[D] dates. Does not contain the integer
    index values that correspond to these dates within the model.
    Can be turned into `SimpleCashflows` via `to_simple_cashflows`.
    Can be created from `IndexedCashflows`, but the reverse
    direction is not possible.
    """

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

    def _split_by_date(self) -> Iterable[Tuple[np.ndarray, str, np.datetime64]]:
        for i, cf in enumerate(self.cashflows.T):
            if self.currencies[i] != _null_ccy:
                for dt in np.unique(cf["date"]):
                    if not np.isnat(dt):
                        cf_dt = cf.copy()
                        cf_dt[cf_dt["date"] != dt] = 0
                        yield cf_dt, self.currencies[i], dt

    def _group_cashflows(self) -> Iterable[Tuple[np.ndarray, str, np.datetime64]]:
        def grpkey(
            entry: Tuple[np.ndarray, str, np.datetime64]
        ) -> Tuple[np.datetime64, str]:
            cf, ccy, dt = entry
            return (dt, ccy)

        for key, entries in groupby(sorted(self._split_by_date(), key=grpkey), grpkey):
            dt, ccy = key
            sum_cf = sum(cf["value"] for cf, _, _ in entries)
            assert isinstance(
                sum_cf, np.ndarray
            )  # appease mypy that sum_cf is actually an array, not 0
            yield sum_cf, ccy, dt

    def to_simple_cashflows(self) -> SimpleCashflows:
        grouped_cf = list(self._group_cashflows())
        assert grouped_cf, "Got no cashflows to convert"
        numcf = len(grouped_cf)
        cashflows: np.ndarray = np.ndarray((self.nsim, numcf), dtype=np.float64)
        currencies: np.ndarray = np.ndarray((numcf,), dtype=(np.unicode_, _ccy_letters))
        dates: np.ndarray = np.ndarray((numcf,), dtype="datetime64[D]")
        for i, (cf, ccy, dt) in enumerate(grouped_cf):
            cashflows[:, i] = cf
            currencies[i] = ccy
            dates[i] = dt
        return SimpleCashflows.from_arrays(cashflows, currencies, dates)


class IndexedCashflows:
    """Complex data structure to represent simulated cashflows
    bound to a specific model.
    Designed for internal usage within composable contracts.
    Contains full information about the cashflow structure
    indexed by integers corresponding to dates within the associated model.
    Can be turned into `SimulatedCashflows` via `apply_index`.
    """

    dtype: Final = np.dtype([("index", np.int64), ("value", np.float64)])
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
    """Integer index for time points (dates) within a model.
    Index value may be different for each path.
    Mathematically, this could be interpreted as the simulation
    of a *stopping time* stochastic process.
    Index value -1 indicates "never" (as in "this contract is never acquired").
    """

    index: Final[np.ndarray]
    nsim: Final[int]

    def __init__(self, index: np.ndarray):
        assert index.dtype == np.int64
        assert index.ndim == 1
        self.index = index
        self.nsim = index.size

    def index_column(self, observable: np.ndarray) -> np.ndarray:
        """Apply this DateIndes to observable.
        Essentially the same as observable[:, self.index],
        but with proper handling for negative index values."""
        assert observable.ndim == 2
        assert observable.shape[0] == self.nsim
        assert (self.index < observable.shape[1]).all()
        obs = observable[np.arange(self.nsim), self.index.clip(0)]
        obs[self.index < 0] = 0
        assert obs.shape == (self.nsim,)
        return obs

    def next_after(self, obs: np.ndarray) -> "DateIndex":
        """Create a new DateIndex where obs is True
        for the first time after (including) this DateIndex."""
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

    def after_mask(self, ndates: int) -> np.ndarray:
        """Return a boolean array of size (nsim, ndates)
        where values after (including) this index are True."""
        mask = np.repeat(
            np.reshape(np.arange(ndates), (1, ndates)),
            self.nsim,
            axis=0,
        ) >= np.reshape(self.index, (self.nsim, 1))
        mask[self.index < 0, :] = False
        return mask

    def before_mask(self, ndates: int) -> np.ndarray:
        """Return a boolean array of size (nsim, ndates)
        where values before (excluding) this index are True."""
        return ~self.after_mask(ndates)


class TermStructuresModel(ABC):
    @abstractmethod
    def linear_rate(self, frequency: str) -> np.ndarray:
        pass


class Model:
    dategrid: Final[np.ndarray]
    simulated_rates: Final[Mapping[str, TermStructuresModel]]
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
        simulated_rates: Mapping[str, TermStructuresModel],
        simulated_fx: Mapping[Tuple[str, str], np.ndarray],
        simulated_stocks: Mapping[str, np.ndarray],
        numeraire: np.ndarray,
        numeraire_currency: str,
    ):
        assert dategrid.dtype == "datetime64[D]"
        self.ndates = dategrid.size
        assert self.ndates > 0
        self.dategrid = np.reshape(dategrid, (self.ndates, 1))
        assert numeraire.dtype == np.float64
        assert numeraire.ndim == 2
        self.numeraire = numeraire
        self.nsim = numeraire.shape[0]
        self.shape = (self.nsim, self.ndates)
        assert numeraire.shape == self.shape
        self.numeraire_currency = numeraire_currency
        self.simulated_rates = simulated_rates
        for fxkey, val in simulated_fx.items():
            assert (
                val.dtype == np.float64
            ), f"FX spot '{fxkey}' is of dtype {val.dtype}, expecting float"
            assert (
                val.shape == self.shape
            ), f"FX Spot '{fxkey}' has shape {val.shape}, expecting {self.shape}"
        self.simulated_fx = simulated_fx
        for key, val in simulated_stocks.items():
            assert (
                val.dtype == np.float64
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
        return DateIndex(np.zeros((self.nsim,), dtype=np.int64))

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

    def generate_simple_cashflows(self, contract: "Contract") -> SimpleCashflows:
        return self.generate_cashflows(contract).to_simple_cashflows()

    def generate_simple_cashflows_in_currency(
        self, contract: "Contract", currency: str
    ) -> SimpleCashflows:
        return (
            self.in_currency(
                contract.generate_cashflows(self.eval_date_index, self), currency
            )
            .apply_index()
            .to_simple_cashflows()
        )

    def generate_simple_cashflows_in_numeraire_currency(
        self, contract: "Contract"
    ) -> SimpleCashflows:
        return self.generate_simple_cashflows_in_currency(
            contract, self.numeraire_currency
        )

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

    def discount(self, cashflows: IndexedCashflows) -> np.ndarray:
        cfnum = self.in_numeraire_currency(cashflows)
        discounted = np.zeros(cfnum.cashflows.shape, dtype=np.float64)
        num_0 = self.numeraire[:, 0]
        for i, cf in enumerate(cfnum.cashflows.T):
            num_t = DateIndex(cf["index"]).index_column(self.numeraire)
            num_t[cf["index"] < 0] = 1  # cashflow happens never anyway
            discounted[:, i] = cf["value"] / num_t * num_0
        return discounted

    def evaluate(
        self, cashflows_or_contract: Union[IndexedCashflows, "Contract"]
    ) -> float:
        cf: IndexedCashflows
        if isinstance(cashflows_or_contract, IndexedCashflows):
            cf = cashflows_or_contract
        elif isinstance(cashflows_or_contract, Contract):
            cf = cashflows_or_contract.generate_cashflows(self.eval_date_index, self)
        else:
            raise TypeError(
                "Expecting IndexedCashflows or Contract, "
                f"got {type(cashflows_or_contract)}"
            )
        return float(self.discount(cf).sum(axis=1).mean(axis=0))


class ObservableFloat(ABC):
    """Abstract base class for all observables of underlying type float,
    essentially a real-valued stochastic process"""

    @abstractmethod
    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        pass

    def __add__(self, other: Union["ObservableFloat", float]) -> "ObservableFloat":
        if isinstance(other, ObservableFloat):
            return Sum(self, other)
        elif isinstance(other, Real):
            return Sum(self, KonstFloat(other))
        else:
            return NotImplemented

    def __radd__(self, other: float) -> "ObservableFloat":
        return Sum(KonstFloat(other), self)

    def __neg__(self) -> "ObservableFloat":
        return Minus(self)

    def __sub__(self, other: Union["ObservableFloat", float]) -> "ObservableFloat":
        if isinstance(other, ObservableFloat):
            return Sum(self, -other)
        elif isinstance(other, Real):
            return Sum(self, -KonstFloat(other))
        else:
            return NotImplemented

    def __rsub__(self, other: float) -> "ObservableFloat":
        return Sum(KonstFloat(other), -self)

    def __mul__(self, other: Union["ObservableFloat", float]) -> "ObservableFloat":
        if isinstance(other, ObservableFloat):
            return Product(self, other)
        elif isinstance(other, Real):
            return Product(self, KonstFloat(other))
        else:
            return NotImplemented

    def __rmul__(self, other: float) -> "ObservableFloat":
        return Product(KonstFloat(other), self)

    def __truediv__(self, other: Union["ObservableFloat", float]) -> "ObservableFloat":
        if isinstance(other, ObservableFloat):
            return Quotient(self, other)
        elif isinstance(other, Real):
            return Quotient(self, KonstFloat(other))
        else:
            return NotImplemented

    def __rtruediv__(self, other: float) -> "ObservableFloat":
        return Quotient(KonstFloat(other), self)

    def __pow__(self, other: Union["ObservableFloat", float]) -> "ObservableFloat":
        if isinstance(other, ObservableFloat):
            return Power(self, other)
        elif isinstance(other, Real):
            return Power(self, KonstFloat(other))
        else:
            return NotImplemented

    def __rpow__(self, other: float) -> "ObservableFloat":
        return Power(KonstFloat(other), self)

    def __ge__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        if isinstance(other, ObservableFloat):
            return GreaterOrEqualThan(self, other)
        elif isinstance(other, Real):
            return GreaterOrEqualThan(self, KonstFloat(other))
        else:
            return NotImplemented

    def __gt__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        if isinstance(other, ObservableFloat):
            return GreaterThan(self, other)
        elif isinstance(other, Real):
            return GreaterThan(self, KonstFloat(other))
        else:
            return NotImplemented

    def __le__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        return ~(self > other)

    def __lt__(self, other: Union["ObservableFloat", float]) -> "ObservableBool":
        return ~(self >= other)


@dataclass
class Sum(ObservableFloat):
    """Equal to the sum of two observables"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) + self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"({self.observable1}) + ({self.observable2})"


@dataclass
class Minus(ObservableFloat):
    """Negative value of observable"""

    observable: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return -self.observable.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"-{self.observable}"


@dataclass
class Product(ObservableFloat):
    """Equal to the product (multiplication) of two observables"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) * self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"({self.observable1}) * ({self.observable2})"


@dataclass
class Quotient(ObservableFloat):
    """Equal to the quotient (division) of two observables"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) / self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"({self.observable1}) / ({self.observable2})"


@dataclass
class Power(ObservableFloat):
    """Equal to observable1 to the power of observable2"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) ** self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"({self.observable1}) ** ({self.observable2})"


@dataclass
class Maximum(ObservableFloat):
    """Equal to the maximum of two observables"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return np.maximum(
            self.observable1.simulate(first_observation_idx, model),
            self.observable2.simulate(first_observation_idx, model),
        )

    def __str__(self) -> str:
        return f"Maximum({self.observable1}, {self.observable2})"


@dataclass
class Minimum(ObservableFloat):
    """Equal to the minimum of two observables"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return np.minimum(
            self.observable1.simulate(first_observation_idx, model),
            self.observable2.simulate(first_observation_idx, model),
        )

    def __str__(self) -> str:
        return f"Minimum({self.observable1}, {self.observable2})"


def _simulate_accumulate(
    accfn: np.ufunc,
    observable: ObservableFloat,
    first_observation_idx: DateIndex,
    model: Model,
) -> np.ndarray:
    underlying = observable.simulate(first_observation_idx, model).copy()
    running_accumulate = underlying.copy()
    mask = first_observation_idx.before_mask(model.ndates)
    running_accumulate[mask] = np.nan
    running_accumulate = accfn.accumulate(running_accumulate, axis=1)
    running_accumulate[mask] = underlying[mask]
    return running_accumulate


@dataclass
class RunningMax(ObservableFloat):
    """Running maximum of observable over time, seen from first_observation_idx."""

    observable: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return _simulate_accumulate(
            np.fmax, self.observable, first_observation_idx, model
        )

    def __str__(self) -> str:
        return f"RunningMax({self.observable})"


@dataclass
class RunningMin(ObservableFloat):
    """Running minimum of observable over time, seen from first_observation_idx."""

    observable: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return _simulate_accumulate(
            np.fmin, self.observable, first_observation_idx, model
        )

    def __str__(self) -> str:
        return f"RunningMin({self.observable})"


@dataclass
class FixedAfter(ObservableFloat):
    """Equal to observable, but remains constant as soon as
    fixing_condition becomes true after (including) first_observation_idx."""

    fixing_condition: "ObservableBool"
    observable: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        underlying = self.observable.simulate(first_observation_idx, model).copy()
        fixing_times = self.fixing_condition.simulate(first_observation_idx, model)
        fixing_idx = first_observation_idx.next_after(fixing_times)
        for i in range(1, underlying.shape[1]):
            fixing_mask = (i > fixing_idx.index) & (fixing_idx.index >= 0)
            underlying[fixing_mask, i] = underlying[fixing_mask, i - 1]
        return underlying

    def __str__(self) -> str:
        return f"FixedAfter({self.fixing_condition}, {self.observable})"


@dataclass
class Stock(ObservableFloat):
    """Value of the stock identified by identifier"""

    identifier: str

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return model.simulated_stocks[self.identifier]

    def __str__(self) -> str:
        return f"Stock({self.identifier})"


@dataclass
class FX(ObservableFloat):
    """Value of the currency spot between base_currency and counter_currency,
    i.e. 'one unit counter_currency' / 'one unit of base_currency'"""

    base_currency: str
    counter_currency: str

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return model.get_simulated_fx(self.base_currency, self.counter_currency)

    def __str__(self) -> str:
        return f"FX({self.base_currency}/{self.counter_currency})"


@dataclass
class LinearRate(ObservableFloat):
    """Value of the linear rate (e.g. a LIBOR) with payment frequency
    in currency"""

    currency: str
    frequency: str

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return model.simulated_rates[self.currency].linear_rate(self.frequency)

    def __str__(self) -> str:
        return f"{self.currency}{self.frequency})"


@dataclass
class KonstFloat(ObservableFloat):
    """Always equal to constant"""

    constant: float

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.constant * np.ones(model.shape, dtype=np.float64)

    def __str__(self) -> str:
        return str(self.constant)


class ObservableBool(ABC):
    """Abstract base class for all observables of underlying type bool"""

    @abstractmethod
    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
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

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return ~self.observable.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"~({self.observable})"


@dataclass
class AndObservable(ObservableBool):
    """True if and only if both observables are True"""

    observable1: ObservableBool
    observable2: ObservableBool

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) & self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"({self.observable1}) & ({self.observable2})"


@dataclass
class OrObservable(ObservableBool):
    """True if either or both observable are True"""

    observable1: ObservableBool
    observable2: ObservableBool

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) | self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"({self.observable1}) | ({self.observable2})"


@dataclass
class GreaterOrEqualThan(ObservableBool):
    """True if and only if observable1 is greater or equal than observable2"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) >= self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"{self.observable1} >= {self.observable2}"


@dataclass
class GreaterThan(ObservableBool):
    """True if and only if observable1 is strictly greater than observable2"""

    observable1: ObservableFloat
    observable2: ObservableFloat

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        return self.observable1.simulate(
            first_observation_idx, model
        ) > self.observable2.simulate(first_observation_idx, model)

    def __str__(self) -> str:
        return f"{self.observable1} > {self.observable2}"


@dataclass
class At(ObservableBool):
    """True only at date"""

    date: np.datetime64

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        mask = (model.dategrid == self.date).reshape((1, model.ndates))
        assert mask.any(), f"{self.date} not contained in dategrid"
        # ToDo: Should we assert that self.date is after first_observation_idx?
        return np.repeat(mask, model.nsim, axis=0)

    def __str__(self) -> str:
        return str(self.date)


class Contract(ABC):
    """Abstract base class for all contracts"""

    @abstractmethod
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        pass

    def __add__(self, other: "Contract") -> "Contract":
        return And(self, other)

    def __neg__(self) -> "Contract":
        return Give(self)

    def __sub__(self, other: "Contract") -> "Contract":
        return And(self, Give(other))

    def __mul__(self, observable: Union[ObservableFloat, float]) -> "Contract":
        if isinstance(observable, ObservableFloat):
            return Scale(observable, self)
        elif isinstance(observable, Real):
            return Scale(KonstFloat(observable), self)
        else:
            return NotImplemented

    def __rmul__(self, observable: Union[ObservableFloat, float]) -> "Contract":
        return self.__mul__(observable)

    def __or__(self, other: "Contract") -> "Contract":
        return Or(self, other)


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

    def __str__(self) -> str:
        return "Zero"


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

    def __str__(self) -> str:
        return f"One({self.currency})"


@dataclass
class Give(Contract):
    """Receive all obligations of the underlying contract and pay all rights,
    i.e. invert the underlying contract"""

    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        return -self.contract.generate_cashflows(acquisition_idx, model)

    def __str__(self) -> str:
        return f"Give({self.contract})"


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

    def __str__(self) -> str:
        return f"And({self.contract1}, {self.contract2})"


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

    def __str__(self) -> str:
        return f"Or({self.contract1}, {self.contract2})"


@dataclass
class Cond(Contract):
    """If observable is True at acquisition, obtain contract1, otherwise contract2"""

    observable: ObservableBool
    contract1: Contract
    contract2: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        obs = acquisition_idx.index_column(
            self.observable.simulate(acquisition_idx, model)
        )
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

    def __str__(self) -> str:
        return f"Cond({self.observable}, {self.contract1}, {self.contract2})"


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
        obs = acquisition_idx.index_column(
            self.observable.simulate(acquisition_idx, model)
        )
        assert cf.cashflows.ndim == 2
        assert cf.cashflows.shape[0] == model.nsim
        assert obs.ndim == 1
        assert obs.shape[0] == model.nsim
        return cf * obs

    def __str__(self) -> str:
        return f"Scale({self.observable}, {self.contract})"


@dataclass
class When(Contract):
    """Obtain the underlying contract as soon as observable
    becomes True after acquisition"""

    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        idx = acquisition_idx.next_after(
            self.observable.simulate(acquisition_idx, model)
        )
        return self.contract.generate_cashflows(idx, model)

    def __str__(self) -> str:
        return f"When({self.observable}, {self.contract})"


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

    def __str__(self) -> str:
        return f"Anytime({self.observable}, {self.contract})"


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
        ko_idx = acquisition_idx.next_after(
            self.observable.simulate(acquisition_idx, model)
        )
        return cf.zero_after(ko_idx)

    def __str__(self) -> str:
        return f"Until({self.observable}, {self.contract})"


class ResolvableContract(Contract):
    @abstractmethod
    def resolve(self) -> Contract:
        pass

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        return self.resolve().generate_cashflows(acquisition_idx, model)

    def __str__(self) -> str:
        return str(self.resolve())


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
