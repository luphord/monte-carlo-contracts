from typing import Final, Iterable, Tuple, Union, Callable
from itertools import groupby

import numpy as np
import pandas as pd


CURRENCY_LETTER_COUNT: Final[int] = 3
NULL_CURRENCY: Final[str] = "NNN"
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
        assert currencies.dtype == (np.unicode_, CURRENCY_LETTER_COUNT)
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
        assert currencies.dtype == (np.unicode_, CURRENCY_LETTER_COUNT)
        assert currencies.shape == (cashflows.shape[1],)
        self.cashflows = cashflows
        self.currencies = currencies
        self.nsim = cashflows.shape[0]
        self.ncashflows = cashflows.shape[1]

    def _split_by_date(self) -> Iterable[Tuple[np.ndarray, str, np.datetime64]]:
        for i, cf in enumerate(self.cashflows.T):
            if self.currencies[i] != NULL_CURRENCY:
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
        currencies: np.ndarray = np.ndarray(
            (numcf,), dtype=(np.unicode_, CURRENCY_LETTER_COUNT)
        )
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
        assert currencies.dtype == (np.unicode_, CURRENCY_LETTER_COUNT)
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

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IndexedCashflows)
            and (self.cashflows == other.cashflows).all()
            and (self.currencies == other.currencies).all()
            and (self.dategrid == other.dategrid).all()
        )

    def zero_after(self, date_idx: "DateIndex") -> "IndexedCashflows":
        assert self.nsim == date_idx.nsim
        zeroedcf = self.cashflows.copy()
        for i, cf in enumerate(self.cashflows.T):
            ko_mask = (cf["index"] >= date_idx.index) & (date_idx.index >= 0)
            zeroedcf["value"][ko_mask, i] = 0
        return IndexedCashflows(zeroedcf, self.currencies, self.dategrid)

    def delay(self, date_idx: "DateIndex") -> "IndexedCashflows":
        """Delay all cashflows before date_idx to date_idx (no discounting applied,
        just move the cashflow date)."""
        assert self.nsim == date_idx.nsim
        shiftedcf = self.cashflows.copy()
        for i, cf in enumerate(shiftedcf.T):
            before_mask = (
                (cf["index"] <= date_idx.index)
                & (date_idx.index >= 0)
                & (cf["index"] >= 0)
            )
            shiftedcf["index"][before_mask, i] = date_idx.index[before_mask]
        return IndexedCashflows(shiftedcf, self.currencies, self.dategrid)

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
