from abc import ABC, abstractmethod
from typing import (
    Final,
    Mapping,
    Tuple,
    Set,
    FrozenSet,
)
import numpy as np
from dataclasses import dataclass

from .cashflows import (
    NULL_CURRENCY,
    CURRENCY_LETTER_COUNT,
    DateIndex,
    IndexedCashflows,
)


class TermStructuresModel(ABC):
    @abstractmethod
    def linear_rate(self, frequency: str) -> np.ndarray:
        pass


@dataclass
class ModelRequirements:
    """Requirements to a model to compute cashflows for a contract."""

    currencies: FrozenSet[str]
    stocks: FrozenSet[str]
    linear_rates: FrozenSet[Tuple[str, str]]
    dates: FrozenSet[np.datetime64]

    def union(self, other: "ModelRequirements") -> "ModelRequirements":
        """Non-destructively merge to ModelRequirements.
        Result contains union of currencies, stocks, linear_rates and dates."""
        return ModelRequirements(
            self.currencies.union(other.currencies),
            self.stocks.union(other.stocks),
            self.linear_rates.union(other.linear_rates),
            self.dates.union(other.dates),
        )

    def __add__(self, other: "ModelRequirements") -> "ModelRequirements":
        """Non-destructively merge to ModelRequirements.
        Result contains union of currencies, stocks, linear_rates and dates."""
        return self.union(other)


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
            base_currency != NULL_CURRENCY and counter_currency != NULL_CURRENCY
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

    def in_currency(
        self, cashflows: IndexedCashflows, currency: str
    ) -> IndexedCashflows:
        assert currency != NULL_CURRENCY, "Cannot convert to null currency NNN"
        currencies = np.zeros(
            cashflows.currencies.shape, dtype=(np.unicode_, CURRENCY_LETTER_COUNT)
        )
        currencies[:] = currency
        converted = np.zeros(cashflows.cashflows.shape, dtype=IndexedCashflows.dtype)
        for i, cf in enumerate(cashflows.cashflows.T):
            converted["index"][:, i] = cf["index"]
            if cashflows.currencies[i] == NULL_CURRENCY:
                currencies[i] = NULL_CURRENCY
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

    def evaluate(self, cashflows: IndexedCashflows) -> float:
        return float(self.discount(cashflows).sum(axis=1).mean(axis=0))
