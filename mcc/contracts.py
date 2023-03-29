from abc import ABC, abstractmethod
from functools import reduce
from operator import add
from typing import Union, Iterable, List
from numbers import Real
import numpy as np
from dataclasses import dataclass

from .cashflows import (
    IndexedCashflows,
    DateIndex,
    CURRENCY_LETTER_COUNT,
    NULL_CURRENCY,
)

from .model import Model, ModelRequirements

from .observables import ObservableBool, ObservableFloat, KonstFloat


class ResolvableContract(ABC):
    @abstractmethod
    def resolve(self) -> "Contract":
        pass

    def __str__(self) -> str:
        return str(self.resolve())


class Contract(ResolvableContract):
    """Abstract base class for all contracts"""

    @abstractmethod
    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        pass

    @abstractmethod
    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        pass

    def resolve(self) -> "Contract":
        return self

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
        ccys = np.array([NULL_CURRENCY], dtype=(np.unicode_, CURRENCY_LETTER_COUNT))
        return IndexedCashflows(cf, ccys, model.dategrid)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        return ModelRequirements(frozenset(), frozenset(), frozenset(), frozenset())

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
        ccys = np.array([self.currency], dtype=(np.unicode_, CURRENCY_LETTER_COUNT))
        return IndexedCashflows(cf, ccys, model.dategrid)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        return ModelRequirements(
            frozenset([self.currency]), frozenset(), frozenset(), frozenset()
        )

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        return self.contract.get_model_requirements(earliest, latest)

    def __str__(self) -> str:
        return f"Give({self.contract})"


@dataclass
class And(Contract):
    """Obtain rights and obligations of all underlying contracts"""

    contracts: List[Contract]

    def __init__(self, *contracts: Contract) -> None:
        self.contracts = list(contracts)
        assert any(self.contracts), "At least one contract is required"
        self.contracts = list(self._flattened_contracts)

    @property
    def _flattened_contracts(self) -> Iterable[Contract]:
        for contract in self.contracts:
            if isinstance(contract, And):
                yield from contract.contracts
            else:
                yield contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cfs = [
            contract.generate_cashflows(acquisition_idx, model)
            for contract in self.contracts
        ]
        assert cfs
        sum_cfs = sum(cfs[1:], cfs[0])
        assert sum_cfs
        return sum_cfs

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        reqs = [
            contract.get_model_requirements(earliest, latest)
            for contract in self.contracts
        ]
        assert reqs
        sum_reqs = sum(reqs[1:], reqs[0])
        assert sum_reqs
        return sum_reqs

    def __str__(self) -> str:
        contracts = [str(contract) for contract in self.contracts]
        return f"And({', '.join(contracts)})"


@dataclass
class Or(Contract):
    """Choose at acquisition between the underlying contracts"""

    contracts: List[Contract]

    def __init__(self, *contracts: Contract) -> None:
        self.contracts = list(contracts)
        assert any(self.contracts), "At least one contract is required"
        self.contracts = list(self._flattened_contracts)

    @property
    def _flattened_contracts(self) -> Iterable[Contract]:
        for contract in self.contracts:
            if isinstance(contract, Or):
                yield from contract.contracts
            else:
                yield contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cfs = [
            contract.generate_cashflows(acquisition_idx, model)
            for contract in self.contracts
        ]
        ccys = np.unique(np.concatenate([cf.currencies.flatten() for cf in cfs]))
        ccys = ccys[ccys != "NNN"].flatten()
        concatenated_cf = reduce(add, cfs)
        assert isinstance(concatenated_cf, IndexedCashflows)
        for cf in concatenated_cf.cashflows.T:
            if ((acquisition_idx.index != cf["index"]) & (cf["index"] >= 0)).any():
                raise NotImplementedError(
                    "Cashflow generation for OR contract at any moment"
                    " other than cashflow date is not implemented"
                )
        cfsums = [
            model.in_numeraire_currency(cf).cashflows["value"].sum(axis=1) for cf in cfs
        ]
        choose_idx = np.argmax(cfsums, axis=0)
        for i, cf in enumerate(cfs):
            choose_this = choose_idx == i
            cf.cashflows["index"][~choose_this] = -1
            cf.cashflows["value"][~choose_this] = 0
        return reduce(add, cfs)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        reqs = [
            contract.get_model_requirements(earliest, latest)
            for contract in self.contracts
        ]
        assert reqs
        sum_reqs = sum(reqs[1:], reqs[0])
        assert sum_reqs
        return sum_reqs

    def __str__(self) -> str:
        contracts = [str(contract) for contract in self.contracts]
        return f"Or({', '.join(contracts)})"


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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"When({self.observable}, {self.contract})"


@dataclass
class Delay(Contract):
    """Obtain the underlying contract and delay all payments
    to first occurence of observable."""

    observable: ObservableBool
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        idx = acquisition_idx.next_after(
            self.observable.simulate(acquisition_idx, model)
        )
        return self.contract.generate_cashflows(acquisition_idx, model).delay(idx)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"ShiftTo({self.observable}, {self.contract})"


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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"Until({self.observable}, {self.contract})"


@dataclass
class Exchange(Contract):
    """Exchange cashflows resulting from contract to currency
    at the current spot rate"""

    currency: str
    contract: Contract

    def generate_cashflows(
        self, acquisition_idx: DateIndex, model: Model
    ) -> IndexedCashflows:
        cf = self.contract.generate_cashflows(acquisition_idx, model)
        return model.in_currency(cf, self.currency)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"Exchange({self.currency}, {self.contract})"
