from abc import ABC, abstractmethod
from typing import (
    Union,
)
from numbers import Real
import numpy as np
from dataclasses import dataclass

from .cashflows import (
    DateIndex,
)

from .model import Model, ModelRequirements


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

    @abstractmethod
    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        return self.observable.get_model_requirements(earliest, latest)

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        return self.observable1.get_model_requirements(earliest, latest).union(
            self.observable2.get_model_requirements(earliest, latest)
        )

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        return self.observable1.get_model_requirements(earliest, latest).union(
            self.observable2.get_model_requirements(earliest, latest)
        )

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()

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

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        # ToDo: Should we assert that earliest <= self.date <= latest?
        return ModelRequirements(
            frozenset(), frozenset(), frozenset(), frozenset([self.date])
        )

    def __str__(self) -> str:
        return str(self.date)
