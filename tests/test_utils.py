import numpy as np
from dataclasses import dataclass
from mcc import (
    DateIndex,
    ModelRequirements,
    Model,
    TermStructuresModel,
    ObservableBool,
    At,
    One,
    When,
    Contract,
    ResolvableContract,
)


@dataclass
class DummyTermStructureModel(TermStructuresModel):
    rate: np.ndarray

    def linear_rate(self, frequency: str) -> np.ndarray:
        return self.rate


def make_model(nsim: int = 100) -> Model:
    dategrid = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-10"),
        dtype="datetime64[D]",
    )
    numeraire = np.ones((nsim, dategrid.size), dtype=np.float64)
    rnd = np.random.RandomState(123)
    rate = rnd.normal(size=(nsim, dategrid.size))
    eurusd = rnd.lognormal(size=(nsim, dategrid.size))
    abc = rnd.lognormal(size=(nsim, dategrid.size))
    defg = rnd.lognormal(size=(nsim, dategrid.size))
    return Model(
        dategrid,
        {"EUR": DummyTermStructureModel(rate)},
        {("EUR", "USD"): eurusd},
        {"ABC": abc, "DEFG": defg},
        numeraire,
        "EUR",
    )


@dataclass
class MyContract(ResolvableContract):
    maturity: np.datetime64
    notional: float

    def resolve(self) -> Contract:
        return When(At(self.maturity), self.notional * One("EUR"))


class AlternatingBool(ObservableBool):
    def __init__(self, start_with_false: bool = True):
        self.offset = 0 if start_with_false else 1

    def simulate(self, first_observation_idx: DateIndex, model: Model) -> np.ndarray:
        mask = np.array(
            (np.arange(model.nsim) + self.offset) % 2, dtype=np.bool_
        ).reshape((model.nsim, 1))
        return np.repeat(mask, model.ndates, axis=1)

    def get_model_requirements(
        self, earliest: np.datetime64, latest: np.datetime64
    ) -> ModelRequirements:
        raise NotImplementedError()
