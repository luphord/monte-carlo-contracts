from dataclasses import dataclass
import numpy as np

from .observables import At

from .contracts import Contract, ResolvableContract, When, One, Zero


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
