from typing import Final, Optional, Callable
import numpy as np

from ..model import TermStructuresModel
from .stochastic_processes import BrownianMotion
from .financial import get_year_fractions


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
        self.yearfractions = get_year_fractions(dategrid)
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
