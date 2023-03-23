from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


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
