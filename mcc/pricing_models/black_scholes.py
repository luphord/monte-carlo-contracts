import numpy as np

from ..model import Model
from .stochastic_processes import GeometricBrownianMotion
from .financial import get_year_fractions


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
    yearfractions = get_year_fractions(dategrid)
    gbm = GeometricBrownianMotion(lambda t: r * t, sigma)
    s = S0 * (
        gbm.simulate_with_moment_matching(yearfractions, n, rnd)
        if use_moment_matching
        else gbm.simulate(yearfractions, n, rnd)
    )
    numeraire = np.repeat(np.exp(r * yearfractions).reshape((1, ndates)), n, axis=0)
    return Model(dategrid, {}, {}, {stock: s}, numeraire, currency)
