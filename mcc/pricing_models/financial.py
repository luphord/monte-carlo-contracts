import numpy as np


def get_year_fractions(dategrid: np.ndarray) -> np.ndarray:
    assert dategrid.dtype == "datetime64[D]"
    return (dategrid - dategrid[0]).astype(np.float64) / 365
