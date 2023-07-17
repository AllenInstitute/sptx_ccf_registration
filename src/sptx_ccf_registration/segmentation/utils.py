import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_alpha_range(
    min_alpha: float, max_alpha: float, ratio: float = 1.25
) -> np.ndarray:
    """Generate a range of alpha values that will be used for
    optimize_alpha

    Parameters
    ----------
    min_alpha : float
        Lower boundary of alpha range.
    max_alpha : float
        Upper boundary of alpha range.
    ratio : float, optional
        Ratio between consecutive alpha values, by default 1.25

    Returns
    -------
    numpy.ndarray
        Array of alpha values.
    """
    lower_bound = min_alpha
    upper_bound = max_alpha

    alpha_values = [lower_bound]
    while lower_bound < upper_bound:
        interval = alpha_values[-1] * (ratio - 1)
        lower_bound += interval
        if lower_bound <= upper_bound:
            alpha_values.append(np.round(lower_bound, 3))

    alpha_range = np.array(alpha_values)
    return alpha_range
