import logging

import numpy as np
from alphashape import alphashape
from rasterio.features import rasterize

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


def label_points_to_binary_mask(
    label_points: np.ndarray, alpha: float, y_dim: int, x_dim: int
):
    """
    Use alphashape to generate a concave hull around the points and
    convert it to a binary mask.

    Parameters
    ----------
    label_points : numpy.ndarray
        The points to be segmented by concave hull

    Returns
    -------
    numpy.ndarray
        The binary mask of the polygons.
    """
    # Generate concave hull
    try:
        alpha_shape = alphashape(label_points, alpha)
    except:
        logger.warning("Could not compute concave hull")
        return None
    # Convert to binary mask
    if not alpha_shape.is_empty:
        mask = rasterize([(alpha_shape, 1)], out_shape=(y_dim, x_dim))
    else:
        return None
    return mask.T
