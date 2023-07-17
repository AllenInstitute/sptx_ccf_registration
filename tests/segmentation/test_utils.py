import numpy as np

from sptx_ccf_registration.segmentation.utils import get_alpha_range


def test_get_alpha_range():
    # Test with default ratio
    alpha_range = get_alpha_range(1.0, 5.0)
    assert isinstance(alpha_range, np.ndarray)
    assert alpha_range[0] == 1.0
    assert alpha_range[-1] <= 5.0
    assert np.all(np.diff(alpha_range) >= 0)
    assert np.all(np.round(alpha_range, 3) == alpha_range)

    # Test with non-default ratio
    alpha_range = get_alpha_range(1.0, 5.0, 1.5)
    assert isinstance(alpha_range, np.ndarray)
    assert alpha_range[0] == 1.0
    assert alpha_range[-1] <= 5.0
    assert np.all(np.diff(alpha_range) >= 0)
    assert np.all(np.round(alpha_range, 3) == alpha_range)
