from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from sptx_ccf_registration.metrics_dashboard.metrics import (
    dice_coefficient,
    load_nii_gz_image,
    masks_intersection,
    parse_itksnap_file,
)


@patch("nibabel.load", autospec=True)
def test_load_nii_gz_image(mocked_nib_load):
    mocked_img = MagicMock()
    mocked_img.get_fdata.return_value = np.array([1, 2, 3, 4])
    mocked_nib_load.return_value = mocked_img

    file_path = "tests/test_image.nii.gz"
    result = load_nii_gz_image(file_path)

    mocked_nib_load.assert_called_once_with(file_path)
    mocked_img.get_fdata.assert_called_once()
    assert np.array_equal(result, np.array([1, 2, 3, 4]))


def test_masks_intersection():
    binary_mask1 = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    binary_mask2 = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])

    result = masks_intersection(binary_mask1, binary_mask2)
    assert result == 2


def test_dice_coefficient():
    binary_mask1 = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    binary_mask2 = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])

    result = dice_coefficient(binary_mask1, binary_mask2)
    intersection = 2
    union = 9
    dice = 2 * intersection / union
    assert result == dice


def test_parse_itksnap_file():
    itksnap_file_path = Path("tests/sample_data/itksnap_ccf_landmark_file.txt")
    result = parse_itksnap_file(itksnap_file_path)
    assert result[51] == "AD - 64"
