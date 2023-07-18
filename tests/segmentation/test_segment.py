import json
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from sptx_ccf_registration.segmentation.segment import SegmentSlice, get_alpha_range


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


class TestSegmentSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_path = Path(__file__).parent.parent / "sample_data"
        cls.args = {
            "segmented_label_output_file": sample_path
            / "mfish_labels_landmark_segmented.nii.gz",
            "input_paths": {
                "unsegmented_label_file": sample_path
                / "mfish_labels_landmark_unsegmented.nii.gz",
                "ccf_file": sample_path / "ccf_labels_landmark.nii.gz",
                "itksnap_file_path": sample_path / "itksnap_ccf_landmark_file.txt",
                "alpha_selection_path": sample_path
                / "mfish_labels_landmark_alpha_selection_all.json",
            },
            "optimize_alpha": True,
            "save_alpha_qc": True,
        }

        cls.unsegmented_labels_slice = (
            nib.load(cls.args["input_paths"]["unsegmented_label_file"])
            .get_fdata()[:, :, 0]
            .astype(np.uint8)
        )
        cls.segmented_labels_slice = (
            nib.load(cls.args["segmented_label_output_file"])
            .get_fdata()[:, :, 0]
            .astype(np.uint8)
        )
        cls.ccf_slice = nib.load(cls.args["input_paths"]["ccf_file"]).get_fdata()[
            :, :, 0
        ]
        with open(cls.args["input_paths"]["alpha_selection_path"], "r") as fp:
            cls.alpha_selection = json.load(fp)
        cls.z = 0

    def setUp(self):
        self.segment_slice = SegmentSlice(
            self.unsegmented_labels_slice,
            self.z,
            ccf_slice=self.ccf_slice,
            optimize_alpha=True,
            alpha_selection=self.alpha_selection,
        )

    def test_initialization(self):
        self.assertEqual(
            self.segment_slice.y_dim, self.unsegmented_labels_slice.shape[0]
        )
        self.assertEqual(
            self.segment_slice.x_dim, self.unsegmented_labels_slice.shape[1]
        )
        self.assertIsNotNone(self.segment_slice.points)
        self.assertIsNotNone(self.segment_slice.labels)
        self.assertIsNotNone(self.segment_slice.alpha_range)
        self.assertEqual(self.segment_slice.z, self.z)

    def test_unique_labels(self):
        unique_labels = self.segment_slice.unique_labels
        self.assertIsInstance(unique_labels, list)
        self.assertTrue(all(isinstance(label, np.int64) for label in unique_labels))

    def test_find_optimal_alpha(self):
        # pick one label to test
        label = self.segment_slice.unique_labels[0]
        alpha = self.segment_slice.find_optimal_alpha(label)
        self.assertIsInstance(alpha, float)
        self.assertTrue(
            self.segment_slice.min_alpha <= alpha <= self.segment_slice.max_alpha
        )
        # test happy case
        self.assertEqual(alpha, 0.2)

    @mock.patch.object(SegmentSlice, "find_optimal_alpha", return_value=0.1)
    def test_z_label_to_alpha(self, mock_find_optimal_alpha):
        # happy case with alpha selection
        z_label_to_alpha = self.segment_slice.z_label_to_alpha
        self.assertIsInstance(z_label_to_alpha, dict)
        self.assertTrue(
            all(isinstance(alpha, (float, str)) for alpha in z_label_to_alpha.values())
        )
        self.assertEqual(z_label_to_alpha.get(str((self.z, 64))), 0.04)
        # ignores optimal_alpha and picks from alpha_selection
        mock_find_optimal_alpha.assert_not_called()
        self.assertEqual(z_label_to_alpha.get(str((self.z, 64))), 0.04)

        # happy case without alpha selection
        # test with optimize_alpha = True
        segment_slice = SegmentSlice(
            self.unsegmented_labels_slice,
            self.z,
            ccf_slice=self.ccf_slice,
            optimize_alpha=True,
            alpha_selection=None,
        )
        z_label_to_alpha = segment_slice.z_label_to_alpha
        mock_find_optimal_alpha.assert_called()
        mock_find_optimal_alpha.reset_mock()
        self.assertEqual(z_label_to_alpha.get(str((self.z, 64))), 0.1)

        # test with optimize_alpha = False
        segment_slice = SegmentSlice(
            self.unsegmented_labels_slice,
            self.z,
            ccf_slice=self.ccf_slice,
            optimize_alpha=False,
            alpha_selection=None,
        )
        z_label_to_alpha = segment_slice.z_label_to_alpha
        mock_find_optimal_alpha.assert_not_called()
        self.assertEqual(
            z_label_to_alpha.get(str((self.z, 57))), self.segment_slice.default_alpha
        )

    def test_max_density_labels(self):
        max_density_labels = self.segment_slice.max_density_labels
        self.assertIsInstance(max_density_labels, np.ndarray)
        self.assertEqual(max_density_labels.shape, self.unsegmented_labels_slice.shape)

    def test__label_points_to_binary_mask(self):
        label_points = self.segment_slice.points[
            self.segment_slice.labels == self.segment_slice.unique_labels[0]
        ]
        # test alpha that reuturns a mask
        alpha = 0.2
        binary_mask = self.segment_slice._label_points_to_binary_mask(
            label_points, alpha, self.segment_slice.y_dim, self.segment_slice.x_dim
        )
        self.assertIsInstance(binary_mask, np.ndarray)
        self.assertEqual(binary_mask.shape, self.unsegmented_labels_slice.shape)

        # test alpha that returns None (alpha_shape.is_empty==True)
        alpha = 1
        binary_mask = self.segment_slice._label_points_to_binary_mask(
            label_points, alpha, self.segment_slice.y_dim, self.segment_slice.x_dim
        )
        self.assertIsNone(binary_mask)

    def test_segmented_labels_slice(self):
        segmented_labels_slice = self.segment_slice.segmented_labels_slice
        self.assertIsInstance(segmented_labels_slice, np.ndarray)
        self.assertEqual(
            segmented_labels_slice.shape, self.unsegmented_labels_slice.shape
        )
        # assert saved validated results
        # np.testing.assert_array_equal(segmented_labels_slice,
        #     self.segmented_labels_slice)
