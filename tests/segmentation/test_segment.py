import unittest

import numpy as np

from sptx_ccf_registration.segmentation.segment import SegmentSlice


class TestSegmentSlice(unittest.TestCase):

    def setUp(self):
        # Test data to be used across all tests
        self.unsegmented_labels_slice = np.array([[0, 1], [2, 3]])
        self.z = 2

    def test_initialization(self):
        slice_obj = SegmentSlice(self.unsegmented_labels_slice, self.z)
        self.assertEqual(slice_obj.z, self.z)
        np.testing.assert_array_equal(slice_obj.unsegmented_labels_slice, self.unsegmented_labels_slice)

    def test_invalid_input_shape(self):
        with self.assertRaises(ValueError):
            SegmentSlice(np.array([0, 1]), self.z)  # 1D array not 2D

    def test_ccf_slice_mismatched_shape(self):
        with self.assertRaises(ValueError):
            SegmentSlice(self.unsegmented_labels_slice, self.z, ccf_slice=np.array([[0, 1, 2], [2, 3, 4]]))

    def test_alpha_range(self):
        slice_obj = SegmentSlice(self.unsegmented_labels_slice, self.z, min_alpha=0.1, max_alpha=0.5)
        alpha_range = slice_obj.alpha_range
        self.assertIsInstance(alpha_range, np.ndarray)
        self.assertTrue(np.all(alpha_range >= 0.1))
        self.assertTrue(np.all(alpha_range <= 0.5))

    def test_find_optimal_alpha(self):
        slice_obj = SegmentSlice(self.unsegmented_labels_slice, self.z, min_alpha=0.1, max_alpha=0.5, optimize_alpha=True, ccf_slice=self.unsegmented_labels_slice)

        # Test the optimal alpha value for a specific label.
        label = 1
        alpha = slice_obj.find_optimal_alpha(label)

        # Due to the complexity of find_optimal_alpha function, it might be hard to calculate an expected value directly.
        # So we check if the output is within the expected range instead.
        self.assertTrue(0.1 <= alpha <= 0.5)

        # Test the case where there is no ccf_area (it's zero).
        slice_obj.ccf_slice = np.zeros_like(self.unsegmented_labels_slice)
        alpha = slice_obj.find_optimal_alpha(label)

        # In this case, the function should return the default_alpha
        self.assertEqual(alpha, slice_obj.default_alpha)


if __name__ == "__main__":
    unittest.main()
