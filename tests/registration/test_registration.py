import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import ants
import nibabel as nib
import numpy as np

from sptx_ccf_registration.registration.registration import Registration, TransformType


class TestRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_path = Path(__file__).parent.parent / "sample_data"
        cls.args = {
            "output_name": "test",
            "output_path": "tests/registration/test_output",
            "merfish_files": {
                "labels_broad_segmented": sample_path
                / "mfish_labels_broad_segmented.nii.gz",
                "labels_broad_unsegmented": sample_path
                / "mfish_labels_broad_unsegmented.nii.gz",
                "labels_landmark_segmented": sample_path
                / "mfish_labels_landmark_segmented.nii.gz",
                "labels_landmark_unsegmented": sample_path
                / "mfish_labels_landmark_unsegmented.nii.gz",
                "right_hemisphere": sample_path / "mfish_labels_right_hemisphere.nii.gz",
            },
            "ccf_files": {
                "labels_broad": sample_path / "ccf_labels_broad.nii.gz",
                "labels_landmark": sample_path / "ccf_labels_landmark.nii.gz",
                "right_hemisphere": sample_path / "ccf_labels_right_hemisphere.nii.gz",
            },
            "labels_level": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "labels_replace_to": [1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
            "iteration_labels": [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [72, 73],
                [74],
                [75, 76],
                [113, 114],
                [84],
                [79],
                [123, 54, 64, 121, 120, 78, 94],
                [124],
            ],
        }

    def setUp(self):
        self.reg = Registration(**self.args)

    def test___get_registration_params(self):
        assert self.reg.__get_registration_params(0) == ("SyN", (40, 20, 10), None)
        assert self.reg.__get_registration_params(1) == (
            "SyNOnly",
            (40, 20, 10),
            "Identity",
        )
        assert self.reg.__get_registration_params(2) == (
            "SyNOnly",
            (70, 40, 20),
            "Identity",
        )

    def test_read_images(self):
        merfish_images = self.reg.__read_images(self.reg.merfish_files)
        for key in self.reg.merfish_files.keys():
            assert isinstance(merfish_images[key], ants.core.ANTsImage)

    def test_is_empty(self):
        empty_img = np.zeros((5, 5))
        assert self.reg.is_empty(empty_img) is True
        not_empty_img = np.ones((5, 5))
        assert self.reg.is_empty(not_empty_img) is False

    def test___select_images_and_labels(self):
        merfish_images = {
            k: nib.load(str(v)).get_fdata()
            for k, v in self.args["merfish_files"].items()
        }
        ccf_images = {
            k: nib.load(str(v)).get_fdata() for k, v in self.args["ccf_files"].items()
        }

        for iteration in range(10):  # assuming there are 10 iterations
            selected_images = self.reg.__select_images_and_labels(
                iteration, merfish_images, ccf_images
            )

            # Test keys are preserved
            self.assertEqual(set(selected_images.keys()), set(["mFish", "CCF"]))

            # Test selected images have correct labels and labels are subsetted
            # and merged correctly
            for key, img in selected_images.items():
                unique_labels = np.unique(img)
                if self.reg.labels_replace_to[iteration] > 0 and iteration > 0:
                    self.assertTrue(
                        np.array_equal(
                            unique_labels,
                            np.array([0, self.reg.labels_replace_to[iteration]]),
                        )
                    )
                elif iteration > 0:
                    expected_labels = set(self.reg.iteration_labels[iteration] + [0])
                    self.assertTrue(set(unique_labels).issubset(expected_labels))

    def test___handle_empty_merfish_slice(self):
        mock_merfish_slice = np.zeros((5, 5))
        mock_ccf_slice = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        merfish_slice, ccf_slice = self.reg.__handle_empty_merfish_slice(
            mock_merfish_slice, mock_ccf_slice
        )
        assert (merfish_slice == mock_ccf_slice).all()
        assert (ccf_slice == mock_ccf_slice).all()

    def test___get_transform_path(self):
        result = self.reg.__get_transform_path(10, 1, "SYN")
        assert "iter1" in result
        assert "slice_transformations" in result
        assert "test_SYN_slice10" in result

    def test_create_transforms_slice(self):
        for iteration in [0, 1]:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                def mocked_registration(*args, **kwargs):
                    # Create the affine and warp transform dummy files
                    fwd_affine_path = tmpdir_path / "fwd_affine_transform.mat"
                    fwd_warp_path = tmpdir_path / "fwd_warp_transform.nii.gz"
                    inv_affine_path = tmpdir_path / "inv_affine_transform.mat"
                    inv_warp_path = tmpdir_path / "inv_warp_transform.nii.gz"
                    with open(fwd_affine_path, "w") as f:
                        f.write("dummy affine data")
                    with open(fwd_warp_path, "w") as f:
                        f.write("dummy warp data")
                    with open(inv_affine_path, "w") as f:
                        f.write("dummy affine data")
                    with open(inv_warp_path, "w") as f:
                        f.write("dummy warp data")

                    # Return a dictionary with the paths to the dummy files
                    return {
                        "fwdtransforms": [str(fwd_warp_path), str(fwd_affine_path)],
                        "invtransforms": [str(inv_warp_path), str(inv_affine_path)],
                    }

                with patch("ants.registration", side_effect=mocked_registration):
                    merfish_slice = np.array(
                        [[1, 2], [3, 4]], dtype=np.float32
                    )  # dummy data
                    ccf_slice = np.array(
                        [[5, 6], [7, 8]], dtype=np.float32
                    )  # dummy data
                    z = 1  # dummy data

                    # Create another temp directory for shutil.move
                    with tempfile.TemporaryDirectory() as tmpdir2:
                        self.reg.__get_transform_path = MagicMock(
                            return_value=Path(tmpdir2) / "transform.mat"
                        )
                        self.reg.create_transforms_slice(
                            iteration, merfish_slice, ccf_slice, z
                        )

                        # Check if the files are correctly moved
                        self.assertTrue((Path(tmpdir2) / "transform.mat").is_file())

                        # Retrieve the arguments used for ants.registration call
                        args, kwargs = ants.registration.call_args

                        # assertions for the arguments
                        self.assertTrue(
                            np.array_equal(kwargs["fixed"], ants.from_numpy(ccf_slice))
                        )
                        self.assertTrue(
                            np.array_equal(
                                kwargs["moving"], ants.from_numpy(merfish_slice)
                            )
                        )

                        if iteration == 0:
                            self.assertEqual(
                                kwargs["type_of_transform"], TransformType.SYN.value
                            )  # replace 'SyN' with your value
                        elif iteration > 0:
                            self.assertEqual(
                                kwargs["type_of_transform"], TransformType.SYNONLY.value
                            )  # replace 'BSplineSyN' with your value
