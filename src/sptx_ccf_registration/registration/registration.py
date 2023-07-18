import logging
import os
import shutil
from enum import Enum
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import ants
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpolatorType(Enum):
    LINEAR = "linear"
    NEAREST_NEIGHBOR = "nearestNeighbor"


class TransformType(Enum):
    SYN = "SyN"
    SYNONLY = "SyNOnly"


class TransformSubType(Enum):
    FORWARD_AFFINE = "fwd_affine"
    FORWARD_DEFORMATION = "fwd_def"
    INVERSE_DEFORMATION = "inv_def"


class Registration:
    def __init__(
        self,
        output_name: str,
        output_path: str,
        merfish_files: Dict,
        ccf_files: Dict,
        labels_level: Tuple[int],
        labels_replace_to: Tuple[int],
        iteration_labels: Tuple[Tuple[int]],
        n_processes: int = cpu_count(),
    ):
        """
        Initialize the Registration class.

        This class handles the registration process for neuroimaging data. It takes
        input files, performs image transformations, and saves the output.

        Parameters:
        ----------
        output_name (str):
            The name to be used for the output files.
        output_path (str):
            The path where the output files will be stored.
        merfish_files (Dict):
            A dictionary containing the paths of the MERFISH data files.
            The keys required are:
            'labels_broad_segmented', 'labels_landmark_segmented',
            'labels_broad_unsegmented', 'labels_landmark_unsegmented',
            'stack', and 'right_hemisphere'.
        ccf_files (Dict):
            A dictionary containing the paths of the CCF data files
            The keys required are:
            'labels_broad', 'labels_landmark', and 'right_hemisphere'.
        labels_level (Tuple[int]):
            A tuple containing the label levels.
        labels_replace_to (Tuple[int]):
            A tuple defining the values to which the original labels will be mapped.
        iteration_labels (Tuple[Tuple[int]]):
            A nested tuple containing the labels for each iteration.
        """
        self.output_name = output_name
        self.output_path = output_path
        self.merfish_files = merfish_files
        self.ccf_files = ccf_files
        self.__validate_merfish_and_ccf_files()
        self.labels_level = labels_level
        self.labels_replace_to = labels_replace_to
        self.iteration_labels = iteration_labels
        self.slice_transform_dir_name = "slice_transformations"
        self.n_processes = n_processes
        os.makedirs(self.output_path, exist_ok=True)

        self.merfish_files
        self.merfish_label_type_to_interpolator_type = {
            "labels_broad_segmented": InterpolatorType.NEAREST_NEIGHBOR.value,
            "labels_landmark_segmented": InterpolatorType.NEAREST_NEIGHBOR.value,
            "labels_broad_unsegmented": InterpolatorType.NEAREST_NEIGHBOR.value,
            "labels_landmark_unsegmented": InterpolatorType.NEAREST_NEIGHBOR.value,
            "stack": InterpolatorType.LINEAR.value,
            "right_hemisphere": InterpolatorType.LINEAR.value,
        }

    def __validate_merfish_and_ccf_files(self) -> None:
        """
        Validate the MERFISH and CCF files.
        """
        merfish_files_valid_keys = [
            "labels_broad_segmented",
            "labels_landmark_segmented",
            "labels_broad_unsegmented",
            "labels_landmark_unsegmented",
            "stack",
            "right_hemisphere",
        ]
        for key in self.merfish_files.keys():
            if key not in merfish_files_valid_keys:
                raise ValueError(
                    f"Invalid key {key} in merfish_files. Check schema for valid keys"
                )
            merfish_files_valid_keys.remove(key)
        if len(merfish_files_valid_keys) > 0:
            raise ValueError(
                f"Missing keys {merfish_files_valid_keys} in merfish_files."
                "Check schema for valid keys"
            )

        ccf_files_valid_keys = [
            "labels_broad",
            "labels_landmark",
            "right_hemisphere",
        ]
        for key in self.ccf_files.keys():
            if key not in ccf_files_valid_keys:
                raise ValueError(
                    f"Invalid key {key} in ccf_files. Check schema for valid keys"
                )
            ccf_files_valid_keys.remove(key)
        if len(ccf_files_valid_keys) > 0:
            raise ValueError(
                f"Missing keys {ccf_files_valid_keys} in ccf_files."
                "Check schema for valid keys"
            )

    def _get_registration_params(self, iteration: int) -> Tuple[str, Tuple[int], str]:
        """Get registration parameters based on iteration
            1. trans_type: The type of transformation to be used
            2. iter_lvl: The number of iterations to be used
            3. initial_transform: The initial transformation to be used

        Parameters:
        -----------
        iteration (int):
            The iteration number

        Returns:
        --------
        Tuple[str, Tuple[int], str]:
            A tuple containing the registration parameters
        """
        if iteration == 0:
            trans_type = TransformType.SYN.value
            iter_lvl = (40, 20, 10)
            initial_transform = None
        elif iteration == 1:
            trans_type = TransformType.SYNONLY.value
            iter_lvl = (40, 20, 10)
            initial_transform = "Identity"
        else:
            trans_type = TransformType.SYNONLY.value
            iter_lvl = (70, 40, 20)
            initial_transform = "Identity"
        return trans_type, iter_lvl, initial_transform

    def _read_images(self, files) -> Dict[str, ants.ANTsImage]:
        """Reads images from given files

        Parameters:
        -----------
        files (Dict):
            A dictionary containing the paths of the files to be read

        Returns:
        --------
        Dict[str, ants.ANTsImage]:
            A dictionary containing the read images
        """
        return {key: ants.image_read(str(value)) for key, value in files.items()}

    def _select_images_and_labels(
        self, iteration, merfish_images, ccf_images
    ) -> Dict[str, ants.ANTsImage]:
        """Select images and labels based on iteration for estimating
        registration affine and warp params

        Parameters:
        -----------
        iteration (int):
            The iteration number
        merfish_images (Dict):
            A dictionary containing the MERFISH images
        ccf_images (Dict):
            A dictionary containing the CCF images

        Returns:
        --------
        Dict[str, ants.ANTsImage]:
            A dictionary containing the selected,
            subsetted, merged, and right hemisphere transformed images
        """
        selected_images = {}
        right_hemisphere_images = {
            "mFish": merfish_images["right_hemisphere"].copy(),
            "CCF": ccf_images["right_hemisphere"].copy() * 500,
        }
        if self.labels_level[iteration] == 0:
            selected_images["mFish"] = merfish_images["labels_broad_segmented"].copy()
            selected_images["CCF"] = ccf_images["labels_broad"].copy()
        else:
            selected_images["mFish"] = merfish_images["labels_landmark_segmented"].copy()
            selected_images["CCF"] = ccf_images["labels_landmark"].copy()

        for key, img in selected_images.items():
            label_mask = np.isin(img.view(), self.iteration_labels[iteration])
            selected_images[key].view()[~label_mask] = 0

            # merge labels
            if self.labels_replace_to[iteration] > 0:
                selected_images[key].view()[label_mask] = self.labels_replace_to[
                    iteration
                ]

            # add right hemisphere
            if iteration == 0:
                selected_images[key].view()[:, :, :] = (
                    img.view() * 250 + img.view() * right_hemisphere_images[key].view()
                )
        return selected_images

    def is_empty(self, img_slice: np.ndarray) -> bool:
        """Check if slice is empty

        Parameters:
        -----------
        img_slice (np.ndarray):
            The image slice to be checked

        Returns:
        --------
        bool:
            True if slice is empty, False otherwise
        """
        return img_slice.sum() == 0

    def _handle_empty_merfish_slice(
        self, merfish_slice: np.ndarray, ccf_slice: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check if slc is empty, if so simulate an identity transform
        TODO: we need some better logic for this in the future

        Parameters:
        -----------
        merfish_slice (np.ndarray):
            The MERFISH slice to be checked
        ccf_slice (np.ndarray):
            The CCF slice to be checked

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the MERFISH and CCF slices
        """
        if self.is_empty(merfish_slice) or self.is_empty(ccf_slice):
            ccf_slice[0, 0] = 1
            merfish_slice = ccf_slice
        return merfish_slice, ccf_slice

    def _get_transform_path(
        self, z: int, iteration: int, transform_type: TransformSubType
    ) -> str:
        """Get path to transform file

        Parameters:
        -----------
        z (int):
            The slice number
        iteration (int):
            The iteration number
        transform_type (TransformSubType):
            The type of transform

        Returns:
        --------
        str:
            The path to the transform file
        """
        slice_output_path = self.__get_slice_output_path(iteration)
        if transform_type == TransformSubType.FORWARD_AFFINE.value:
            return os.path.join(
                slice_output_path,
                f"{self.output_name}_{transform_type}_slice{z}.mat",
            )
        else:
            return os.path.join(
                slice_output_path,
                f"{self.output_name}_{transform_type}_slice{z}.nii.gz",
            )

    def __get_iteration_output_path(self, iteration: int) -> str:
        """Get path to iteration output

        Parameters:
        -----------
        iteration (int):
            The iteration number

        Returns:
        --------
        str:
            The path to the iteration output
        """
        return os.path.join(
            self.output_path,
            f"iter{iteration}",
        )

    def __get_slice_output_path(self, iteration: int) -> str:
        """Get path to slice output

        Parameters:
        -----------
        iteration (int):
            The iteration number

        Returns:
        --------
        str:
            The path to the slice output
        """
        iteration_output_path = self.__get_iteration_output_path(iteration)
        return os.path.join(
            iteration_output_path,
            self.slice_transform_dir_name,
        )

    def __get_merfish_label_output_path(self, iteration: int, label_name: str) -> str:
        """Get path to merfish label output

        Parameters:
        -----------
        iteration (int):
            The iteration number
        label_name (str):
            The name of the label

        Returns:
        --------
        str:
            The path to the merfish label output
        """
        iteration_output_path = self.__get_iteration_output_path(iteration)
        return os.path.join(
            iteration_output_path,
            f"{self.output_name}_{label_name}_AppliedWarpAllSlc.nii.gz",
        )

    def create_transforms_slice(
        self, iteration: int, merfish_slice: np.ndarray, ccf_slice: np.ndarray, z: int
    ) -> None:
        """Registration between merfish and ccf slices, outputs affine and warp
        transforms that's stored to output_path/iter{iteration}/slice_transformations

        Parameters:
        -----------
        iteration (int):
            The iteration number
        merfish_slice (np.ndarray):
            The MERFISH slice selected for registration (moving image)
        ccf_slice (np.ndarray):
            The CCF slice selected for registration (fixed image)

        Returns:
        --------
        None
        """
        merfish_slice, ccf_slice = self._handle_empty_merfish_slice(
            merfish_slice, ccf_slice
        )
        trans_type, iter_lvl, initial_transform = self._get_registration_params(
            iteration
        )

        registration = ants.registration(
            fixed=ants.from_numpy(ccf_slice),
            moving=ants.from_numpy(merfish_slice),
            initial_transform=initial_transform,
            type_of_transform=trans_type,
            grad_step=0.1,
            flow_sigma=3,
            total_sigma=0,
            reg_iterations=iter_lvl,
            aff_metric="meansquares",
            syn_metric="meansquares",
            verbose=False,
        )

        transform_type_to_key = {
            TransformSubType.FORWARD_AFFINE.value: "fwdtransforms",
            TransformSubType.FORWARD_DEFORMATION.value: "fwdtransforms",
            TransformSubType.INVERSE_DEFORMATION.value: "invtransforms",
        }

        transform_type_to_index = {
            TransformSubType.FORWARD_AFFINE.value: 1,
            TransformSubType.FORWARD_DEFORMATION.value: 0,
            TransformSubType.INVERSE_DEFORMATION.value: 1,
        }

        for transform_type in TransformSubType:
            transform_type = transform_type.value
            transform_key = transform_type_to_key[transform_type]
            transform_index = transform_type_to_index[transform_type]
            transform_path = self._get_transform_path(z, iteration, transform_type)
            shutil.move(registration[transform_key][transform_index], transform_path)

    def _parallel_create_transforms_slice(self, args) -> None:
        """Helper function to parallelize create_transforms_slice"""
        iteration, merfish_slice, ccf_slice, z = args
        self.create_transforms_slice(iteration, merfish_slice, ccf_slice, z)

    def create_transforms(self, iteration: int, selected_images: Dict) -> None:
        """Create transforms for a given iteration across all z-slices

        Parameters:
        -----------
        iteration (int):
            The iteration number
        selected_images (Dict):
            A dictionary containing the selected images for
            registration (mFish and CCF)

        Returns:
        --------
        None
        """
        iteration_output_path = self.__get_iteration_output_path(iteration)
        slice_output_path = self.__get_slice_output_path(iteration)
        os.makedirs(slice_output_path, exist_ok=True)

        for key, img in selected_images.items():
            ants.image_write(
                img, f"{iteration_output_path}/{self.output_name}_selLabels_{key}.nii.gz"
            )

        num_slices = selected_images["mFish"].view().shape[-1]

        pool = Pool(processes=self.n_processes)

        args = [
            (
                iteration,
                selected_images["mFish"].view()[:, :, z],
                selected_images["CCF"].view()[:, :, z],
                z,
            )
            for z in range(num_slices)
        ]

        pool.map(self._parallel_create_transforms_slice, args)
        pool.close()
        pool.join()

    def apply_slice_transforms(
        self,
        iteration: int,
        moving_image: ants.ANTsImage,
        fixed_image: ants.ANTsImage,
        interpolator_type: InterpolatorType,
        trans_type: TransformType,
        output_label_path: str,
    ) -> ants.ANTsImage:
        """
        Perform transformation (affine+warp) by z-slice for a given moving image
        and fixed image.

        Parameters
        -----------
        moving_image : ants.ANTsImage
            The image to be transformed.
        fixed_image : ants.ANTsImage
            The image that serves as the reference during the transformation.
        interpolator_type : InterpolatorType
            The type of interpolation to be used during the transformation.
            Either 'linear' or 'nearest_neighbor'.
        num_transforms : int
            The number of transformations to be applied.


        Output
        ------
        output_registered : ants.core.ants_image.ANTsImage
            The transformed image
        """
        num_slices = moving_image.view().shape[2]
        output_registered = moving_image.copy()

        # Apply transforms to each section
        for z in range(num_slices):
            input_slice = ants.from_numpy(moving_image.view()[:, :, z])
            ccf_slice = ants.from_numpy(fixed_image.view()[:, :, z])

            # Apply transforms
            if trans_type == TransformType.SYNONLY.value:
                invert_array = [False]
                transform_list = [
                    self._get_transform_path(
                        z, iteration, TransformSubType.FORWARD_DEFORMATION.value
                    )
                ]
            else:
                invert_array = [False, False]
                transform_list = [
                    self._get_transform_path(
                        z, iteration, TransformSubType.FORWARD_DEFORMATION.value
                    ),
                    self._get_transform_path(
                        z, iteration, TransformSubType.FORWARD_AFFINE.value
                    ),
                ]

            warped = ants.apply_transforms(
                moving=input_slice,
                fixed=ccf_slice,
                transformlist=transform_list,
                interpolator=interpolator_type,
                whichtoinvert=invert_array,
                verbose=False,
            )
            output_registered.view()[:, :, z] = warped.view()

        # Save and return transformed image
        ants.image_write(output_registered, output_label_path)
        return output_registered

    def transform_merfish_images(
        self, iteration: int, merfish_images: Dict, ccf_image: ants.ANTsImage
    ) -> Tuple[Dict[str, ants.ANTsImage], Dict[str, str]]:
        """Apply transform to merfish images, save transformed images to
        output_path/iter{iteration}

        Parameters:
        -----------
        iteration (int):
            The iteration number
        merfish_images (Dict):
            A dictionary containing the MERFISH images
        ccf_image (ants.ANTsImage):
            The CCF image

        Returns:
        --------
        Tuple[Dict[str, ants.ANTsImage], Dict[str, str]]:
            A tuple containing the transformed MERFISH images and the paths
            to the transformed images
        """
        trans_type, _, __ = self._get_registration_params(iteration)
        transformed_image_paths = {}
        for merfish_label_name in merfish_images.keys():
            transformed_image_paths[
                merfish_label_name
            ] = self.__get_merfish_label_output_path(iteration, merfish_label_name)
            interp_type = self.merfish_label_type_to_interpolator_type[
                merfish_label_name
            ]
            merfish_images[merfish_label_name] = self.apply_slice_transforms(
                iteration,
                merfish_images[merfish_label_name],
                ccf_image,
                interp_type,
                trans_type,
                transformed_image_paths[merfish_label_name],
            )
        return merfish_images, transformed_image_paths

    def transform_merfish_selected_image(
        self, iteration: int, selected_images: Dict
    ) -> None:
        """Apply transformations to selected labels

        Parameters:
        -----------
        iteration (int):
            The iteration number
        selected_images (Dict):
            A dictionary containing the selected images for
            registration (mFish and CCF)

        Returns:
        --------
        None
        """
        iteration_output_path = self.__get_iteration_output_path(iteration)
        selected_images_output_path = os.path.join(
            iteration_output_path, "merfish_selLabels_WarpedAllSlc.nii.gz"
        )
        trans_type, _, __ = self._get_registration_params(iteration)
        _ = self.apply_slice_transforms(
            iteration,
            selected_images["mFish"],
            selected_images["CCF"],
            InterpolatorType.NEAREST_NEIGHBOR.value,
            trans_type,
            selected_images_output_path,
        )

    def register(self) -> None:
        """Apply all iterations of registration between MERFISH images
        and CCF images and save the output to output_path/iter{iteration}
        """
        merfish_images = self._read_images(self.merfish_files)
        ccf_images = self._read_images(self.ccf_files)

        self.iteration_labels = [np.array(labels) for labels in self.iteration_labels]
        num_iter = len(self.iteration_labels)

        output_dict = {"iteration": []}

        for iteration in range(num_iter):
            logger.info(f"Running iteration {iteration}")
            iteration_output_path = os.path.join(self.output_path, f"iter{iteration}")
            os.makedirs(iteration_output_path, exist_ok=True)
            selected_images = self._select_images_and_labels(
                iteration, merfish_images, ccf_images
            )

            logger.info(f"Creating transforms for iteration {iteration}")
            self.create_transforms(iteration, selected_images)
            self.transform_merfish_selected_image(iteration, selected_images)

            logger.info(f"Transforming merfish images for iteration {iteration}")
            merfish_images, transformed_image_paths = self.transform_merfish_images(
                iteration, merfish_images, ccf_images["labels_broad"]
            )
            output_dict["iteration"].append(
                {"iter_num": iteration, "registered_files": transformed_image_paths}
            )
        return output_dict
