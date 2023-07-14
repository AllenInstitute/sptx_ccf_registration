import logging
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_closing, disk

from sptx_ccf_registration.segmentation.utils import (
    get_alpha_range,
    label_points_to_binary_mask,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentSlice:
    def __init__(
        self,
        unsegmented_labels_slice: np.ndarray,
        z: int,
        min_points: int = 5,
        default_alpha: int = 0.2,
        optimize_alpha=False,
        ccf_slice: np.ndarray = None,
        min_alpha: float = 0.04,
        max_alpha: float = 0.5,
        alpha_selection: Dict = None,
        force_binary_closing: bool = False,
        radius: int = 5,
        sigma: int = 5,
    ):
        """
        Parameters
        ----------
        unsegmented_labels_slice : np.ndarray
            2D array representing a slice of unsegmented labels.

        z : int
            Z coordinate for the slice.

        min_points : int, optional
            Minimum number of points for a label to be considered, by default 5.

        default_alpha : int, optional
            Default alpha value, by default 0.2.

        optimize_alpha : bool, optional
            Flag to optimize alpha value, by default False.

        ccf_slice : np.ndarray, optional
            Slice from the CCF volume, must be the same shape as
            unsegmented_labels_slice, by default None.

        min_alpha : float, optional
            Minimum alpha value, by default 0.04.

        max_alpha : float, optional
            Maximum alpha value, by default 0.5.

        alpha_selection : Dict, optional
            Pre-computed dictionary mapping labels to alpha values, by default None.

        force_binary_closing : bool, optional
            Flag to force binary closing on a label, by default False.

        radius : int, optional
            Radius for the structuring element used in binary closing, by default 5.

        sigma : int, optional
            Sigma value for the Gaussian filter used in KDE, by default 5.
        """
        if not len(unsegmented_labels_slice.shape) == 2:
            raise ValueError("unsegmented_labels_slice must be 2D")

        self.unsegmented_labels_slice = unsegmented_labels_slice
        self.y_dim, self.x_dim = unsegmented_labels_slice.shape
        self.points = np.array(np.nonzero(unsegmented_labels_slice)).T
        self.labels = unsegmented_labels_slice[unsegmented_labels_slice != 0].astype(int)
        self.alpha_selection = alpha_selection
        self.alpha_range = get_alpha_range(min_alpha, max_alpha)
        self.sigma = sigma
        self._unique_labels = None
        self._unique_labels_ccf = None
        self._z_label_to_alpha = None
        self._max_density_labels = None
        self.default_alpha = default_alpha
        self.optimize_alpha = optimize_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_points = min_points
        self._segmented_labels_slice = None
        self.force_binary_closing = force_binary_closing
        self.radius = radius
        self.z = z

        if ccf_slice is not None:
            if ccf_slice.shape != unsegmented_labels_slice.shape:
                raise ValueError(
                    "ccf_slice must be the same shape as unsegmented_labels_slice"
                )
            self.ccf_slice = ccf_slice
            self.max_label = unsegmented_labels_slice.max()
        if alpha_selection is None:
            self.alpha_selection = {}
        else:
            self.alpha_selection = alpha_selection

    @property
    def unique_labels(self) -> np.ndarray:
        """
        Get unique labels, ignoring 0 and len(label) < min_points
        """
        if self._unique_labels is None:
            unique_labels = np.unique(self.labels[self.labels != 0])
            self._unique_labels = [
                label
                for label in unique_labels
                if len(self.labels[self.labels == label]) > self.min_points
            ]
        return self._unique_labels

    @property
    def unique_labels_ccf(self) -> np.ndarray:
        if self._unique_labels_ccf is None and self.ccf_slice is not None:
            self._unique_labels_ccf = np.unique(self.ccf_slice[self.ccf_slice != 0])
        return self._unique_labels_ccf

    def find_optimal_alpha(self, label: int) -> float:
        """
        Find the optimal alpha for a given label by minimizing the
        difference between the segmented area and the CCF area.

        Parameters
        ----------
        label : int
            The label to find the optimal alpha for.

        Returns
        -------
        float
            The optimal alpha value.
        """
        min_difference = float("inf")
        ccf_area = np.sum(self.ccf_slice == label)
        best_alpha = self.default_alpha
        if ccf_area != 0:
            label_points = self.points[self.labels == label]
            for alpha in self.alpha_range:
                mask = label_points_to_binary_mask(
                    label_points, alpha, self.y_dim, self.x_dim
                )
                if mask is not None:
                    area = mask.sum()
                    difference = abs(area - ccf_area)
                    if difference < min_difference:
                        min_difference = difference
                        best_alpha = alpha
        return best_alpha

    @property
    def z_label_to_alpha(self) -> Dict:
        """
        Create a dictionary mapping (z, label) to alpha value.
        If alpha_selection is provided, use the alpha value from alpha_selection.
        If optimize_alpha is True, find the optimal alpha value for each label.
        If force_binary_closing is True, use "dilated" as the alpha value
        for each label.
        If alpha_selection is not providedor label is not found in the
        respective ccf slice, use default_alpha as the alpha value for each label.

        Returns
        -------
        Dict
            Dictionary mapping (z, label) to alpha value.
        """
        if self._z_label_to_alpha is None:
            self._z_label_to_alpha = {}

            for label in self.unique_labels:
                best_alpha = self.default_alpha
                if self.force_binary_closing:
                    best_alpha = "dilated"
                elif self.alpha_selection is not None:
                    if self.alpha_selection.get(str((self.z, label))):
                        best_alpha = self.alpha_selection[str((self.z, label))]
                else:
                    best_alpha = self.default_alpha
                    if self.optimize_alpha:
                        best_alpha = self.find_optimal_alpha(label)
                self._z_label_to_alpha[str((self.z, label))] = best_alpha
        return self._z_label_to_alpha

    @property
    def max_density_labels(self) -> np.ndarray:
        """
        Max density labels for each pixel in the 2D slice.
        Compute the KDE (unormalized) for the masks of each label by convolving a
        gaussian kernel of sigma.
        The output of the KDE should be an array of equal size of the mask array.
        Stack the KDE masks and take the argmax across the stack.
        Convert each argmax index to the label value with a dictionary mapping.

        Returns
        -------
        numpy.ndarray
            The 2D slice of integer labels (w x h) where each pixel indicates the
            label with the max density at that pixel
        """

        if self._max_density_labels is None:
            kde_stack = []
            for label in self.unique_labels:
                mask = (self.unsegmented_labels_slice == label).astype(
                    float
                )  # Convert to float
                kde = gaussian_filter(mask, sigma=self.sigma)
                kde_stack.append(kde)

            kde_stack = np.stack(kde_stack)
            kde_stack_argmax = np.argmax(kde_stack, axis=0)
            kde_stack_max = np.max(kde_stack, axis=0)
            zero_density_positions = kde_stack_max == 0
            kde_stack_argmax[zero_density_positions] = -1
            label_map = {i: label for i, label in enumerate(self.unique_labels)}
            label_map[-1] = 0
            self._max_density_labels = np.vectorize(label_map.get)(kde_stack_argmax)

        return self._max_density_labels

    @property
    def segmented_labels_slice(self) -> np.ndarray:
        """
        Segment the 2D slice by concave hull and/or binary_closing for each label.

        Returns
        -------
        numpy.ndarray
            The 2D slice of segmented integer labels (w x h)
        """
        if self._segmented_labels_slice is None:
            self._segmented_labels_slice = np.zeros_like(self.unsegmented_labels_slice)
            for label in self.unique_labels:
                label_points = self.points[self.labels == label]
                alpha = self.z_label_to_alpha[str((self.z, label))]
                if alpha == "dilated":
                    mask = binary_closing(
                        self.unsegmented_labels_slice == label, disk(self.radius)
                    )
                else:
                    mask = label_points_to_binary_mask(
                        label_points, alpha, self.y_dim, self.x_dim
                    )
                if mask is not None:
                    not_overlap_mask = (mask == 1) & (self._segmented_labels_slice == 0)
                    overlap_max_density_mask = (
                        (mask == 1)
                        & (self._segmented_labels_slice != 0)
                        & (self.max_density_labels == label)
                    )
                    self._segmented_labels_slice[not_overlap_mask] = label
                    self._segmented_labels_slice[overlap_max_density_mask] = label
        return self._segmented_labels_slice
