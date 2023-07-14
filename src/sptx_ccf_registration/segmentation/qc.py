import logging
from pathlib import Path
from typing import Dict, List, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sptx_ccf_registration.segmentation.segment import SegmentSlice
from sptx_ccf_registration.utils.file_processing import alpha_to_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentSliceQC:
    def __init__(
        self,
        segment_slice: SegmentSlice,
        label_map: Dict = None,
        cmap="rainbow",
        seed=2021,
        cache_dir: Union[Path, str] = None,
    ):
        self.segment_slice = segment_slice
        self.z = self.segment_slice.z
        self.label_map = label_map
        self.mpl_cmap = cmap
        self._cmap = None
        self._legend_patches = None
        self._dilated_labels_slice = None
        self._concave_hull_qc_slices = None
        self.cache_dir = cache_dir
        self.rng = np.random.default_rng(seed)

    @property
    def dilated_labels_slice(self) -> np.ndarray:
        if self._dilated_labels_slice is None:
            cache_path = Path(self.cache_dir) / "segment_slice_qc_alpha_dilated.nii.gz"
            if cache_path.exists():
                self.self._dilated_labels_slice = nib.load(cache_path).get_fdata()[
                    :, :, self.z
                ]
            else:
                _, seg_slice = SegmentSliceQC._process_segment_qc_slice(
                    "dilated", self.z, self.segment_slice.unsegmented_labels_slice
                )
                self._dilated_labels_slice = seg_slice
        return self._dilated_labels_slice

    @property
    def concave_hull_qc_slices(self) -> dict:
        if self._concave_hull_qc_slices is None:
            self._concave_hull_qc_slices = {}
            for alpha in self.segment_slice.alpha_range:
                cache_path = (
                    Path(self.cache_dir)
                    / f"segment_slice_qc_alpha_{alpha_to_str(alpha)}.nii.gz"
                )
                if cache_path.exists():
                    self.concave_hull_qc_slices[alpha] = nib.load(
                        cache_path
                    ).get_fdata()[:, :, self.z]
                else:
                    _, seg_slice = SegmentSliceQC._process_segment_qc_slice(
                        alpha, self.z, self.segment_slice.unsegmented_labels_slice
                    )
                    self._concave_hull_qc_slices[alpha] = seg_slice
        return self._concave_hull_qc_slices

    @property
    def cmap(self) -> matplotlib.colors.LinearSegmentedColormap:
        """Create a discrete colormap with the same number of colors as
        the number of labels
        """
        if self._cmap is None:
            N = len(self.label_map.keys())
            cmap = plt.get_cmap(self.mpl_cmap, N)
            newcolors = cmap(np.linspace(0, 1, N))
            self.rng.shuffle(newcolors)
            newcolors[0, :] = np.array([0, 0, 0, 1])
            self._cmap = ListedColormap(newcolors)
        return self._cmap

    @property
    def legend_patches(self) -> List[Patch]:
        if self._legend_patches is None:
            self._legend_patches = []
            for lbl in self.segment_slice.unique_labels:
                best_alpha_for_label = self.segment_slice.z_label_to_alpha[
                    str((self.z, lbl))
                ]
                self._legend_patches.append(
                    mpatches.Patch(
                        color=self.cmap(lbl / self.segment_slice.max_label),
                        label=f"{lbl} - {self.label_map[lbl]}, α = {best_alpha_for_label}",  # noqa E501
                    )
                )
            for lbl in np.unique(self.segment_slice.unique_labels_ccf):
                if lbl not in self.segment_slice.unique_labels:
                    self._legend_patches.append(
                        mpatches.Patch(
                            color=self.cmap(lbl / self.segment_slice.max_label),
                            label=f"{lbl} - {self.label_map[lbl]}",
                        )
                    )
        return self._legend_patches

    @staticmethod
    def _process_segment_qc_slice(alpha, z, unsegmented_label):
        if alpha == "dilated":
            seg_obj = SegmentSlice(unsegmented_label, z, force_binary_closing=True)
        else:
            seg_obj = SegmentSlice(unsegmented_label, z, default_alpha=alpha)
        return z, seg_obj.segmented_labels_slice

    def plot_alpha_qc(self):
        num_plots = 4 + len(self.segment_slice.alpha_range)

        # calculate nrows and ncols such that their difference is minimal
        ncols = int(num_plots**0.5)
        nrows = ncols
        while num_plots % nrows != 0:
            nrows -= 1
        ncols = num_plots // nrows

        plt.rcParams["figure.figsize"] = [ncols * 6, nrows * 6]

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        axs = axs.flatten()

        data_list = [
            np.zeros_like(self.segment_slice.unsegmented_labels_slice),
            self.dilated_labels_slice.T,
            self.segment_slice.segmented_labels_slice.T,
            self.segment_slice.ccf_slice.T,
        ] + [
            self.concave_hull_qc_slices[alpha].T
            for alpha in self.segment_slice.alpha_range
        ]

        title_list = ["Unsegmented", "Dilated", "Per label optimization", "CCF"] + [
            f"alpha={alpha}" for alpha in self.segment_slice.alpha_range
        ]

        for i, ax in enumerate(axs[:num_plots]):
            if i == 0:
                ax.imshow(data_list[i], cmap=self.cmap)
                ax.scatter(
                    self.segment_slice.points[:, 0],
                    self.segment_slice.points[:, 1],
                    color=self.cmap(
                        self.segment_slice.labels / self.segment_slice.max_label
                    ),
                    s=0.3,
                    alpha=1,
                )
            else:
                ax.imshow(
                    data_list[i],
                    cmap=self.cmap,
                    vmax=self.segment_slice.max_label,
                    interpolation="none",
                )

            ax.set_title(title_list[i])
            ax.set_xticks([])
            ax.set_yticks([])

        fig.legend(
            handles=self.legend_patches, bbox_to_anchor=(0.9, 0.85), loc="upper left"
        )

        return fig
