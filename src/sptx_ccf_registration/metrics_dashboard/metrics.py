from pathlib import Path
from typing import Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd


def load_nii_gz_image(file_path: Union[str, Path]) -> np.ndarray:
    """Load a .nii.gz image into a numpy array.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the .nii.gz file.

    Returns
    -------
    numpy.ndarray
        Numpy array containing the image data.
    """
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()
    return img_data.astype("uint8")


def masks_intersection(binary_mask1: np.ndarray, binary_mask2: np.ndarray) -> int:
    """Compute the intersection between two binary mask arrays.

    Parameters
    ----------
    binary_mask1 : np.ndarray
        The first binary mask.
    binary_mask2 : np.ndarray
        The second binary mask.

    Returns
    -------
    int
        The number of pixels in the intersection of the two binary masks.
    """
    return np.sum(binary_mask1 * binary_mask2)


def dice_coefficient(binary_mask1: np.ndarray, binary_mask2: np.ndarray) -> float:
    """Compute the Dice coefficient between two binary mask arrays.

    Parameters
    ----------
    binary_mask1 : np.ndarray
        The first binary mask.
    binary_mask2 : np.ndarray
        The second binary mask.

    Returns
    -------
    float
        The Dice coefficient between the two binary mask arrays.

    """
    if binary_mask1.shape != binary_mask2.shape:
        raise ValueError(
            "Shape mismatch: binary_mask1 and binary_mask2 must have the same shape."
        )

    intersection = masks_intersection(binary_mask1, binary_mask2)
    union = np.sum(binary_mask1) + np.sum(binary_mask2)

    if union == 0:
        return np.nan
    else:
        dice_coeff = 2 * intersection / union
        return dice_coeff


def overlap_metrics(binary_mask1: np.ndarray, binary_mask2: np.ndarray) -> Tuple:
    """
    Compute the intersection, area of mask1, area of mask2, fraction of mask1
    that is intersected by mask2, and the Dice coefficient between two binary
    mask arrays.

    Parameters
    ----------
    binary_mask1 : np.ndarray
        The first binary mask.
    binary_mask2 : np.ndarray
        The second binary mask.

    Returns
    -------
    Tuple
        A tuple containing the intersection, area of mask1, area of mask2,
        fraction of mask1 that is intersected by mask2, and the Dice
        coefficient between two binary mask arrays.

    """
    if binary_mask1.shape != binary_mask2.shape:
        raise ValueError(
            "Shape mismatch: binary_mask1 and binary_mask2 must have the same shape."
        )

    intersection = masks_intersection(binary_mask1, binary_mask2)
    area_mask1 = np.sum(binary_mask1)
    area_mask2 = np.sum(binary_mask2)
    union = area_mask1 + area_mask2

    if area_mask1 == 0:
        return np.nan()
    else:
        fraction_intersect_mask1 = intersection / area_mask1
    if union == 0:
        return np.nan()
    else:
        dice_coeff = 2 * intersection / union
        return intersection, area_mask1, area_mask2, fraction_intersect_mask1, dice_coeff


def parse_itksnap_file(label_map_path: Union[str, Path]) -> dict:
    """Create a label to section name mapping dictionary

    Parameters
    ----------
    label_map_path : Union[str, Path]
        Path to the label map file.

    Returns
    -------
    dict
        A dictionary mapping label to section name.
    """
    with open(label_map_path) as f:
        return {int(line.split(" ")[0]): line.split('"')[-2] for line in f.readlines()}


def metrics_df_z_slices(
    mer: Union[str, Path, np.ndarray],
    ccf: Union[str, Path, np.ndarray],
    label_map: dict,
):
    """Generate a dataframe of the overlap metrics for each section

    Parameters
    ----------
    mer : Union[str, Path, np.ndarray]
        Path to the MERFISH image or the MERFISH image array.
    ccf : Union[str, Path, np.ndarray]
        Path to the CCF image or the CCF image array.
    label_map : dict
        A dictionary mapping label to section name.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the overlap metrics for each section.
    """
    if isinstance(mer, np.ndarray):
        img_mer = mer
        img_ccf = ccf
    else:
        img_mer = load_nii_gz_image(mer)
        img_ccf = load_nii_gz_image(ccf)
    labels = []
    z_slices = []
    dice_scores = []
    area_mer = []
    area_ccf = []
    intersection = []
    fraction_intersect = []
    for z in range(img_mer.shape[-1]):
        z_mer = img_mer[:, :, z]
        z_ccf = img_ccf[:, :, z]
        if z_mer.sum() > 0:
            label_set = np.unique(z_mer[z_mer != 0])
            labels.extend([label for label in label_set])
            z_slices.extend([z for _ in label_set])
            metrics = [
                overlap_metrics(z_mer == label, z_ccf == label) for label in label_set
            ]
            intersection.extend([metric[0] for metric in metrics])
            area_mer.extend([metric[1] for metric in metrics])
            area_ccf.extend([metric[2] for metric in metrics])
            fraction_intersect.extend([metric[3] for metric in metrics])
            dice_scores.extend([metric[4] for metric in metrics])

    section_set = [label_map[label] for label in labels]

    df = pd.DataFrame(
        {
            "label": labels,
            "structure": section_set,
            "z-slice": z_slices,
            "MERFISH area (pixels)": area_mer,
            "CCF area (pixels)": area_ccf,
            "intersection": intersection,
            "intersection / MERFISH area": fraction_intersect,
            "dice coefficient": dice_scores,
        }
    )
    return df


def coarse_fine_metric_comparison(
    df_coarse_metrics: pd.DataFrame,
    df_fine_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a dataframe comparing the coarse and fine overlap metrics

    Parameters
    ----------
    df_coarse_metrics : pd.DataFrame
        A dataframe containing the coarse overlap metrics.
    df_fine_metrics : pd.DataFrame
        A dataframe containing the fine overlap metrics.

    Returns
    -------
    pd.DataFrame
        A dataframe comparing the coarse and fine overlap metrics.
    """

    df = df_coarse_metrics.copy()
    df = df.rename(
        columns={
            "dice coefficient": "dice: coarse",
            "MERFISH area (pixels)": "MERFISH area (pixels): coarse",
            "intersection": "intersection: coarse",
            "intersection / MERFISH area": "intersection / MERFISH area: coarse",
        }
    )
    df = pd.merge(
        df, df_fine_metrics, on=["label", "structure", "z-slice", "CCF area (pixels)"]
    )
    df = df.rename(
        columns={
            "dice coefficient": "dice: fine",
            "MERFISH area (pixels)": "MERFISH area (pixels): fine",
            "intersection": "intersection: fine",
            "intersection / MERFISH area": "intersection / MERFISH area: fine",
        }
    )
    df["intersection / MERFISH area: delta"] = (
        df["intersection / MERFISH area: fine"]
        - df["intersection / MERFISH area: coarse"]
    )
    df["intersection / MERFISH area: % change"] = (
        np.round(
            df["intersection / MERFISH area: delta"]
            / df["intersection / MERFISH area: coarse"],
            5,
        )
        * 100
    )
    df["dice: delta"] = df["dice: fine"] - df["dice: coarse"]
    df["dice: % change"] = (
        np.round(((df["dice: fine"] - df["dice: coarse"]) / df["dice: coarse"]), 5) * 100
    )
    df = df.iloc[:, [0, 1, 2, 4, 3, 8, 5, 9, 6, 10, 12, 13, 7, 11, 14, 15]]
    return df


# Generate overlap_metrics output
def get_nii_path(tag, config):
    for dat in config["ccf"]:
        if dat["tag"] == tag:
            return dat["nii_path"]


def get_label_path(tag, config):
    for dat in config["ccf"]:
        if dat["tag"] == tag:
            return dat["label_path"]
