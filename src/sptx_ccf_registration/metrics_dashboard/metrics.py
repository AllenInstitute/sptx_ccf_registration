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
) -> pd.DataFrame:
    """Generate a dataframe of the overlap metrics for each section.

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
    img_mer = mer if isinstance(mer, np.ndarray) else load_nii_gz_image(mer)
    img_ccf = ccf if isinstance(ccf, np.ndarray) else load_nii_gz_image(ccf)

    data = []

    for z in range(img_mer.shape[-1]):
        z_mer = img_mer[:, :, z]
        z_ccf = img_ccf[:, :, z]
        if z_mer.sum() > 0:
            label_set = np.unique(z_mer[z_mer != 0])
            for label in label_set:
                metrics = overlap_metrics(z_mer == label, z_ccf == label)
                data.append([label, label_map[label], z, *metrics])

    df = pd.DataFrame(
        data,
        columns=[
            "label",
            "structure",
            "z-slice",
            "MERFISH area (pixels)",
            "CCF area (pixels)",
            "intersection",
            "intersection / MERFISH area",
            "dice coefficient",
        ],
    )

    return df


# Generate overlap_metrics output
def get_nii_path(tag: str, config: dict) -> str:
    """
    Get the path to the nii file for a given tag.

    Parameters
    ----------
    tag : str
        The tag for the nii file.
    config : dict
        The config dictionary.

    Returns
    -------
    str
        The path to the nii file.
    """
    for dat in config["ccf"]:
        if dat["tag"] == tag:
            return dat["nii_path"]


def get_label_path(tag: str, config: dict) -> str:
    """
    Get the path to the itksnap label file for a given tag.

    Parameters
    ----------
    tag : str
        The tag for the nii file.
    config : dict
        The config dictionary.

    Returns
    -------
    str
        The path to the itksnap label file
    """
    for dat in config["ccf"]:
        if dat["tag"] == tag:
            return dat["label_path"]
