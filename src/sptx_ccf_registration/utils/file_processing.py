import numpy as np


def alpha_to_str(alpha: float, fill_length: int = 4) -> str:
    """
    Convert alpha value to string representation with 2 decimal places.
    Fill string value with trailing zeros to ensure length of n.

    Parameters
    ----------
    alpha : float
        Alpha value to be converted.
    fill_length : int, optional
        Length of the string, by default 4.

    Returns
    -------
    str
        String representation of alpha value.
    """
    return str(np.round(alpha, 2)).ljust(fill_length, "0")


def parse_itksnap_file(label_map_path):
    """Create a label to section name mapping dictionary"""
    with open(label_map_path) as f:
        return {int(line.split(" ")[0]): line.split('"')[-2] for line in f.readlines()}
