import json
import multiprocessing as mp
import re
import subprocess
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from sptx_ccf_registration.metrics_dashboard.metrics import (
    get_label_path,
    get_nii_path,
    metrics_df_z_slices,
    parse_itksnap_file,
)

MERFISH_SCHEMA_TO_TAG_PREFIX = {
    "labels_broad_segmented": "Broad-segmented",
    "labels_landmark_segmented": "Landmark-segmented",
    "labels_broad_unsegmented": "Broad-unsegmented",
    "labels_landmark_unsegmented": "Landmark-unsegmented",
}


def create_format_configs_from_inputs(
    ccf_files: Dict,
    label_paths: Dict,
    registration_output_json: Dict,
    config_path: Union[str, Path],
    file_prefix: str = "config_iter",
) -> List[str]:
    """
    Create a series of format configuration JSON files using the provided input.
    These configuration files are inputs for neuroglancer_interface and for
    compute_metrics

    Parameters
    ----------
    ccf_files (Dict):
        Dictionary with the paths of CCF files.
    label_paths (Dict):
        Dictionary with the paths of label files.
    registration_output_json (Dict):
        Dictionary containing registration output information.
    config_path (Union[str, Path]):
        The path where the generated configuration files should be saved.
    file_prefix (str, optional):
        The prefix of the configuration file names. Defaults to "config_iter".

    Returns:
    List[str]:
        A list of paths to the created configuration files.
    """
    num_iterations = len(registration_output_json["iteration"])
    config_file_paths = []
    for iter in range(num_iterations):
        registered_files_iter = registration_output_json["iteration"][iter][
            "registered_files"
        ]
        format_config = {}
        format_config["template"] = [
            {"tag": "MERFISH-template", "nii_path": registered_files_iter["stack"]},
            {"tag": "CCF-template", "nii_path": ccf_files["atlas"]},
        ]
        format_config["ccf"] = [
            {
                "tag": "Full-CCF-Annotation/CCF",
                "nii_path": ccf_files["full"],
                "label_path": label_paths["full"],
            },
        ]
        for merfish_schema_key, tag_prefix in MERFISH_SCHEMA_TO_TAG_PREFIX.items():
            if "broad" in tag_prefix.lower():
                label_path = label_paths["broad"]
                ccf_path = ccf_files["broad"]
            elif "landmark" in tag_prefix.lower():
                label_path = label_paths["landmark"]
                ccf_path = ccf_files["landmark"]
            format_config["ccf"].append(
                {
                    "tag": f"{tag_prefix}/MERFISH",
                    "nii_path": registered_files_iter[merfish_schema_key],
                    "label_path": label_path,
                }
            )
            format_config["ccf"].append(
                {
                    "tag": f"{tag_prefix}/CCF",
                    "nii_path": ccf_path,
                    "label_path": label_path,
                }
            )
        config_file = Path(config_path) / f"{file_prefix}{iter}.json"
        with open(config_file, "w") as f:
            json.dump(format_config, f)
        config_file_paths.append(config_file)
    return config_file_paths


def _process_iter(
    i: str, tag: Tuple, config: Dict, outpath: Union[str, Path], colnames: List[str]
) -> pd.DataFrame:
    """
    Compute metrics for a single iteration.

    Parameters
    ----------
    i (str):
        The iteration number.
    tag (Tuple):
        A tuple of the MERFISH and CCF tags.
    config (Dict):
        The configuration dictionary.
    outpath (Union[str, Path]):
        The path to the output directory.
    colnames (List[str]):
        The column names for the metrics dataframe.

    Returns
    -------
    pd.DataFrame:
        The metrics dataframe for the iteration.

    """
    mer_path = get_nii_path(tag[0], config)
    ccf_path = get_nii_path(tag[1], config)
    label_path = get_label_path(tag[0], config)
    label_map = parse_itksnap_file(label_path)

    def iter_sub(path):
        return re.sub(r"iter\d+", f"iter{i}", str(path))

    mer_path_ = Path(iter_sub(mer_path))
    ccf_path_ = Path(iter_sub(ccf_path))

    outpath_iter = outpath / f"iter{i}"
    if not outpath_iter.exists():
        outpath_iter.mkdir()
    metrics_df_iter = metrics_df_z_slices(mer_path_, ccf_path_, label_map)
    metrics_df_iter.to_csv(outpath_iter / f"{tag[0].split('/')[0]}.csv")
    colnames_iter = [colnames[3]] + colnames[5:]
    metrics_df_iter = metrics_df_iter.rename(
        columns={col: f"{col}: iter{i}" for col in colnames_iter}
    )
    return metrics_df_iter


def compute_metrics(
    config_file_paths: List[str], outpath: Union[str, Path], n_processes: int = -1
) -> None:
    """
    Compute metrics for each iteration and aggregate the results.

    Parameters
    ----------
    config_file_paths (List[str]):
        A list of paths to the configuration files.
    outpath (Union[str, Path]):
        The path to the output directory.
    cpu_count (int, optional):
    """
    num_iter = len(config_file_paths)

    with open(config_file_paths[0], "r") as f:
        config = json.load(f)

    tags = [
        (f"{tag}/MERFISH", f"{tag}/CCF") for tag in MERFISH_SCHEMA_TO_TAG_PREFIX.values()
    ]

    colnames = [
        "label",
        "structure",
        "z-slice",
        "MERFISH area (pixels)",
        "CCF area (pixels)",
        "intersection",
        "intersection / MERFISH area",
        "dice coefficient",
    ]

    if n_processes == -1:
        n_processes = mp.cpu_count()
    else:
        n_processes = min(n_processes, num_iter)
    pool = mp.Pool(n_processes)

    for tag in tags:
        process_iter_with_fixed_params = partial(
            _process_iter, tag=tag, config=config, outpath=outpath, colnames=colnames
        )

        metrics_df_iter_list = pool.map(process_iter_with_fixed_params, range(num_iter))

        for i, metrics_df_iter in enumerate(metrics_df_iter_list):
            if i == 0:
                metrics_df = metrics_df_iter.copy()
            else:
                metrics_df = pd.merge(
                    metrics_df, metrics_df_iter, on=colnames[:3] + [colnames[4]]
                )
                metrics_df[f"intersection / MERFISH area: cumulative change iter{i}"] = (
                    metrics_df[f"intersection / MERFISH area: iter{i}"]
                    - metrics_df["intersection / MERFISH area: iter0"]
                )
                metrics_df[f"dice coefficient: cumulative change iter{i}"] = (
                    metrics_df[f"dice coefficient: iter{i}"]
                    - metrics_df["dice coefficient: iter0"]
                )
        metrics_df.to_csv(outpath / f"{tag[0].split('/')[0]}_aggregate.csv")

    pool.close()


def run_neuroglancer_formatting(
    config_file_paths: List[str],
    out_path: Union[str, Path],
    tmp_dir: Union[str, Path],
    n_processors: int,
    bucket_prefix: str = "scratch/transpose",
) -> None:
    """
    Run the neuroglancer_interface registration_visualization.visualize module
    to generate neuroglancer visualizations for each iteration.

    Parameters
    ----------
    config_file_paths (List[str]):
        A list of paths to the configuration files.
    out_path (Union[str, Path]):
        The path to the output directory.
    tmp_dir (Union[str, Path]):
        The path to the temporary directory.
    n_processors (int, optional):
        The number of processors to use.
    bucket_prefix (str, optional):
        The prefix of the bucket path. Defaults to "scratch/transpose".
    """
    for iter, config_file_path in enumerate(config_file_paths):
        command = [
            "python",
            "-m",
            "neuroglancer_interface.registration_visualization.visualize",
            "--tmp_dir",
            str(tmp_dir),
            "--bucket_prefix",
            bucket_prefix,
            "--n_processors",
            str(n_processors),
            "--config_path",
            str(config_file_path),
            "--output_path",
            f"{out_path}/neuroglancer_iter{iter}.html",
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"Error running command: {' '.join(command)}")
            raise
