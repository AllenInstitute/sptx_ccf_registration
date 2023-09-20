import json
import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from argschema import ArgSchemaParser

from sptx_ccf_registration.segmentation.qc import SegmentSliceQC
from sptx_ccf_registration.segmentation.schemas import (
    SegmentationSchema,
    SegmentationSchemaOutput,
)
from sptx_ccf_registration.segmentation.segment import SegmentSlice, get_alpha_range
from sptx_ccf_registration.utils.file_processing import alpha_to_str, parse_itksnap_file

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Segmentation(ArgSchemaParser):
    default_schema = SegmentationSchema
    default_output_schema = SegmentationSchemaOutput

    @staticmethod
    def process_slice(
        z,
        unsegmented_data,
        ccf_img,
        min_points,
        default_alpha,
        optimize_alpha,
        min_alpha,
        max_alpha,
        alpha_selection,
        force_binary_closing,
        radius,
        sigma,
        save_selected_alpha,
        save_alpha_qc,
        label_map,
        cache_dir,
        alpha_qc_dir,
        seed,
    ):
        z_dim = unsegmented_data.shape[2]
        logger.info(f"Processing z-slice: {z}/{z_dim}")
        # Get 2D slice
        unsegmented_labels_slice = unsegmented_data[:, :, z]
        if unsegmented_labels_slice.sum() == 0:
            return z, None, None

        if ccf_img is not None:
            ccf_slice = ccf_img[:, :, z]
        else:
            ccf_slice = None

        segment_slice_obj = SegmentSlice(
            unsegmented_labels_slice,
            z,
            min_points,
            default_alpha,
            optimize_alpha,
            ccf_slice,
            min_alpha,
            max_alpha,
            alpha_selection,
            force_binary_closing,
            radius,
            sigma,
        )

        if save_alpha_qc:
            segment_slice_qc_obj = SegmentSliceQC(
                segment_slice_obj, label_map=label_map, cache_dir=cache_dir, seed=seed
            )
            fig = segment_slice_qc_obj.plot_alpha_qc()
            fig.savefig(alpha_qc_dir / f"alpha_qc_{segment_slice_obj.z}.png")
        if save_selected_alpha:
            selected_alpha_slice = segment_slice_obj.z_label_to_alpha
        else:
            selected_alpha_slice = {}

        return z, segment_slice_obj.segmented_labels_slice, selected_alpha_slice

    @staticmethod
    def create_segment_qc_cache(
        unsegmented_label, alpha_range, cache_dir, n_processes: int = cpu_count()
    ):
        logger.info(f"Creating segment QC cache into {cache_dir}")
        z_dim = unsegmented_label.shape[2]
        segmented_labels = np.zeros_like(unsegmented_label)
        for alpha in list(alpha_range) + ["dilated"]:
            if alpha == "dilated":
                logger.info("Creating segment QC cache for dilated")
                cache_path = Path(cache_dir) / "segment_slice_qc_dilated.nii.gz"
            else:
                logger.info(f"Creating segment QC cache for alpha {alpha_to_str(alpha)}")
                cache_path = (
                    Path(cache_dir)
                    / f"segment_slice_qc_alpha_{alpha_to_str(alpha)}.nii.gz"
                )
            if cache_path.exists():
                continue
            with Pool(processes=n_processes) as pool:
                results = [
                    pool.apply_async(
                        SegmentSliceQC._process_segment_qc_slice,
                        (alpha, z, unsegmented_label[:, :, z]),
                    )
                    for z in range(z_dim)
                ]
                for res in results:
                    z, slice_result = res.get()
                    segmented_labels[:, :, z] = slice_result

            nib.save(nib.Nifti1Image(segmented_labels, np.eye(4)), cache_path)

    def run(self):
        # Get input arguments
        output_dir = Path(self.args["output_dir"])
        segmented_label_output_file = self.args["segmented_label_output_file"]
        segmented_label_prefix = segmented_label_output_file.split(".")[0]
        itksnap_file_path = self.args["input_paths"].get("itksnap_file_path", None)
        unsegmented_label_file = Path(self.args["input_paths"]["unsegmented_label_file"])
        ccf_file = self.args["input_paths"].get("ccf_file", None)
        alpha_selection_path = self.args["input_paths"].get("alpha_selection_path", None)
        min_points = self.args["min_points"]
        default_alpha = self.args["default_alpha"]
        optimize_alpha = self.args["optimize_alpha"]
        radius = self.args["radius"]
        sigma = self.args["sigma"]
        min_alpha = self.args["min_alpha"]
        max_alpha = self.args["max_alpha"]
        save_alpha_qc = self.args["save_alpha_qc"]
        force_binary_closing = self.args["force_binary_closing"]
        seed = self.args["seed"]
        n_processes = self.args["n_processes"]

        if n_processes == -1:
            n_processes = cpu_count()

        output_dir.mkdir(exist_ok=True)

        if ccf_file is not None:
            ccf_file = Path(ccf_file)
            ccf_img = nib.load(ccf_file).get_fdata()
        else:
            ccf_img = None

        if itksnap_file_path is not None:
            label_map = parse_itksnap_file(itksnap_file_path)
        else:
            label_map = None

        if alpha_selection_path is not None:
            with open(alpha_selection_path, "r") as f:
                alpha_selection = json.load(f)
        else:
            alpha_selection = None

        unsegmented_img = ants.image_read(str(unsegmented_label_file))
        unsegmented_data = unsegmented_img.view()

        segmented_labels = unsegmented_img.copy()
        segmented_labels.view()[:] = 0

        save_selected_alpha = optimize_alpha is not False or alpha_selection is not None
        selected_alpha = {}

        z_dim = unsegmented_data.shape[2]

        pool = Pool(processes=n_processes)

        results = []

        if save_alpha_qc:
            alpha_qc_dependencies = ["ccf_file", "itksnap_file_path"]
            if None in alpha_qc_dependencies:
                raise ValueError(
                    "save_alpha_qc requires ccf_file and itksnap_file_path"
                    "to be specified."
                )
            alpha_qc_dir = output_dir / "alpha_qc"
            alpha_qc_dir.mkdir(exist_ok=True)
            cache_dir = alpha_qc_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            alpha_range = get_alpha_range(min_alpha, max_alpha)
            Segmentation.create_segment_qc_cache(
                unsegmented_data, alpha_range, cache_dir
            )
        else:
            alpha_qc_dir = None
            cache_dir = None

        for z in range(z_dim):
            result = pool.apply_async(
                Segmentation.process_slice,
                (
                    z,
                    unsegmented_data,
                    ccf_img,
                    min_points,
                    default_alpha,
                    optimize_alpha,
                    min_alpha,
                    max_alpha,
                    alpha_selection,
                    force_binary_closing,
                    radius,
                    sigma,
                    save_selected_alpha,
                    save_alpha_qc,
                    label_map,
                    cache_dir,
                    alpha_qc_dir,
                    seed,
                ),
            )
            results.append(result)

        for result in results:
            (
                z,
                labels_slice,
                selected_alpha_slice,
            ) = result.get()  # this line will block until the result is ready
            if labels_slice is not None:
                segmented_labels.view()[:, :, z] = labels_slice
            if selected_alpha_slice is not None:
                selected_alpha.update(selected_alpha_slice)

        pool.close()
        pool.join()

        # Save segmented labels
        output_path = str(output_dir / segmented_label_output_file)
        ants.image_write(
            segmented_labels,
            output_path,
        )
        output_dict = {"segmented_label_output_file": output_path}
        if save_selected_alpha:
            # Save selected alpha values for each (z, label)
            selected_alpha_path = (
                output_dir / f"selected_alpha_{segmented_label_prefix}.csv"
            )
            with open(selected_alpha_path, "w") as f:
                json.dump(selected_alpha, f, indent=2)

            output_dict["selected_alpha"] = str(selected_alpha_path)

        if save_alpha_qc:
            output_dict["alpha_qc_dir"] = str(alpha_qc_dir)

        # save input arguments to segmentation_input.json
        file_prefix = self.args["input_json"].split("/")[-1].split(".")[0]
        input_json_path = output_dir / f"{file_prefix}_input.json"
        with open(input_json_path, "w") as f:
            json.dump(self.args, f, indent=4)

        # save output arguments to registration_output.json
        self.args["output_json"] = output_dir / f"{file_prefix}_output.json"
        self.output(output_dict, indent=4)


if __name__ == "__main__":
    seg = Segmentation()
    seg.run()
