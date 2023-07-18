import json
import logging
import os
import shutil
from pathlib import Path

from argschema import ArgSchemaParser

from sptx_ccf_registration.metrics_dashboard.dashboard import (
    compute_metrics,
    create_format_configs_from_inputs,
    run_neuroglancer_formatting,
)
from sptx_ccf_registration.metrics_dashboard.schemas import (
    GenerateMetricsDashboardOutputSchema,
    GenerateMetricsDashboardSchema,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateMetricsDashboard(ArgSchemaParser):
    """
    Class to generate metrics dashboard.
    """

    default_schema = GenerateMetricsDashboardSchema
    default_output_schema = GenerateMetricsDashboardOutputSchema

    def load_json(self, path):
        """Loads data from a JSON file"""
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def run(self):
        """
        Run the process to generate metrics dashboard.
        """

        # Get input arguments
        registration_output_json = self.load_json(self.args["registration_output_json"])
        ccf_files = self.args["ccf_files"]
        out_path = Path(self.args["out_path"])
        label_paths = self.args["label_paths"]
        tmp_dir = Path(self.args["tmp_dir"])
        n_processes = self.args["n_processes"]

        if n_processes == -1:
            n_processes = os.cpu_count()
        if (
            os.environ.get("AWS_ACCESS_KEY_ID") is None
            or os.environ.get("AWS_SECRET_ACCESS_KEY") is None
        ):
            raise ValueError(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID"
                " and AWS_SECRET_ACCESS_KEY environment variables. See"
                " https://docs.aws.amazon.com/cli/latest/userguide/"
                "cli-configure-envvars.html for more information."
            )

        os.makedirs(out_path, exist_ok=True)

        dashboard_path = out_path / "ccf_merfish_metrics"
        shutil.rmtree(dashboard_path, ignore_errors=True)
        shutil.copytree(Path(__file__).parent / "ccf_merfish_metrics", dashboard_path)

        config_path = out_path / "config"
        os.makedirs(config_path, exist_ok=True)

        data_path = dashboard_path / "data"
        os.makedirs(data_path, exist_ok=True)

        os.makedirs(tmp_dir, exist_ok=True)

        neuroglancer_path = data_path / "neuroglancer"
        os.makedirs(neuroglancer_path, exist_ok=True)

        config_file_paths = create_format_configs_from_inputs(
            ccf_files,
            label_paths,
            registration_output_json,
            config_path,
        )

        compute_metrics(config_file_paths, data_path)

        run_neuroglancer_formatting(
            config_file_paths, neuroglancer_path, tmp_dir, n_processes
        )

        self.args["output_json"] = str(out_path / "dashboard_output.json")
        output = {"dashboard_path": str(dashboard_path)}
        self.output(output, indent=4)

        input_json_path = out_path / "dashboard_input.json"
        with open(input_json_path, "w") as f:
            json.dump(self.args, f, indent=4)


if __name__ == "__main__":
    metrics = GenerateMetricsDashboard()
    metrics.run()
