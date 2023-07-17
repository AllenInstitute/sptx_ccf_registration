import json
import logging
import os
from typing import Dict

from argschema import ArgSchemaParser

from sptx_ccf_registration.registration.registration import Registration
from sptx_ccf_registration.registration.schemas import (
    RegistrationOutputSchema,
    RegistrationSchema,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegistrationRunner(ArgSchemaParser):
    default_schema = RegistrationSchema
    default_output_schema = RegistrationOutputSchema

    def __save_json(self, data, path):
        """Saves data as a JSON file"""
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def __pop_arg(self, arg: str) -> Dict:
        """Pops 'log_level' from args if it exists"""
        self.args[arg].pop("log_level", None)
        return self.args[arg]

    def run(self):
        output_name = self.args["output_name"]
        output_path = self.args["output_path"]
        merfish_files = self.__pop_arg("merfish_files")
        ccf_files = self.__pop_arg("ccf_files")
        labels_level = self.args["labels_level"]
        labels_replace_to = self.args["labels_replace_to"]
        iteration_labels = self.args["iteration_labels"]
        os.makedirs(output_path, exist_ok=True)

        reg_obj = Registration(
            output_name,
            output_path,
            merfish_files,
            ccf_files,
            labels_level,
            labels_replace_to,
            iteration_labels,
        )
        output_dict = reg_obj.register()

        # save input arguments to registration_input.json
        self.__save_json(self.args, os.path.join(output_path, "registration_input.json"))

        # save output arguments to registration_output.json
        self.args["output_json"] = os.path.join(output_path, "registration_output.json")

        self.output(output_dict)


if __name__ == "__main__":
    reg = RegistrationRunner()
    reg.run()
