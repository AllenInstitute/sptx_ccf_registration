from argschema import ArgSchema
from argschema.fields import Int, List, Nested, String


class MERFISHFilesSchema(ArgSchema):
    labels_broad_segmented = String(
        required=True, description="Path to the segmented broad labels for MERFISH"
    )
    labels_landmark_segmented = String(
        required=True, description="Path to the segmented landmark labels for MERFISH"
    )
    labels_broad_unsegmented = String(
        required=True, description="Path to the unsegmented broad labels for MERFISH"
    )
    labels_landmark_unsegmented = String(
        required=True, description="Path to the unsegmented landmark labels for MERFISH"
    )
    stack = String(required=True, description="Path to the stacked MERFISH file")
    right_hemisphere = String(
        required=True, description="Path to the right hemisphere of the MERFISH data"
    )


class CCFFilesSchema(ArgSchema):
    labels_broad = String(required=True, description="Path to the broad labels for CCF")
    labels_landmark = String(
        required=True, description="Path to the landmark labels for CCF"
    )
    right_hemisphere = String(
        required=True, description="Path to the right hemisphere of the CCF data"
    )


class IterationSchema(ArgSchema):
    iter_num = Int(required=True, description="Iteration of registration")
    registered_files = Nested(MERFISHFilesSchema, description="Registered MERFISH files")


class RegistrationOutputSchema(ArgSchema):
    iteration = List(
        Nested(IterationSchema),
        description="Registered MERFISH files for each iteration",
    )


class RegistrationSchema(ArgSchema):
    output_name = String(
        required=True, description="General prefix attached to the set of experiments"
    )
    output_path = String(required=True, description="Base directory for output data")
    merfish_files = Nested(
        MERFISHFilesSchema, description="Paths to MERFISH input files"
    )
    ccf_files = Nested(CCFFilesSchema, description="Paths to CCF input files")
    labels_level = List(
        Int,
        cli_as_single_argument=True,
        required=True,
        description="Level of labels to use for registration. 0 is broad labels, 1"
        "is landmark labels",
    )
    labels_replace_to = List(
        Int,
        cli_as_single_argument=True,
        required=True,
        description="Labels to replace to. Set to 1 if merging labels, set to -1"
        "if using all labels in range as distinct labels",
    )
    iteration_labels = List(
        List(Int),
        cli_as_single_argument=True,
        required=True,
        description="Labels to use at each iteration of registration",
    )
    n_processes = Int(
        default=-1,
        description="Number of processes to use." "If -1, use all available cores.",
    )
