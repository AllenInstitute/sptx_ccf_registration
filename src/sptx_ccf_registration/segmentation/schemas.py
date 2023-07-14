from argschema import ArgSchema
from argschema.fields import Bool, Float, Int, Nested, String


class InputPaths(ArgSchema):
    unsegmented_label_file = String(required=True, description="Input file.")
    ccf_file = String(required=False, description="Path of the CCF file.")
    itksnap_file_path = String(required=False, description="Path of the itksnap file.")
    alpha_selection_path = String(
        required=False, description="Path of the alpha selection file."
    )


class SegmentationSchemaOutput(ArgSchema):
    segmented_label_output_file = String(required=True, description="Output file.")
    selected_alpha = String(
        required=False,
        description="Path to csv file to store alpha values used for segmentation.",
    )
    alpha_qc_dir = String(required=False, description="Path to alpha_qc directory.")


class SegmentationSchema(ArgSchema):
    input_paths = Nested(InputPaths, required=True, description="Input files.")
    output_dir = String(required=True, description="Output directory.")
    segmented_label_output_file = String(required=True, description="Output file.")
    default_alpha = Float(
        default=0.2,
        description="Default concave hull alpha value if alpha is not selected"
        "through optimize_alpha or alpha_selection_path",
    )
    optimize_alpha = Bool(
        default=False,
        description="Optimize concave hull alpha parameter for each"
        "(z-slice, label) by minimizing difference in area with respective CCF.",
    )
    min_alpha = Float(
        default=0.04, description="Lower boundary of search in optimize_alpha."
    )
    max_alpha = Float(
        default=0.45, description="Upper boundary of search in optimize_alpha."
    )
    min_points = Int(
        default=10,
        description="Minimum number of points in a label required for segmentation.",
    )
    sigma = Int(
        default=5,
        description="Sigma parameter used for gaussian smoothing to estimate density"
        "of each label. For overlapping labels, the densest estimated label is"
        "selected.",
    )
    radius = Int(
        default=5,
        description="Radius parameter used for binary closing to dilate labels.",
    )
    save_alpha_qc = Bool(default=False, description="Whether to save alpha QC images.")
    force_binary_closing = Bool(
        default=False,
        description="If true, all segmentation will be done using binary_closing instead"
        "of concave hull. All concave hull related parameters will be ignored",
    )
    seed = Int(default=2021, description="Seed for random number generator.")
