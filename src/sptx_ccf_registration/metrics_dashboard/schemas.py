from argschema import ArgSchema
from argschema.fields import Int, Nested, String


class LabelPaths(ArgSchema):
    full = String(required=True, description="Path to full itksnap labels")
    broad = String(required=True, description="Path to broad itksnap labels")
    landmark = String(required=True, description="Path to landmark itksnap labels")


class GenerateMetricsDashboardOutputSchema(ArgSchema):
    dashboard_path = String(required=True, description="Path to metrics dashboard")


class CCFFilesSchema(ArgSchema):
    full = String(required=True, description="Path to full CCF image")
    atlas = String(required=True, description="Path to atlas CCF image")
    broad = String(required=True, description="Path to broad CCF image")
    landmark = String(required=True, description="Path to landmark CCF image")


class GenerateMetricsDashboardSchema(ArgSchema):
    registration_output_json = String(
        required=True, description="Path to registration output json"
    )
    out_path = String(required=True, description="Path to output directory")
    ccf_files = Nested(CCFFilesSchema, description="Paths to CCF input files")
    label_paths = Nested(LabelPaths, description="Paths to itksnap label files")
    tmp_dir = String(
        required=True,
        description="Path to temporary scratch directory for"
        "storing processed neuroglancer formatted images",
    )
    n_processors = Int(
        default=-1,
        description="Number of processors to use for"
        "multiprocessing. -1 uses all available processors.",
    )
