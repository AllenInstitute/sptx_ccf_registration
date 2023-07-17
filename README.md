# Spatial Transcriptomics to Common Coordinate Framework Registration

## Overview
sptx_ccf_registration is an image processing pipeline for registering spatial transcriptomics data to the common coordinate framework (CCF). This pipeline consists of three modules:  

1. **Segmentation** of MERFISH point cloud data consisting of cell label and position into regions
2. **Registration** of MERFISH labels to CCF labels
3. **Dashboard** for analyzing the registration through metrics and image visualization

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/AllenInstitute/sptx_ccf_registration
   ```
2. Navigate to the directory:
   ```
   cd sptx_ccf_registration
   ```
3. Install the package:
   ```
   pip install .
   ```
4. Install dependencies for Neuroglancer
   ```
   pip install -r requirements_neuroglancer.txt
   ```
5. For Neuroglancer, obtain AWS credentials and export to your environment

## Usage
For detailed usage instructions, see the documentation for individual modules in `docs`  

You can use the following commands to get help on the input params for each module:
```
python -m sptx_ccf_registration.segmentation --help
python -m sptx_ccf_registration.registration --help
python -m sptx_ccf_registration.metrics_dashboard --help
```

You can configure these parameters with an input_json file for each module. Examples of these can be found in the `sample_config` directory.

Here are examples of launching the modules with the input_json files.
```
python -m sptx_ccf_registration.segmentation --input_json sample_config/segmentation/segmentation_broad.json
python -m sptx_ccf_registration.segmentation --input_json sample_config/segmentation/segmentation_landmark.json
python -m sptx_ccf_registration.registration --input_json sample_config/segmentation/registration.json
python -m sptx_ccf_registration.metrics_dashboard --input_json sample_config/segmentation/metrics_dashboard.json
```