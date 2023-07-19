# Registration: sptx_ccf_registration.registration

## Quick Start

1. **Perform registration**

   Setup input parameters in `sample_config/registration/registration.json`
   Run the module

    ```bash
    python -m sptx_ccf_registration.registration --input_json sample_config/registration/registration.json
    ```

## Overview

This module performs affine and/or deformation registration iteratively on selected labels from either broad level or landmark level labels.

1. For each iteration, select whether broad or landmark label level are used `labels_level`, subset specific labels with `iteration_labels` and merge them by selecting the replacement value `labels_replace_to`.
2. Perform registration by using gradient descent to estimate the affine and/or warp parameters that minimize the mean squared error. MERFISH is used as the moving image while CCF is used as the fixed image.
3. Store affine/warp parameters in `{output_path}/iteration{n}/slice_transformations`
4. Apply affine/warp to MERFISH input images (segmented, unsegmented, stack, atlas) and save to `{output_path}/iteration{n}`

The current implementation includes hardcoded rules and configurable rules.

### Hardcoded rules

1. Iteration 0:  
   * Performs both affine and deformation (SyN) registration
   * reg_iterations = (40,20,10)
   * Applies a right hemisphere mask transformation to MERFISH and CCF images where 
     `img = img*250 + img*right_hemisphere_mask*500`
2. Iteration 1:
   * Performs deformation (SyN) registration
   * reg_iterations = (40,20,10)
3. Iteration 2+:
   * Performs deformation (SyN) registration
   * reg_iterations = (70,40,20)

   *\*`reg_iteration`: a vector of iterations for SyN that underlies the smoothing and multi resolution parameters.*

## Usage

Parameter setup:

```
RegistrationSchema:
  --output_name OUTPUT_NAME
                        General prefix attached to the set of experiments (REQUIRED)
  --output_path OUTPUT_PATH
                        Base directory for output data (REQUIRED)
  --labels_level LABELS_LEVEL
                        Level of labels to use for registration. 0 is broad labels, 1is landmark labels (REQUIRED)
  --labels_replace_to LABELS_REPLACE_TO
                        Labels to replace to. Set to 1 if merging labels, set to -1if using all labels in range as distinct labels (REQUIRED)
  --iteration_labels ITERATION_LABELS
                        Labels to use at each iteration of registration (REQUIRED)
  --n_processes N_PROCESSES
                        Number of processes to use.If -1, use all available cores. (default=-1)

merfish_files:
  Paths to MERFISH input files

  --merfish_files.labels_broad_segmented MERFISH_FILES.LABELS_BROAD_SEGMENTED
                        Path to the segmented broad labels for MERFISH (REQUIRED)
  --merfish_files.labels_landmark_segmented MERFISH_FILES.LABELS_LANDMARK_SEGMENTED
                        Path to the segmented landmark labels for MERFISH (REQUIRED)
  --merfish_files.labels_broad_unsegmented MERFISH_FILES.LABELS_BROAD_UNSEGMENTED
                        Path to the unsegmented broad labels for MERFISH (REQUIRED)
  --merfish_files.labels_landmark_unsegmented MERFISH_FILES.LABELS_LANDMARK_UNSEGMENTED
                        Path to the unsegmented landmark labels for MERFISH (REQUIRED)
  --merfish_files.stack MERFISH_FILES.STACK
                        Path to the stacked MERFISH file (REQUIRED)
  --merfish_files.right_hemisphere MERFISH_FILES.RIGHT_HEMISPHERE
                        Path to the right hemisphere of the MERFISH data (REQUIRED)

ccf_files:
  Paths to CCF input files

  --ccf_files.labels_broad CCF_FILES.LABELS_BROAD
                        Path to the broad labels for CCF (REQUIRED)
  --ccf_files.labels_landmark CCF_FILES.LABELS_LANDMARK
                        Path to the landmark labels for CCF (REQUIRED)
  --ccf_files.right_hemisphere CCF_FILES.RIGHT_HEMISPHERE
                        Path to the right hemisphere of the CCF data (REQUIRED)
```
