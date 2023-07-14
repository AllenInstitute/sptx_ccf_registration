# Metrics Dashboard: sptx_ccf_registration.metrics_dashboard

### Metrics dashboard
[Dashboard](doc_assets/dashboard.png)

### Neuroglancer
[Neuroglancer](doc_assets/neuroglancer.png)

## Quickstart

1. **Generate the metrics dashboard with Neuroglancer visualizations**

    Setup paths in `sample_config/metrics_dashboard/metrics_dashboard.json`  
    *Note: `registration_input_json` and `registration_output_json` are stored in the output_directory_base path from your registraion run*

    Run the module

    ```bash
    python -m sptx_ccf_registration.metrics_dashboard --input_json sample_config/metrics_dashboard/metrics_dashboard.json
    ```

2. **Edit webapp text**

    The text is defined in `app.py` under the section, `MARKDOWN_TEXT`

3. **Launch the webapp locally**

    Navigate to the webapp directory

    ```bash
    cd <out_path>/ccf_merfish_metrics
    ```

    Launch the webapp backend

    ```bash
    python app.py
    ```

4. View the webapp in your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Webapp Usage

* Select a plot from the tab on the top
* Hover over a trace to see its associated structure, z-slice, and metrics
* Click on a structure in the legend to hide/show it
* Double click on a structure in the legend to isolate it

## Overview

The metrics dashboard displays the cumulative change in dice coefficient between the registered labels and their CCF counterpart. This allows one to track how the dice coefficient is improving or worsening after each iteration of registration. 