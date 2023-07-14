from pathlib import Path

import markdown
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from bs4 import BeautifulSoup
from flask import Flask, render_template


def extract_url(file_path):
    with open(file_path, "r") as f:
        contents = f.read()

    soup = BeautifulSoup(contents, "lxml")
    url = soup.find("a", href=True)["href"]
    return url


NUM_ITERS = 10
MARKDOWN_TEXT = f"""
{{
# MERFISH - CCF Registration Metrics
This is a dashboard for plotting metrics from each iteration of registration between MERFISH and CCF.
Each trace represents a structure and z-slice. The color of the trace indicates the structure.
### Dataset
RegIterN_10iters_concavehull_patchedv2_mouse3_v2
#### Inputs
* MERFISH (regapped) with midline annotation
* CCF (using old affine)
#### Iterations (links to neuroglancer visualization)
0. [Broad region (affine + def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter0.html")})
1. [Broad region (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter1.html")})
2. [Landmark: Layer2/3/4 (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter2.html")})
3. [Landmark: Layer5 (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter3.html")})
4. [Landmark: Layer6a/6b (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter4.html")})
5. [Landmark: STRd + STRv (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter5.html")})
6. [Landmark: MH (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter6.html")})
7. [Landmark: LH (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter7.html")})
8. [Landmark: TH-Glut + AV + AD + LD + LGd + LGv + PF (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter8.html")})
9. [Landmark: RT (def)]({extract_url(Path(__file__).parent / "data/neuroglancer/neuroglancer_iter9.html")})

### Instructions
1. Select a plot from the tab above
2. Hover over a trace to see its associated structure, z-slice, and metrics
3. Click on a structure in the legend to hide/show it
4. Double click on a structure in the legend to isolate it
}}
"""
app = Flask(__name__)


def create_plot(df):
    df["structure"] = [val.split(" - ")[0] for val in df["structure"]]
    df["dice coefficient: cumulative change iter0"] = [0] * len(df)

    # Add a 'label' column as a concatenation of 'structure' and 'z-slice'
    df["label"] = df["structure"] + "-z" + df["z-slice"].astype(str)

    cumulative_columns_to_melt = [
        f"dice coefficient: cumulative change iter{num}" for num in range(NUM_ITERS)
    ]

    # Create a unique identifier for each row
    df["unique_id"] = df.index

    # Create a color map for the 'structure' column
    color_map = {}
    unique_structures = df["structure"].unique()
    for i, structure in enumerate(unique_structures):
        color_map[structure] = px.colors.qualitative.Plotly[
            i % len(px.colors.qualitative.Plotly)
        ]

    # Initialize the plot
    fig = go.Figure()

    # Iterate through each unique_id and plot the line
    for unique_id in df["unique_id"].unique():
        row_df = df[df["unique_id"] == unique_id]
        structure = row_df["structure"].values[0]
        label = row_df["label"].values[0]

        fig.add_trace(
            go.Scatter(
                x=list(range(NUM_ITERS)),
                y=row_df[cumulative_columns_to_melt].values[0],
                mode="lines",
                name=structure,
                line=dict(color=color_map[structure]),
                customdata=[structure] * NUM_ITERS,
                hovertemplate="Structure: %{customdata}<br>Structure-Z-Slice: %{text}<br>Iteration: %{x}<br>Cumulative Change: %{y}",
                text=[label] * NUM_ITERS,
                legendgroup=structure,
                showlegend=bool(
                    unique_id == df[df["structure"] == structure]["unique_id"].min()
                ),
            )
        )

    fig.update_layout(
        autosize=True,
        width=None,
        height=1000,
        title="Cumulative Change in Dice Coefficient vs Iteration",
        xaxis_title="Iteration",
        yaxis_title="Cumulative Change in Dice Coefficient",
    )
    return pio.to_html(fig, full_html=False)


@app.route("/")
def index():
    # Your markdown text
    markdown_text = MARKDOWN_TEXT
    # Convert the markdown text to HTML
    html_text = markdown.markdown(markdown_text)
    return render_template("index.html", text=html_text)


@app.route("/plot/<plot_type>")
def plot(plot_type):
    if plot_type == "broad_segmented":
        df_broad_segmented = pd.read_csv(
            Path(__file__).parent / "data/Broad-segmented_aggregate.csv"
        )
        plot_html = create_plot(df_broad_segmented)
    elif plot_type == "landmark_segmented":
        df_landmark_segmented = pd.read_csv(
            Path(__file__).parent / "data/Landmark-segmented_aggregate.csv"
        )
        plot_html = create_plot(df_landmark_segmented)
    elif plot_type == "broad_unsegmented":
        df_broad_unsegmented = pd.read_csv(
            Path(__file__).parent / "data/Broad-unsegmented_aggregate.csv"
        )
        plot_html = create_plot(df_broad_unsegmented)
    elif plot_type == "landmark_unsegmented":
        df_landmark_unsegmented = pd.read_csv(
            Path(__file__).parent / "data/Landmark-unsegmented_aggregate.csv"
        )
        plot_html = create_plot(df_landmark_unsegmented)
    else:
        return "Invalid plot type.", 400

    return plot_html


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
