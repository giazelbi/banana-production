"""
plot_tropical_fruit_production_shares.py

Module for loading FAOSTAT tropical fruit production data and generating
stacked area charts (raw values and normalized shares) using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyses.FAOSTAT import load_faostat_data
from config import FIGURE_PATH


def plot_tropical_fruit_production(filename: str="FAOSTAT_tropical_fruit_production.csv"):
    """
    Generate a two-panel stacked area chart of tropical fruit production.

    Left subplot: raw production values.
    Right subplot: normalized production shares.

    Parameters
    ----------
    filename : str
    Path to FAOSTAT CSV file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    Figure object that can be further customized.
    """
    tropical_fruits_prod_df = load_faostat_data(filename)
    tropical_fruits_prod_df = tropical_fruits_prod_df[
        tropical_fruits_prod_df['Year'].isin(range(2007, 2016))
    ]

    # Pivot to wide format
    pivot_df = tropical_fruits_prod_df.pivot_table(
        index="Year", columns="Item", values="Value", aggfunc="sum"
    ).fillna(0)

    # Sort items by total area (largest at bottom)
    order = pivot_df.sum(axis=0).sort_values(ascending=False).index
    pivot_df = pivot_df[order]

    # Normalize for the second subplot
    pivot_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # Assign consistent colors using Plotly Express color scale
    color_map = {
        item: color
        for item, color in zip(pivot_df.columns, px.colors.qualitative.Plotly)
    }

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=False,
        subplot_titles=("Raw production over the years", "Production shares over the years")
    )

    # ---------------- Raw Values ----------------
    for item in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[item],
            mode="lines",
            line=dict(width=0.5, color=color_map[item]),
            stackgroup="one",
            name=item,
            hovertemplate=(
                f"Tons of {item}:<br>"
                "%{customdata:.2e}<extra></extra>"
            ),
            customdata=pivot_df[item].values
        ), row=1, col=1)

    # ---------------- Normalized Values ----------------
    for item in pivot_norm.columns:
        fig.add_trace(go.Scatter(
            x=pivot_norm.index,
            y=pivot_norm[item],
            mode="lines",
            line=dict(width=0.5, color=color_map[item]),
            stackgroup="one",
            name=item,
            showlegend=False,  # only show legend once
            hovertemplate=(
                f"Item: {item}<br>"
                "Share: %{y:.2%}<br><extra></extra>"
            ),
            customdata=pivot_df[item].values
        ), row=1, col=2)

    # Layout
    fig.update_layout(
        title_text="Production of Bananas and other Tropical Fruits:",
        hovermode="x unified",
        legend=dict(title="Item", x=1.05, y=0.5)
    )

    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_yaxes(title_text="Tons", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Share", row=1, col=2)

    return fig

if __name__ == "__main__":
    # Example usage
    figure = plot_tropical_fruit_production("FAOSTAT_tropical_fruit_production.csv")
    figure.show(renderer="browser")
    figure.write_image(FIGURE_PATH / "FAOSTAT" / "tropical_fruit_production.png",
                       width=2000,
                       height=800)
