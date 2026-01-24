"""Generate interactive HTML chart for Options Returns."""

import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# Get the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "_data"
OUTPUT_DIR = PROJECT_ROOT / "_output"


def generate_options_returns_chart():
    """Generate SPX options portfolio returns time series chart."""
    # Load HKM option portfolio returns data
    df = pd.read_parquet(DATA_DIR / "ftsfr_hkm_option_returns.parquet")

    # Get a subset of unique_ids for cleaner visualization
    unique_ids = df['unique_id'].unique()[:10]
    df_subset = df[df['unique_id'].isin(unique_ids)]

    # Create line chart
    fig = px.line(
        df_subset.sort_values("ds"),
        x="ds",
        y="y",
        color="unique_id",
        title="SPX Option Portfolio Returns (He, Kelly, Manela 2017 Methodology)",
        labels={
            "ds": "Date",
            "y": "Return",
            "unique_id": "Portfolio"
        }
    )

    # Update layout
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save chart
    output_path = OUTPUT_DIR / "options_returns_replication.html"
    fig.write_html(str(output_path))
    print(f"Chart saved to {output_path}")

    return fig


if __name__ == "__main__":
    generate_options_returns_chart()
