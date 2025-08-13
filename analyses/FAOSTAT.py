"""
FAOSTAT Data Processing Module

This module provides functionality to load and process agricultural data from FAOSTAT.
Functions handle CSV data loading and restructuring into crop-specific dataframes.
"""

from pathlib import Path
#import __main__
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PATH, FIGURE_PATH

def load_faostat_data(filename="FAOSTAT_data_en_8-12-2025.csv"):
    """
    Load FAOSTAT CSV data into a pandas DataFrame.
    
    Args:
        filename (str): Name of the FAOSTAT CSV file
        
    Returns:
        pandas.DataFrame: Loaded FAOSTAT data
    """
    df = pd.read_csv(
        Path(DATA_PATH) / "FAOSTAT" / filename,
        dtype={'Year': int,
               'Domain Code': str, 'Domain': str,
               'Area Code (M49)': str, 'Area': str,
               'Element Code': str, 'Element': str,
               'Item Code (CPC)': str, 'Item': str,
               'Year Code': str,
               'Unit': str,
               'Value': float,
               'Flag': str, 'Flag Description': str,
               'Note': str}
    )
    return df

def build_fao_dfs(df):
    """
    Build a nested dictionary of DataFrames for each crop and metric.
    """
    crops = df['Item'].unique()
    metrics = ['Production', 'Area harvested']#, No 'Yield'. We calculate that one.

    # Build nested dictionary: fao_dfs[crop][element]
    fao_dfs_dict = {
        crop: {
            metric: (
                df[
                    (df['Item'] == crop) &
                    (df['Element'] == metric)
                ]
                .set_index("Year", drop=True)
            )
            for metric in metrics
        }
        for crop in crops
    }

    # Function to keep same value or sum otherwise
    def same_or_sum(series):
        vals = series.dropna().unique()
        return vals[0] if len(vals) == 1 else series.sum()

    # Add "All crops"
    fao_dfs_dict["All crops"] = {
        metric: (
            df[df['Element'] == metric]
            .drop(columns=["Item", "Element"])
            .groupby("Year", group_keys=False)
            .agg(same_or_sum)
        )
        for metric in metrics
    }

    # Add 'Yield' for each crop and 'All crops'
    for crop, metrics_dict in fao_dfs_dict.items():
        prod_df = metrics_dict['Production']
        area_df = metrics_dict['Area harvested']

        # Division element-wise; avoid division by zero
        yield_df = prod_df[['Value']] / area_df[['Value']]
        yield_df.rename(columns={'Value': 'Value'}, inplace=True)

        # Add Unit column based on existing units
        if 'Unit' in prod_df.columns and 'Unit' in area_df.columns:
            prod_unit = prod_df['Unit'].iloc[0] if not prod_df['Unit'].empty else ''
            area_unit = area_df['Unit'].iloc[0] if not area_df['Unit'].empty else ''
            yield_df['Unit'] = f"{prod_unit}/{area_unit}"
        else:
            yield_df['Unit'] = ''

        metrics_dict['Yield'] = yield_df

    return fao_dfs_dict

def onclick(event):
    """Callback function to print x and y coordinates on click."""
    if event.inaxes:  # Check if the click was within the plot axes
        x = event.xdata
        y = event.ydata
        print(f"Clicked coordinates: x={x:.2f}, y={y:.2f}")

def plot_fao_data(dict_dfs, years=None, figsize=(15, 10), save=False):
    """
    Plot FAO data for each crop and element.
    """
    crops = dict_dfs.keys()
    elements = dict_dfs[list(crops)[0]].keys()
    if years:
        start_year, end_year = years
    else:
        first_crop = list(crops)[0]
        first_element = list(elements)[0]
        df = dict_dfs[first_crop][first_element]
        start_year = df.index.min()
        end_year = df.index.max()

    fig, axes = plt.subplots(len(crops), len(elements), figsize=figsize, sharex=True)
    fig.suptitle(f"FAO Data {start_year}-{end_year}", fontsize=16, y=1.02)
    tab10_cmap = plt.get_cmap('tab10')
    colors = tab10_cmap.colors

    for col_idx, crop in enumerate(crops):
        for row_idx, element in enumerate(elements):
            ax = axes[row_idx, col_idx]
            df = dict_dfs[crop][element]

            # Filter year range
            df = df.loc[(df.index >= start_year) & (df.index <= end_year)]

            # Plot
            ax.plot(df.index, df['Value'], marker='o', ms=3,
                    color=colors[col_idx % len(colors)])
            ax.set_title(f"{crop} - {element}", fontsize=10)
            ax.grid(True, linestyle='--', lw=0.3)

            if df.index.size > 30:
                continue
            else:
                ax.set_xticks(df.index)
                ax.set_xticklabels([str(int(t)) if t % 2 == 0 else '' for t in df.index],fontsize=8)

            # Label only first column with y-axis label
            if col_idx == 0:
                ax.set_ylabel(df['Unit'].iloc[0] if 'Unit' in df.columns else 'Value')

            # Label only last row with x-axis label
            if row_idx == len(crops) - 1:
                ax.set_xlabel("Year")

    # After creating the figure and axes, connect the event
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_PATH / "fao_data_plot.png")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load FAOSTAT data
    faostat_df = load_faostat_data()
    fao_dfs = build_fao_dfs(faostat_df)
    save = True
    plot_fao_data(fao_dfs, years=(2003, 2020), figsize=(11, 8), save=save)
