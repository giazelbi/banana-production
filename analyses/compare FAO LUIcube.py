"""
compare_FAO_LUI.py

This script compares FAOSTAT agricultural data with LUIcube spatially explicit land-use data.
Specifically, it compares area harvested, production, and yield for bananas / plantains in Ecuador.

Outputs:
- A figure with three subplots (area, production, yield) saved to FIGURE_PATH.

Modules required:
- geopandas
- pandas
- matplotlib
- config (local module with DATA_PATH, FIGURE_PATH, REGION_CONFIG, clean_region_shapefile)
- analyses.FAOSTAT (local FAOSTAT processing module)
"""


from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from config import (REGION_CONFIG, clean_region_shapefile,
                    DATA_PATH, FIGURE_PATH)
import analyses.FAOSTAT as faostat_data

# --------------------------------------------------
# Config
# --------------------------------------------------
COLORS = {
    'bananas': 'blue',
    'plantains': 'red',
    'combined': 'green',
    'luicube': 'k'
}

KWARGS = {
    'luicube': {'label': 'LUIcube', 'ls': "-", 'marker': "o", 'ms': 3},
    'fao': {'ls': '--', 'lw': .8, 'marker': "o", 'ms': 3}
}

# --------------------------------------------------
# Data loading functions
# --------------------------------------------------
def load_lui_data(country: str):
    """
    Load LUIcube satellite area and HANPPharv data 
    for the given country and crop, aggregated at national level.
    """
    geodf = gpd.read_file(REGION_CONFIG[country]["path"])
    adm2_geodf = clean_region_shapefile(geodf, country)

    area_df = pd.read_csv(
                Path(DATA_PATH) / "spatially_explicit_land_use" / f"BANP_{country}_area.csv",
                usecols=['ADM1', 'ADM2', 'year', 'area_sum', 'crop'],
                dtype={'year': int}
            ).rename({'area_sum': 'area'}, axis=1)

    hanpp_df = pd.read_csv(
                Path(DATA_PATH) / "spatially_explicit_land_use" / f"BANP_{country}_HANPPharv.csv",
                usecols=['ADM1', 'ADM2', 'year', 'HANPPharv_sum', 'crop'],
                dtype={'year': int}
            ).rename({'HANPPharv_sum': 'HANPPharv'}, axis=1)

    adm2_data = (adm2_geodf.merge(area_df, how='outer')
                 .merge(hanpp_df, how='outer'))

    country_area = adm2_data.groupby('year').agg({'area': 'sum'})
    country_hanpp = adm2_data.groupby('year').agg({'HANPPharv': 'sum'})

    return country_area["area"], country_hanpp["HANPPharv"]


def load_fao_data():
    """
    Load and preprocess FAOSTAT data using local helper functions.
    """
    faostat_df = faostat_data.load_faostat_data()
    return faostat_data.build_fao_dfs(faostat_df)


# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------
def setup_axis(ax, title, ylabel, years):
    """Configure common axis appearance for plots."""
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(t)) if float(t).is_integer() else '' for t in years], fontsize=7)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(lw=0.2, color='grey')
    ax.set_ylim(bottom=0)

# --------------------------------------------------
# Main execution
# --------------------------------------------------
def main(country="ecuador"):
    """
    Main routine: load data, compute metrics, generate comparison plots.
    """
    # --- Load Data ---
    area_lui, prod_lui = load_lui_data(country)
    fao_dfs = load_fao_data()

    # --- FAO Series ---
    area_faob = fao_dfs["Bananas"]["Area harvested"]["Value"] / 100
    area_faop = fao_dfs["Plantains and cooking bananas"]["Area harvested"]["Value"] / 100
    area_faocombined = area_faob + area_faop

    prod_faob = fao_dfs["Bananas"]["Production"]["Value"]
    prod_faop = fao_dfs["Plantains and cooking bananas"]["Production"]["Value"]
    prod_faocombined = prod_faob + prod_faop

    # --- Yields ---
    yield_lui = prod_lui / area_lui
    yield_faob = prod_faob / area_faob
    yield_faop = prod_faop / area_faop
    yield_faocombined = prod_faocombined / area_faocombined

    # --- Common Years ---
    common_years = (area_lui.dropna().index
                    .intersection(area_faob.dropna().index)
                    .intersection(area_faop.dropna().index))

    # --- Build Figure ---
    fig, axs = plt.subplot_mosaic(
            [
            ["0", "1"],     # top row
            ["2", "legend"] # bottom row
            ],
            figsize=(10, 7), constrained_layout=True
    )
    # Map string keys "0","1","2" back to integers
    ax = [axs["0"], axs["1"], axs["2"]]

    # --- Subplot 0: Area harvested ---
    ax[0].plot(common_years, area_faob.loc[common_years], label="FAO Bananas",
               color=COLORS['bananas'], **KWARGS['fao'])
    ax[0].plot(common_years, area_faop.loc[common_years], label="FAO Plantains",
               color=COLORS['plantains'], **KWARGS['fao'])
    ax[0].plot(common_years, area_faocombined.loc[common_years], label="FAO Bananas + Plantains",
               color=COLORS['combined'], **KWARGS['fao'])
    ax[0].plot(common_years, area_lui.loc[common_years],
               color=COLORS['luicube'], **KWARGS['luicube'])
    setup_axis(ax[0], "Area harvested (km²)", "Area (km²)", common_years)

    # --- Subplot 1: Production ---
    ax[1].plot(common_years, prod_faob.loc[common_years], label="FAO Bananas",
               color=COLORS['bananas'], **KWARGS['fao'])
    ax[1].plot(common_years, prod_faop.loc[common_years], label="FAO Plantains",
               color=COLORS['plantains'], **KWARGS['fao'])
    ax[1].plot(common_years, prod_faocombined.loc[common_years], label="FAO Bananas + Plantains",
               color=COLORS['combined'], **KWARGS['fao'])

    ax1 = ax[1].twinx()
    ax1.plot(common_years, prod_lui.loc[common_years],
             color=COLORS['luicube'], **KWARGS['luicube'])
    ax1.set_ylabel('HANPPharv (tC)', fontweight='bold')
    ax1.set_ylim(bottom=0)
    setup_axis(ax[1], "Production", "Tons", common_years)

    # --- Subplot 2: Yield ---
    ax[2].plot(common_years, yield_faob.loc[common_years], label="FAO Bananas",
               color=COLORS['bananas'], **KWARGS['fao'])
    ax[2].plot(common_years, yield_faop.loc[common_years], label="FAO Plantains",
               color=COLORS['plantains'], **KWARGS['fao'])
    ax[2].plot(common_years, yield_faocombined.loc[common_years], label="FAO Bananas + Plantains",
               color=COLORS['combined'], **KWARGS['fao'])

    ax2 = ax[2].twinx()
    ax2.plot(common_years, yield_lui.loc[common_years],
             color=COLORS['luicube'], **KWARGS['luicube'])
    ax2.set_ylabel('HANPPharv/km² (tC/km²)', fontweight='bold')
    ax2.set_ylim(bottom=0)
    setup_axis(ax[2], "Yield", "ton/km²", common_years)

    # --- Vertical reference line ---
    for a in ax:
        a.axvline(2012, zorder=-2, color='k')

    # --- Legend ---
    handles, labels = ax[0].get_legend_handles_labels()
    axs["legend"].axis("off")
    axs["legend"].legend(handles, labels, loc="center")

    # --- Save & Show ---
    plt.savefig(Path(FIGURE_PATH) / "area_production_yield_FAO_LUI.png", format="png", dpi=400)
    plt.show()


if __name__ == "__main__":
    main()
