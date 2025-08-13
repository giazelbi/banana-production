"""
Generates heatmap density maps (count and revenue) for firms in Ecuador provinces/cantons.

Loads:
- Shapefiles
- Grammatically corrected city-province pairs
- Firm nodelist
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import DATA_PATH, FIGURE_PATH

# ============================ Config ================================
SECTOR = "A0122"        # Example: 'A0122' or ''
CONTR_TYPE = "sociedades"  # 'personas', 'sociedades', or 'all'
YEAR = 2015
SHOW_GALAPAGOS = False
LOGSCALE = True

print(f'Drawing heatmap for {SECTOR} {CONTR_TYPE} firms in Ecuador ({YEAR})')
print(f'Log scale: {LOGSCALE}', f'Galápagos included: {SHOW_GALAPAGOS}')

# ============================ Load shapefiles ================================
# Load GeoJSON or SHP of Ecuador cantons (ADM2)
cantons_gdf = gpd.read_file(DATA_PATH / "geography" / "ecu_adm_2024" / "ecu_adm_adm2_2024.shp")

# Load ADM1 boundaries (provinces)
adm1_gdf = gpd.read_file(DATA_PATH / "geography" / "ecu_adm_2024" / "ecu_adm_adm1_2024.shp")

# Corrected canton/province names
cantons_gdf_goodnames = pd.read_csv(DATA_PATH / "geography" / "renamed_adm2_city_prov.csv", index_col=0)
cantons_gdf[['ADM2_ES','ADM1_ES']] = cantons_gdf_goodnames


# ============================ Load firm data ================================
firm_df = pd.read_csv(
    DATA_PATH / "firm-level" / f"nodelist_{SECTOR}{CONTR_TYPE}.csv",
    sep="\t",
    dtype={"out_strength": float}
)
# Lower case for match with shp
firm_df["province"] = firm_df["province"].str.strip().str.lower()
firm_df["canton"] = firm_df["canton"].str.strip().str.lower()

# For the moment, drop years != YEAR. Future: expand the geoanalysis to temporal dimension.
firm_df = firm_df[firm_df["date"] == YEAR]  # Filter year

# ============================ Filter Galápagos ================================

if SHOW_GALAPAGOS:
    adm1_gdf_regions_ok = adm1_gdf.copy()
    cantons_gdf_regions_ok = cantons_gdf.copy()
else:
    adm1_gdf_regions_ok = adm1_gdf[adm1_gdf['ADM1_ES'] != 'Galápagos']
    cantons_gdf_regions_ok = cantons_gdf[cantons_gdf['ADM1_ES'] != 'galapagos']

# ============================ Aggregate and merge with shapes =================
# Do the count per (province, canton) and merge count to the corresponding row for the heatmap
# All the provinces and cantones of firm_df_norm are in cantons_gdf_norm.
firm_counts = (
    firm_df
    .groupby(["province", "canton"])
    .agg(
        contr_count=('firm_id','size'),
        cuml_s_out = ('cw_s_out','sum'))
    .reset_index()
)

# Merge firm counts with geodata
merged_gdf = cantons_gdf_regions_ok.merge(
    firm_counts,
    left_on=["ADM1_ES", "ADM2_ES"],
    right_on=["province", "canton"],
    how="left"
)

# No filling, we want to know the cantones where there's no data
# merged_gdf["firm_count"] = merged_gdf["firm_counts"].fillna(0)


# ============================ Plot ================================
# Compute summary stats
text_box = [
    ('Total count:   ', int(merged_gdf["contr_count"].sum())),
    ('Total out str: ', format(merged_gdf["cuml_s_out"].sum(), ".2e"))
]


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14, 7))

for i, col in enumerate(["contr_count", "cuml_s_out"]):

    # In principle there's no location with element at col = 0... but to be sure:
    data = merged_gdf[col].replace(0, np.nan)  # avoid log(0)
    valid_data = data.dropna()

    if LOGSCALE:
        # Log-transform the data safely
        merged_gdf.loc[valid_data.index, f"log_{col}"] = np.log10(valid_data)

        # Define log-spaced bins
        bins = np.logspace(np.log10(valid_data.min()), np.log10(valid_data.max()), num=7)
        labels = [f"{bins[j]:.1e} - {bins[j+1]:.1e}" for j in range(len(bins)-1)]
        # Digitize original (not log) values
        merged_gdf[f"{col}_binned"] = pd.cut(data, bins=bins, labels=labels, include_lowest=True)

        PLOT_COL = f"{col}_binned"
    else:
        PLOT_COL = col

    # Plot the canton-level heatmap
    merged_gdf.plot(
        column=PLOT_COL,
        cmap="OrRd",
        linewidth=0.3,
        ax=ax[i],
        edgecolor="black",
        linestyle="--",
        legend=True,
        missing_kwds={"color": "lightgray", "label": "Zero"}
    )

    # Overlay ADM1 boundaries
    adm1_gdf_regions_ok.boundary.plot(
        ax=ax[i],
        edgecolor="black",
        linewidth=1,
        linestyle="-"     # Optional: make province borders dashed
    )
    # Add total count text box
    ax[i].text(
        0.475, 0.05,  # X and Y position (axes fraction)
        f"Sector:        {SECTOR}\n"
        f"Contrib. type: {CONTR_TYPE}\n"
        f"{text_box[i][0]}{text_box[i][1]}",
        transform=ax[i].transAxes,
        fontsize=12,
        family='monospace',  # Use monospaced font for alignment
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )
    ax[i].axis("off")

# Titles and cleanup
ax[0].set_title("Tax contributor density (2015) by canton \n in continental Ecuador", fontsize=14)
ax[1].set_title("Tax contributor revenue (2015) by canton \n in continental Ecuador", fontsize=14)

FIGURE_PATH.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH / f"gelocated_2015_{SECTOR}{CONTR_TYPE}.png", format="png", dpi=600)
plt.show()
