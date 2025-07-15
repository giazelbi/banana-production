"""
This module loads
- shapefiles,
- grammatically corrected city_province couples,
- nodelist
and draws the heatmap density (count and revenue) of the firms in provinces-cantones of ecuador
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_PATH = './data'
print(os.getcwd())

# ============================ Load shapefiles ================================
# Load GeoJSON or SHP of Ecuador cantons (ADM2)
cantons_gdf = gpd.read_file(f"{DATA_PATH}/geography/ecu_adm_2024/ecu_adm_adm2_2024.shp")

# Load ADM1 boundaries (provinces)
adm1_gdf = gpd.read_file(f"{DATA_PATH}/geography/ecu_adm_2024/ecu_adm_adm1_2024.shp")

# Load fixed names data
cantons_gdf_goodnames = pd.read_csv(f"{DATA_PATH}/geography/renamed_adm2_city_prov.csv",index_col=0)
# Replace in old df
cantons_gdf[['ADM2_ES','ADM1_ES']] = cantons_gdf_goodnames


# ============================ Choose the nodelist for the heatmap ================================

SECTOR='A0122' # 'A0122' ''
CONTR_TYPE = 'sociedades' #'personas' # 'sociedades' # 'all'
firm_df = pd.read_csv(f'{DATA_PATH}/firm-level/nodelist_{SECTOR}{CONTR_TYPE}2015.csv', sep='\t',
                      dtype={'out_strength':float})

"""
SECTOR='1st and sub.'
CONTR_TYPE = 'sociedades'
firm_df = pd.read_csv('./ec_viz/data/nodelist_NoboaSubsidiariesSectors2015.csv', sep='\t')
"""

# Lower case for match with shp
firm_df["province"] = firm_df["province"].str.strip().str.lower()
firm_df["city"]     = firm_df["city"].str.strip().str.lower()

# Remove 'Galápagos' islands?
SHOW_GALAPAGOS = False
if SHOW_GALAPAGOS:
    adm1_gdf_regions_ok = adm1_gdf.copy()
    cantons_gdf_regions_ok = cantons_gdf.copy()
else:
    adm1_gdf_regions_ok = adm1_gdf[adm1_gdf['ADM1_ES'] != 'Galápagos']
    cantons_gdf_regions_ok = cantons_gdf[cantons_gdf['ADM1_ES'] != 'galapagos']


# ============================ Merge with shapes ================================
# Do the count per (province, canton) and merge count to the corresponding row for the heatmap
# All the provinces and cantones of firm_df_norm are in cantons_gdf_norm.
firm_counts = (
    firm_df
    .groupby(["province", "city"])
    .agg(contr_count=('id','size'),
         cuml_out_strength = ('out_strength','sum'))
    .reset_index()
)

# Merge firm counts with geodata
merged_gdf = cantons_gdf_regions_ok.merge(
    firm_counts,
    left_on=["ADM1_ES", "ADM2_ES"],
    right_on=["province", "city"],
    how="left"
)

# No filling, we want to know where we don't have data
# merged_gdf["firm_count"] = merged_gdf["firm_counts"].fillna(0)


# ============================ Plot ================================
# Compute summary stats
text_box = [('Total count:   ', int(merged_gdf["contr_count"].sum())),
            ('Total out str: ', format(merged_gdf["cuml_out_strength"].sum(), ".2e"))]

# Plotting
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14, 7))
ax.flatten()
LOGSCALE = True

for i, col in enumerate(["contr_count", "cuml_out_strength"]):

    # In principle there's no location with element at col = 0
    data = merged_gdf[col].replace(0, np.nan)  # avoid log(0)
    valid_data = data.dropna()

    if LOGSCALE:
        # Log-transform the data safely
        LOG_COL = f"log_{col}"
        merged_gdf.loc[valid_data.index, LOG_COL] = np.log10(valid_data)

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
        missing_kwds={
            "color": "lightgray",
            "label": "Zero"
        }
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
#plt.savefig(f'./ec_viz/data/figures/gelocated_2015_{SECTOR}{CONTR_TYPE}.png',
#            format='png', dpi=600)
plt.show()
