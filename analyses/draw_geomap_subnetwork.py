import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tests.VAT import load_firm_data_sector
from config import REGION_CONFIG, clean_region_shapefile, FIGURE_PATH

# ============================ Load Data =============================
def load_shapefiles():
    """
    Load Ecuador canton (ADM2) and province (ADM1) shapefiles as GeoDataFrames.
    """
    geodf = gpd.read_file(REGION_CONFIG['ecuador']["path"])
    adm2_geodf = clean_region_shapefile(geodf, 'ecuador')
    adm1_geodf = adm2_geodf.dissolve(by=['ADM0', 'ADM1'], as_index=False)
    return adm1_geodf, adm2_geodf


# ============================ Processing ============================
def aggregate_firm_data(data_df, adm2_geodf, agg_col, agg_func):
    """
    Aggregate firms data by canton and merge with ADM2 (canton) shapefile.
    """
    agg_df = (data_df.groupby(["date", "ADM1", "ADM2"], as_index=False)
                     .agg({agg_col : agg_func}))

    # Merge with shapefile
    return adm2_geodf.merge(agg_df, how='outer')


# ============================ Plotting ==============================
def plot_firmdata_heatmap(ax,
                          firm_geodata_df,
                          agg_col,
                          logscale,
                          adm1_boundaries,
                          textbox,
                          cmap="viridis"):
    """
    Plot a canton-level heatmap for a firm data column.
    """
    values = firm_geodata_df[agg_col].replace(0, np.nan)
    total = firm_geodata_df[agg_col].sum()
    if total > 500000: # Temporary distintion between strength and count...
        total = str(round(total /1e9,2)) + ' B$'
    if agg_col == 'firm_id':
        total = int(total)

    # Optional log binning
    if logscale and not values.empty:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), num=7)
        bins[-1] *= 1 + 1e-12  # ensure upper inclusion
        labels = [f"{bins[j]:.1e} - {bins[j+1]:.1e}" for j in range(len(bins)-1)]
        firm_geodata_df[f"{agg_col}_binned"] = pd.cut(values, bins=bins, labels=labels,
                                                      include_lowest=True)
        plot_col = f"{agg_col}_binned"
    else:
        plot_col = agg_col

    # Draw main heatmap
    firm_geodata_df.plot(
        column=plot_col,
        cmap=cmap,
        linewidth=0.3,
        ax=ax,
        edgecolor="black",
        linestyle="--",
        legend=True,
        missing_kwds={"color": "lightgray", "label": "Zero"}
    )
    # Overlay ADM1 boundaries
    adm1_boundaries.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

    # Add title
    year = str(firm_geodata_df['date'].unique()[0])
    ax.set_title(f"Tax contributor {agg_col} ({year}) by canton\n"
                    f" in continental Ecuador", fontsize=14)

    # Add text box
    ax.text(
        0.475, 0.05,
        f"Sector:        {textbox['sector']}\n"
        f"Contrib. type: {textbox['contrib']}\n"
        f"Total {agg_col}: {total}",
        transform=ax.transAxes,
        fontsize=12,
        family='monospace',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )
    ax.axis("off")


def canton_aggregated_count_and_column(firm_data_df,
                                       adm2_geodf, adm1_geodf, logscale,
                                       year, sector=None, contrib=None,
                                       col_fun1 = ('cw_s_out', 'sum'),
                                       save=False, figsize=(14, 7)):
    """
    Aggregate firm data at ADM2 level and plot count and value heatmaps.
    """

    # Textbox in the bottom of the ax
    if not sector: sector=''
    if not contrib: contrib=''
    textbox = {'sector':sector, 'contrib':contrib}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # ax 0
    # Aggregate firm column data at the ADM2 level.
    col_fun = ('firm_id', 'size')
    adm2_data_geodf = aggregate_firm_data(firm_data_df, adm2_geodf, col_fun[0], col_fun[1])

    # Plot column data in each canton
    plot_firmdata_heatmap(ax[0], adm2_data_geodf, col_fun[0], logscale,
                        adm1_geodf, textbox=textbox)


    # ax 1
    # Aggregate firm column data at the ADM2 level.
    adm2_data_geodf = aggregate_firm_data(firm_data_df, adm2_geodf, col_fun1[0], col_fun1[1])

    # Plot column data in each canton
    plot_firmdata_heatmap(ax[1], adm2_data_geodf, col_fun1[0], logscale,
                        adm1_geodf, textbox=textbox)

    if save:
        FIGURE_PATH.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_PATH / "VAT" / f"geolocated_{year}_{sector}{contrib}.png",
                    format="png", dpi=600)
    fig.show()


if __name__ == "__main__":
    SECTOR = "A0122"
    YEAR = 2015
    LOGSCALE = True

    # Load shapefiles for Ecuador
    adm1_geodf, adm2_geodf = load_shapefiles()

    # Preload once
    firms_df = load_firm_data_sector(year=YEAR, contr_type='all', sector=SECTOR)

    for CONTR_TYPE in ['all', 'Personas naturales', 'Sociedades']:

        # Filtering firms
        if CONTR_TYPE != 'all':
            df = firms_df[firms_df['descrip_1'] == CONTR_TYPE].copy()
        else:
            df = firms_df.copy()

        print(f"Drawing heatmaps for {SECTOR} {CONTR_TYPE} firms in Ecuador ({YEAR})")
        print(f"Log scale: {LOGSCALE}")

        canton_aggregated_count_and_column(df,
                                       YEAR, SECTOR, CONTR_TYPE,
                                       adm2_geodf, adm1_geodf, LOGSCALE,
                                       col_fun1 = ('cw_s_out', 'sum'),
                                       save=False)
