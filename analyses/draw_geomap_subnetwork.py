import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from config import REGION_CONFIG, clean_region_shapefile, DATA_PATH, FIGURE_PATH

# ============================ Config ================================
SECTOR = "A0122"          # Example: 'A0122' or ''
CONTR_TYPE = "all"        # 'personas', 'sociedades', or 'all'
YEAR = 2015
LOGSCALE = True

# ============================ Load Data =============================
def load_shapefiles():
    """
    Load Ecuador canton (ADM2) and province (ADM1) shapefiles as GeoDataFrames.
    """
    geodf = gpd.read_file(REGION_CONFIG['ecuador']["path"])
    adm2_geodf = clean_region_shapefile(geodf, 'ecuador')
    adm1_geodf = adm2_geodf.dissolve(by=['ADM0', 'ADM1'], as_index=False)
    return adm1_geodf, adm2_geodf


def load_firm_data(year=None, contr_type='all', sector='A0122'):
    """
    Load firm-level data for the given sector, contributor type, and year, returning a DataFrame.
    """
    filepath = Path(DATA_PATH) / f"firm-level/nodelist_{sector}{contr_type}.csv"
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    firms_df = pd.read_csv(filepath, sep='\t',
                           dtype={'firm_id': str, 'date': int, "out_strength": float})

    # filter year
    if year:
        firms_df = firms_df[firms_df["date"] == year]
    # filter contributor type
    if contr_type == 'personas':
        name_contr = 'Personas naturales'
    elif contr_type == 'sociedades':
        name_contr = 'Sociedades'
    else:
        name_contr = 'all'
    if contr_type and contr_type != 'all':
        firms_df = firms_df[firms_df["descrip_1"] == name_contr]
    # filter sector
    if sector:
        firms_df = firms_df[firms_df["ISIC4"] == sector] 
    return firms_df.rename({'province': 'ADM1', 'canton': 'ADM2'}, axis=1)


# ============================ Processing ============================
def aggregate_firm_data(firms_df, adm2_geodf):
    """Aggregate firm counts and revenue by canton and merge with ADM2 shapefile."""
    agg_df = (firms_df.groupby(["date", "ADM1", "ADM2"], as_index=False)
                    .agg(counts=('firm_id', 'size'),
                         cuml_s_out=('cw_s_out', 'sum')))

    # Merge with shapefile
    return adm2_geodf.merge(agg_df, how='outer')


# ============================ Plotting ==============================
def plot_heatmaps(adm2_data_geodf, adm1_geodf, year, sector, contr_type):
    """
    Plot canton-level heatmaps of firm density and revenue, optionally using log-scaled bins.
    """
    text_box = [
        ('Total count:   ', int(adm2_data_geodf["counts"].sum())),
        ('Total out str: ', format(adm2_data_geodf["cuml_s_out"].sum(), ".2e"))
    ]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    for i, col in enumerate(["counts", "cuml_s_out"]):
        data = adm2_data_geodf[col].replace(0, np.nan)
        print(data)
        if LOGSCALE and not data.empty:
            adm2_data_geodf.loc[data.index, f"log_{col}"] = np.log10(data)
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), num=7)
            labels = [f"{bins[j]:.1e} - {bins[j+1]:.1e}" for j in range(len(bins)-1)]
            adm2_data_geodf[f"{col}_binned"] = pd.cut(data, bins=bins,
                                                      labels=labels, include_lowest=True)
            plot_col = f"{col}_binned"
        else:
            plot_col = col

        # Plot canton-level heatmap
        adm2_data_geodf.plot(
            column=plot_col,
            cmap="viridis",
            linewidth=0.3,
            ax=ax[i],
            edgecolor="black",
            linestyle="--",
            legend=True,
            missing_kwds={"color": "lightgray", "label": "Zero"}
        )

        # Overlay ADM1 boundaries
        adm1_geodf.boundary.plot(ax=ax[i], edgecolor="black", linewidth=1)

        # Add text box
        ax[i].text(
            0.475, 0.05,
            f"Sector:        {sector}\n"
            f"Contrib. type: {contr_type}\n{text_box[i][0]}{text_box[i][1]}",
            transform=ax[i].transAxes,
            fontsize=12,
            family='monospace',
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
        )
        ax[i].axis("off")

    ax[0].set_title(f"Tax contributor density ({year}) by canton\n"
                    f" in continental Ecuador", fontsize=14)
    ax[1].set_title(f"Tax contributor revenue ({year}) by canton\n"
                    f" in continental Ecuador", fontsize=14)

    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH / f"geolocated_{year}_{sector}{contr_type}.png", format="png", dpi=600)
    plt.show()


# ============================ Main ================================
def draw_geomap_subnetwork(sector=SECTOR, contr_type=CONTR_TYPE,
                           year=YEAR, logscale=LOGSCALE):
    print(f"Drawing heatmap for {sector} {contr_type} firms in Ecuador ({year})")
    print(f"Log scale: {logscale}")

    adm1_geodf, adm2_geodf = load_shapefiles()
    firms_df = load_firm_data(year=year, contr_type=contr_type, sector=sector)
    adm2_data_geodf = aggregate_firm_data(firms_df, adm2_geodf)
    plot_heatmaps(adm2_data_geodf, adm1_geodf, year, sector, contr_type)


if __name__ == "__main__":
    draw_geomap_subnetwork()
