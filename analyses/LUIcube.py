"""
Geospatial crop and land-use analysis utilities.

This module provides tools for working with spatially explicit land-use data 
and crop production metrics. It includes functions to:

- Build dictionaries of raster (.tif) paths for multiple crops and metrics.
- Load, clean, and standardize shapefiles for supported regions.
- Clip and plot raster data within administrative boundaries.
- Compute zonal statistics (per administrative unit) for crop rasters.
- Extract and save temporal zonal statistics as CSV files.
- Plot the temporal evolution of crop metrics per administrative level.

Supported regions are defined in REGION_CONFIG with custom rules for column 
renaming, filtering, and attribute retention.

Dependencies:
    - rasterio, rasterstats, geopandas, pandas, matplotlib, numpy
    - config.py (defines DATA_PATH, FIGURE_PATH)

Typical usage example:
    >>> raster_dict = build_raster_path_dict()
    >>> gdf = gpd.read_file(REGION_CONFIG["ecuador"]["path"])
    >>> gdf = clean_region_shapefile(gdf, "ecuador")
    >>> plot_crop_metric_in_region(raster_dict, "BANP", "HANPPharv", "ecuador", "2005")

Author: Zelbi
"""
import re
import collections
import logging
from pathlib import Path

import rasterio
import rasterio.mask
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

from config import (REGION_CONFIG, clean_region_shapefile,
                    DATA_PATH, FIGURE_PATH)

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ----------------------------------------------------------------------------
# Raster paths dictionary builder
# ----------------------------------------------------------------------------
def build_raster_path_dict(data_path=DATA_PATH):
    """
    Build a nested dictionary of raster (.tif) paths.
    Structure: raster_dict[crop][metric][year] = path_to_tif

    Args:
    data_path (str | Path): Base data directory

    Returns:
    dict: Nested dictionary of raster paths
    """

    # Available crops in the original dataset:
    crops=[
     'WHEA', # (wheat)
     'RICE', # (rice)
     'BARL', # (barley)
     'MAIZ', # (maize)
     'REST', # (rest of crops)
     'MILL', # (millet)
     'SORG', # (sorghum)
     'POTA', # (potato)
     'SWPY', # (sweet potato and yam)
     'CASS', # (cassava)
     'SUGC', # (sugarcane)
     'SUGB', # (sugarbeet)
     'BEAN', # (beans)
     'OPUL', # (other pulses)
     'SOYB', # (soybean)
     'GROU', # (groundnuts)
     'OOIL', # (other oilcrops)
     'COTT', # (cotton)
     'BANP', # (banana / plantain)
     'COFF', # (coffee)
     'VEFR', # (vegetables / fruits)
     'OFIB', # (other fibers)
     'FODD', # (fodder crops)
     'FALL'] # (fallow)

    base_metrics = ['area', 'HANPPharv', 'HANPPluc', 'NPPeco']

    raster_dict = collections.defaultdict(lambda: collections.defaultdict(dict))

    for crop in crops:
        metrics = base_metrics.copy()

        if crop == 'FALL': # Special case: fallow (maggese) has no harvest metric
            metrics.remove('HANPPharv')

        for metric in metrics:
            unit = 'km2' if metric == 'area' else 'tC'
            folder = Path(data_path) / "spatially_explicit_land_use" / "tif" / f"CL-{crop}" / metric

            if not folder.is_dir():
                continue

            # Extract available years from filenames
            years = sorted({
                match.group(1)
                for f in folder.iterdir()
                if f.is_file() and (match := re.match(rf'(\d{{4}}){metric}_{unit}_CL-{crop}\.tif',
                                                      f.name))
            })

            if not years:
                logging.warning("No raster files found for %s/%s", crop, metric)
                continue

            for year in years:
                raster_path = folder / f"{year}{metric}_{unit}_CL-{crop}.tif"
                raster_dict[crop][metric][year] = str(raster_path)

    return raster_dict

# ----------------------------------------------------------------------------
# Raster visualization
# ----------------------------------------------------------------------------
def plot_raster_within_shape(raster_path: str, gdf: gpd.GeoDataFrame, ax):
    """
    Clip and plot a raster within shapefile boundaries.

    Args:
    raster_path (str): Path to raster (.tif)
    gdf (GeoDataFrame): Shapefile
    ax (matplotlib axis): Axis to plot

    Returns:
    (imshow, float): Image handle and pixel sum
    """
    with rasterio.open(raster_path) as src:
        # Coordinate reference system (e.g. EPSG:4326 = WGS84 lat/lon)
        # Reproject shapefile to match raster CRS
        # (required because raster and vector must share the same coordinate system)
        coord_ref_syst = src.crs
        gdf = gdf.to_crs(coord_ref_syst)

        # Mask / clip raster to shapefile
        masked_data, transform = rasterio.mask.mask(
            src,
            gdf.geometry,  # geometry(ies) used to clip
            crop=True,       # crop raster extent to polygon bbox (drop everything outside)
            filled=True,     # fill pixels outside polygon
            nodata=np.nan    # use NaN as "nodata" fill value
        )
        # masked_data     → NumPy array of clipped pixels
        # transform → affine transform describing new raster’s spatial location

    # --- Use masked_transform to georeference raster correctly ---
    # --> Without extent, imshow would just plot in pixel space (0..N),
    #     not in geographic coordinates.
    im = ax.imshow(
        masked_data[0],
        cmap='viridis',
        extent=[
            transform[2],  # xmin
            transform[2] + transform[0] * masked_data[0].shape[1],  # xmax
            transform[5] + transform[4] * masked_data[0].shape[0],  # ymin
            transform[5]   # ymax
        ]
    )

    # Calculate pixel sum and add textbox
    pixel_sum = np.nansum(masked_data[0])
    ax.text(
        0.60, 0.1,
        #0.05, 0.7,
        f'Total: {pixel_sum:.3e}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox={'boxstyle':'round', 'facecolor':'white', 'alpha':0.8}
        )

    return im, pixel_sum, coord_ref_syst

# ----------------------------------------------------------------------------
# Main plot function
# ----------------------------------------------------------------------------
def plot_crop_metric_in_region(raster_dict, crop, metric, region, year):
    """
    Plot raster for a crop/metric/year clipped to a region.
    """
    try:
        raster_path = raster_dict[crop][metric][year]
    except KeyError:
        logging.error("No data for %s/%s/%s", crop, metric, year)
        return None

    unit = 'km2' if metric == 'area' else 'tC'

    gdf = gpd.read_file(REGION_CONFIG[region]["path"])
    gdf = clean_region_shapefile(gdf, region)

    with rasterio.open(raster_path) as src:
        gdf = gdf.to_crs(src.crs)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im, pixel_sum = plot_raster_within_shape(raster_path, gdf, ax)

    fig.colorbar(im, ax=ax, label=f'{metric.capitalize()} ({unit})')
    ax.set_title(f'{crop} {metric} in {region.capitalize()}, {year}')
    plt.tight_layout()

    Path(FIGURE_PATH).mkdir(parents=True, exist_ok=True)
    out_file = Path(FIGURE_PATH) / f"{crop}_{metric}_{region}_{year}.png"
    plt.savefig(out_file, dpi=600)
    plt.close()

    logging.info("Saved plot to %s", out_file)
    return pixel_sum

# ----------------------------------------------------------------------------
# Zonal statistics
# ----------------------------------------------------------------------------
def compute_zonal_stats(gdf, raster_path):
    """
    Compute zonal statistics for a shapefile over a raster.

    Args:
    gdf (GeoDataFrame): Shapefile
    raster_path (str): Path to raster

    Returns:
    GeoDataFrame: Stats with ADM columns and sums
    """
    #with rasterio.open(raster_path) as src:
    #    # Clip the raster to the bounding geometry of the shape
    #    out_image, out_transform = rasterio.mask.mask(
    #        src, shape.geometry, crop=True, nodata=np.nan
    #    )
    #    out_meta = src.meta.copy()
    #    out_meta.update({
    #        "height": out_image.shape[1],
    #        "width": out_image.shape[2],
    #        "transform": out_transform
    #    })
    ## zonal_stats can take ndarray + affine directly
    #zone_stats = zonal_stats(
    #    vectors=shape,
    #    raster=out_image[0],     # first band
    #    affine=out_transform,
    #    stats=["sum"],
    #    all_touched=False,
    #    geojson_out=True,
    #    nodata=np.nan
    #)
    stats = zonal_stats(
        vectors=gdf,
        raster=raster_path,
        stats=["sum"],
        # Include a pixel if more than half of it is within the shape
        # (we lose something at the outer border)
        all_touched=False,

        geojson_out=True,  # must be True for from_features
    )
    stats_gdf = gpd.GeoDataFrame.from_features(stats)
    return stats_gdf[["ADM0", "ADM1", "ADM2", "sum"]]

# ----------------------------------------------------------------------------
# Temporal zonal stats
# ----------------------------------------------------------------------------
def extract_temporal_zonal_stats(raster_dict, gdf, region, crop, metrics=None, years=None):
    """
    Compute temporal zonal stats for crop metrics.

    Args:
    raster_dict (dict): Raster dictionary
    gdf (GeoDataFrame): Region shapefile
    region (str): Region name
    crop (str): Crop code
    metrics (list, optional): List of metrics. Defaults to all.
    years (list, optional): List of years. Defaults to all.
    """
    if metrics is None:
        # List of 'area', 'HANPPharv'...
        metrics = list(raster_dict[crop].keys())

    for metric in metrics:
        out_file = Path(DATA_PATH, "spatially_explicit_land_use", f"{crop}_{region}_{metric}.csv")

        # Skip if file already exists
        if out_file.exists():
            logging.info("Skipping %s, already exists at %s", metric, out_file)
            continue
        if years is None:
            years = sorted(raster_dict[crop][metric].keys())

        # Dataframe where I store the zonal stats
        results_df = pd.DataFrame()

        for year in years:
            #if year not in ["2000", "2001"]: continue #debugging
            raster_path = raster_dict[crop][metric][year]
            logging.info("Processing %s/%s, year %s", crop, metric, year)

            yearly_stats = compute_zonal_stats(gdf, raster_path)
            yearly_stats['year'] = year
            yearly_stats['crop'] = crop
            yearly_stats['metric'] = metric
            results_df = pd.concat([results_df, yearly_stats], ignore_index=True)

        # Finalize DataFrame with yearly zonal stats data
        results_df = results_df.rename(columns={'sum': f'{metric}_sum'})
        out_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_file, index=False)
        logging.info("Saved temporal stats to %s", out_file)

    return out_file

# ----------------------------------------------------------------------------
# Time series plots
# ----------------------------------------------------------------------------
def plot_temporal_evolution(df, metric, adm_level, ax):
    """
    Plot temporal evolution of crop metric per administrative level.
    """

    pivoted = (df.groupby(['year', adm_level])
               .agg({f'{metric}_sum': 'sum'})
               .reset_index()
               .pivot(index='year', columns=adm_level, values=f'{metric}_sum')
               )

    crop = df['crop'].unique()[0]
    unit = 'km²' if metric == 'area' else 'tC'
    region = df['ADM0'].unique()[0]

    # Get 23 colors from a colormap
    colors = (plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors)[:23]# take first 23 # pylint: disable=no-member

    # Apply to plot
    pivoted.plot(marker='o', color=colors, ax=ax)

    #ax.set_xticks(rotation=90)
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{metric.upper()} ({unit})')
    ax.set_title(f'{crop} {metric.upper()} over time in {region} ({adm_level})')
    ax.grid('grey', lw=.5)

    # Legend outside
    ax.legend(
        title=adm_level.title(),
        bbox_to_anchor=(1.05, 1), loc="upper left",
        ncol=2
    )

# ----------------------------------------------------------------------------
# Stats in year choropleth visualization
# ----------------------------------------------------------------------------
def color_polygons_metric_in_year(
    merged_gdf: gpd.GeoDataFrame,
    year: str | int,
    cmap: str = "viridis",
    ax: plt.Axes = None
):
    """
    Plot a choropleth heatmap of zonal stats merged with polygons for a given year.

    Args:
        merged_gdf (GeoDataFrame): GeoDataFrame that already contains polygons merged
            with zonal stats, including columns ['crop', 'metric', 'ADM0', 'year', '<metric>_sum'].
        year (str | int): Year to filter stats for (used in title).
        cmap (str, optional): Colormap for visualization (default: "viridis").
        ax (plt.Axes, optional): Axis to plot on. Must be provided.

    Returns:
        matplotlib.colorbar.Colorbar: The colorbar handle
    """
    if ax is None:
        raise ValueError("An axis (ax) must be provided for plotting.")

    crop = merged_gdf['crop'].unique()[0]
    metric = merged_gdf['metric'].unique()[0]
    region = merged_gdf['ADM0'].unique()[0]
    unit = 'km2' if metric == 'area' else 'tC'

    # Plot choropleth
    merged_gdf.plot(
        column=f"{metric}_sum",
        ax=ax,
        legend=False,
        cmap=cmap,
    )

    ax.set(title=f"{crop} {metric} in {region}, year {year}")

    # Colorbar setup
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=merged_gdf[f"{metric}_sum"].min(),
            vmax=merged_gdf[f"{metric}_sum"].max(),
        ),
    )
    sm._A = []  # hack to enable ScalarMappable without direct data
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{crop} {metric} ({unit})")

    return cbar


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load FAOSTAT data folder names
    tif_raster_dict = build_raster_path_dict()

    # Choose crop, metric, shapefile
    CROP = 'BANP'
    REGION = 'ecuador'
    #CROP = 'VEFR'
    #REGION = 'austria'
    METRIC = 'area' #'area' 'HANPPharv'
    UNIT = 'km2' if METRIC == 'area' else 'tC'

    logging.info("Loading and dissolving shapefile")
    geodf = gpd.read_file(REGION_CONFIG[REGION]["path"])
    adm2_geodf = clean_region_shapefile(geodf, REGION)
    adm1_geodf = adm2_geodf.dissolve(by=['ADM0', 'ADM1'], as_index=False)

    # --------------------------------------------------------
    # GeoSnapshot of metric, no aggregation (raster within boundaries)
    # --------------------------------------------------------
    logging.info("Plotting raster")
    shapefile_kwargs = {'edgecolor': 'red', 'color': 'none',
                        'linewidth': 0.3, 'linestyle': '-'}
    YEAR = 2015
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = plot_raster_within_shape(
        raster_path = tif_raster_dict[CROP][METRIC][str(YEAR)],
        gdf = adm2_geodf,
        ax=ax)
    fig.colorbar(im, ax=ax, label=f'{METRIC.capitalize()} value ({UNIT})')
    ax.set_title(f'{CROP} {METRIC} in {REGION.capitalize()}, {YEAR}' + '\nRaster within Boundaries')

    # Overlay shapefile boundaries
    adm1_geodf.plot(ax=ax, **{'edgecolor': 'yellow', 'color': 'none', 'linewidth': 0.3, 'linestyle': '-'})
    adm2_geodf.plot(ax=ax, **{'edgecolor': 'orange', 'color': 'none', 'linewidth': 0.1, 'linestyle': '--'})

    plt.tight_layout()
    plt.savefig(Path(FIGURE_PATH) / "LUIcube" / f"{REGION}_{CROP}_{METRIC}_raster_{YEAR}.png",
                format='png', dpi=600)
    plt.show()
    """
    # --------------------------------------------------------
    # Here I build the csv with the metric aggregated at ADM2
    # --------------------------------------------------------
    logging.info("Extracting temporal stats")
    extract_temporal_zonal_stats(tif_raster_dict,
                                 adm2_geodf,
                                 REGION, CROP, metrics=[METRIC],
                                 )

    # --------------------------------------------------------
    # Then, I load it and make plots
    # --------------------------------------------------------
    stats_df = pd.read_csv(Path(DATA_PATH) /
                    "spatially_explicit_land_use" / f"{CROP}_{REGION}_{METRIC}.csv",
                    dtype={'year': str})

    # --------------------------------------------------------
    # Temporal evolution, metric aggregated at ADM0, ADM1
    # --------------------------------------------------------
    for adm in ['ADM0', 'ADM1']:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_temporal_evolution(stats_df, METRIC, adm, ax)
        plt.tight_layout()
        plt.savefig(Path(FIGURE_PATH) / "LUIcube" / f"{REGION}_{CROP}_{METRIC}_temporal_evolution_{adm}.png",
                    format='png', dpi=600)
        plt.show()

    # --------------------------------------------------------
    # GeoSnapshot of metric, aggregated at ADM2
    # --------------------------------------------------------
    logging.info("Drawing ADM2-aggregated %s in %s", METRIC, YEAR)

    # Merge stats with polygons for the given year
    adm2_geogdf_plot = adm2_geodf.merge(stats_df[stats_df['year'] == str(YEAR)],
                                        how='outer')

    fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    color_polygons_metric_in_year(adm2_geogdf_plot, YEAR, ax=ax3)

    # Internal borders
    adm1_geodf.plot(**shapefile_kwargs, ax=ax3)
    ax3.set_axis_off()

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "LUIcube" / f"{REGION}_{CROP}_{METRIC}_ADM2_aggregated_{YEAR}.png",
                format='png', dpi=600)
    plt.show()
