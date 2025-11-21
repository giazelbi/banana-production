from pathlib import Path
import json
import geopandas as gpd

# Path to the data and figure folders inside the package
DATA_PATH = Path(__file__).parent / "data"
FIGURE_PATH = Path(__file__).parent / "figures"

# Ensure they exist (optional)
DATA_PATH.mkdir(exist_ok=True)
FIGURE_PATH.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Shapefile configuration
# ----------------------------------------------------------------------------
REGION_CONFIG = {
    "ecuador": {
        "path": Path(DATA_PATH) / "geography" / "ecu_adm_2024" / "ecu_adm_adm2_2024.shp",
        "rename": {"ADM0_ES": "ADM0", "ADM1_ES": "ADM1", "ADM2_ES": "ADM2"},
        "filter": lambda df: df[df["ADM1"] != "Galapagos"], # filter to remove certain rows
        "keep": ["ADM0", "ADM1", "ADM2", "geometry"],  # keep only these
    },
    "austria": {
        "path": Path(DATA_PATH) / "geography" / "VGD_Oesterreich_gst_20230403" / "VGD.shp",
        "rename": {"ST": "ADM0", "BL": "ADM1", "PB": "ADM2"},
        "filter": lambda df: df,#[df["ADM1"].isin(["Wien", "Niederösterreich"])],
        "keep": ["ADM0", "ADM1", "ADM2", "geometry"],
    },
}

def clean_region_shapefile(gdf: gpd.GeoDataFrame, region: str) -> gpd.GeoDataFrame:
    """
    Clean and standardize shapefile based on region config.

    Args:
    gdf (GeoDataFrame): Input shapefile
    region (str): Region key from REGION_CONFIG

    Returns:
    GeoDataFrame: Cleaned shapefile
    """
    cfg = REGION_CONFIG.get(region)
    if not cfg:
        raise ValueError(f"Unknown region: {region}")

    # --- Ecuador-specific remapping with conventional code:name dict---
    if region == 'ecuador':
        # Load dictionaries
        with open(Path(DATA_PATH) / "ADM1_PCODE_description.json", "r", encoding="utf-8") as f:
            province_dict = json.load(f)
        with open(Path(DATA_PATH) / "ADM2_PCODE_description.json", "r", encoding="utf-8") as f:
            canton_dict = json.load(f)
        # First rename columns (e.g., ADM1_ES → ADM1)
        gdf = gdf.rename(cfg["rename"], axis=1)

        # Replace codes with human-readable names
        gdf["ADM1"] = gdf["ADM1_PCODE"].map(province_dict).fillna("Unknown")
        gdf["ADM2"] = gdf["ADM2_PCODE"].map(canton_dict).fillna("Unknown")
    else:
        # For other regions, just rename
        gdf = gdf.rename(cfg["rename"], axis=1)

    # --- Apply common cleaning steps ---
    gdf = cfg["filter"](gdf)

    # Dissolve by ADM2 (aggregate shapes that are smaller than ADM2)
    gdf = gdf.dissolve(by=["ADM0", "ADM1", "ADM2"], as_index=False)

    # Keep only region-specific subset of columns (if defined)
    if "keep" in cfg:
        cols = [c for c in cfg["keep"] if c in gdf.columns]
        # Ensure geometry is last and not duplicated
        if "geometry" not in cols:
            cols.append("geometry")
        gdf = gdf[cols]

    return gdf
