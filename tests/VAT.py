import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_PATH
from tests.NW_validation_modules import compute_io_table_sector_level

def load_firm_data_cw(year=None, contr_type=None):
    """
    Load country-wide firm-level data ('cw'), returning a DataFrame.

    Args:
        year (int, optional): Specific year to filter the data.

    Returns:
        pd.DataFrame: Filtered firm data with renamed administrative columns.
    """
    filename = "nodelist_cw.csv"
    if contr_type:
        filename = f"nodelist_cw{contr_type}.csv"
    filepath = Path(DATA_PATH) / "firm-level" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading file: {filename}")
    firms_df = pd.read_csv(
        filepath,
        sep='\t',
        usecols= ['firm_id', 'date', 'descrip_1', 'ISIC4', 'descrip_n4', 'province', 'canton',
                  'cw_s_out', 'cw_s_in'],
        dtype={'firm_id': str, 'date': int, "out_strength": float},
        encoding='utf-8'
    )

    # Filter by year (optional)
    if year:
        firms_df = firms_df[firms_df["date"] == year]

    # Round strength to cents
    for col in ['cw_s_out', 'cw_s_in']:
        firms_df[col] = firms_df[col].round(2)
    #firms_df['cw_s_tot'] = firms_df['cw_s_out'] + firms_df['cw_s_in']

    # Rename administrative columns
    return firms_df.rename({'province': 'ADM1', 'canton': 'ADM2'}, axis=1)

def load_firm_data_sector(sector, contr_type='all', year=None):
    """
    Load firm-level data for a specific ISIC sector and contributor type, returning a DataFrame.

    Args:
        sector (str): ISIC code like 'A0122' (MUST NOT be 'cw').
        contr_type (str): Contributor type: 'all', 'sociedades', or 'personas'.
        year (int, optional): Specific year to filter the data.

    Returns:
        pd.DataFrame: Filtered firm data with renamed administrative columns.
    
    Raises:
        ValueError: If 'sector' is set to 'cw'.
    """
    if sector == "cw":
        # Raise an error to enforce separation of concerns
        raise ValueError("Use the 'load_firm_data_cw()' function for 'cw' data.")

    filename = f"nodelist_{sector}{contr_type}.csv"
    filepath = Path(DATA_PATH) / "firm-level" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading file: {filename}")
    firms_df = pd.read_csv(
        filepath,
        sep='\t',
        usecols= ['firm_id', 'date', 'descrip_1', 'ISIC4', 'descrip_n4', 'province', 'canton',
                  'cw_s_out', 'cw_s_in'],
        dtype={'firm_id': str, 'date': int, "out_strength": float}
    )

    # 1. Filter by year (optional)
    if year:
        firms_df = firms_df[firms_df["date"] == year]

    # 2. Filter by contributor type (skip if contr_type='all')
    if contr_type in ("personas", "sociedades"):
        name_contr = "Personas naturales" if contr_type == "personas" else "Sociedades"
        firms_df = firms_df[firms_df["descrip_1"] == name_contr]

    # 3. Filter by ISIC sector code
    firms_df = firms_df[firms_df["ISIC4"] == sector]

    # Round strength to cents
    for col in ['cw_s_out', 'cw_s_in']:
        firms_df[col] = firms_df[col].round(2)
    #firms_df['cw_s_tot'] = firms_df['cw_s_out'] + firms_df['cw_s_in']

    # Rename administrative columns
    return firms_df.rename({'province': 'ADM1', 'canton': 'ADM2'}, axis=1)

def load_link_data_cw(year=None, contr_type=None):
    """
    Load country-wide link-level data ('cw'), returning a DataFrame.

    Args:
        year (int, optional): Specific year to filter the data.

    Returns:
        pd.DataFrame: Filtered link data (edges).
    """
    filename = "edgelist_cw.csv"
    if contr_type:
        filename = f"edgelist_cw{contr_type}.csv"
    filepath = Path(DATA_PATH) / "firm-level" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading file: {filename}")
    links_df = pd.read_csv(
        filepath,
        sep='\t',
        dtype={'id_supplier': str,
               'id_customer': str,
               'weight': float,
               'date': int}
    )

    # Round values to cents
    links_df['weight'] = links_df['weight'].round(2)

    # Filter by year (optional)
    if year:
        links_df = links_df[links_df["date"] == year]

    return links_df

def load_link_data_sector(sector, contr_type='all', year=None):
    """
    Load link-level data for a specific ISIC sector and contributor type, returning a DataFrame.

    Args:
        sector (str): ISIC code like 'A0122' (MUST NOT be 'cw').
        contr_type (str): Contributor type: 'all', 'sociedades', or 'personas'.
        year (int, optional): Specific year to filter the data.

    Returns:
        pd.DataFrame: Filtered link data (edges).

    Raises:
        ValueError: If 'sector' is set to 'cw'.
    """
    if sector == "cw":
        # Raise an error to enforce separation of concerns
        raise ValueError("Use the 'load_link_data_cw()' function for 'cw' data.")

    filename = f"edgelist_{sector}{contr_type}.csv"
    filepath = Path(DATA_PATH) / "firm-level" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading file: {filename}")
    links_df = pd.read_csv(
        filepath,
        sep='\t',
        dtype={'id_supplier': str,
               'id_customer': str,
               'weight': float,
               'date': int}
    )

    # Filter by year (optional)
    if year:
        links_df = links_df[links_df["date"] == year]

    # Round values to cents
    links_df['weight'] = links_df['weight'].round(2)

    return links_df

# Some links remain very weird. I want to update the cw strengths after removing links.
def recalculate_strengths_from_links(firm_register_df, links_records_df, prefix='subnw_'):
    """
    Recalculates the in-strength (s_in) and out-strength (s_out) for each firm 
    in each year based on the link weights.

    Args:
        firm_register_df (pd.DataFrame): DataFrame containing firm IDs and years 
                                         (columns: 'firm_id', 'date').
        links_records_df (pd.DataFrame): DataFrame containing link details 
                                         (columns: 'id_supplier', 'id_customer', 'weight', 'date').

    Returns:
        pd.DataFrame: The original firm_register_df with two new columns: 
                      's_in' (in-strength) and 's_out' (out-strength).
    """

    # --- 1. Calculate Out-Strength (s_out) ---
    # Group by the supplier (id_supplier) and year (date), and sum the weight.
    s_out_df = links_records_df.groupby(['id_supplier', 'date'])['weight'].sum().reset_index()
    s_out_df.rename(columns={'id_supplier': 'firm_id', 'weight': f'{prefix}s_out'}, inplace=True)

    # --- 2. Calculate In-Strength (s_in) ---
    # Group by the customer (id_customer) and year (date), and sum the weight.
    s_in_df = links_records_df.groupby(['id_customer', 'date'])['weight'].sum().reset_index()
    s_in_df.rename(columns={'id_customer': 'firm_id', 'weight': f'{prefix}s_in'}, inplace=True)

    # --- 3. Merge Results into Firm Register ---

    # Start with the firm register (node list)
    result_df = firm_register_df.copy()

    # Merge s_out data (Left Join ensures all firms in the register are kept)
    result_df = pd.merge(
        result_df,
        s_out_df,
        on=['firm_id', 'date'],
        how='left'
    )

    # Merge s_in data
    result_df = pd.merge(
        result_df,
        s_in_df,
        on=['firm_id', 'date'],
        how='left'
    )

    # Replace NaN values (for firms with no links in a given year) with 0.0
    result_df[f'{prefix}s_out'] = result_df[f'{prefix}s_out'].fillna(0.0)
    result_df[f'{prefix}s_in'] = result_df[f'{prefix}s_in'].fillna(0.0)

    return result_df

def split_and_aggregate_edgelist(cw_edgelist: pd.DataFrame,
                                 firm_PN_IDs: set, verbose: bool=False):
    """
    Split cw_edgelist into three subsets (PN-PN, PN–RoE, RoE-only),
    compute an aggregated RoE row, and return the combined global edgelist.

    Parameters
    ----------
    cw_edgelist : pd.DataFrame
        Input edgelist (countrywide) containing id_supplier, id_customer, and weight.
    firm_PN_IDs : set
        Set of firm IDs considered part of the PN.

    Returns
    -------
    global_edgelist : pd.DataFrame
        Combined DataFrame of PN, PN+RoE, and aggregated RoE-only links.
    pn_edgelist : pd.DataFrame
    pn_and_roe_edgelist : pd.DataFrame
    roe_edgelist : pd.DataFrame
    """
    year = cw_edgelist['date'].unique()[0]

    # Identify PN membership
    in_sup = cw_edgelist['id_supplier'].isin(firm_PN_IDs)
    in_cus = cw_edgelist['id_customer'].isin(firm_PN_IDs)

    cw_edgelist = cw_edgelist.copy()
    cw_edgelist['inPN_supplier'] = in_sup
    cw_edgelist['inPN_customer'] = in_cus

    # Split indices
    idx_both = cw_edgelist.index[in_sup & in_cus]       # PN → PN
    idx_one  = cw_edgelist.index[in_sup ^ in_cus]       # PN ↔ RoE (XOR)
    idx_none = cw_edgelist.index[~in_sup & ~in_cus]     # RoE → RoE

    # Subsets
    pn_edgelist = cw_edgelist.loc[idx_both]
    pn_and_roe_edgelist = cw_edgelist.loc[idx_one]
    roe_edgelist = cw_edgelist.loc[idx_none]

    if verbose:
        print("Firm-level PN link count:", pn_edgelist.shape[0])
        print("Firm-level PN link to and from Ecuador count:", pn_and_roe_edgelist.shape[0])

    # Aggregated RoE row
    aggregated_row = {
        'id_supplier': 'RoE_supplier',
        'id_customer': 'RoE_customer',
        'weight': roe_edgelist['weight'].sum(),

        'date': year,

        'sector_supplier': np.nan,
        'sector_customer': np.nan,
        'ADM_supplier': np.nan,
        'ADM_customer': np.nan,

        'inPN_supplier': False,
        'inPN_customer': False,
    }

    roe_aggregated = pd.DataFrame([aggregated_row])

    # Combine all parts
    global_edgelist = pd.concat(
        [pn_edgelist, pn_and_roe_edgelist, roe_aggregated],
        ignore_index=True
    )

    return global_edgelist, pn_edgelist, pn_and_roe_edgelist, roe_edgelist


if __name__ == "__main__":

    firms_df = load_firm_data_cw(year=2012, contr_type='Sociedades')
    links_df = load_link_data_cw(year=2012, contr_type='Sociedades')

    ID_TO_SECTOR = dict(zip(firms_df["firm_id"], firms_df["ISIC4"]))

    links_df['sector_supplier'] = links_df['id_supplier'].map(ID_TO_SECTOR)
    links_df['sector_customer'] = links_df['id_customer'].map(ID_TO_SECTOR)

    table = compute_io_table_sector_level(links_df)