"""
visual_flow_diagram_helpers.py

Utility functions for validating IDs, mapping nodes for visualization,
aggregating flows, and hiding unwanted edges in production-network graphs.
"""
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# ID validation
# ---------------------------------------------------------
def validate_ids(not_aggregated_IDs, firm_PN_IDs):
    """
    Validate that all visualization-level firm IDs exist in the
    full production network firm ID set.

    Parameters
    ----------
        not_aggregated_IDs : iterable
        IDs the user wants to include in the visualization.
        firm_PN_IDs : iterable
        Complete set of IDs available in the production network.

    Returns
    -------
        str
        Confirmation message if all IDs are valid.

    Raises
    ------
        ValueError
        If one or more visualization IDs are not found in firm_PN_IDs.
    """
    missing = set(not_aggregated_IDs) - set(firm_PN_IDs)
    if missing:
        raise ValueError(f"These IDs are not in prod. net. ID set: {missing}")
    return (
        "firm_visual_IDs is subset of firm_PN_IDs: "
        f"{set(not_aggregated_IDs).issubset(firm_PN_IDs)}"
    )

# ---------------------------------------------------------
# Node mapping helper
# ---------------------------------------------------------
def map_node(row, side, not_aggregated_IDs):
    """
    Map a supplier or customer to its visualization-level node label.
    
    Returns
    -------
    str
        Either:
        - A specific firm ID (if in PN and selected for visualization),
        - Its sector (if in PN but not selected for visualization),
        - Or an RoE placeholder string ("RoE_supplier" or "RoE_customer").
    """
    inPN = row[f"inPN_{side}"]
    firm_id = row[f"id_{side}"]
    sect = row[f"sector_{side}"]

    if not inPN:
        return f"RoE_{side}"

    if firm_id in not_aggregated_IDs:
        return firm_id

    return sect

# ---------------------------------------------------------
# Map all firms into visualization nodes
# ---------------------------------------------------------
def map_edge_ends_into_sankey_nodes(df, not_aggregated_IDs):
    """
    Convert raw supplier/customer IDs into visualization-level node labels
    using `map_node`.

    Parameters
    ----------
        df : pd.DataFrame
        Input edgelist with id/sector/inPN columns.
        not_aggregated_IDs : set
        IDs that should appear as firm-level nodes.

    Returns
    -------
        pd.DataFrame
        DataFrame with new columns 'sankey_supplier' and 'sankey_customer'.
    """
    df = df.copy()
    df["sankey_supplier"] = df.apply(lambda r: map_node(r, "supplier", not_aggregated_IDs), axis=1)
    df["sankey_customer"] = df.apply(lambda r: map_node(r, "customer", not_aggregated_IDs), axis=1)
    vis_nodes_set = set(df['sankey_supplier']).union(set(df['sankey_customer']))
    return df, vis_nodes_set

# ---------------------------------------------------------
# Remove RoE → RoE edges
# ---------------------------------------------------------
def hide_roe_to_roe(df):
    """
    Mark edges between RoE_supplier → RoE_customer as not visible.

    Parameters
    ----------
        df : pd.DataFrame
        Must contain 'sankey_', 'sankey_customer', and 'visible'.

    Returns
    -------
        pd.DataFrame
        DataFrame with RoE→RoE edges set to visible=False.
    """
    mask = (
        (df["visible"] == True) &
        (df["sankey_supplier"] == "RoE_supplier") &
        (df["sankey_customer"] == "RoE_customer")
    )
    df.loc[mask, "visible"] = False
    return df

# ---------------------------------------------------------
# Remove reverse-edge (repayment or counterflow)
# ---------------------------------------------------------
def hide_repayment(df, factor):
    """
    Hide edges whose reverse-edge weight is too large compared to the forward edge.
    Only considers currently visible edges.

    Logic
    -----
        - Construct a reversed version of the edge list.
        - Merge it back onto the original.
        - For visible edges, set visible=False if:
        reverse_weight > factor * weight

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - 'sankey_supplier'
        - 'sankey_customer'
        - 'weight'
        - 'visible'
        factor : float
        Ratio above which the reverse edge is considered dominant.

    Returns
    -------
        pd.DataFrame
        DataFrame with updated visibility status.
    """
    reverse = (
        df[df["visible"] == True][['sankey_supplier', 'sankey_customer', 'weight']]
        #.drop("visible", axis=1)
        .rename(columns={
            "sankey_supplier": "sankey_customer",
            "sankey_customer": "sankey_supplier",
            "weight": "reverse_weight",
        })
    )

    merged = df.merge(reverse, on=["sankey_supplier", "sankey_customer"], how="left")

    mask_update = merged["visible"] == True
    merged.loc[mask_update, "visible"] = (
        merged.loc[mask_update, "reverse_weight"].isna() |
        (
            merged.loc[mask_update, "reverse_weight"] <=
            factor * merged.loc[mask_update, "weight"]
        )
    )

    merged = merged.drop("reverse_weight", axis=1)
    return merged

# ---------------------------------------------------------
# Remove very small edges
# ---------------------------------------------------------
def hide_tiny_edges(df, factor):
    """
    Hide edges whose weight is below a fraction of the largest visible edge.
    Only considers currently visible edges.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'visible' and 'weight' columns.
    factor : float
        Edges smaller than `factor * max_visible_weight` will have their 
        'visible' column set to False.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'visible' column based on weight threshold.
    """
    largest_edge = df[df["visible"] == True]["weight"].max()
    mask = (df["visible"] == True) & (df["weight"] <= factor * largest_edge)
    df.loc[mask, "visible"] = False
    return df

# ---------------------------------------------------------
# Hide all edges involving RoE (either endpoint)
# ---------------------------------------------------------
def hide_all_roe_edges(df):
    """
    Hide any edge where either the supplier or customer is a RoE node.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - 'sankey_supplier'
        - 'sankey_customer'
        - 'visible'

    Returns
    -------
    pd.DataFrame
        DataFrame with visibility updated.
    """

    mask = (
        (df["visible"] == True) &
        (
            df["sankey_supplier"].str.contains("RoE", case=False, regex=False) |
            df["sankey_customer"].str.contains("RoE", case=False, regex=False)
        )
    )

    df.loc[mask, "visible"] = False
    return df

# ---------------------------------------------------------
# Count the firm IDs in each visualization node
# ---------------------------------------------------------
def compute_total_unique_firms_per_visual_node(df):
    """
    Simpler: returns sankey_node_label -> total_unique_firms (union of supplier & customer IDs).
    """
    sup = df[["id_supplier", "sankey_supplier"]].drop_duplicates().rename(
        columns={"id_supplier": "id", "sankey_supplier": "sankey_node_label"}
    )
    cust = df[["id_customer", "sankey_customer"]].drop_duplicates().rename(
        columns={"id_customer": "id", "sankey_customer": "sankey_node_label"}
    )

    union = pd.concat([sup, cust], ignore_index=True).drop_duplicates(subset=["id", "sankey_node_label"])

    return union.groupby("sankey_node_label", as_index=False)["id"].nunique().rename(columns={"id": "total_unique_firms"})

# ---------------------------------------------------------
# Aggregate firm-level flows to the visualization nodes level
# ---------------------------------------------------------
def aggregate_flows_between_sankey_nodes(df):
    """
    Aggregate transaction flows by visualization-level node pairs.

    Parameters
    ----------
        df : pd.DataFrame
        Must contain columns:
        - 'sankey_supplier'
        - 'sankey_customer'
        - 'weight'

    Returns
    -------
        pd.DataFrame
        Aggregated flows with column 'visible' set to True.
    """
    df = (
        df#[["weight", "sankey_supplier", "sankey_customer"]]
        .groupby(["sankey_supplier", "sankey_customer"], as_index=False)
        .agg(weight = ('weight','sum'),
             trans_uni = ('weight','count'),
             supp_uni = ('id_supplier', 'nunique'),
             cust_uni = ('id_customer', 'nunique')
            )
    )

    df["visible"] = True
    return df

# ---------------------------------------------------------
# Compute attributes (s_in, adm) for the aggregated nodes
# ---------------------------------------------------------
def compute_cw_strengths_of_sankey_nodes(flows_df, ID_TO_SECTOR, SECTOR_TO_DESCR, ID_TO_ADM):
    """
    Calculate node-level metrics from edge list.
    
    Parameters
    ----------
    flows_df : pd.DataFrame
        Edge list with columns: sankey_supplier, sankey_customer, weight
    ID_TO_SECTOR : dict
        Mapping from firm ID to ISIC4 sector
    SECTOR_TO_DESCR : dict
        Mapping from ISIC4 code to description
    ID_TO_ADM : dict
        Mapping from firm ID to administrative region
    
    Returns
    -------
    pd.DataFrame
        Node-level dataframe with columns: firm, s_out, s_in, ISIC4, ISIC4_descr, ADM
    """
    # Calculate out-strength
    out_strength = flows_df.groupby('sankey_supplier')['weight'].sum().reset_index()
    out_strength.columns = ['firm', 's_out']

    # Calculate in-strength
    in_strength = flows_df.groupby('sankey_customer')['weight'].sum().reset_index()
    in_strength.columns = ['firm', 's_in']

    # Merge
    nodes_df = out_strength.merge(in_strength, on='firm', how='outer').fillna(0)

    # Add attributes
    nodes_df['ISIC4'] = nodes_df['firm'].map(ID_TO_SECTOR).fillna(nodes_df['firm'])
    nodes_df['ISIC4_descr'] = nodes_df['ISIC4'].map(SECTOR_TO_DESCR).fillna('Aggregated')
    nodes_df['ADM'] = nodes_df['firm'].map(ID_TO_ADM).fillna('Aggregated')

    return nodes_df

# ---------------------------------------------------------
# Extract nodes that are present in the visible flows
# ---------------------------------------------------------
def extract_visible_nodes(flows_df):
    """
    Extract unique visible nodes from flows.
    """
    visible_flows = flows_df[flows_df['visible'] == True]
    vis_nodes = pd.concat([
        visible_flows['sankey_supplier'],
        visible_flows['sankey_customer']
    ]).unique()

    return pd.DataFrame({
        'sankey_node_label': vis_nodes,
        'x': None,
        'y': None
    })

# ---------------------------------------------------------
# Map nodes to indices, and get dictionary map for the flows
# ---------------------------------------------------------
def create_node_index_mapping(vis_nodes_df):
    """
    Create mapping from node labels to indices and add to dataframe.
    """
    label_to_idx = {label: i for i, label in enumerate(vis_nodes_df['sankey_node_label'])}
    vis_nodes_df['idx'] = vis_nodes_df['sankey_node_label'].map(label_to_idx)
    return vis_nodes_df, label_to_idx

# ---------------------------------------------------------
# Add firm data to visible nodes dataframe
# ---------------------------------------------------------
def enrich_nodes_with_flow_data(vis_nodes_df, nodes_df):
    """Merge firm information into visible nodes."""
    return vis_nodes_df.merge(
        nodes_df.rename(columns={'firm': 'sankey_node_label'}),
        on='sankey_node_label',
        how='left'
    )

def calculate_strength_captured_in_pn(vis_nodes_df):
    """Calculate what percentage of total strength is inside pn."""
    vis_nodes_df['capt_cw_s_out_perc'] = (
        100 * vis_nodes_df['pn_s_out'] / vis_nodes_df['cw_s_out']).round(1).fillna(0)

    vis_nodes_df['capt_cw_s_in_perc'] = (
        100 * vis_nodes_df['pn_s_in'] / vis_nodes_df['cw_s_in']).round(1).fillna(0)
    return vis_nodes_df
# ---------------------------------------------------------
# Add custom positions to visible nodes fo the plot
# ---------------------------------------------------------
def set_custom_node_positions(vis_nodes_df):
    """Set custom x,y positions for specific nodes in the visualization."""
    position_map = {
        'RoE_supplier': (0.01, 0.01),
        'RoE_customer': (0.99, 0.01),
        'A0122': (0.5, 0.1),
        'C1702': (0.1, 0.6),
        '1914591': (0.001, 0.7),
        '1798870': (0.001, 0.9),
        '1956583': (0.8, 0.9),
        '1797601': (0.8, 0.7)
    }

    position_map = {
        'RoE_supplier': (0.01, 0.5),
        'RoE_customer': (0.99, 0.5),
        'A0122': (0.5, 0.5),
        'G4669': (0.2, 0.9),
        'S9609': (0.2, 0.8),
        'C2220': (0.2, 0.3),
        'C1701': (0.3, 0.2),
        'C1702': (0.45, 0.01),
        'G4620': (0.6, 0.99),
        'A0163': (0.6, 0.85),
        'G4772': (0.6, 0.7),
        'G4630': (0.8, 0.01),
    }

    for node_label, (x, y) in position_map.items():
        mask = vis_nodes_df['sankey_node_label'] == node_label
        vis_nodes_df.loc[mask, ['x', 'y']] = (x, y)

    return vis_nodes_df

# ---------------------------------------------------------
# Map flow end nodes with indices and get plotly df
# ---------------------------------------------------------
def build_visible_flows_with_indices(flows_df, label_to_idx):
    """
    Build dataframe of visible flows with mapped indices and log weights.
    """
    vis_flows_df = flows_df[flows_df['visible'] == True].copy()

    # Map node labels to indices
    vis_flows_df['source_idx'] = vis_flows_df['sankey_supplier'].map(label_to_idx)
    vis_flows_df['target_idx'] = vis_flows_df['sankey_customer'].map(label_to_idx)

    # Calculate log weights for visualization
    vis_flows_df['log_weight'] = np.log10(vis_flows_df['weight'])

    return vis_flows_df

# ---------------------------------------------------------
# Quantify visible data
# ---------------------------------------------------------
def calculate_visible_flow_strengths(vis_flows_df):
    """
    Not all the aggregated (!) PN flows are visible: some links
    are hidden to simply the representation.
    Calculate in/out strengths from visible flows only.
    """
    # Outgoing strength from visible flows
    visible_s_out = vis_flows_df.groupby('sankey_supplier')['weight'].sum().reset_index()
    visible_s_out.columns = ['sankey_node_label', 'visible_s_out']

    # Incoming strength from visible flows
    visible_s_in = vis_flows_df.groupby('sankey_customer')['weight'].sum().reset_index()
    visible_s_in.columns = ['sankey_node_label', 'visible_s_in']

    # Merge and fill missing values
    return visible_s_out.merge(visible_s_in, on='sankey_node_label', how='outer').fillna(0)

def aggregate_firm_data(firm_data_df, sankey_nodes_set):
    sankey_ids_df = (firm_data_df[firm_data_df['firm_id'].isin(sankey_nodes_set)]
                        .rename({'firm_id':'sankey_node_label'}, axis=1))
    sankey_aggregated_df = (firm_data_df.drop(sankey_ids_df.index)
                        .groupby(['ISIC4', 'descrip_n4'])
                        .agg({'cw_s_out': 'sum', # is cw_socied_&_pers_nat_s_out
                            'cw_s_in': 'sum',
                            'pn_s_in': 'sum',
                            'pn_k_in': 'sum',
                            'pn_s_out': 'sum',
                            'pn_k_out': 'sum',
                            'descrip_1': 'unique',
                            'firm_id': 'nunique',
                            'ADM2': 'nunique', 'ADM1': 'nunique'
                            }).reset_index()
                    )
    sankey_aggregated_df['sankey_node_label'] = sankey_aggregated_df['ISIC4']
    aggr_firm_data_df = pd.concat([sankey_ids_df, sankey_aggregated_df], join='inner')
    return aggr_firm_data_df

# ---------------------------------------------------------
# Quantify visible data and add it visible nodes df
# ---------------------------------------------------------
def add_visibility_percentages(vis_nodes_df):
    """Calculate what percentage of total strength is visible."""
    vis_nodes_df['visible_cw_s_out_perc'] = (
        100 * vis_nodes_df['visible_s_out'] / vis_nodes_df['cw_s_out']).round(1).fillna(0)

    vis_nodes_df['visible_cw_s_in_perc'] = (
        100 * vis_nodes_df['visible_s_in'] / vis_nodes_df['cw_s_in']).round(1).fillna(0)
    return vis_nodes_df



# ---------------------------------------------------------
# Quantify link strength share
# ---------------------------------------------------------
"""
supplier_out = (
    visual_flows_df
    .groupby("sankey_supplier")["weight"]
    .transform("sum")
)

visible_flows_df['suppl_s_out_share'] = round(100 * visible_flows_df["weight"] / supplier_out, 2)
visible_flows_df
"""