"""
Module gathers the operations to visualize features of the production network.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from config import DATA_PATH, FIGURE_PATH
from tests.visual_flow_diagram_helpers import (map_edge_ends_into_visual_nodes,
                                               compute_total_unique_firms_per_visual_node,
                                               aggregate_flows, hide_roe_to_roe, hide_repayment,
                                               hide_all_roe_edges,
                                               hide_tiny_edges, compute_aggregated_node_attrs,
                                               extract_visible_nodes, create_node_index_mapping,
                                               enrich_nodes_with_firm_data,
                                               set_custom_node_positions,
                                               build_visible_flows_with_indices,
                                               calculate_visible_flow_strengths,
                                               add_visibility_percentages)
# =========================================================
# Input ouput flow table at sector level inside the production network
# =========================================================
def compute_io_table_sector_level(links_df, isic_lev = 4, verbose=False):
    """
    Computes a sector-level Input-Output (IO) table from firm-level link data.

    This function aggregates the total weight, the number of unique suppliers, 
    and the number of unique customers for each unique pair of supplier sector 
    and customer sector in each year.

    Args:
        links_df (pd.DataFrame): DataFrame containing link-level records with 
                                 at least these columns: 
                                 'date' (int, year of the link), 
                                 'sector_supplier' (str or int, sector of the source firm), 
                                 'sector_customer' (str or int, sector of the destination firm), 
                                 'weight' (float, strength/weight of the link),
                                 'id_supplier' (str, unique ID of the supplier firm),
                                 'id_customer' (str, unique ID of the customer firm).

    Returns:
        pd.DataFrame: An aggregated DataFrame representing the IO table at the 
                      sector level, with columns: 
                      'date', 'sector_supplier', 'sector_customer', 
                      'weight' (total trade volume), 
                      'id_supplier' (count of unique supplier firms), 
                      'id_customer' (count of unique customer firms).
    """

    # ------------------ Sector - sector aggregation
    if isic_lev == 1:
        char = isic_lev
    else:
        char = isic_lev + 1

    df = links_df.copy()
    df['sector_supplier'] = df['sector_supplier'].str[:char]
    df['sector_customer'] = df['sector_customer'].str[:char]
    result_df = df.groupby(['date', 'sector_supplier', 'sector_customer']
                                 ).agg({'weight': 'sum',
                                        'id_supplier': 'nunique', 'id_customer': 'nunique',}
                                        ).reset_index()


    # ------------------ Pivot the data into a matrix form suitable for imshow
    pivot = (result_df
             .pivot(index="sector_supplier", columns="sector_customer", values="weight")
             .fillna(0))
    if verbose:
        print('(row i, col j) = (sector_supplier, sector_customer)')
    # Replace zeros with NaN
    pivot = pivot.replace(0, np.nan)
    return pivot

def plot_io_pivottable(pivot, logscale, xlabel=None, ylabel=None, title='I/O table'):
    """
    Plots a pivot table as an input-output matrix
    
    :pivot: Description
    :logscale:
        Whether the colorscale should evaluate values or log(values)
    :xlabel: Description
    :ylabel: Description
    :title: Description
    """
    # Extract axis labels and matrix
    x_labels = pivot.columns
    y_labels = pivot.index
    matrix = pivot.values

    # Apply log scale
    data_matrix = matrix
    if logscale:
        data_matrix = np.log10(matrix)

    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap="viridis")

    # Ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(y_labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Annotate each cell with the value
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                if logscale:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="w")
                else:
                    ax.text(j, i, f"{val:.1e}", ha="center", va="center", color="w", fontsize=5)

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()

# =========================================================
# Provincial aggregation of firms in the production network
# compared with official data (CFN for the moment)
# =========================================================
def group_sectorfirms_per_province(banana_pn_nodelist, col_sum, sector=None, notable_prov=None):
    """
    Groups firms by sector, province, and year.
    """
    years = sorted(banana_pn_nodelist['date'].unique())
    sectors = sorted(banana_pn_nodelist['ISIC4'].dropna().unique())

    if not notable_prov:
        notable_prov = banana_pn_nodelist['ADM1'].unique()

    records = []

    for y in years:
        df_y = banana_pn_nodelist[banana_pn_nodelist['date'] == y]

        for sec in sectors:
            df_ys = df_y[df_y['ISIC4'] == sec]

            if df_ys.empty:
                continue

            g = (df_ys
                 .groupby('ADM1')
                 .agg(count=('ADM1', 'size'),
                      total=(col_sum, 'sum')
                      )
                )

            # notable provinces
            for prov in notable_prov:
                if prov not in g.index:
                    continue

                count_val = g.loc[prov, 'count']
                sum_val   = g.loc[prov, 'total']

                if count_val > 0:
                    records.append({
                        'province': prov,
                        'year': int(y),
                        'sector': sec,
                        'element': 'count',
                        'value': count_val
                    })
                    records.append({
                        'province': prov,
                        'year': int(y),
                        'sector': sec,
                        'element': col_sum,
                        'value': sum_val
                    })

            # Others
            other = g.loc[~g.index.isin(notable_prov)]

            if not other.empty:
                other_count = other['count'].sum()
                other_sum   = other['total'].sum()

                if other_count > 0:
                    records.append({
                        'province': 'Others',
                        'year': int(y),
                        'sector': sec,
                        'element': 'count',
                        'value': other_count
                    })
                    records.append({
                        'province': 'Others',
                        'year': int(y),
                        'sector': sec,
                        'element': col_sum,
                        'value': other_sum
                    })

    province_df = pd.DataFrame(records)

    # optional sector filter AFTER grouping
    if sector is not None:
        province_df = province_df[province_df['sector'] == sector]

    province_df = (
        province_df
        .sort_values(['year', 'sector', 'element', 'value'],
                     ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    province_df['source'] = 'VAT subnetwork'
    return province_df

def load_cfn_empresas_per_provincia(sector=None):
    """
    Load csv transcripted from official reports of
    'Corporacion Financiera Nacional' (public development bank in Ecuador)
    """
    df = pd.read_csv(DATA_PATH / "Corporacion Financiera Nacional" / "empresas.csv")

    # Calculate 'empresas' counts per region from percentages
    new_rows = []

    # Group by year and comment to process each group
    for (_, _), group in df.groupby(['year', 'comment']):
        # Find the Ecuador total for this group
        ecuador_row = group[group['province'] == 'Ecuador']

        if not ecuador_row.empty:
            total_value = ecuador_row['value'].values[0]
            unit_type = ecuador_row['element'].values[0]

            # For each region with percentage, calculate count
            percentage_rows = group[group['element'] == '%']
            for _, row in percentage_rows.iterrows():
                count_value = round(row['value'] * total_value / 100)
                new_rows.append({
                    'year': row['year'],
                    'province': row['province'],
                    'comment': row['comment'],
                    'element': unit_type,
                    'value': count_value
                })

    df_filled = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df_filled = (df_filled.sort_values(['year', 'comment', 'province', 'element'])
                 .reset_index(drop=True))

    # Optional sector filter
    if sector:
        df_filled = df_filled[df_filled['comment'].str[0:5] == f'{sector}']
    df_filled = df_filled[df_filled['province'] != 'Ecuador']

    # Drop percentage rows
    df_filled = df_filled[df_filled['element'] != '%']

    df_filled['source'] = 'cfn'
    return df_filled

def plot_stacked_bar_comparison(pivot_1, pivot_2, sector, column_color):
    """Plot stacked bar comparison between two pivot tables"""
    all_years = range(int(min(pivot_1.index.min(), pivot_2.index.min())),
                      int(max(pivot_1.index.max(), pivot_2.index.max())) + 1)

    def lighten_color(color, amount=0.5):
        """Lighten a color by blending it with white"""
        c = mcolors.to_rgb(color)
        return tuple(c[i] + (1 - c[i]) * amount for i in range(3))

    _, ax = plt.subplots(figsize=(10, 5))

    # In case of overlapping years/bars, adjust
    offset, width, rotation, perc = (0.2, 0.3, 90, 3)
    offset, width, rotation, perc = (0, 0.8, 0, 6)

    # Sort provinces by total values (descending)
    pivot_1_totals = pivot_1.sum().sort_values(ascending=False)
    pivot_2_totals = pivot_2.sum().sort_values(ascending=False)
    pivot_1 = pivot_1[pivot_1_totals.index]
    pivot_2 = pivot_2[pivot_2_totals.index]

    # Plot both pivots (base and lightened colors)
    bottom_data1 = [0] * len(pivot_1.index)
    bottom_data2 = [0] * len(pivot_2.index)

    for col in pivot_1.columns:
        color = column_color[col]
        ax.bar(x=pivot_1.index - offset, height=pivot_1[col], width=width,
               bottom=bottom_data1, color=color, label=col)
        bottom_data1 = [b + h for b, h in zip(bottom_data1, pivot_1[col])]

    for col in pivot_2.columns:
        color = column_color[col]
        ax.bar(x=pivot_2.index + offset, height=pivot_2[col], width=width,
               bottom=bottom_data2, color=lighten_color(color, amount=0.5))
        bottom_data2 = [b + h for b, h in zip(bottom_data2, pivot_2[col])]

    # Add percentage labels
    for year in pivot_1.index:
        cumulative_height = 0
        year_total = pivot_1.loc[year].sum()
        for col in pivot_1.columns:
            height = pivot_1.loc[year, col]
            percentage = (height / year_total) * 100 if year_total > 0 else 0
            if percentage >= perc:
                ax.text(year - offset, cumulative_height + height/2, f'{int(percentage)}%',
                       ha='center', va='center', fontsize=8, color='black', 
                       weight='bold', rotation=rotation)
            cumulative_height += height

    for year in pivot_2.index:
        cumulative_height = 0
        year_total = pivot_2.loc[year].sum()
        for col in pivot_2.columns:
            height = pivot_2.loc[year, col]
            percentage = (height / year_total) * 100 if year_total > 0 else 0
            if percentage >= perc:
                ax.text(year + offset, cumulative_height + height/2, f'{int(percentage)}%',
                       ha='center', va='center', fontsize=8, color='black', 
                       weight='bold', rotation=rotation)
            cumulative_height += height

    ax.set(xlabel="Year", ylabel='Count',
           xticks=all_years,
           title=f"Provincial Counts by Year - Sector {sector} - Comparison",
           axisbelow=True)
    ax.set_xticklabels(all_years, rotation=0)
    ax.grid(lw=.5)
    ax.set_ylim(0, max(max(bottom_data1), max(bottom_data2)) * 1.05)
    # Sort legend alphabetically
    handles, labels = ax.get_legend_handles_labels()
    labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
    labels_sorted, handles_sorted = zip(*labels_handles)
    ax.legend(handles_sorted, labels_sorted, loc='upper left', ncols=2, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH/'model_validation'/f'provincial_{sector}_count_comparison.png', dpi=300)
    plt.show()

def plot_stacked_bar(pivot, column_color, title_descr=None):
    """Plot a single stacked bar chart from one pivot table"""

    all_years = range(int(pivot.index.min()), int(pivot.index.max()) + 1)

    _, ax = plt.subplots(figsize=(10, 5))

    # Sort provinces by total values (descending)
    pivot_totals = pivot.sum().sort_values(ascending=False)
    pivot = pivot[pivot_totals.index]

    bottom_data = [0] * len(pivot.index)

    for col in pivot.columns:
        ax.bar(x=pivot.index, height=pivot[col], width=0.8,
               bottom=bottom_data, color=column_color[col], label=col)
        bottom_data = [b + h for b, h in zip(bottom_data, pivot[col])]

    # Add percentage labels
    for year in pivot.index:
        cumulative_height = 0
        year_total = pivot.loc[year].sum()
        for col in pivot.columns:
            height = pivot.loc[year, col]
            percentage = (height / year_total) * 100 if year_total > 0 else 0
            if percentage >= 3:
                ax.text(year, cumulative_height + height/2, f'{int(percentage)}%',
                       ha='center', va='center', fontsize=8, color='black', 
                       weight='bold')
            cumulative_height += height

    ax.set(xlabel="Year", ylabel="Count",
           xticks=all_years,
           title=f"Provincial counts by Year - {title_descr}",
           axisbelow=True,
    )
    ax.set_xticklabels(all_years, rotation=0)
    ax.grid(lw=.5)
    ax.set_ylim(0, max(bottom_data) * 1.05)
    # Sort legend alphabetically
    handles, labels = ax.get_legend_handles_labels()
    labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
    labels_sorted, handles_sorted = zip(*labels_handles)
    ax.legend(handles_sorted, labels_sorted, loc='upper left', ncols=2, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH/'model_validation'/f'provincial_{title_descr}_count.png', dpi=300)
    plt.show()
# =========================================================
# Sankey diagram of production network flows
# =========================================================
def derive_visualization_data(global_edgelist, firm_PN_visual_IDs: set,
                              ID_TO_SECTOR, SECTOR_TO_DESCR, ID_TO_ADM,
                              sector_color_map=None,
                              show_roe=True):
    """
    Derive the visualization data (nodes and links) from the global edgelist
    and the set of firm IDs that should be kept separate in the visualization.
    """

    # ---------------------------------------------------------
    # Reduce PN system edgelist into visualization edgelist
    # ---------------------------------------------------------
    # Map edgelist ends into their visualization node.
    visual_flows_df = map_edge_ends_into_visual_nodes(global_edgelist, firm_PN_visual_IDs)

    # Count firms in each visualization node (1 if IDs, many if sector)
    firm_counts_per_visual_node_df = compute_total_unique_firms_per_visual_node(visual_flows_df)

    # Now aggregate global edgelist into flows between visualization diagram nodes
    visual_flows_df = aggregate_flows(visual_flows_df)

    # Hide messy flows
    visual_flows_df = hide_roe_to_roe(visual_flows_df)
    visual_flows_df = hide_repayment(visual_flows_df, factor=10)
    if not show_roe:
        visual_flows_df = hide_all_roe_edges(visual_flows_df)

    # Hide flows to RoE
    mask = (visual_flows_df['vis_customer'] == 'RoE_customer') & (visual_flows_df['vis_supplier'] == 'G4630')
    #visual_flows_df.loc[mask, 'visible'] = False
    mask = (visual_flows_df['vis_supplier'] == 'RoE_supplier') & (visual_flows_df['vis_customer'] == 'G4630') # Why do we hide it??
    #visual_flows_df.loc[mask, 'visible'] = False
    mask = (visual_flows_df['vis_customer'] == 'RoE_customer') & (visual_flows_df['vis_supplier'] == 'G4669')
    #visual_flows_df.loc[mask, 'visible'] = False

    visual_flows_df = hide_tiny_edges(visual_flows_df, 0.02)


    # ---------------------------------------------------------
    # Build visualization diagram nodes and links dataframes
    # ---------------------------------------------------------
    # Compute node attributes from full aggregated data
    nodes_df = compute_aggregated_node_attrs(visual_flows_df, ID_TO_SECTOR, SECTOR_TO_DESCR, ID_TO_ADM)

    # Step 1: Extract visible nodes
    visible_nodes_df = extract_visible_nodes(visual_flows_df)

    # Step 2: Create index mapping
    visible_nodes_df, label_to_idx = create_node_index_mapping(visible_nodes_df)

    # Step 3: Enrich with firm data
    visible_nodes_df = enrich_nodes_with_firm_data(visible_nodes_df, nodes_df)

    # Step 4: Add firm counts in visualization node
    visible_nodes_df = visible_nodes_df.merge(firm_counts_per_visual_node_df, how='left')

    # Step 5: Set custom positions
    visible_nodes_df = set_custom_node_positions(visible_nodes_df)

    # Step 6: Build visible flows with indices
    visible_flows_df = build_visible_flows_with_indices(visual_flows_df, label_to_idx)

    # Step 7: Calculate visible flow strengths
    visible_strengths = calculate_visible_flow_strengths(visible_flows_df)
    visible_nodes_df = (visible_nodes_df.merge(visible_strengths, on='visual_node_label', how='left')
                                        .fillna(0))

    # Step 8: Add visibility percentages
    visible_nodes_df = add_visibility_percentages(visible_nodes_df)

    if sector_color_map:
        # Step 9: Add node colors
        visible_nodes_df['color'] = visible_nodes_df['ISIC4'].map(sector_color_map).fillna('black')
    return visible_nodes_df, visible_flows_df

def plot_flows_sankey(visible_nodes_df, visible_flows_df):
    """
    Plots a Sankey diagram of the visible flows between nodes.

    Parameters:
    - visible_nodes_df: DataFrame containing node information.
    - visible_flows_df: DataFrame containing flow information.

    Returns:
    - fig: Plotly Figure object representing the Sankey diagram.
    """
    customdata_nodes = visible_nodes_df[['ISIC4', 'ISIC4_descr',
                                        'ADM',
                                        'total_unique_firms',
                                        's_in', 'vis_s_in_perc', 'vis_s_in',
                                        's_out', 'vis_s_out_perc', 'vis_s_out',
                                        ]].values

    customdata_flows = visible_flows_df[['vis_supplier', 'vis_customer','weight',
                                    'trans_uni',
                                    'supp_uni', 'cust_uni'
                                    ]].values

    fig = go.Figure(go.Sankey(
        #arrangement = "fixed",
        node = {
            "label": visible_nodes_df['visual_node_label'],
            "x": [val if val else val for val in visible_nodes_df['x'].values],
            "y": [val if val else val for val in visible_nodes_df['y'].values],
            "customdata": customdata_nodes,
            "hovertemplate": 
                "<b>%{label}</b><br>" +
                "ISIC4, descr: %{customdata[0]}, %{customdata[1]}<br>" +
                "Canton, Region: %{customdata[2]} (# of firms: %{customdata[3]})<br>" +
                "Costs (s-in): %{customdata[4]:.2e}$, visible: %{customdata[5]}%<br>" +
                "Sales (s-out): %{customdata[7]:.2e}$, visible: %{customdata[8]}%<br>" +
                "<extra></extra>" # this suppresses the second box
            },
        link = {
            "source": visible_flows_df['source_idx'],
            "target": visible_flows_df['target_idx'],
            "value":  visible_flows_df['weight'],
            'customdata': customdata_flows,
            "hovertemplate": 
                "source: %{customdata[0]}<br>" +
                "target: %{customdata[1]}<br>" +
                "# of links: %{customdata[3]}<br>" +
                "# of unique suppliers: %{customdata[4]}<br>" +
                "# of unique customers: %{customdata[5]}"
            }))

    # Apply node and link colour choices
    if 'color' in visible_nodes_df.columns:
        fig.update_traces(node_color = visible_nodes_df['color'].values)
    fig.show()
