"""
This module performs single-year and multi-year analysis of
the specialization and relevance of firms supplying from A0122 sector'.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Needed for np.sort, np.arange
from config import DATA_PATH, FIGURE_PATH # Assuming this is correct for your environment

# ==============================================================================
## 1. Data Loading and Initial Processing
# ==============================================================================

# --- Load Firm Data (Nodelist) ---
print("Loading firm data...")
firms_filepath = Path(DATA_PATH) / "firm-level" / 'nodelist_cw.csv'
firms_df = pd.read_csv(
    firms_filepath,
    sep='\t',
    dtype={'firm_id': str, 'date': int, "out_strength": float}
)
# Rename columns for administrative area codes
firms_df = firms_df.rename({'province': 'ADM1', 'canton': 'ADM2'}, axis=1)

# --- Load Link Data (Edgelist) ---
print("Loading link data...")
links_filepath = Path(DATA_PATH) / "firm-level" / 'edgelist_cw.csv'
links_df = pd.read_csv(
    links_filepath,
    sep='\t',
    dtype={'id_supplier': str, 'id_customer': str, 'weight': float, 'date': int}
)
# Sort links by weight (optional, but maintained from original script)
links_df = links_df.sort_values('weight', ascending=False)

# --- Merge Sector Information ---
# Create mapping from firm_id (node) to ISIC4 (sector)
ID_TO_SECTOR = dict(zip(firms_df["firm_id"], firms_df["ISIC4"]))

# Map supplier/customer ids to their sectors in the links data
links_df.loc[:, "sector_supplier"] = links_df["id_supplier"].map(ID_TO_SECTOR)
links_df.loc[:, "sector_customer"] = links_df["id_customer"].map(ID_TO_SECTOR)
print("Data loading and initial processing complete.\n")


# ==============================================================================
## 2. Single-Year Analysis (Year: 2012)
# ==============================================================================

TARGET_YEAR = 2012
TARGET_SECTOR = 'A0122'
print(f"Starting single-year analysis for {TARGET_YEAR} and sector {TARGET_SECTOR}...")

# Filter all links originating from the target sector ('A0122') across all years
links_A0122 = links_df[links_df['sector_supplier'] == TARGET_SECTOR].copy()

# Filter data for the specific year
links_df_y = links_A0122[links_A0122['date'] == TARGET_YEAR].copy()
firms_df_y = firms_df[firms_df['date'] == TARGET_YEAR].copy()

# Filter links originating *from* the target sector ('A0122')
links_df_y_from_A0122 = links_df_y[links_df_y['sector_supplier'] == TARGET_SECTOR].copy()

# --- Calculate Total 'A0122' Input per Customer (Absolute) ---
customers_A0122_inputs = (
    links_df_y_from_A0122
    .groupby('id_customer', as_index=False)
    .agg(
        weight=('weight', 'sum'),
        date=('date', 'first'),
        sector_supplier=('sector_supplier', 'first'),
        sector_customer=('sector_customer', 'first')
    )
    .rename(columns={'weight': 'abs_input_from_A0122'})
)

# --- Calculate Relative Expense Share ---
# Get the customer's total input ('cw_s_in') from the firm data
# Get the firm data for the IDs of customers who bought from 'A0122'.
customers_A0122_total_inputs = (firms_df_y[['firm_id', 'cw_s_in']]
                                [firms_df_y['firm_id'].isin(
                                    set(links_df_y_from_A0122['id_customer'])
                                    )]
)

# Merge absolute input from 'A0122' with the customer's total input
customers_A0122_inputs = customers_A0122_inputs.merge(
    customers_A0122_total_inputs,
    left_on='id_customer',
    right_on='firm_id',
    how='left'
)

# Calculate the share of 'A0122' input in the customer's total inputs
customers_A0122_inputs['share_of_inputs_from_A0122'] = (
    customers_A0122_inputs['abs_input_from_A0122'] / customers_A0122_inputs['cw_s_in']
)

# Clean up and sort
customers_A0122_inputs = (
    customers_A0122_inputs
    .drop('firm_id', axis=1)
    .sort_values(by='share_of_inputs_from_A0122', ascending=False)
    .rename(columns={'abs_input_from_A0122': 'weight'}) # Revert to 'weight' for historical plotting compatibility
)

# --- Visualize Distribution (Histogram) ---
plt.figure(figsize=(7, 5))
plt.hist(
    customers_A0122_inputs['share_of_inputs_from_A0122'].dropna(),
    bins=20,
    label=TARGET_YEAR
)
plt.xlabel(f'Fraction of firm inputs supplied from {TARGET_SECTOR}')
plt.ylabel('Firm count in bin (log scale)')
plt.yscale('log')
plt.title(f'Distribution of Input Shares from Sector {TARGET_SECTOR} in {TARGET_YEAR}')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()
print("Single-year analysis complete.\n")


# ==============================================================================
## 3. Multi-Year Analysis and Visualization
# ==============================================================================
print("Starting multi-year analysis...")

# --- Aggregate Input per Customer and Date ---
# This calculates the total absolute input ('weight') each customer received from 'A0122' per year
cust_inputs = (
    links_A0122
    .groupby(['date', 'id_customer'], as_index=False)
    .agg(
        weight=('weight', 'sum'),
        sector_supplier=('sector_supplier', 'first'),
        sector_customer=('sector_customer', 'first')
    )
)

# --- Merge with Total Customer Inputs ('cw_s_in') ---
# Merge with firm data to get the total input received by the customer
cust_inputs = cust_inputs.merge(
    firms_df[['date', 'firm_id', 'cw_s_in']],
    left_on=['date', 'id_customer'],
    right_on=['date', 'firm_id'],
    how='left'
)

# --- Calculate Relative Expense Share (Multi-Year) ---
cust_inputs['abs_input_from_A0122'] = cust_inputs['weight'] / cust_inputs['cw_s_in']

# Sort for better data review
cust_inputs = cust_inputs.sort_values(
    by=['date', 'sector_customer', 'weight'],
    ascending=(True, True, False)
)


# --- CCDF Plotting ---
# --- Visualization 1: CCDF of Input Share ('abs_input_from_A0122') ---
fig, ax = plt.subplots(figsize=(6,4))
for y, df_y in cust_inputs.groupby('date'):
    sorted_data = np.sort(df_y['abs_input_from_A0122'])
    ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, ccdf, 'o-', ms=2, lw=1, label=str(y))
ax.set(
    xlabel=f"Share of total input costs sourced from sector {TARGET_SECTOR}",
    ylabel='CCDF',
    yscale='log',
    title=f"CCDF of {TARGET_SECTOR} Customers by Input Share")
ax.legend(ncols=3); ax.grid(True)
plt.tight_layout()
save_path = Path(FIGURE_PATH) / "VAT" / 'ccdf_A0122_customers_input_share.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {save_path}")
plt.show()


# --- Visualization 2: CCDF of Relative Output Demanded ('weight' normalized) ---
fig, ax = plt.subplots(figsize=(6,4))
for y, df_y in cust_inputs.groupby('date'):
    sorted_data = np.sort(df_y['weight'])
    sorted_data = sorted_data / sorted_data.sum()
    ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, ccdf, 'o-', ms=2, lw=1, label=str(y))

ax.set(
    xlabel=f'Share of total {TARGET_SECTOR} output demanded by each customer firm',
    ylabel='CCDF',
    yscale='log', xscale='log',
    title=f'CCDF of Customer Firms by Share of Total {TARGET_SECTOR} Output Demanded'
)
ax.legend(title='Year', ncols=2); ax.grid(True)
plt.tight_layout()
save_path = Path(FIGURE_PATH) / "VAT" / 'ccdf_A0122_output_share_demanded.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {save_path}")
plt.show()

# --- Visualization 3: Scatter Plot (Input Share vs. Relative Output Demanded) ---
fig, ax = plt.subplots(figsize=(7, 5))
for y, df_y in cust_inputs.groupby('date'):
    df_y_clean = df_y.dropna(subset=['abs_input_from_A0122', 'weight'])
    A0122_abs_output = df_y_clean['weight'].sum()
    abs_input_from_A0122 = df_y_clean['abs_input_from_A0122']

    # Calculate relative output demanded for the cleaned data
    abs_output_demanded = df_y_clean['weight']
    rel_output_demanded = abs_output_demanded / abs_output_demanded.sum()
    
    ax.scatter(abs_input_from_A0122, rel_output_demanded, s=10, alpha=0.7,
               label=str(y) + f' (output: {A0122_abs_output / 1e9:.1f}B$)')

ax.set(
    xlabel=f"Share of customer firm inputs sourced from sector {TARGET_SECTOR}",
    ylabel=f'Share of tot. {TARGET_SECTOR} output\ndemanded by customer firm',
    yscale='log',
    xscale='log',
    title=f'{TARGET_SECTOR} Customers: Relevance and Specialization'
)
ax.legend(title='Year', ncols=2); ax.grid(True, which='both', ls='--')
plt.tight_layout()
if FIGURE_PATH:
    save_path = Path(FIGURE_PATH) / "VAT" / 'scatter_input_vs_output_share_A0122.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")
plt.show()
print("Multi-year visualizations complete.\n")


# ==============================================================================
## 4. Reporting/Summary Statistics
# ==============================================================================

# Threshold for 'best' customers
INPUT_SHARE_THRESHOLD = 0.2

print(f"Starting summary statistics (Best Customers with input share > {INPUT_SHARE_THRESHOLD})...")

for y, df_y in cust_inputs.groupby('date'):
    # 1. Identify "Best Customers" based on the expense share
    best_cust = df_y[df_y['abs_input_from_A0122'] > INPUT_SHARE_THRESHOLD].copy()
    best_cust_ids = set(best_cust['id_customer'])
    total_customers = set(df_y['id_customer'])

    # 2. Calculate the total product (revenue) captured by these customers
    best_cust_captured_product = best_cust['weight'].sum()

    # 3. Calculate the number of A0122 producers connected to these customers
    links_A0122_y = links_A0122[links_A0122['date'] == y]
    links_to_best_cust_ids = links_A0122_y[links_A0122_y['id_customer'].isin(best_cust_ids)]
    best_cust_captured_suppliers = len(links_to_best_cust_ids['id_supplier'].unique())

    # 4. Calculate total A0122 product and total producers for the year
    A0122producers_product = links_A0122_y['weight'].sum()
    A0122producers_number = len(links_A0122_y['id_supplier'].unique())

    # 5. Print results
    if A0122producers_product > 0 and A0122producers_number > 0:
        product_share = round(best_cust_captured_product / A0122producers_product * 100, 1)
        supplier_share = round(best_cust_captured_suppliers / A0122producers_number * 100, 1)

        print(f"\n--- Year: {y} ---")
        print(f"Customers considered: {len(best_cust_ids)} out of {len(total_customers)}")
        print(f"They are responsible for {best_cust_captured_product:.2e}$ of {TARGET_SECTOR} revenue, which is {product_share}% of the total ({A0122producers_product:.2e}$)")
        print(f"They are connected to {best_cust_captured_suppliers} {TARGET_SECTOR} producers, the {supplier_share}% (total is {A0122producers_number})")
    else:
        print(f"\n--- Year: {y} ---")
        print("Data is insufficient for analysis in this year (Total A0122 product/producers is zero).")

print("\nScript execution finished.")
