"""
Initial EDA — Behavioral Economics Survey Dataset
Investigator: Tanmay
Outputs: outputs/tanmay/
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Try to set a nice style
for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
    try:
        plt.style.use(style)
        break
    except:
        continue

USERNAME = 'tanmay'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', USERNAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# [1/8] LOADING ALL SURVEY FILES
# ============================================================
print("=" * 60)
print("[1/8] Loading all survey files...")
print("=" * 60)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f != '.gitkeep'])

file_info = []
for f in files:
    fpath = os.path.join(DATA_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    file_info.append((f, size_mb))
    print(f"  {f}: {size_mb:.2f} MB")

# Load numbers files (coded responses) — these are what we analyze
# Load labels files just to extract question text mapping
print("\nLoading numeric response files...")
surveys_num = {}
surveys_lab = {}
for f in files:
    fpath = os.path.join(DATA_DIR, f)
    # Row 0 = short column names, Row 1 = full question text (multiline quoted)
    # For numbers files: skip row 1 (question text), use row 0 as header
    if '_numbers' in f:
        key = f.replace('_numbers.csv', '')
        df = pd.read_csv(fpath, header=0, low_memory=False, encoding='utf-8-sig')
        # Strip whitespace from column names (many have trailing spaces)
        df.columns = df.columns.str.strip()
        # Row 1 contains multiline question text — drop it, then coerce to numeric
        df = df.iloc[1:].reset_index(drop=True)
        pid = df['person_id']
        df = df.apply(pd.to_numeric, errors='coerce')
        df['person_id'] = pid
        surveys_num[key] = df
        print(f"  {key} numbers: {df.shape[0]} rows x {df.shape[1]} cols, "
              f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    elif '_labels' in f:
        key = f.replace('_labels.csv', '')
        # Read just first 2 rows to get the label mapping
        df_lab = pd.read_csv(fpath, header=0, nrows=1, low_memory=False, encoding='utf-8-sig')
        df_lab.columns = df_lab.columns.str.strip()
        surveys_lab[key] = df_lab

# --- VALIDATION: verify skiprows worked (no question text leaked into data) ---
print("\n--- Validating data load ---")
for key, df in surveys_num.items():
    # person_id should be numeric if load was correct
    non_numeric_pid = pd.to_numeric(df['person_id'], errors='coerce').isna().sum()
    if non_numeric_pid > 0:
        print(f"  WARNING: {key} has {non_numeric_pid} non-numeric person_id values — "
              "question text row may not have been skipped correctly!")
    else:
        print(f"  {key}: person_id column is clean (all numeric) — load OK")

    # Check a sample of other columns for text leakage
    sample_cols = [c for c in df.columns if c != 'person_id'][:5]
    for col in sample_cols:
        if df[col].dtype == 'object':
            # Check if values look like question text (long strings)
            max_len = df[col].dropna().astype(str).str.len().max() if len(df[col].dropna()) > 0 else 0
            if max_len > 200:
                print(f"  WARNING: {key}.{col} has very long text values (max {max_len} chars) — "
                      "possible question text leakage")

# --- VALIDATION: check for duplicate column names across surveys ---
print("\n--- Checking for column name conflicts across surveys ---")
all_col_sets = {}
for key in sorted(surveys_num.keys()):
    cols = set(surveys_num[key].columns) - {'person_id'}
    all_col_sets[key] = cols

conflicts_found = False
survey_keys = sorted(all_col_sets.keys())
for i in range(len(survey_keys)):
    for j in range(i + 1, len(survey_keys)):
        overlap = all_col_sets[survey_keys[i]] & all_col_sets[survey_keys[j]]
        if overlap:
            print(f"  CONFLICT: {survey_keys[i]} and {survey_keys[j]} share {len(overlap)} columns: "
                  f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}")
            conflicts_found = True

if not conflicts_found:
    print("  No column name conflicts — safe to merge.")

# --- VALIDATION: check person_id overlap across surveys ---
print("\n--- Checking person_id overlap ---")
pid_sets = {key: set(df['person_id'].dropna()) for key, df in surveys_num.items()}
for key, pids in sorted(pid_sets.items()):
    print(f"  {key}: {len(pids)} unique person_ids")
all_pids = set()
for pids in pid_sets.values():
    all_pids |= pids
common_pids = pid_sets[survey_keys[0]]
for key in survey_keys[1:]:
    common_pids = common_pids & pid_sets[key]
print(f"  Union: {len(all_pids)} | Intersection (all 4 surveys): {len(common_pids)}")
if len(common_pids) < len(all_pids):
    print(f"  NOTE: {len(all_pids) - len(common_pids)} respondents did NOT complete all 4 surveys")

# ============================================================
# [2/8] MERGING INTO ONE MASTER DATAFRAME
# ============================================================
print("\n" + "=" * 60)
print("[2/8] Merging surveys on person_id...")
print("=" * 60)

# Merge all numeric surveys on person_id
master = None
for key in sorted(surveys_num.keys()):
    df = surveys_num[key]
    if master is None:
        master = df.copy()
    else:
        master = master.merge(df, on='person_id', how='outer')
    print(f"  After merging {key}: {master.shape}")

# Downcast numerics to save memory
print("\nDowncasting numeric dtypes...")
for col in master.select_dtypes(include=['float64']).columns:
    master[col] = pd.to_numeric(master[col], downcast='float', errors='coerce')
for col in master.select_dtypes(include=['int64']).columns:
    master[col] = pd.to_numeric(master[col], downcast='integer', errors='coerce')

mem_mb = master.memory_usage(deep=True).sum() / 1024**2
print(f"Master dataset: {master.shape[0]} rows x {master.shape[1]} cols")
print(f"Memory usage: {mem_mb:.1f} MB")

# Free individual survey DataFrames
del surveys_num
gc.collect()

# ============================================================
# [3/8] DATA PROFILING
# ============================================================
print("\n" + "=" * 60)
print("[3/8] Data profiling...")
print("=" * 60)

print(f"\nShape: {master.shape[0]} rows x {master.shape[1]} columns")
print(f"Person IDs: {master['person_id'].nunique()} unique")

# Dtype summary
dtype_counts = master.dtypes.value_counts()
print(f"\nDtype distribution:")
for dt, count in dtype_counts.items():
    print(f"  {dt}: {count} columns")

# Null analysis
null_counts = master.isnull().sum()
null_pcts = (master.isnull().sum() / len(master) * 100).round(1)
cols_with_nulls = null_pcts[null_pcts > 0].sort_values(ascending=False)
print(f"\nColumns with missing data: {len(cols_with_nulls)} / {master.shape[1]}")
print(f"Overall missing: {master.isnull().sum().sum()} / {master.size} ({master.isnull().sum().sum()/master.size*100:.1f}%)")

if len(cols_with_nulls) > 0:
    print(f"\nTop 20 columns by missing %:")
    for col, pct in cols_with_nulls.head(20).items():
        print(f"  {col[:50]:50s} {pct:6.1f}%  ({null_counts[col]} nulls)")

# Identify column groups by prefix
print("\n--- SCALE/CONSTRUCT GROUPS ---")
# Extract prefixes (everything before the last _N or space+N)
col_groups = {}
for col in master.columns:
    if col == 'person_id':
        continue
    # Try to find the group name
    parts = col.rsplit('_', 1)
    if len(parts) == 2 and parts[1].strip().isdigit():
        group = parts[0].strip()
    elif col.startswith('Q') and col[1:].replace('_', '').replace('.', '').isdigit():
        group = 'Standalone Q items'
    else:
        group = col.strip()
    col_groups.setdefault(group, []).append(col)

print(f"Found {len(col_groups)} construct groups:")
for group in sorted(col_groups.keys()):
    items = col_groups[group]
    print(f"  {group[:50]:50s} {len(items):3d} items")

# Descriptive stats for numeric columns
num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
if 'person_id' in num_cols:
    num_cols.remove('person_id')
cat_cols = master.select_dtypes(include=['object']).columns.tolist()
if 'person_id' in cat_cols:
    cat_cols.remove('person_id')

print(f"\nNumeric columns: {len(num_cols)}")
print(f"Categorical/text columns: {len(cat_cols)}")

# Quick descriptive stats
desc = master[num_cols].describe().T
print(f"\nDescriptive stats summary (numeric):")
print(f"  Mean range: {desc['mean'].min():.2f} to {desc['mean'].max():.2f}")
print(f"  Most columns appear to be Likert scales (1-5 or 1-7 range)")

# ============================================================
# [4/8] MISSING DATA VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("[4/8] Missing data visualization...")
print("=" * 60)

cols_with_missing = null_pcts[null_pcts > 0].sort_values(ascending=False)
if len(cols_with_missing) > 0:
    # Show top 40 columns with most missing data
    plot_cols = cols_with_missing.head(40)
    fig, ax = plt.subplots(figsize=(14, max(6, len(plot_cols) * 0.3)))
    # Truncate column names for display
    labels = [c[:40] for c in plot_cols.index]
    ax.barh(range(len(plot_cols)), plot_cols.values, color='coral', edgecolor='darkred', alpha=0.8)
    ax.set_yticks(range(len(plot_cols)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Missing %')
    ax.set_title(f'Missing Data — Top {len(plot_cols)} Columns\n({master.shape[0]} respondents)')
    ax.invert_yaxis()
    plt.savefig(f'{OUTPUT_DIR}/missing_data.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {OUTPUT_DIR}/missing_data.png")
else:
    print("  No missing data — skipping chart.")

# ============================================================
# [5/8] DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("[5/8] Distribution analysis...")
print("=" * 60)

# Sample if large
sample_note = ""
if len(master) > 10000:
    plot_df = master.sample(n=10000, random_state=42)
    sample_note = " (sampled 10k rows)"
else:
    plot_df = master
    sample_note = f" ({len(master)} rows)"

# Numeric distributions — split into batches of 12
if len(num_cols) > 0:
    batch_size = 12
    for batch_idx in range(0, len(num_cols), batch_size):
        batch_cols = num_cols[batch_idx:batch_idx + batch_size]
        n = len(batch_cols)
        ncols_grid = 4
        nrows_grid = (n + ncols_grid - 1) // ncols_grid
        fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, nrows_grid * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(batch_cols):
            ax = axes[i]
            data = plot_df[col].dropna()
            if len(data) == 0:
                ax.set_title(col[:30] + '\n(all null)', fontsize=8)
                continue
            # If few unique values (Likert), use bar chart
            nunique = data.nunique()
            if nunique <= 10:
                vc = data.value_counts().sort_index()
                ax.bar(vc.index.astype(str), vc.values, color='steelblue', edgecolor='navy', alpha=0.8)
            else:
                ax.hist(data, bins=min(30, nunique), color='steelblue', edgecolor='navy', alpha=0.8)
            ax.set_title(col[:30], fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        batch_num = batch_idx // batch_size + 1
        fig.suptitle(f'Numeric Distributions — Batch {batch_num}{sample_note}', fontsize=12, y=1.02)
        plt.tight_layout()
        fname = f'{OUTPUT_DIR}/numeric_distributions_{batch_num}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {fname}")
    gc.collect()

# Categorical distributions
if len(cat_cols) > 0:
    # Filter out high-cardinality columns
    cat_plot_cols = [c for c in cat_cols if master[c].nunique() <= 100]
    if len(cat_plot_cols) > 0:
        n = len(cat_plot_cols)
        fig, axes = plt.subplots(n, 1, figsize=(12, n * 4))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(cat_plot_cols):
            vc = master[col].value_counts().head(15)
            labels = [str(v)[:40] for v in vc.index]
            axes[i].barh(labels, vc.values, color='teal', edgecolor='darkcyan', alpha=0.8)
            axes[i].set_title(col[:50], fontsize=10)
            axes[i].invert_yaxis()
            axes[i].tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {OUTPUT_DIR}/categorical_distributions.png")
    else:
        print("  All categorical columns are high-cardinality — skipping.")
    # Note high-cardinality ones
    high_card = [c for c in cat_cols if master[c].nunique() > 100]
    if high_card:
        print(f"  Skipped high-cardinality categorical cols: {high_card}")
else:
    print("  No categorical columns found.")

# ============================================================
# [6/8] CORRELATION MATRIX
# ============================================================
print("\n" + "=" * 60)
print("[6/8] Correlation analysis...")
print("=" * 60)

if len(num_cols) >= 2:
    corr = master[num_cols].corr()

    # Save full correlation as CSV
    corr.to_csv(f'{OUTPUT_DIR}/full_correlation.csv')
    print(f"  Saved full correlation matrix: {OUTPUT_DIR}/full_correlation.csv")

    # For plot: select top 30 columns with highest mean absolute correlation
    mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
    top_corr_cols = mean_abs_corr.head(30).index.tolist()

    corr_subset = corr.loc[top_corr_cols, top_corr_cols]
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=[c[:25] for c in top_corr_cols],
                yticklabels=[c[:25] for c in top_corr_cols],
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Correlation Matrix — Top 30 Most Correlated Columns{sample_note}', fontsize=12)
    ax.tick_params(labelsize=7)
    plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {OUTPUT_DIR}/correlation_matrix.png")

    # Print top 20 strongest correlations (excluding self)
    print("\n  Top 20 strongest pairwise correlations:")
    corr_pairs = corr.unstack().reset_index()
    corr_pairs.columns = ['col1', 'col2', 'r']
    corr_pairs = corr_pairs[corr_pairs['col1'] < corr_pairs['col2']]  # upper triangle only
    corr_pairs['abs_r'] = corr_pairs['r'].abs()
    top_pairs = corr_pairs.nlargest(20, 'abs_r')
    for _, row in top_pairs.iterrows():
        print(f"    {row['col1'][:30]:30s} <-> {row['col2'][:30]:30s}  r={row['r']:.3f}")
    del corr_pairs
    gc.collect()
else:
    print("  Fewer than 2 numeric columns — skipping correlation.")

# ============================================================
# [7/8] OUTLIER DETECTION (BOX PLOTS)
# ============================================================
print("\n" + "=" * 60)
print("[7/8] Outlier detection (box plots)...")
print("=" * 60)

if len(num_cols) > 0:
    batch_size = 12
    for batch_idx in range(0, len(num_cols), batch_size):
        batch_cols = num_cols[batch_idx:batch_idx + batch_size]
        n = len(batch_cols)
        ncols_grid = 4
        nrows_grid = (n + ncols_grid - 1) // ncols_grid
        fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, nrows_grid * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(batch_cols):
            ax = axes[i]
            data = plot_df[col].dropna()
            if len(data) == 0:
                ax.set_title(col[:30] + '\n(all null)', fontsize=8)
                continue
            ax.boxplot(data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', edgecolor='navy'),
                       medianprops=dict(color='red'))
            ax.set_title(col[:30], fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        batch_num = batch_idx // batch_size + 1
        fig.suptitle(f'Outlier Box Plots — Batch {batch_num}{sample_note}', fontsize=12, y=1.02)
        plt.tight_layout()
        fname = f'{OUTPUT_DIR}/outlier_boxplots_{batch_num}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {fname}")
    gc.collect()

# ============================================================
# [8/8] SUMMARY REPORT
# ============================================================
print("\n" + "=" * 60)
print("[8/8] Writing summary report...")
print("=" * 60)

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("INITIAL EDA SUMMARY — BEHAVIORAL ECONOMICS SURVEY")
summary_lines.append(f"Investigator: {USERNAME}")
summary_lines.append(f"Date: 2026-03-23")
summary_lines.append("=" * 60)

summary_lines.append(f"\n--- DATASET OVERVIEW ---")
summary_lines.append(f"Files: 8 CSV files (4 surveys x 2 versions: labels + numbers)")
summary_lines.append(f"Merged master: {master.shape[0]} respondents x {master.shape[1]} variables")
summary_lines.append(f"Memory usage: {mem_mb:.1f} MB")
summary_lines.append(f"Unique person IDs: {master['person_id'].nunique()}")

summary_lines.append(f"\n--- FORENSICS CONCLUSION ---")
summary_lines.append("Domain: Behavioral Economics / Cognitive Psychology")
summary_lines.append("Source: Large-scale online survey (likely MTurk/Prolific)")
summary_lines.append("Geography: United States (demographics ask about US regions, citizenship)")
summary_lines.append("Confidence: HIGH")
summary_lines.append("Constructs measured:")
summary_lines.append("  - Personality: Big Five, Empathy, Individualism, Conscientiousness")
summary_lines.append("  - Mental Health: Beck Anxiety Inventory (BAI), Beck Depression Inventory (BDI)")
summary_lines.append("  - Cognitive: CRT, Numeracy, Financial Literacy, Syllogisms, Wason Selection")
summary_lines.append("  - Decision Biases: Risk Aversion, Loss Aversion, Present Bias, Discount Rate,")
summary_lines.append("    Sunk Cost, Allais Paradox, Base Rate Neglect, Linda Problem, Denominator Neglect")
summary_lines.append("  - Social: Ultimatum Game, Trust Game, Dictator Game")
summary_lines.append("  - Other: Need for Closure, Need for Cognition, Need for Uniqueness,")
summary_lines.append("    Minimalism, Maximization, Self-Monitoring, Regulatory Focus, GREEN, Agentic Communal")

summary_lines.append(f"\n--- DATA QUALITY ---")
summary_lines.append(f"Columns with nulls: {len(cols_with_nulls)} / {master.shape[1]}")
total_missing_pct = master.isnull().sum().sum() / master.size * 100
summary_lines.append(f"Overall missing rate: {total_missing_pct:.1f}%")
if len(cols_with_missing) > 0:
    summary_lines.append(f"Most missing column: {cols_with_missing.index[0]} ({cols_with_missing.iloc[0]:.1f}%)")

summary_lines.append(f"\n--- NUMERIC COLUMNS ---")
summary_lines.append(f"Count: {len(num_cols)}")
summary_lines.append(f"Most appear to be Likert-scale responses (ordinal 1-5 or 1-7)")

summary_lines.append(f"\n--- CATEGORICAL COLUMNS ---")
summary_lines.append(f"Count: {len(cat_cols)}")
for col in cat_cols:
    summary_lines.append(f"  {col}: {master[col].nunique()} unique values")

summary_lines.append(f"\n--- TOP CORRELATIONS ---")
if len(num_cols) >= 2:
    corr_full = master[num_cols].corr()
    cp = corr_full.unstack().reset_index()
    cp.columns = ['c1', 'c2', 'r']
    cp = cp[cp['c1'] < cp['c2']]
    cp['abs_r'] = cp['r'].abs()
    for _, row in cp.nlargest(10, 'abs_r').iterrows():
        summary_lines.append(f"  {row['c1'][:35]} <-> {row['c2'][:35]}  r={row['r']:.3f}")
    del cp, corr_full

summary_lines.append(f"\n--- CONSTRUCT GROUPS ---")
for group in sorted(col_groups.keys()):
    items = col_groups[group]
    summary_lines.append(f"  {group}: {len(items)} items")

if sample_note:
    summary_lines.append(f"\nNote: Charts based on{sample_note}")

summary_text = '\n'.join(summary_lines)
with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
    f.write(summary_text)
print(f"  Saved: {OUTPUT_DIR}/eda_summary.txt")

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("=" * 60)
