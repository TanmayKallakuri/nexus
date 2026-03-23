# /investigate — Dataset Forensics & Initial EDA

You are a senior data scientist. A new dataset has just been dropped into the `data/` folder. Your job is to figure out everything about it — where it came from, what domain it belongs to, and produce a comprehensive initial EDA.

## Step 0: Identity
- If you don't know which team member you're working with, ask first.
- Set `USERNAME` to their name (lowercase) for file outputs.

## Step 1: Scan
- List all files in `data/`.
- For each file, note: filename, extension, **file size** (in MB).
- **Before loading**: if file > 500MB, warn the user and use chunked reading or load specific columns.
- Read the first few rows and the last few rows of each dataset.
- Report: number of rows, number of columns, column names, data types, **memory usage after loading**.
- Immediately downcast numeric dtypes to reduce memory footprint.

## Step 2: Forensics — Figure Out the Source
Analyze the dataset like a detective. Check:

**Column Names:**
- What naming convention is used? (camelCase, snake_case, spaces, abbreviations)
- What domain do these column names suggest? (retail, healthcare, finance, telecom, HR, logistics, etc.)
- Do any column names match well-known public datasets?

**Value Fingerprinting:**
- Date formats and ranges — what time period does this cover?
- Currency symbols or numeric scales — what economy/market?
- Country codes, region names, zip codes — what geography?
- ID formats — sequential, UUID, coded?
- Categorical values — product categories, status codes, industry-specific terms?

**Statistical Shape:**
- Row count — does it match any known dataset sizes?
- Column count and combination — recognizable schema?
- Distribution patterns — synthetic or real-world data?
- Unique value cardinality across columns.

**Metadata Clues:**
- File encoding, delimiter, header conventions.
- Are there any comments, notes, or metadata rows?

**Present your forensics conclusion:**
- "This dataset appears to be: [domain], likely from [source/type], covering [time period], focused on [geography/market]."
- Confidence level: high / medium / low, and why.
- **If you cannot identify the source, that's perfectly fine.** Say so honestly, note what domain/context clues you DID find, and move straight to EDA. Forensics is a bonus, not a blocker. The real value is in the analysis.

## Step 3: Initial EDA Script
Write a SINGLE comprehensive Python script called `scripts/initial_eda.py` that does ALL of the following.

**The script MUST handle large datasets (100k+ rows, 50+ columns) gracefully. Follow the Large Dataset Protocol in CLAUDE.md.**

```python
import matplotlib
matplotlib.use('Agg')
import gc
```

The script must include progress markers throughout: `print("[1/6] Loading data...")` etc.

**3a. Data Profiling**
- Shape, dtypes, memory usage (print in MB)
- Downcast numeric dtypes immediately after loading to save memory
- Null counts and percentages per column
- Descriptive statistics (mean, median, std, min, max, quartiles) for numeric columns
- Unique value counts for categorical columns (top 20 only if > 50 unique values)
- Print a clean summary table to terminal

**3b. Missing Data Visualization**
- Heatmap or bar chart of missing values per column
- If > 50 columns: only show columns that actually have nulls
- Save to `outputs/{USERNAME}/missing_data.png`
- `plt.close('all')` after saving

**3c. Distribution Analysis**
- **If rows > 10,000: sample 10,000 rows** (`df.sample(n=10000, random_state=42)`) for all plots
- If numeric columns > 20: split into multiple chart files (`numeric_distributions_1.png`, `_2.png` etc.), max 12 subplots per figure
- Histogram for every numeric column (subplot grid)
- Save to `outputs/{USERNAME}/numeric_distributions.png`
- Bar chart of value counts for categorical columns (top 15 values each, skip columns with > 100 unique values like IDs)
- Save to `outputs/{USERNAME}/categorical_distributions.png`
- `plt.close('all')` after each figure save

**3d. Correlation Matrix**
- Compute correlation on ALL numeric columns (pandas handles this efficiently)
- If > 30 numeric columns: only PLOT the top 20 columns with highest absolute mean correlation. Save full correlation as CSV to `outputs/{USERNAME}/full_correlation.csv`
- Correlation heatmap for numeric columns
- Save to `outputs/{USERNAME}/correlation_matrix.png`
- `plt.close('all')` after saving

**3e. Outlier Detection**
- **If rows > 10,000: sample 10,000 rows** for box plots
- If numeric columns > 20: split into multiple chart files, max 12 per figure
- Box plots for numeric columns
- Save to `outputs/{USERNAME}/outlier_boxplots.png`
- `plt.close('all')` after saving

**3f. Summary Report**
- Write a text summary to `outputs/{USERNAME}/eda_summary.txt`
- Include: row count, column count, memory usage, null percentages, top correlations, potential outlier columns, key observations
- If data was sampled for plots, note this: "Charts based on 10,000 row sample from {total} total rows"

ALL charts must:
- Use `dpi=300, bbox_inches='tight'`
- Have clear titles and axis labels
- Use style with fallback chain: try `seaborn-v0_8-whitegrid`, then `seaborn-whitegrid`, then `ggplot`, then skip
- Be readable and presentation-quality
- Note in the title if sampled: "Distribution (sampled 10k rows)"

**Edge cases the script MUST handle:**
- **0 rows**: detect early, write summary noting empty dataset, skip all charts
- **1 row**: warn user, skip distribution/correlation/boxplot charts (not meaningful), still write profiling summary
- **All-null columns**: skip from visualizations, flag in summary
- **No numeric columns**: skip correlation, distributions, boxplots — generate only categorical charts
- **No categorical columns**: skip categorical chart — generate only numeric charts
- **High-cardinality categoricals** (>100 unique values like IDs, dates): skip from categorical charts, note in summary
- **Special characters in column names** (`$`, `%`, `/`, `&`, parentheses): handle in chart titles by truncating to 30 chars
- **Mixed-type columns**: let pandas infer, note oddities in summary
- **Very large files (>500MB)**: warn user before loading, suggest column subset if needed

## Step 4: Run the Script
- Create `outputs/{USERNAME}/` directory if it doesn't exist.
- Run `scripts/initial_eda.py`.
- If it fails, debug and fix. Do not move on until it runs successfully.
- Show the terminal output to the user.

## Step 5: Update brief.md
- Fill in the **Dataset Origin** section with your forensics conclusion.
- Fill in the **Dataset Overview** section with shape, columns, dtypes, key quality issues.
- Do NOT touch any other sections.

## Step 6: Present to User
Summarize your findings conversationally:
1. "Here's what I think this dataset is..."
2. "Key stats: X rows, Y columns, Z% missing data overall"
3. "Interesting things I noticed: ..."
4. "Charts saved to outputs/{USERNAME}/ — here's what they show"
5. "Recommended next steps for analysis: ..."

Then suggest: "Want to /sync these findings so your teammates can see them?"
