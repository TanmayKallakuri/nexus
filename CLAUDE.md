# Nexus — Hackathon Collaboration Hub

## Team
- **Tanmay** (host)
- **Jasjyot**
- **Tolendi**

## Project Structure
```
nexus/
├── data/              # Raw dataset(s) — drop zone. NEVER modify originals.
├── scripts/           # All analysis scripts (descriptive names only)
├── outputs/           # Generated charts/CSVs, organized by team member
│   ├── tanmay/
│   ├── jasjyot/
│   └── tolendi/
├── brief.md           # Shared brain — all findings via /sync
├── report.docx        # Final deliverable — generated via /report
├── requirements.txt   # Python dependencies
└── CLAUDE.md          # This file
```

## Rules for Every Session

### Identity
- On first interaction, if the user hasn't identified themselves, ask: "Which team member am I working with — Tanmay, Jasjyot, or Tolendi?"
- Remember the name for the entire session. All file outputs and sync entries are tagged with this name.

### File Organization
- **Data files** go in `data/`. NEVER modify original data files. Create copies in `scripts/` or `outputs/` if transformation is needed.
- **Scripts** go in `scripts/`. Use descriptive names: `revenue_correlation.py`, `churn_by_segment.py` — not `test.py` or `analysis.py`.
- **Outputs** go in `outputs/{username}/`. Charts as PNG, processed data as CSV.
- If you detect files placed in the wrong location, suggest moving them to the correct folder.
- If a teammate uploads files via SCP/SSH, detect them on next /sync and ask what they contain.

### EDA & Analysis Conventions
- Python 3.13. Use virtual environment if available.
- ALWAYS use `matplotlib.use('Agg')` at the top of every script. No GUI. Headless only.
- Use matplotlib and seaborn for visualizations. Save all charts as PNG to `outputs/{username}/`.
- Use high-resolution output: `plt.savefig(..., dpi=300, bbox_inches='tight')`.
- Print key statistics to terminal so the user sees results immediately.
- Every script must be self-contained and independently runnable.
- Include clear comments in scripts explaining what each section does.

### Data Handling
- Support both CSV (pandas.read_csv) and Excel (pandas.read_excel with openpyxl).
- Always start analysis by profiling the dataset: shape, columns, dtypes, null counts, basic descriptive stats.
- Never assume what columns mean. Read column names, check unique values, and investigate before drawing conclusions.

### Large Dataset Protocol
Datasets can be large (100k+ rows, 50+ columns). Every script MUST handle this gracefully:

**Reading:**
- Always check file size FIRST (`os.path.getsize()`). Print it before loading.
- For files > 500MB: use `pd.read_csv(..., low_memory=False)` and consider chunked reading.
- Immediately downcast dtypes after loading to reduce memory: `df[col] = pd.to_numeric(df[col], downcast='float')` for floats, similar for ints.
- For Excel files > 100MB: read with `openpyxl` engine and consider sheet-by-sheet loading.
- Print memory usage after loading so the user knows the footprint.

**Visualization — ALWAYS sample for plots:**
- If rows > 10,000: sample 10,000 rows for scatter plots, histograms, and box plots. Use `df.sample(n=10000, random_state=42)`.
- Use `plt.close('all')` after EVERY figure save to free memory. Never hold multiple figures open.
- If numeric columns > 20: split charts into multiple pages/files instead of one giant subplot grid. Max 12 subplots per figure.
- If categorical columns have > 50 unique values: show only top 20 by frequency.
- For correlation matrix with > 30 numeric columns: compute full correlation but only PLOT the top 20 most correlated pairs, or cluster the heatmap.

**Computation:**
- For correlation on large data: use `df[num_cols].corr()` directly — pandas handles this efficiently. Do NOT iterate row-by-row.
- For value counts and groupbys: these are fine at any scale. No sampling needed.
- For descriptive stats (`describe()`): fine at any scale. No sampling needed.
- Add `print()` progress markers throughout long-running scripts so the user knows it's working: `print("[1/6] Loading data...")`, `print("[2/6] Computing statistics...")`, etc.

**Memory management:**
- Delete intermediate DataFrames with `del df_temp` and call `gc.collect()` after large operations.
- If total memory usage > 2GB: warn the user and suggest working with a subset or specific columns.
- Never load the full dataset multiple times. Load once, reuse.

### Sync Discipline
- Every 4-5 exchanges, if the user hasn't synced recently, suggest: "Want to /sync your findings?"
- After any significant discovery or completed analysis, nudge: "That's a good finding — worth syncing."
- Before starting a new analysis direction, suggest syncing to check if a teammate already covered it.

### Report Standard
- The final report targets C-level stakeholders.
- Language must be professional, concise, and insight-driven.
- Every finding must answer "so what?" — why should a business leader care?
- Charts must be clean, labeled, and tell a story on their own.
- No jargon without explanation. No raw code in the report.
