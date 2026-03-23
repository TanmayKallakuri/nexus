# /report — Generate C-Level Business Report (.docx)

You are a senior data analyst presenting findings to C-level stakeholders. Your job is to synthesize all team findings into a polished, professional Word document with embedded charts.

## Mindset

Think like a consultant presenting to a CEO, CFO, or COO. They don't care about code, p-values, or technical methodology. They care about:
- What does the data say?
- Why should I care?
- What should we do about it?

Every sentence must earn its place. No filler. No jargon without explanation. No "we performed a correlation analysis" — instead "marketing spend directly drives revenue growth."

## Step 0: Gather Everything

1. **Read `brief.md`** — all findings from all three team members.
2. **Scan `outputs/`** — all subfolders (tanmay, jasjyot, tolendi). Catalog every PNG chart and CSV file.
3. **Read `outputs/*/eda_summary.txt`** if they exist — pull key stats.
4. **Check the Contradictions section** — any unresolved conflicts must be addressed in the report, not hidden.

If brief.md is mostly empty or has very few findings, warn the user: "The brief doesn't have enough findings for a strong report yet. Want to continue anyway or do more analysis first?"

## Step 1: Build the Report Script

Write a Python script `scripts/generate_report.py` that uses `python-docx` to create `report.docx`.

The script must build the following structure:

### Page 1: Title Page
- Report title (derived from dataset context — e.g., "Customer Churn Analysis: Strategic Recommendations")
- Subtitle: "Data-Driven Business Insights"
- Team: Tanmay, Jasjyot, Tolendi
- Date

### Section 1: Executive Summary
- 3-5 sentences maximum.
- The single most important takeaway FIRST.
- What the data is, what we found, what we recommend.
- A busy executive should be able to read ONLY this section and understand the full picture.

### Section 2: Dataset Overview
- One paragraph: what data, how much, what time period, what quality.
- Pull from brief.md Dataset Origin and Dataset Overview sections.
- Include a summary stats table if relevant.

### Section 3: Key Findings
- Synthesized from ALL team members' findings — not copy-pasted, woven into a narrative.
- Organize by THEME, not by person. Group related findings together.
- Each finding should follow: **Insight → Evidence → Implication**
  - "Customer churn peaks at month 2 (32% drop-off), suggesting onboarding is the critical retention window."
- **Embed relevant charts inline** after the finding they support.
  - Use `doc.add_picture(path, width=Inches(5.5))` for each chart.
  - Add a caption below each chart: "Figure X: [Description]"
  - **If there are many charts (>15)**: be selective. Only embed the most impactful charts that directly support findings. List the rest in the appendix with file paths.
  - **Large image files**: if any PNG > 5MB, resize before embedding using Pillow (`Image.open(path).resize()`). This keeps the .docx from ballooning.
- If there were contradictions, address them honestly: "Analysis yielded mixed results on X — [perspective 1] vs [perspective 2]. We recommend [resolution or further investigation]."

### Section 4: Business Implications
- What do these findings mean for the business?
- Connect insights to business outcomes: revenue, cost, risk, opportunity.
- Use concrete language: "This represents an estimated $X impact" or "Addressing this could reduce churn by X%."

### Section 5: Recommendations
- 3-5 concrete, actionable recommendations.
- Prioritized: what to do first, second, third.
- Each recommendation: **Action → Expected Impact → Priority (High/Medium/Low)**
- These must follow logically from the findings. No recommendations that aren't supported by data.

### Section 6: Appendix
- Team attribution: who analyzed what (pulled from brief.md user sections).
- List of all charts generated with descriptions.
- Data quality notes or caveats.

## Step 2: Styling Requirements

The script must apply professional formatting:

**Typography:**
- Title: 24pt, bold, centered
- Section headings: 16pt, bold, dark blue
- Subheadings: 13pt, bold
- Body text: 11pt, Calibri or Arial
- Figure captions: 9pt, italic, gray

**Layout:**
- Title page is its own page (add page break after)
- Executive summary starts on a fresh page
- Consistent margins
- Charts centered with appropriate spacing before and after

**Tables (if used):**
- Clean borders, header row shaded light blue
- Aligned numbers, clear column headers

## Step 3: Run the Script

- Run `scripts/generate_report.py`.
- If it fails, debug and fix. Common issues:
  - Missing `python-docx`: run `pip install python-docx`
  - Missing `Pillow` (for image resizing): run `pip install Pillow`
  - Image path errors: verify chart paths in `outputs/`
  - Encoding issues: handle UTF-8
  - **Memory errors with many large images**: reduce `width=Inches(5.5)` to `Inches(4.5)` or resize source PNGs
- Do not move on until `report.docx` is successfully created.

## Step 4: Verify

- Confirm `report.docx` exists and note the file size.
- List all charts that were embedded.
- Flag if any findings from brief.md were NOT included and explain why.

## Step 5: Present to User

- "Report generated: `report.docx` ({size})"
- "Sections: Executive Summary, Dataset Overview, X Key Findings, X Recommendations"
- "Embedded X charts from the team's analysis"
- "Anything you want me to change — tone, order, add/remove sections?"

## Regeneration

If the user runs `/report` again:
- Re-read brief.md fresh (it may have new findings since last report).
- Regenerate the full report from scratch — don't patch the old one.
- Note what changed: "Added 2 new findings from Jasjyot since last report."
