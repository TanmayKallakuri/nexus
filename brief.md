# Nexus — Hackathon Brief

## Dataset Origin
Behavioral economics / cognitive psychology omnibus survey. 30+ validated instruments. US-only sample (Prolific/MTurk), post-2019, likely UC Davis lab. 233 respondents, Qualtrics-administered. _(filled by Tanmay)_

## Dataset Overview
- **3 data formats:** personas_text (233 .txt), personas_json (233 .json), personas_csv (8 CSVs across 4 surveys)
- **Master table:** 125,115 rows x 14 columns — every (person, question, answer) triple
- **531 unique questions** — 61% ordinal, 31% categorical, 4% text, 4% multi-select
- **Blocks:** Personality (65K rows), Economic preferences (42K), Cognitive tests (15K), Demographics (3K)
- **Missing data:** 27% overall, mostly text-entry fields coerced to NaN in numeric files
- **Person profiles:** 233 rows x 92 features (Model 1 complete)

## Findings

### Tanmay (last synced: 11:50 AM)
- Initial EDA complete — 234 merged rows, 915 variables, top correlations are within-scale (Loss Aversion r=0.97) _(synced at 11:50 AM)_
- Dataset forensics — 30+ instruments including Big Five, Beck Anxiety/Depression, CRT, economic games, Forward Flow creativity, classic heuristics battery _(synced at 11:50 AM)_
- Built master table from 233 persona JSONs — 125,115 rows, 531 unique questions _(synced at 11:50 AM)_
- Architecture decision: ML backbone (LightGBM bootstrap ensemble) with LLM as fallback only _(synced at 11:50 AM)_
- Scoring insight: correlation is Pearson r per question across 233 people — must preserve person-level ranking per construct, not just average accuracy _(synced at 11:50 AM)_
- Rejected synthetic bootstrapped respondents — destroys within-person cross-construct coherence, hurts correlation _(synced at 11:50 AM)_

### Jasjyot (last synced: 11:45 AM)
- Model 1 complete — 233 rows x 92 columns of person-level features _(synced at 11:45 AM)_
- Features include: coverage, response style, personality constructs, economic preferences, cognitive scores, demographics _(synced at 11:45 AM)_
- Known risk: hard-coded answer keys for CRT/numeracy/financial literacy need sanity check _(synced at 11:45 AM)_
- Outputs: person_response_profiles.csv + data dictionary _(synced at 11:45 AM)_

### Tolendi
_No findings yet — working on bootstrap ensemble._

## Contradictions
_No conflicting findings._

## Open Questions
- Do hard-coded cognitive answer keys in Model 1 match the actual correct answers? _(raised by Jasjyot)_
- What Pearson r does the bootstrap ensemble achieve on held-out questions? _(raised by Tanmay)_
- How well does question embedding similarity match actual construct groupings? _(raised by Tanmay)_

## Business Implications
_Key insights that matter to C-level stakeholders. Synthesized in /report._
