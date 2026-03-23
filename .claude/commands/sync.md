# /sync — Collaborative Findings Sync

You are the sync engine for a 3-person hackathon team (Tanmay, Jasjyot, Tolendi). Your job is to keep the shared brain (`brief.md`) accurate, deduplicated, and conflict-aware.

## Step 0: Identity

**First sync of a session:**
- If the user provides their name with the command (e.g., `/sync Tanmay`), register that identity.
- If no name is provided and you don't know who this is, ask: "Which team member am I working with — Tanmay, Jasjyot, or Tolendi?"
- Remember `USERNAME` for all subsequent syncs in this session.

**Subsequent syncs:**
- You already know who this is. Skip straight to Step 1.

## Step 1: Pull — Read the Shared Brain

Read `brief.md` in full.

**Identify what's new from teammates:**
- Compare the current state of OTHER team members' sections to what you've seen before in this conversation.
- If there are new entries from teammates since the last sync (or since session start), present them clearly:

```
--- Updates from teammates ---
Jasjyot (synced at 2:41 PM):
  - Customer churn peaks in month 2 of subscription
  - Top 3 segments account for 78% of revenue

Tolendi (synced at 2:38 PM):
  - No new updates.
---
```

- If nothing is new: "Your teammates haven't synced anything new since your last check."

## Step 2: Detect New Files

Scan `outputs/` and `scripts/` for files that aren't referenced in `brief.md`.

For each unreferenced file:
- Note who it belongs to (based on folder: `outputs/jasjyot/` → Jasjyot)
- Note the filename and type
- Report: "Jasjyot uploaded `revenue_trend.png` — not yet documented in brief."

If the new files belong to the CURRENT user, ask: "You have undocumented outputs — want me to include these in your sync?"

## Step 3: Extract — What Did You Find?

Look through the CURRENT conversation (everything discussed in this session since the last /sync or since session start).

Auto-extract findings:
- Any analysis results, statistical observations, patterns, insights
- Any charts generated and where they're saved
- Any scripts written and what they do

Present the extracted findings to the user:

```
--- From your session so far ---
1. Revenue correlates strongly with marketing spend (r=0.85)
2. Generated: outputs/tanmay/revenue_vs_marketing.png
3. Script: scripts/revenue_analysis.py
---
```

Ask: "Does this capture your work? Want to add, remove, or rephrase anything before I sync?"

## Step 4: Duplicate & Contradiction Check

**THIS IS CRITICAL. Do not skip.**

Before writing ANYTHING, compare each new finding against ALL existing findings in `brief.md` from ALL team members.

**Duplicate check:**
- If a finding is semantically the same as something already in brief.md (even if worded differently):
- Flag it: "This looks similar to what [teammate] already found: '[their finding]'. Skip this duplicate?"
- If the user says add anyway, note it but respect their choice.

**Contradiction check:**
- If a finding directly contradicts or conflicts with an existing finding:
- Flag it clearly: "Heads up — this conflicts with [teammate]'s finding: '[their finding]'. Your finding says [X], theirs says [Y]."
- Ask: "Want to: (a) add it as a conflicting finding so the team can resolve it, (b) replace the old finding, or (c) skip it?"
- If they choose (a), add the finding to the user's section AND add an entry to the **Contradictions** section of brief.md:
  ```
  - **[Topic]**: Tanmay found [X] vs. Jasjyot found [Y]. Needs team resolution.
  ```

## Step 4.5: Lock Before Writing

**Before writing to brief.md, implement a simple file lock to prevent race conditions:**

1. Check if `.brief.lock` exists in the project root.
2. If it exists, read the lock file — it contains who holds the lock and when.
   - If the lock is older than 30 seconds, it's stale — delete it and proceed.
   - If the lock is fresh, wait 3 seconds and check again. Retry up to 3 times.
   - If still locked after retries, warn the user: "Another session is writing to brief.md right now. Try again in a few seconds."
3. If no lock exists, create `.brief.lock` with content: `{USERNAME} - {timestamp}`
4. **Re-read brief.md AFTER acquiring the lock** — someone may have updated it between your read and your lock.
5. Proceed to write.
6. **Delete `.brief.lock` immediately after writing.**

This prevents the scenario where two people sync at the same time and one overwrites the other's findings.

## Step 5: Write — Update Your Section Only

After the user confirms and the lock is acquired:

**Update ONLY the current user's section** under `## Findings > ### {USERNAME}`.

Format each finding as:
```
- [Finding text] _(synced at {time})_
```

**Rules:**
- NEVER modify another team member's section. Ever.
- NEVER delete previous findings from your own section unless the user explicitly asks.
- Append new findings below existing ones.
- Update the "last synced" timestamp: `### Tanmay (last synced: 3:15 PM)`

**Also update if relevant:**
- **Dataset Overview** — if new profiling info was discovered
- **Open Questions** — if the analysis raised questions worth investigating
- Tag open questions with who raised them: `- Why does churn spike in month 2? _(raised by Tanmay)_`

## Step 6: Confirm

After writing:
- Show a brief summary: "Synced 3 findings to your section. Brief is up to date."
- If contradictions were flagged, remind: "There's 1 open contradiction the team should resolve."
- Suggest next steps if obvious: "Jasjyot found X — might be worth building on that."

## Edge Cases

**Empty sync (no new findings):**
- Still do the Pull (Step 1) so the user sees teammate updates.
- Say: "Nothing new to sync from your side, but here's what your teammates have been up to."

**Session restart (user re-identifies):**
- Read brief.md to see their existing findings.
- Say: "Welcome back, {name}. Here's where you left off: [their existing findings]."

**Large brief.md:**
- If brief.md exceeds 100 lines, summarize teammate sections instead of showing line-by-line.
- Always show full detail for contradictions.
