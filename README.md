# Nexus — Hackathon Collaboration Hub

A collaborative data analysis and reporting platform for hackathon teams. Each team member explores datasets independently, syncs findings, and generates polished C-level reports.

## Team

- **Tanmay** (host)
- **Jasjyot**
- **Tolendi**

## Project Structure

```
nexus/
├── data/              # Raw datasets — drop zone (never modify originals)
├── scripts/           # Analysis scripts (descriptive names only)
├── outputs/           # Generated charts/CSVs, organized by team member
│   ├── tanmay/
│   ├── jasjyot/
│   └── tolendi/
├── brief.md           # Shared findings log
├── report.docx        # Final deliverable
└── requirements.txt   # Python dependencies
```

## Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/TanmayKallakuri/nexus.git
   cd nexus
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Drop your dataset** into the `data/` folder.

4. **Start analyzing** — use Claude Code with the built-in slash commands:
   - `/investigate` — Run EDA and dataset forensics
   - `/sync` — Share findings with teammates via `brief.md`
   - `/report` — Generate the final C-level business report

## Conventions

- Python 3.13, headless matplotlib (`Agg` backend)
- All charts saved as high-res PNGs to `outputs/{your_name}/`
- Scripts must be self-contained and independently runnable
- Never modify raw data files in `data/`
