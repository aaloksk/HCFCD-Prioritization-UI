# HCFCD Project Prioritization Platform

**Developed by infraTECH Engineers & Innovators, LLC**

A professional, analyst-ready platform for prioritizing capital projects using transparent, configurable, and repeatable scoring methods. The application supports direct-weight scoring, AHP-based weight derivation, and TOPSIS ranking, with a configurable criteria framework and an auditable workflow.

## Overview
This Streamlit application enables teams to:
- Load project datasets and manage schema changes.
- Define and adjust weighting strategies.
- Apply multiple ranking methods (Weighted Sum and TOPSIS).
- Compare methods side by side.
- Generate reproducible, exportable ranking outputs.

The platform is tailored for project prioritization workflows used by agencies such as HCFCD and similar organizations.

## Key Features
- **Prioritization Database**: upload/edit project data and manage schema updates.
- **Direct Weights**: intuitive weight entry with validation.
- **AHP Weights**: pairwise comparisons with consistency checks.
- **Ranking**: Weighted Sum and TOPSIS options, plus method comparison.
- **Custom Criteria Mapping**: add custom parameters and map them into the scoring framework.
- **Exportable Results**: download ranked outputs for reporting.

## Project Structure
- `app.py` – Streamlit UI and workflow orchestration
- `engine.py` – Scoring logic, AHP, and TOPSIS
- `scoring_config.json` – criteria mappings, weights, and labels
- `input_template.csv` – starter data template
- `CandidateProjects.csv` – sample input data (if provided)

## Quick Start (Local)
1. Create and activate a virtual environment (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the application:
```powershell
streamlit run app.py
```

## Deployment (Streamlit Cloud)
1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app:
   - Repository: `aaloksk/HCFCD-Prioritization-UI`
   - Branch: `main`
   - File path: `app.py`

## Notes
- `requirements.txt` defines all dependencies.
- `runtime.txt` sets the Python version for Streamlit Cloud (if needed).
- Weighting logic and criteria definitions are fully configurable via `scoring_config.json`.

## Support
For deployment assistance or customization requests, please open an issue or contact infraTECH Engineers & Innovators, LLC.
