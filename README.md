# Project Prioritization Tool

An interactive Streamlit decision-support tool for evaluating, scoring, analyzing, and ranking drainage improvement projects. The application supports structured project data intake, spatial review, HCFCD-style parameter scoring, custom criteria, direct weighting, pairwise comparison using AHP, ranking with TOPSIS, and sensitivity review across alternative decision scenarios.

The tool was developed to support transparent, data-driven project prioritization for drainage planning and flood mitigation studies. It combines tabular project data, GIS layers, configurable scoring logic, and stakeholder-accessible decision methods in one workflow.

## Key Features

- Spatial project review with precinct boundary, unincorporated area, candidate project, planning-level study area, and HCFCD project boundary layers.
- Importable CSV and JSON workspaces for project databases, scoring inputs, and saved analysis states.
- Automated scoring for core prioritization parameters.
- Direct user-adjustable weights.
- Pairwise Comparison (AHP) workflow with CSV import/export and consistency ratio review.
- TOPSIS and weighted-sum ranking workflows.
- Parameter analysis tools, including correlation, regression, distribution review, and data quality summaries.
- Portable Windows launcher for local desktop use.
- DigitalOcean deployment bundle for hosting the Streamlit app behind Nginx with HTTPS.

## Repository Structure

```text
.
+-- Start Prioritization Tool.bat
+-- Support Files/
|   +-- app.py
|   +-- engine.py
|   +-- requirements.txt
|   +-- scoring_config.json
|   +-- input_template.csv
|   +-- HC_P1.jpg
|   +-- ITE_Logo.png
|   +-- landing_background.png
|   +-- Importable Database/
|   +-- SHPs/
+-- WebHost/
    +-- README.md
    +-- first_time_precinct1_setup.sh
    +-- FutureUpdate/
    +-- nginx/
    +-- precinct1project/
    +-- systemd/
```

## Local Windows Usage

For normal local use, run:

```powershell
.\Start Prioritization Tool.bat
```

The launcher starts the Streamlit application from `Support Files/app.py`. On first use, it may create or update the local Python environment under `Support Files/.venv`.

## Manual Local Run

If you prefer to run the app manually:

```powershell
cd "Support Files"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Main Application Files

- `Support Files/app.py`: Streamlit user interface, spatial viewer, data tools, AHP, TOPSIS, workspace handling, and access controls.
- `Support Files/engine.py`: scoring, AHP, TOPSIS, and core calculation helpers.
- `Support Files/scoring_config.json`: scoring mappings, labels, weights, and efficiency bins.
- `Support Files/Importable Database/`: default/importable CSV and JSON data files.
- `Support Files/SHPs/`: shapefile layers used by the Spatial View tab.

## Deployment

The production deployment is designed for an Ubuntu VPS using:

- Streamlit as the app runtime.
- `systemd` to run the app as a service.
- Nginx as a reverse proxy.
- Let's Encrypt certificates through Certbot.

The active hosted app is configured for:

```text
https://www.hydrodecisions.com/Precinct1Project/
```

Detailed deployment instructions are in:

```text
WebHost/README.md
```

Future update scripts and notes are in:

```text
WebHost/FutureUpdate/
```

## Git Hygiene

The repository intentionally excludes local environments, generated builds, archives, and runtime logs. Do not commit:

- `Support Files/.venv/`
- `Archive/`
- `EXE/`
- `dist/`
- `build/`
- access history or runtime log files

These are ignored through `.gitignore`.

## Notes

This project contains working GIS and prioritization datasets used by the application. Treat the repository as a source and deployment bundle, not as a place to store generated executable builds or local runtime history.
