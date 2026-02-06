# HCFCD Prioritization UI

Streamlit app for project prioritization used by HCFCD. This repository contains the Streamlit UI, scoring engine, and input templates.

Quick start (local)

1. Create a Python environment (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the app:

```powershell
streamlit run app.py
```

Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io, sign in, create a new app and select:
   - Repository: `aaloksk/HCFCD-Prioritization-UI`
   - Branch: `main`
   - File path: `app.py`

Notes
- `requirements.txt` lists Python packages required by the app.
- `input_template.csv` and `scoring_config.json` contain the data and scoring configuration used by the app.
- If you need a specific Python version on Streamlit Cloud, `runtime.txt` is included.

Contact

If you need help deploying or customizing the app, open an issue in the repo or contact the author.
# HCFCD-Prioritization-UI