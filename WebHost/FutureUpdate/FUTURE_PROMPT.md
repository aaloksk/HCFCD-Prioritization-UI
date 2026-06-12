# Future Update Prompt

Use this prompt in a future Codex chat when you want help updating the live `Precinct1Project` deployment without redoing the initial VPS setup.

---

I have an already-deployed Streamlit app called `Precinct1Project` running on an Ubuntu 24.04 DigitalOcean VPS.

Current live domain:
- `https://www.hydrodecisions.com/Precinct1Project/`

Current redirect behavior:
- `https://hydrodecisions.com` -> `https://www.hydrodecisions.com/Precinct1Project/`
- `https://www.hydrodecisions.com` -> `https://www.hydrodecisions.com/Precinct1Project/`

Server details:
- Provider: DigitalOcean
- Ubuntu: 24.04
- Public IP: `162.243.40.71`
- App root: `/opt/hydrodecisions/precinct1project`
- Live code folder: `/opt/hydrodecisions/precinct1project/current`
- Virtual environment: `/opt/hydrodecisions/precinct1project/venv`
- systemd service: `precinct1project.service`
- Nginx site file: `/etc/nginx/sites-available/hydrodecisions.com`

Streamlit runtime:
- Entrypoint: `/opt/hydrodecisions/precinct1project/current/app.py`
- Port: `8501`
- Address: `127.0.0.1`
- baseUrlPath: `Precinct1Project`

Important context:
- Initial VPS setup is already complete.
- Python, pip, venv, nginx, certbot, and rsync are already installed.
- The app is already live and working.
- Do not redo initial package installation unless there is a new dependency requirement.
- Do not recreate SSL unless there is a real certificate problem.
- Focus only on updating the live app safely.

Local repo on Windows:
- `D:\HCFCD_Prioritization_UI`

Current deployable files come from:
- `Support Files/app.py`
- `Support Files/engine.py`
- `Support Files/requirements.txt`
- `Support Files/scoring_config.json`
- `Support Files/input_template.csv`
- `Support Files/HC_P1.jpg`
- `Support Files/ITE_Logo.png`
- `Support Files/landing_background.png`
- `Support Files/Importable Database/`
- `Support Files/SHPs/`

Do not deploy:
- `.git`
- `Archive`
- `.venv`
- `__pycache__`
- `.bat`
- `WebHost`

What I want you to help with now:
- Review the current local changes I made.
- Tell me whether I need to upload only code/data/assets, or also reinstall Python dependencies.
- If `requirements.txt` changed, only then include dependency update steps.
- Give me the exact commands to update the live website safely.
- Prefer the existing update scripts in `WebHost/FutureUpdate/` if they still fit the situation.
- If needed, update those scripts instead of making me repeat manual deployment steps.
- If there is risk of breaking the live site, explain the safest rollout/rollback path.

Please start by:
1. Reviewing what changed locally.
2. Telling me whether `requirements.txt` changed in a way that affects deployment.
3. Giving me the exact safest update steps for the current situation.

---
