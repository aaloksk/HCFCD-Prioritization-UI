# Deployment Reference

This document records the deployment state that was successfully established for `Precinct1Project` so future updates do not repeat the initial VPS setup.

## Production identity
- Domain: `hydrodecisions.com`
- Canonical app URL: `https://www.hydrodecisions.com/Precinct1Project/`
- Public IP at time of setup: `162.243.40.71`
- VPS provider: DigitalOcean
- OS: Ubuntu 24.04

## Live server layout
- App root: `/opt/hydrodecisions/precinct1project`
- Live app files: `/opt/hydrodecisions/precinct1project/current`
- Python virtual environment: `/opt/hydrodecisions/precinct1project/venv`
- Requirements hash marker used by update script: `/opt/hydrodecisions/precinct1project/.requirements.sha256`

## Live service and routing
- systemd service: `precinct1project.service`
- Streamlit port: `8501`
- Streamlit bind address: `127.0.0.1`
- Streamlit base URL path: `Precinct1Project`
- Nginx site file: `/etc/nginx/sites-available/hydrodecisions.com`

## Redirect behavior that is working
- `https://hydrodecisions.com` redirects to `https://www.hydrodecisions.com/Precinct1Project/`
- `https://www.hydrodecisions.com` redirects to `https://www.hydrodecisions.com/Precinct1Project/`
- `https://www.hydrodecisions.com/Precinct1Project/` serves the Streamlit app

## Repo source of deployed files
Upload these from the repo:
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

Do not upload:
- `Support Files/.venv/`
- `Support Files/__pycache__/`
- `.git/`
- `Archive/`
- `Start Prioritization Tool.bat`
- `WebHost/`

## Installed packages already present on server
Initial VPS setup already installed:
- `python3`
- `python3-pip`
- `python3-venv`
- `nginx`
- `certbot`
- `python3-certbot-nginx`
- `rsync`
- `unzip`

Do not reinstall these as part of routine updates unless there is a server-level reason.

## Initial deployment facts
- App files were uploaded manually from Windows using OpenSSH `scp.exe`
- Windows `ssh` was not available on PATH, so the absolute path was used:
  - `C:\Windows\System32\OpenSSH\ssh.exe`
  - `C:\Windows\System32\OpenSSH\scp.exe`
- The service successfully reached `active (running)`
- Nginx config tested successfully with `nginx -t`
- HTTP routing worked before SSL
- Certbot completed and HTTPS redirects worked correctly
- The site worked on phone; a desktop FortiGuard warning was likely due to corporate filtering, not the VPS setup

## Resource notes recorded at deployment time
- Disk usage:
  - root volume size: `48G`
  - used: `3.1G`
  - available: `45G`
- Memory snapshot:
  - total: `1.9 GiB`
  - available: `1.4 GiB`
  - swap: `0B`
- Conclusion at that time:
  - storage is not a limiting factor
  - 2 GB RAM is acceptable for this single app
  - concurrency and additional apps may eventually justify moving to 4 GB / 2 vCPU

## Routine update approach going forward
Use the update tools in this folder:
- `deploy_update.ps1`
- `deploy_update.bat`

What the update script does:
- uploads the current app files from `Support Files/`
- compares local `requirements.txt` hash to the server marker
- runs `pip install -r requirements.txt` only if the requirements hash changed or the venv is missing
- restarts `precinct1project.service`
- prints recent service logs

## Standard verification commands after any update
On the VPS:
```bash
systemctl status precinct1project.service
journalctl -u precinct1project.service -n 100 --no-pager
nginx -t
systemctl status nginx
```

From any machine:
```bash
curl -I https://hydrodecisions.com
curl -I https://www.hydrodecisions.com
curl -I https://www.hydrodecisions.com/Precinct1Project/
```

## If a future change adds Python packages
- Update `Support Files/requirements.txt`
- Run `WebHost/FutureUpdate/deploy_update.ps1`
- The script will detect the hash change and install only then

## If a future change is code/data/assets only
- Run `WebHost/FutureUpdate/deploy_update.ps1`
- It should skip dependency install and only restart the service
