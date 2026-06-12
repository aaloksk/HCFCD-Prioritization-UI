# HydroDecisions VPS Deployment Bundle

This bundle is now narrowed to the one app that exists today:
- `Precinct1Project`

Final recommendation: use `/opt/hydrodecisions`

Why:
- it matches the live domain name
- it avoids carrying the old `waterdecisions` naming into production
- it keeps future apps under one clean product root

## 1. Streamlit app review

### Active Streamlit app entry file
- `Support Files/app.py`

### Base URL path verification
The systemd service is configured with:
- `--server.baseUrlPath Precinct1Project`

That is the correct Streamlit setting for serving the app at:
- `https://www.hydrodecisions.com/Precinct1Project/`

## 2. Server folder structure

Use this exact structure:

```text
/opt/hydrodecisions/
    precinct1project/
        current/
            app.py
            engine.py
            requirements.txt
            scoring_config.json
            input_template.csv
            HC_P1.jpg
            ITE_Logo.png
            landing_background.png
            Importable Database/
            SHPs/
        venv/
```

## 3. Final files to use from `WebHost`

Use these files now:
- `WebHost/README.md`
- `WebHost/precinct1project/requirements.txt`
- `WebHost/precinct1project/upload_manifest.txt`
- `WebHost/systemd/precinct1project.service`
- `WebHost/nginx/hydrodecisions.com.http.conf`
- `WebHost/nginx/hydrodecisions.com.ssl.conf`

## 4. Exact repo files to upload for `Precinct1Project`

Upload these from your repo:
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

## 5. Ubuntu install commands

### Update packages
```bash
sudo apt update && sudo apt upgrade -y
```

### Install Python, venv, nginx, certbot, rsync
```bash
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx rsync unzip
```

## 6. Create deployment directories

```bash
sudo mkdir -p /opt/hydrodecisions/precinct1project/current
sudo chown -R $USER:$USER /opt/hydrodecisions
```

## 7. Upload the app files

Run this from your local machine in the repo root:

```bash
rsync -avz --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'Archive' \
  --exclude '*.bat' \
  --exclude 'WebHost' \
  "Support Files/" root@162.243.40.71:/opt/hydrodecisions/precinct1project/current/
```

If you prefer `scp`:
```bash
scp -r "Support Files/app.py" "Support Files/engine.py" "Support Files/requirements.txt" \
  "Support Files/scoring_config.json" "Support Files/input_template.csv" \
  "Support Files/HC_P1.jpg" "Support Files/ITE_Logo.png" "Support Files/landing_background.png" \
  "Support Files/Importable Database" "Support Files/SHPs" \
  root@162.243.40.71:/opt/hydrodecisions/precinct1project/current/
```

## 8. Create the Python virtual environment on the VPS

SSH in:
```bash
ssh root@162.243.40.71
```

Create and populate the venv:
```bash
python3 -m venv /opt/hydrodecisions/precinct1project/venv
source /opt/hydrodecisions/precinct1project/venv/bin/activate
pip install --upgrade pip wheel
pip install -r /opt/hydrodecisions/precinct1project/current/requirements.txt
```

## 9. Install the systemd service

From your local machine:
```bash
scp WebHost/systemd/precinct1project.service root@162.243.40.71:/tmp/precinct1project.service
```

On the VPS:
```bash
sudo mv /tmp/precinct1project.service /etc/systemd/system/precinct1project.service
sudo systemctl daemon-reload
sudo systemctl enable precinct1project.service
sudo systemctl start precinct1project.service
sudo systemctl status precinct1project.service
```

View logs if needed:
```bash
journalctl -u precinct1project.service -f
```

## 10. Install the initial Nginx HTTP config

From your local machine:
```bash
scp WebHost/nginx/hydrodecisions.com.http.conf root@162.243.40.71:/tmp/hydrodecisions.com.conf
```

On the VPS:
```bash
sudo mv /tmp/hydrodecisions.com.conf /etc/nginx/sites-available/hydrodecisions.com
sudo ln -sf /etc/nginx/sites-available/hydrodecisions.com /etc/nginx/sites-enabled/hydrodecisions.com
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 11. Obtain the SSL certificate

On the VPS:
```bash
sudo certbot --nginx -d hydrodecisions.com -d www.hydrodecisions.com
```

This may modify the Nginx file automatically. Since you want a precise redirect setup, replace it after certbot with the prepared SSL config below.

## 12. Install the final SSL Nginx config

From your local machine:
```bash
scp WebHost/nginx/hydrodecisions.com.ssl.conf root@162.243.40.71:/tmp/hydrodecisions.com.ssl.conf
```

On the VPS:
```bash
sudo mv /tmp/hydrodecisions.com.ssl.conf /etc/nginx/sites-available/hydrodecisions.com
sudo nginx -t
sudo systemctl restart nginx
```

## 13. Final verification commands

### Verify Streamlit is listening only on localhost
```bash
ss -tulpn | grep 8501
```
Expected: `127.0.0.1:8501`

### Verify systemd service health
```bash
systemctl status precinct1project.service
```

### Verify Nginx config
```bash
sudo nginx -t
```

### Verify certificate renewal timer
```bash
systemctl status certbot.timer
```

### Verify URLs
```bash
curl -I http://hydrodecisions.com
curl -I https://hydrodecisions.com
curl -I https://www.hydrodecisions.com
curl -I https://www.hydrodecisions.com/Precinct1Project/
```

Expected behavior:
- `http://hydrodecisions.com` -> redirect toward HTTPS / certbot-managed path
- `https://hydrodecisions.com` -> `https://www.hydrodecisions.com/Precinct1Project/`
- `https://www.hydrodecisions.com` -> `https://www.hydrodecisions.com/Precinct1Project/`
- `https://www.hydrodecisions.com/Precinct1Project/` -> app loads

## 14. Final deployment checklist in exact order

1. Confirm Namecheap A records for `@` and `www` point to `162.243.40.71`
2. SSH into the Ubuntu VPS
3. Run system updates
4. Install Python, venv, nginx, certbot, and rsync
5. Create `/opt/hydrodecisions/precinct1project/current`
6. Upload the `Precinct1Project` files from `Support Files/`
7. Create `/opt/hydrodecisions/precinct1project/venv`
8. Install Python dependencies into that venv
9. Upload `WebHost/systemd/precinct1project.service`
10. Enable and start `precinct1project.service`
11. Confirm the app is running locally on `127.0.0.1:8501`
12. Upload `WebHost/nginx/hydrodecisions.com.http.conf`
13. Enable the Nginx site and restart Nginx
14. Run `certbot` for `hydrodecisions.com` and `www.hydrodecisions.com`
15. Upload `WebHost/nginx/hydrodecisions.com.ssl.conf`
16. Test and restart Nginx
17. Open `https://www.hydrodecisions.com/Precinct1Project/`
18. Verify the root domain redirects correctly to the app path
19. Verify certbot renewal timer is active
20. Save the working commands for future updates

## 15. Safe future updates

Recommended pattern:
- upload a new release folder
- reinstall requirements if changed
- swap the `current` folder
- restart the service
- roll back if needed

Example:
```bash
ssh root@162.243.40.71
mkdir -p /opt/hydrodecisions/precinct1project/releases/2026-04-01-01
exit
```

```bash
rsync -avz --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'Archive' \
  --exclude '*.bat' \
  --exclude 'WebHost' \
  "Support Files/" root@162.243.40.71:/opt/hydrodecisions/precinct1project/releases/2026-04-01-01/
```

```bash
ssh root@162.243.40.71
source /opt/hydrodecisions/precinct1project/venv/bin/activate
pip install -r /opt/hydrodecisions/precinct1project/releases/2026-04-01-01/requirements.txt
mv /opt/hydrodecisions/precinct1project/current /opt/hydrodecisions/precinct1project/current.bak
mv /opt/hydrodecisions/precinct1project/releases/2026-04-01-01 /opt/hydrodecisions/precinct1project/current
sudo systemctl restart precinct1project.service
sudo systemctl status precinct1project.service
```

Rollback if needed:
```bash
ssh root@162.243.40.71
rm -rf /opt/hydrodecisions/precinct1project/current
mv /opt/hydrodecisions/precinct1project/current.bak /opt/hydrodecisions/precinct1project/current
sudo systemctl restart precinct1project.service
```
