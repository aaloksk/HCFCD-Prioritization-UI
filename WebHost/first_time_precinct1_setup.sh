#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="/opt/hydrodecisions/precinct1project"
APP_CURRENT="$APP_ROOT/current"
APP_VENV="$APP_ROOT/venv"
SERVICE_NAME="precinct1project"
DOMAIN_ROOT="hydrodecisions.com"
DOMAIN_WWW="www.hydrodecisions.com"
STREAMLIT_PATH="Precinct1Project"
STREAMLIT_PORT="8501"
NGINX_SITE="/etc/nginx/sites-available/hydrodecisions.com"
NGINX_ENABLED="/etc/nginx/sites-enabled/hydrodecisions.com"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run this script as root: sudo bash first_time_precinct1_setup.sh"
  exit 1
fi

echo "==> Updating apt package lists"
apt update

echo "==> Installing system packages"
apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx rsync unzip

echo "==> Creating deployment folders"
mkdir -p "$APP_CURRENT"
mkdir -p "$APP_VENV"
mkdir -p /opt/hydrodecisions

# Make the deployment tree readable by nginx/systemd service user.
chown -R www-data:www-data /opt/hydrodecisions
chmod -R 755 /opt/hydrodecisions

echo "==> Checking whether app files are present"
MISSING=0
for required in \
  "$APP_CURRENT/app.py" \
  "$APP_CURRENT/engine.py" \
  "$APP_CURRENT/requirements.txt" \
  "$APP_CURRENT/scoring_config.json" \
  "$APP_CURRENT/input_template.csv" \
  "$APP_CURRENT/Importable Database" \
  "$APP_CURRENT/SHPs"
  do
  if [[ ! -e "$required" ]]; then
    echo "   Missing: $required"
    MISSING=1
  fi
done

echo "==> Writing systemd service file"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=HydroDecisions Precinct1Project Streamlit App
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=${APP_CURRENT}
Environment=PYTHONUNBUFFERED=1
ExecStart=${APP_VENV}/bin/streamlit run app.py --server.port ${STREAMLIT_PORT} --server.address 127.0.0.1 --server.baseUrlPath ${STREAMLIT_PATH} --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "==> Writing initial HTTP nginx config"
cat > "$NGINX_SITE" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN_ROOT} ${DOMAIN_WWW};

    client_max_body_size 100M;

    location = / {
        return 301 http://${DOMAIN_WWW}/${STREAMLIT_PATH}/;
    }

    location = /${STREAMLIT_PATH} {
        return 301 http://${DOMAIN_WWW}/${STREAMLIT_PATH}/;
    }

    location /${STREAMLIT_PATH}/ {
        proxy_pass http://127.0.0.1:${STREAMLIT_PORT}/${STREAMLIT_PATH}/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOF

echo "==> Enabling nginx site"
ln -sf "$NGINX_SITE" "$NGINX_ENABLED"
rm -f /etc/nginx/sites-enabled/default

echo "==> Testing nginx config"
nginx -t

echo "==> Enabling nginx service"
systemctl enable nginx
systemctl restart nginx

echo "==> Reloading systemd"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"

if [[ "$MISSING" -eq 0 ]]; then
  echo "==> App files found. Creating/updating Python virtual environment"
  if [[ ! -x "${APP_VENV}/bin/python" ]]; then
    rm -rf "$APP_VENV"
    python3 -m venv "$APP_VENV"
  fi

  source "${APP_VENV}/bin/activate"
  pip install --upgrade pip wheel
  pip install -r "${APP_CURRENT}/requirements.txt"

  echo "==> Starting ${SERVICE_NAME} service"
  systemctl restart "${SERVICE_NAME}.service"
  systemctl status "${SERVICE_NAME}.service" --no-pager || true
else
  echo "==> App files are not uploaded yet, so the Streamlit service was enabled but not started."
fi

echo
if [[ "$MISSING" -eq 1 ]]; then
  cat <<NEXTSTEPS
NEXT STEPS AFTER YOU UPLOAD FILES
================================
1. Upload these repo items into ${APP_CURRENT}/
   - Support Files/app.py
   - Support Files/engine.py
   - Support Files/requirements.txt
   - Support Files/scoring_config.json
   - Support Files/input_template.csv
   - Support Files/HC_P1.jpg
   - Support Files/ITE_Logo.png
   - Support Files/landing_background.png
   - Support Files/Importable Database/
   - Support Files/SHPs/

2. Then run these commands on the VPS:
   sudo rm -rf ${APP_VENV}
   sudo python3 -m venv ${APP_VENV}
   sudo ${APP_VENV}/bin/pip install --upgrade pip wheel
   sudo ${APP_VENV}/bin/pip install -r ${APP_CURRENT}/requirements.txt
   sudo chown -R www-data:www-data /opt/hydrodecisions
   sudo systemctl restart ${SERVICE_NAME}.service
   sudo systemctl status ${SERVICE_NAME}.service
   curl -I http://${DOMAIN_WWW}/${STREAMLIT_PATH}/

3. After HTTP is confirmed working, obtain SSL:
   sudo certbot --nginx -d ${DOMAIN_ROOT} -d ${DOMAIN_WWW}
NEXTSTEPS
else
  cat <<NEXTSTEPS
HTTP VERIFICATION COMMANDS
==========================
Run these now:
  systemctl status ${SERVICE_NAME}.service
  ss -tulpn | grep ${STREAMLIT_PORT}
  curl -I http://${DOMAIN_ROOT}
  curl -I http://${DOMAIN_WWW}
  curl -I http://${DOMAIN_WWW}/${STREAMLIT_PATH}/

AFTER HTTP IS CONFIRMED, RUN SSL
================================
  sudo certbot --nginx -d ${DOMAIN_ROOT} -d ${DOMAIN_WWW}
NEXTSTEPS
fi
