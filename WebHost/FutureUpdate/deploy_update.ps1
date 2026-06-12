param(
    [string]$ServerIp = "162.243.40.71",
    [string]$ServerUser = "root",
    [string]$LocalRepoRoot = "D:\HCFCD_Prioritization_UI",
    [string]$RemoteAppRoot = "/opt/hydrodecisions/precinct1project",
    [switch]$SkipDependencyCheck,
    [switch]$SkipServiceRestart
)

$ErrorActionPreference = "Stop"

$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$scp = "C:\Windows\System32\OpenSSH\scp.exe"

if (-not (Test-Path $ssh)) {
    throw "ssh.exe not found at $ssh"
}
if (-not (Test-Path $scp)) {
    throw "scp.exe not found at $scp"
}

$remoteCurrent = "$RemoteAppRoot/current"
$remoteVenv = "$RemoteAppRoot/venv"
$remoteReqHashFile = "$RemoteAppRoot/.requirements.sha256"
$serviceName = "precinct1project.service"

$uploadItems = @(
    "Support Files/app.py",
    "Support Files/engine.py",
    "Support Files/requirements.txt",
    "Support Files/scoring_config.json",
    "Support Files/input_template.csv",
    "Support Files/HC_P1.jpg",
    "Support Files/ITE_Logo.png",
    "Support Files/landing_background.png",
    "Support Files/Importable Database",
    "Support Files/SHPs"
) | ForEach-Object { Join-Path $LocalRepoRoot $_ }

foreach ($item in $uploadItems) {
    if (-not (Test-Path $item)) {
        throw "Required local file/folder missing: $item"
    }
}

Write-Host "==> Ensuring remote app folder exists"
& $ssh "$ServerUser@$ServerIp" "mkdir -p '$remoteCurrent'"

Write-Host "==> Uploading updated app files"
$scpArgs = @("-r") + $uploadItems + @("${ServerUser}@${ServerIp}:${remoteCurrent}/")
& $scp @scpArgs

$localReq = Join-Path $LocalRepoRoot "Support Files\requirements.txt"
$localReqHash = (Get-FileHash -Algorithm SHA256 $localReq).Hash.ToLowerInvariant()
$runDependencyCheck = if ($SkipDependencyCheck) { "0" } else { "1" }
$runRestart = if ($SkipServiceRestart) { "0" } else { "1" }

$remoteTemplate = @'
set -euo pipefail
REMOTE_CURRENT="__REMOTE_CURRENT__"
REMOTE_VENV="__REMOTE_VENV__"
REMOTE_REQ_HASH_FILE="__REMOTE_REQ_HASH_FILE__"
LOCAL_REQ_HASH="__LOCAL_REQ_HASH__"
RUN_DEP_CHECK="__RUN_DEP_CHECK__"
RUN_RESTART="__RUN_RESTART__"
SERVICE_NAME="__SERVICE_NAME__"

if [ "$RUN_DEP_CHECK" = "1" ]; then
  NEED_INSTALL=0
  if [ ! -f "$REMOTE_REQ_HASH_FILE" ]; then
    NEED_INSTALL=1
  else
    CURRENT_HASH=$(cat "$REMOTE_REQ_HASH_FILE" 2>/dev/null || true)
    if [ "$CURRENT_HASH" != "$LOCAL_REQ_HASH" ]; then
      NEED_INSTALL=1
    fi
  fi

  if [ ! -x "$REMOTE_VENV/bin/python" ]; then
    echo "==> Virtual environment missing; recreating"
    rm -rf "$REMOTE_VENV"
    python3 -m venv "$REMOTE_VENV"
    NEED_INSTALL=1
  fi

  if [ "$NEED_INSTALL" = "1" ]; then
    echo "==> Installing/updating Python dependencies"
    "$REMOTE_VENV/bin/pip" install --upgrade pip wheel
    "$REMOTE_VENV/bin/pip" install -r "$REMOTE_CURRENT/requirements.txt"
    printf '%s' "$LOCAL_REQ_HASH" > "$REMOTE_REQ_HASH_FILE"
  else
    echo "==> requirements.txt unchanged; skipping pip install"
  fi
fi

if [ "$RUN_RESTART" = "1" ]; then
  echo "==> Restarting Streamlit service"
  systemctl restart "$SERVICE_NAME"
  systemctl status "$SERVICE_NAME" --no-pager
fi

echo "==> Recent service logs"
journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
'@

$remoteScript = $remoteTemplate.Replace('__REMOTE_CURRENT__', $remoteCurrent).
    Replace('__REMOTE_VENV__', $remoteVenv).
    Replace('__REMOTE_REQ_HASH_FILE__', $remoteReqHashFile).
    Replace('__LOCAL_REQ_HASH__', $localReqHash).
    Replace('__RUN_DEP_CHECK__', $runDependencyCheck).
    Replace('__RUN_RESTART__', $runRestart).
    Replace('__SERVICE_NAME__', $serviceName)

Write-Host "==> Running remote update steps"
$remoteScript | & $ssh "$ServerUser@$ServerIp" "bash -s"

Write-Host ""
Write-Host "Update complete. Test:" -ForegroundColor Green
Write-Host "https://www.hydrodecisions.com/Precinct1Project/"
