@echo off
setlocal
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0deploy_update.ps1" %*
endlocal
