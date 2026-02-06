@echo off
cd /d "%~dp0"

REM ---- Initialize Anaconda ----
call "C:\Users\akafle\AppData\Local\anaconda3\Scripts\activate.bat"

REM ---- Activate your environment ----
call conda activate PrioritznFrmwrkHCFCD

REM ---- Run Streamlit app ----
streamlit run app.py

pause