@echo off
REM GiftxAI — one-command local launch.
REM Creates a virtual environment on first run, installs dependencies, starts the app.
REM The full retrieval stack pulls PyTorch and sentence-transformers (~2 GB).
REM Pass "ui" to install only Streamlit if you just want to look at the interface:
REM     run.bat ui
setlocal
cd /d "%~dp0"

set VENV=.venv
set PY=%VENV%\Scripts\python.exe

if not exist "%PY%" (
    echo Creating virtual environment...
    py -3.12 -m venv %VENV% 2>nul || py -3 -m venv %VENV% || python -m venv %VENV%
    if not exist "%PY%" (
        echo.
        echo Could not create a virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
    "%PY%" -m pip install --upgrade pip
    if /i "%~1"=="ui" (
        echo Installing interface only...
        "%PY%" -m pip install streamlit
    ) else (
        echo Installing dependencies. This one takes a while...
        "%PY%" -m pip install -r requirements.txt
    )
)

echo.
echo Starting GiftxAI at http://localhost:8503
echo Set GROQ_API_KEY beforehand, or paste a key into the sidebar.
echo Press Ctrl+C to stop.
echo.
"%PY%" -m streamlit run app.py --server.port 8503
endlocal
