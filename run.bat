@echo off
REM AIn - Windows Run Script

echo AIn
echo ========================

REM Bypass SSL for pip (Netskope/corporate proxy)
set PYTHONHTTPSVERIFY=0
set PIP_TRUSTED_HOST=pypi.org pypi.python.org files.pythonhosted.org

REM Store the project root
set PROJECT_ROOT=%~dp0

REM Check if venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate and install deps
echo Installing dependencies...
call .venv\Scripts\activate.bat
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

REM Start backend in new window (using full path to venv python)
echo Starting FastAPI backend on port 8000...
start "AIn Backend" cmd /k "cd /d %PROJECT_ROOT%backend && %PROJECT_ROOT%.venv\Scripts\python.exe main.py"

REM Wait for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend (using full path to venv python)
echo Starting Streamlit frontend on port 8501...
cd /d %PROJECT_ROOT%frontend
%PROJECT_ROOT%.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
