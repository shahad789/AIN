#!/usr/bin/env python3
"""
AIn - Cross-platform Run Script
Works on Windows, Linux, and macOS
"""

import os
import sys
import subprocess
import time
import platform
from pathlib import Path


def setup_ssl_bypass():
    """Bypass SSL verification for pip in corporate proxy environments (Netskope, etc.)."""
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["PIP_TRUSTED_HOST"] = "pypi.org pypi.python.org files.pythonhosted.org"


def get_venv_python():
    """Get the path to the venv Python executable."""
    if platform.system() == "Windows":
        return Path(".venv/Scripts/python.exe")
    return Path(".venv/bin/python")


def get_pip_path():
    """Get the path to pip in venv."""
    if platform.system() == "Windows":
        return Path(".venv/Scripts/pip.exe")
    return Path(".venv/bin/pip")


def run_command(cmd, shell=True, cwd=None):
    """Run a command and return success status."""
    try:
        subprocess.run(cmd, shell=shell, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("AIn")
    print("=" * 40)

    # Bypass SSL for corporate proxy environments
    setup_ssl_bypass()

    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)

    venv_python = get_venv_python()
    pip_path = get_pip_path()

    # Create venv if it doesn't exist
    if not venv_python.exists():
        print("Creating virtual environment...")
        if not run_command("uv venv"):
            # Fallback to standard venv
            run_command(f"{sys.executable} -m venv .venv")

    # Install dependencies with trusted hosts
    print("Installing dependencies...")
    trusted_hosts = "--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org"
    run_command(f"{pip_path} install {trusted_hosts} -r requirements.txt")

    # Start backend
    print("\nStarting FastAPI backend on port 8000...")
    backend_dir = project_root / "backend"

    if platform.system() == "Windows":
        subprocess.Popen(
            f'start "AIn Backend" cmd /k "{venv_python} main.py"',
            shell=True,
            cwd=backend_dir
        )
    else:
        subprocess.Popen(
            [str(venv_python), "main.py"],
            cwd=backend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # Wait for backend
    print("Waiting for backend to start...")
    time.sleep(3)

    # Start frontend
    print("\nStarting Streamlit frontend on port 8501...")
    print("Press Ctrl+C to stop\n")

    frontend_dir = project_root / "frontend"

    try:
        subprocess.run(
            [str(venv_python), "-m", "streamlit", "run", "app_unified.py", "--server.port", "8501"],
            cwd=frontend_dir
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
