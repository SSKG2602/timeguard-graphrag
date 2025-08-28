#!/usr/bin/env python3
"""
Zero-dependency bootstrap script for TimeGuard Graph-RAG
Handles complete environment setup and application startup using only standard library
- Creates virtual environment
- Installs dependencies
- Downloads spaCy model
- Starts backend and frontend services
- Uses urllib instead of requests to avoid circular dependencies
"""

import os
import sys
import subprocess
import venv
import platform
import time
import json
from urllib import request as urlrequest, error as urlerror

# Configuration constants
PY_MIN = (3, 10)  # Minimum Python version required
VENV_DIR = ".venv"  # Virtual environment directory
REQ = "requirements.txt"  # Dependencies file
SPACY_MODEL = "en_core_web_sm"  # spaCy language model
HEALTH_URL = "http://127.0.0.1:8000/healthz"  # Backend health check URL


def log(*a): 
    """Simple logging function with consistent prefix"""
    print("[run.py]", *a)


def exe(win, nix):
    """
    Get platform-specific executable path within virtual environment
    Args:
        win: Windows executable name
        nix: Unix/Linux executable name
    """
    return os.path.join(VENV_DIR, "Scripts", win) if platform.system() == "Windows" else os.path.join(VENV_DIR, "bin", nix)


# Platform-specific executable getters
def PY(): return exe("python.exe", "python")
def PIP(): return exe("pip.exe", "pip")
def STREAMLIT(): return exe("streamlit.exe", "streamlit")


def ensure_python():
    """
    Verify Python version meets minimum requirements
    Exits with error if version is insufficient
    """
    if sys.version_info < PY_MIN:
        sys.exit(
            f"Python {PY_MIN[0]}.{PY_MIN[1]}+ required, found {platform.python_version()}")


def ensure_venv():
    """
    Create virtual environment if it doesn't exist
    Uses built-in venv module for maximum compatibility
    """
    if not os.path.isdir(VENV_DIR):
        log(f"Creating {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)


def run(cmd, env=None, check=True):
    """
    Execute subprocess command with logging
    Args:
        cmd: Command list to execute
        env: Environment variables
        check: Whether to raise on non-zero exit
    """
    log(">>", " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env)


def install_deps():
    """
    Install all project dependencies in virtual environment
    Upgrades pip first, then installs requirements, then spaCy model
    """
    env = os.environ.copy()
    # Set protobuf implementation to avoid compatibility issues
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    
    # Upgrade pip and build tools first
    run([PY(), "-m", "pip", "install", "--upgrade",
        "pip", "wheel", "setuptools"], env=env)
    
    # Install project requirements
    run([PY(), "-m", "pip", "install", "-r", REQ], env=env)
    
    # Download spaCy language model (best-effort, continues on failure)
    try:
        run([PY(), "-m", "spacy", "download", SPACY_MODEL], env=env)
    except subprocess.CalledProcessError:
        log("⚠️ spaCy model download failed; continuing.")


def http_get_json(url: str, timeout: float = 3.0):
    """
    Make HTTP GET request and parse JSON response
    Uses urllib to avoid dependency on requests library
    Args:
        url: Target URL
        timeout: Request timeout in seconds
    Returns:
        Parsed JSON dict or None on failure
    """
    try:
        req = urlrequest.Request(
            url, headers={"User-Agent": "timeguard-bootstrap"})
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def wait_until_ready(timeout_sec=900):
    """
    Poll backend health endpoint until service is ready
    Args:
        timeout_sec: Maximum time to wait in seconds
    Returns:
        True if backend becomes ready, False on timeout or error
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        # Check backend health status
        j = http_get_json(HEALTH_URL, timeout=3.0)
        if j and j.get("status") == "ready":
            return True
        
        # Handle different status states
        state = j.get("status") if isinstance(j, dict) else "down"
        if state == "error":
            log(f"Backend status: error ({j.get('error')}) …")
            return False
        
        log(f"Backend status: {state} …")
        time.sleep(2.0)
    return False


def start():
    """
    Start the complete TimeGuard Graph-RAG system
    Sets up environment, starts backend, waits for readiness, then starts frontend
    """
    # Prepare environment with sensible defaults
    env = os.environ.copy()
    env.setdefault("TIMEZONE", "Asia/Kolkata")
    env.setdefault("DAYFIRST", "1")
    env.setdefault("FISCAL_YEAR_START_MONTH", "4")
    env.setdefault("FISCAL_YEAR_START_DAY", "1")
    env.setdefault("QWEN_MODEL", "")  # Auto-select best model if empty
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("BACKEND_URL", "http://127.0.0.1:8000")

    # Override with .env file settings if present
    if os.path.exists(".env"):
        for line in open(".env", "r", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    # Debug information for troubleshooting
    log("File:", os.path.abspath(__file__))
    log("CWD :", os.getcwd())
    log("PyEX:", sys.executable)
    log("VENV PY:", PY())
    log("VENV ST:", STREAMLIT())

    # Start backend server process
    backend = subprocess.Popen([PY(), "server.py"], env=env)

    # Wait for backend to initialize and load models
    log("⏳ Waiting for backend to become ready (models load in background)…")
    if not wait_until_ready():
        log("Backend didn't become ready in time.")
        backend.terminate()
        sys.exit(1)

    # Start Streamlit frontend process
    front = subprocess.Popen([STREAMLIT(
    ), "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"], env=env)

    # Display success message with URLs
    print("\n Backend: http://127.0.0.1:8000/docs\n Frontend: http://127.0.0.1:8501\n")
    
    # Wait for both processes and handle shutdown
    try:
        backend.wait()
        front.wait()
    except KeyboardInterrupt:
        backend.terminate()
        front.terminate()


# Main execution flow
if __name__ == "__main__":
    ensure_python()    # Check Python version
    ensure_venv()      # Create virtual environment
    install_deps()     # Install dependencies
    start()           # Start services
