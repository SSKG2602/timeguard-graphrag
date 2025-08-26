#!/usr/bin/env python3
"""
Zero-dependency bootstrap (std lib only):
- Creates .venv
- pip install -r requirements.txt
- Downloads spaCy model
- Starts backend (server.py)
- Polls /healthz using urllib (no 'requests' import!)
- Starts Streamlit with the SAME .venv interpreter
"""

import os
import sys
import subprocess
import venv
import platform
import time
import json
from urllib import request as urlrequest, error as urlerror

PY_MIN = (3, 10)
VENV_DIR = ".venv"
REQ = "requirements.txt"
SPACY_MODEL = "en_core_web_sm"
HEALTH_URL = "http://127.0.0.1:8000/healthz"


def log(*a): print("[run.py]", *a)


def exe(win, nix):
    return os.path.join(VENV_DIR, "Scripts", win) if platform.system() == "Windows" else os.path.join(VENV_DIR, "bin", nix)


def PY(): return exe("python.exe", "python")
def PIP(): return exe("pip.exe", "pip")
def STREAMLIT(): return exe("streamlit.exe", "streamlit")


def ensure_python():
    if sys.version_info < PY_MIN:
        sys.exit(
            f"Python {PY_MIN[0]}.{PY_MIN[1]}+ required, found {platform.python_version()}")


def ensure_venv():
    if not os.path.isdir(VENV_DIR):
        log(f"Creating {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)


def run(cmd, env=None, check=True):
    log(">>", " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env)


def install_deps():
    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    # upgrade pip tooling
    run([PY(), "-m", "pip", "install", "--upgrade",
        "pip", "wheel", "setuptools"], env=env)
    # install requirements
    run([PY(), "-m", "pip", "install", "-r", REQ], env=env)
    # spaCy model (best-effort)
    try:
        run([PY(), "-m", "spacy", "download", SPACY_MODEL], env=env)
    except subprocess.CalledProcessError:
        log("⚠️ spaCy model download failed; continuing.")


def http_get_json(url: str, timeout: float = 3.0):
    try:
        req = urlrequest.Request(
            url, headers={"User-Agent": "timeguard-bootstrap"})
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def wait_until_ready(timeout_sec=900):
    start = time.time()
    while time.time() - start < timeout_sec:
        j = http_get_json(HEALTH_URL, timeout=3.0)
        if j and j.get("status") == "ready":
            return True
        state = j.get("status") if isinstance(j, dict) else "down"
        log(f"Backend status: {state} …")
        time.sleep(2.0)
    return False


def start():
    env = os.environ.copy()
    # sensible defaults; .env can override
    env.setdefault("TIMEZONE", "Asia/Kolkata")
    env.setdefault("DAYFIRST", "1")
    env.setdefault("FISCAL_YEAR_START_MONTH", "4")
    env.setdefault("FISCAL_YEAR_START_DAY", "1")
    env.setdefault("QWEN_MODEL", "")  # auto-pick best Qwen if empty
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("BACKEND_URL", "http://127.0.0.1:8000")

    # .env overrides
    if os.path.exists(".env"):
        for line in open(".env", "r", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    # DEBUG: prove which file and interpreter are used
    log("File:", os.path.abspath(__file__))
    log("CWD :", os.getcwd())
    log("PyEX:", sys.executable)
    log("VENV PY:", PY())
    log("VENV ST:", STREAMLIT())

    # start backend with the .venv python
    backend = subprocess.Popen([PY(), "server.py"], env=env)

    log("⏳ Waiting for backend to become ready (models load in background)…")
    if not wait_until_ready():
        log("❌ Backend didn't become ready in time.")
        backend.terminate()
        sys.exit(1)

    # start Streamlit with the .venv streamlit
    front = subprocess.Popen([STREAMLIT(
    ), "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"], env=env)

    print("\n✅ Backend: http://127.0.0.1:8000/docs\n✅ Frontend: http://127.0.0.1:8501\n")
    try:
        backend.wait()
        front.wait()
    except KeyboardInterrupt:
        backend.terminate()
        front.terminate()


if __name__ == "__main__":
    ensure_python()
    ensure_venv()
    install_deps()
    start()
