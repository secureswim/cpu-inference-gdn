"""Shared setup gate; deliberately uses only the Python standard library."""
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "build/system.json"
SCHEMA = 1
SETUP_HINT = "Run python scripts/configure.py first (python3 on macOS/Linux)."


def identity():
    return {"os": platform.system(), "architecture": platform.machine().lower(),
            "host": platform.node(), "bits": struct.calcsize("P") * 8,
            "root": str(ROOT), "logical_cpus": os.cpu_count(),
            "allowed_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def source_signature():
    paths = [ROOT / "CMakeLists.txt", ROOT / "requirements.txt"]
    paths += sorted(path for path in (ROOT / "src").rglob("*") if path.is_file())
    paths += sorted((ROOT / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_configuration(for_build=False):
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA or data["identity"] != identity():
            raise ValueError("configuration belongs to another system or checkout")
        if data["source_signature"] != source_signature():
            raise ValueError("project setup has changed")
        if data["status"] not in (("building", "ready") if for_build else ("ready",)):
            raise ValueError("setup has not completed successfully")
        python = Path(data["python"])
        if not python.is_file():
            raise ValueError("project Python environment is missing")
        if not for_build:
            for name in ("gdn", "machine", "stream", "ceilings"):
                if not executable(data, name).is_file():
                    raise ValueError(f"{name} executable is missing")
        return data
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"System configuration required: {error}. {SETUP_HINT}") from None


def executable(config, name):
    return Path(config["bin_dir"]) / (name + (".exe" if os.name == "nt" else ""))


def use_project_python():
    """Allow users to run entry points without manually activating the venv."""
    config = require_configuration()
    if Path(sys.prefix).resolve() != Path(config["python"]).parent.parent.resolve():
        import subprocess
        raise SystemExit(subprocess.call([config["python"], *sys.argv]))
    return config


if __name__ == "__main__":
    require_configuration(for_build="--for-build" in sys.argv)
