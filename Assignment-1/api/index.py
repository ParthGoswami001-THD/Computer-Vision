from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_FILE = ROOT_DIR / "src" / "demo" / "app.py"

spec = importlib.util.spec_from_file_location("demo_app", APP_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load Flask app from {APP_FILE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.APP
