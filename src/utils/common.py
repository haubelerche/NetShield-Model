
import json, joblib, os
from pathlib import Path

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_pickle(obj, path):
    from joblib import dump
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(obj, path)

def load_pickle(path):
    from joblib import load
    return load(path)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
