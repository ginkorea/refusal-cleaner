import os

# Repo root (two levels up from this file: refusal_cleaner/__init__.py → src → repo root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
