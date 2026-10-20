"""Shared paths for the revision scripts (repo-relative, no hard-coded home paths).

REPO        repository root (two levels above this file)
DATA        REPO/data
OUT         REPO/results/revision      -- every intermediate result (JSON/CSV) lives here
MANUSCRIPT  LaTeX manuscript folder; set CQG_MANUSCRIPT to override
            (default: REPO/manuscript)
TABLE_DIR   MANUSCRIPT/table           -- where generated .tex tables are written
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SCRIPTS = REPO / "scripts"
OUT = REPO / "results" / "revision"
MANUSCRIPT = Path(os.environ.get("CQG_MANUSCRIPT", REPO / "manuscript"))
TABLE_DIR = Path(os.environ.get("CQG_TABLE_DIR", MANUSCRIPT / "table"))
OUT.mkdir(parents=True, exist_ok=True)
