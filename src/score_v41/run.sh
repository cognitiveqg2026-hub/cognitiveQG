#!/usr/bin/env bash
# CognitiveQG Combined Trace Score reproduction -- Mac/Linux.
# Creates a venv, installs pinned deps, verifies input checksums, runs the
# full pipeline, runs the tests, then zips results_<hostname>_<date>.zip.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=${PYTHON:-python3}

echo "=== 1/6: creating venv (.venv) ==="
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "=== 2/6: verifying input checksums ==="
CHECKSUM_LOG=checksum_verify.log
if command -v shasum >/dev/null 2>&1; then
  (cd data && shasum -a 256 -c SHA256SUMS) | tee "$CHECKSUM_LOG"
elif command -v sha256sum >/dev/null 2>&1; then
  (cd data && sha256sum -c SHA256SUMS) | tee "$CHECKSUM_LOG"
else
  echo "WARNING: no sha256sum/shasum found; skipping checksum verification" | tee "$CHECKSUM_LOG"
fi

echo "=== 3/6: running pipeline (src/run_all.py) ==="
python src/run_all.py

echo "=== 4/6: running tests ==="
python -m pip install --quiet pytest
python -m pytest tests/ -v | tee pytest_report.txt

echo "=== 5/6: writing env.txt (already written by run_all.py; appending uname) ==="
{ echo "--- uname -a ---"; uname -a; } >> out/env.txt

echo "=== 6/6: building results zip ==="
HOSTNAME_SAFE=$(hostname | tr -c 'A-Za-z0-9._-' '_')
DATE_SAFE=$(date +%Y%m%d_%H%M%S)
RESULTS_ZIP="results_${HOSTNAME_SAFE}_${DATE_SAFE}.zip"
rm -f "$RESULTS_ZIP"
zip -q -r "$RESULTS_ZIP" out/ pytest_report.txt "$CHECKSUM_LOG"

echo
echo "Done. Send this file back: $RESULTS_ZIP"
echo "($(du -h "$RESULTS_ZIP" | cut -f1))"
