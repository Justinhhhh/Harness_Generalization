#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/verify_setup.py

if [[ -n "${ALFWORLD_DATA:-}" && -x "${HARNESS_ENV:-}/bin/python" ]]; then
  ALFWORLD_DATA="$ALFWORLD_DATA" \
    "$HARNESS_ENV/bin/python" -c 'import alfworld, textworld; print("OK: ALFWorld imports")'
else
  echo "INFO: set ALFWORLD_DATA and HARNESS_ENV for package import checks"
fi

if [[ -n "${META_HARNESS:-}" ]]; then
  PYTHONPATH="$META_HARNESS/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3.12 -c 'import metaharness; print("OK: Meta-Harness imports")'
else
  echo "INFO: set META_HARNESS for Meta-Harness import check"
fi
