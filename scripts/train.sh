#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PWD/EasyR1:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" scripts/train.py "$@"
