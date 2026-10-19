#!/usr/bin/env bash
# Small, side-effect-free helpers shared by our experiment launchers.

count_training_iterations() {
  # grep prints 0 AND returns 1 for no matches. Do not append a second zero:
  # callers use this result in integer comparisons while waiting for training.
  local count
  if [[ -r "$1" ]]; then
    count=$(tr '\r' '\n' < "$1" | grep -c "Learning iteration") || true
  fi
  printf '%s\n' "${count:-0}"
}

select_checkpoint() {
  # Expand one glob safely, then choose by mtime (newest) or lexical order
  # (first), preserving each launcher's original selection policy.
  "$PYTHON_BIN" - "$1" "${2:-newest}" <<'PY'
import glob
import os
import sys

paths = [path for path in glob.glob(sys.argv[1]) if os.path.isfile(path)]
if paths:
    selected = min(paths) if sys.argv[2] == "first" else max(paths, key=os.path.getmtime)
    print(selected)
PY
}
