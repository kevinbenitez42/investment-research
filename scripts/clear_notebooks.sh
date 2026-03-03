#!/usr/bin/env bash

set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Python was not found on PATH." >&2
  exit 1
fi

export PYTHONUTF8=1

stripped_count=0
unchanged_count=0
error_count=0

while IFS= read -r -d '' notebook_path; do
  result="$("$PYTHON_BIN" - "$notebook_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    notebook = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"error: {path} ({exc})")
    raise SystemExit(0)

changed = False

for cell in notebook.get("cells", []):
    if cell.get("cell_type") != "code":
        continue

    if cell.get("outputs"):
        cell["outputs"] = []
        changed = True

    if cell.get("execution_count") is not None:
        cell["execution_count"] = None
        changed = True

if changed:
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"stripped: {path}")
else:
    print(f"unchanged: {path}")
PY
)"

  printf '%s\n' "$result"

  case "$result" in
    stripped:*)
      stripped_count=$((stripped_count + 1))
      ;;
    unchanged:*)
      unchanged_count=$((unchanged_count + 1))
      ;;
    error:*)
      error_count=$((error_count + 1))
      ;;
  esac
done < <(find . -type f -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' -print0)

printf 'summary: stripped=%s unchanged=%s errors=%s\n' \
  "$stripped_count" \
  "$unchanged_count" \
  "$error_count"
