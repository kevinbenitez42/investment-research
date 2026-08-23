#!/usr/bin/env python3
"""Hide or show code inputs in Jupyter notebooks.

This edits each code cell's metadata:

    metadata.jupyter.source_hidden = true

Outputs are left visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_notebooks(paths: list[Path]):
    for path in paths:
        if path.is_file() and path.suffix == ".ipynb":
            yield path
            continue

        if path.is_dir():
            for notebook_path in path.rglob("*.ipynb"):
                if ".ipynb_checkpoints" not in notebook_path.parts:
                    yield notebook_path


def _set_code_input_visibility(path: Path, *, hidden: bool, dry_run: bool) -> tuple[int, bool]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    code_cell_count = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        code_cell_count += 1
        metadata = cell.setdefault("metadata", {})
        jupyter_metadata = metadata.setdefault("jupyter", {})

        if hidden:
            if jupyter_metadata.get("source_hidden") is not True:
                jupyter_metadata["source_hidden"] = True
                changed = True
        elif "source_hidden" in jupyter_metadata:
            del jupyter_metadata["source_hidden"]
            changed = True
            if not jupyter_metadata:
                del metadata["jupyter"]
            if not metadata:
                cell["metadata"] = {}

    if changed and not dry_run:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    return code_cell_count, changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Hide code inputs in one or more Jupyter notebooks.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Notebook files or directories to update recursively.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Remove source_hidden metadata instead of adding it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files.",
    )
    args = parser.parse_args()

    notebooks = sorted(set(_iter_notebooks(args.paths)))
    if not notebooks:
        print("No notebooks found.")
        return 1

    total_code_cells = 0
    changed_notebooks = 0
    action = "show" if args.show else "hide"

    for notebook_path in notebooks:
        code_cell_count, changed = _set_code_input_visibility(
            notebook_path,
            hidden=not args.show,
            dry_run=args.dry_run,
        )
        total_code_cells += code_cell_count
        changed_notebooks += int(changed)
        status = "would update" if args.dry_run and changed else "updated" if changed else "unchanged"
        print(f"{status}: {notebook_path} ({code_cell_count} code cells)")

    print(
        f"{action} inputs complete: {changed_notebooks}/{len(notebooks)} notebooks changed, "
        f"{total_code_cells} code cells scanned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
