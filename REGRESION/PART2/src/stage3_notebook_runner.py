"""Execute, validate, and atomically promote a cached Stage 3 notebook run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat

from stage3_recovery_utils import PROGRESS_PATH, REPORTS, ROOT, update_progress, utc_now
from stage3_tree_utils import write_json


NOTEBOOK = ROOT / "REGRESSION_PART3_TREE_MODELS.ipynb"
EXECUTION_REPORT = REPORTS / "stage3_notebook_executions.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["run1", "run2", "final"], required=True)
    args = parser.parse_args()
    progress = update_progress()
    attempt = int(progress.get("full_notebook_execution_attempts", 0)) + 1
    if attempt > 3:
        raise RuntimeError("The three-attempt Stage 3 full-notebook budget is exhausted.")
    update_progress(
        full_notebook_execution_attempts=attempt, status="notebook_running",
        current_model="cached_notebook", sensitive_mode=None, fold=None,
        start_time_utc=utc_now(), elapsed_seconds=0,
        next_action=f"execute cached notebook {args.label}",
    )
    output = ROOT / f"REGRESSION_PART3_TREE_MODELS.{args.label}.ipynb"
    log_path = REPORTS / f"stage3_notebook_{args.label}.log"
    environment = os.environ.copy(); environment["STAGE3_EXECUTION_LABEL"] = args.label
    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        NOTEBOOK.name, "--output", output.name, "--ExecutePreprocessor.timeout=1800",
    ]
    started_at = utc_now(); start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            text=True, timeout=2400,
        )
    duration = time.perf_counter() - start
    entries = json.loads(EXECUTION_REPORT.read_text(encoding="utf-8"))["executions"] if EXECUTION_REPORT.exists() else []
    record = {
        "label": args.label, "attempt": attempt, "started_at_utc": started_at,
        "ended_at_utc": utc_now(), "duration_seconds": duration,
        "returncode": process.returncode, "log_path": str(log_path.relative_to(ROOT)),
        "promoted": False,
    }
    if process.returncode == 0 and output.exists():
        notebook = nbformat.read(output, as_version=4)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        executed = sum(cell.execution_count is not None for cell in code)
        errors = sum(out.output_type == "error" for cell in code for out in cell.get("outputs", []))
        cells_with_outputs = sum(bool(cell.get("outputs")) for cell in code)
        record.update({
            "code_cells": len(code), "executed_code_cells": executed,
            "cells_with_outputs": cells_with_outputs, "error_outputs": errors,
        })
        if executed != len(code) or errors:
            record["validation_error"] = "Execution counts or error outputs failed validation."
        else:
            backup = ROOT / "artifacts/backups" / f"REGRESSION_PART3_TREE_MODELS_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
            shutil.copy2(output, backup)
            os.replace(output, NOTEBOOK)
            record["promoted"] = True
            record["backup_path"] = str(backup.relative_to(ROOT))
    entries.append(record)
    write_json(EXECUTION_REPORT, {"executions": entries})
    if not record["promoted"]:
        update_progress(status="notebook_failed", elapsed_seconds=round(duration, 2), last_completed_artifact=None)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Notebook {args.label} failed or was not promotable.\n{tail}")
    update_progress(
        status="notebook_success", elapsed_seconds=round(duration, 2),
        last_completed_artifact=str(NOTEBOOK.relative_to(ROOT)),
        next_action="run second cached notebook" if args.label == "run1" else "perform independent review",
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
