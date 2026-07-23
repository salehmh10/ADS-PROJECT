"""Run the authorized Stage 4A recovery executions from clean kernels."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import stage4_boosting_utils as s4


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb"
REPORT = ROOT / "artifacts/reports/stage4a_notebook_runner.json"
OUTPUT_AUDIT = ROOT / "artifacts/reports/stage4a_notebook_output_audit.json"
BACKUPS = ROOT / "artifacts/backups"
RECOVERY_ID = s4.STAGE4A_RECOVERY_ID
MAXIMUM_RECOVERY_ATTEMPTS = 4
REQUIRED_RECOVERY_SUCCESSES = 2
MAXIMUM_TOTAL_SECONDS = 90 * 60


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_notebook(notebook, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, path)


def load_history() -> dict:
    if REPORT.is_file():
        history = json.loads(REPORT.read_text(encoding="utf-8"))
    else:
        history = {}
    if "historical" not in history:
        old_attempts = list(history.get("attempts", []))
        history["historical"] = {
            "maximum_attempts": int(history.get("maximum_attempts", 3)),
            "required_successful_runs": int(history.get("required_successful_runs", 2)),
            "attempts": old_attempts,
            "successful_runs": sum(item.get("status") == "success" for item in old_attempts),
            "preserved": True,
        }
    recovery = history.setdefault("recovery", {})
    if recovery.get("recovery_id") not in (None, RECOVERY_ID):
        raise RuntimeError("A different Stage 4A recovery session already exists.")
    recovery.update({
        "recovery_id": RECOVERY_ID,
        "authorized_at": "2026-07-14",
        "maximum_attempts": MAXIMUM_RECOVERY_ATTEMPTS,
        "required_successful_runs": REQUIRED_RECOVERY_SUCCESSES,
    })
    recovery.setdefault("attempts", [])
    return history


history = load_history()
BACKUPS.mkdir(parents=True, exist_ok=True)
recovery = history["recovery"]
started_recovery = time.monotonic()

while (
    len(recovery["attempts"]) < MAXIMUM_RECOVERY_ATTEMPTS
    and sum(item.get("status") == "success" for item in recovery["attempts"]) < REQUIRED_RECOVERY_SUCCESSES
    and time.monotonic() - started_recovery < MAXIMUM_TOTAL_SECONDS
):
    attempt_number = len(recovery["attempts"]) + 1
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    implementation_digest = s4.stage4a_implementation_digest(ROOT, NOTEBOOK)
    previous_recovery_id = os.environ.get("STAGE4A_RECOVERY_ID")
    previous_digest = os.environ.get("STAGE4A_IMPLEMENTATION_DIGEST")
    os.environ["STAGE4A_RECOVERY_ID"] = RECOVERY_ID
    os.environ["STAGE4A_IMPLEMENTATION_DIGEST"] = implementation_digest
    try:
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
        audit = s4.audit_stage4a_notebook_outputs(notebook)
        if audit["status"] != "PASS":
            raise AssertionError(f"Notebook output audit failed: {audit}")
        execution_history = json.loads(
            (ROOT / "artifacts/reports/stage4a_notebook_executions.json").read_text(encoding="utf-8")
        )
        matching_records = [
            item for item in execution_history.get("runs", [])
            if item.get("status") == "success"
            and item.get("recovery_id") == RECOVERY_ID
            and item.get("implementation_digest") == implementation_digest
        ]
        if not matching_records:
            raise AssertionError("The notebook did not record a matching recovery success.")
        record = matching_records[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUPS / f"REGRESSION_PART4_BOOSTING_FOUNDATION_recovery_run{attempt_number}_{timestamp}.ipynb"
        temporary_notebook = NOTEBOOK.with_suffix(f".recovery-{os.getpid()}.ipynb")
        nbformat.write(notebook, temporary_notebook)
        validated = nbformat.read(temporary_notebook, as_version=4)
        validated_audit = s4.audit_stage4a_notebook_outputs(validated)
        if validated_audit["status"] != "PASS":
            raise AssertionError("The saved temporary notebook failed its output audit.")
        save_notebook(validated, backup)
        os.replace(temporary_notebook, NOTEBOOK)
        attempt = {
            "attempt": attempt_number,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "wall_seconds": time.monotonic() - started,
            "implementation_digest": implementation_digest,
            "snapshot_digest": record["snapshot_digest"],
            "code_cells": audit["total_code_cells"],
            "executed_code_cells": audit["executed_code_cells"],
            "cells_with_outputs": audit["cells_with_outputs"],
            "error_outputs": audit["error_output_count"],
            "cache_reused": True,
            "backup": str(backup.relative_to(ROOT)),
            "output_audit": audit,
        }
        recovery["attempts"].append(attempt)
    except Exception as exc:
        recovery["attempts"].append({
            "attempt": attempt_number,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "wall_seconds": time.monotonic() - started,
            "implementation_digest": implementation_digest,
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        if previous_recovery_id is None:
            os.environ.pop("STAGE4A_RECOVERY_ID", None)
        else:
            os.environ["STAGE4A_RECOVERY_ID"] = previous_recovery_id
        if previous_digest is None:
            os.environ.pop("STAGE4A_IMPLEMENTATION_DIGEST", None)
        else:
            os.environ["STAGE4A_IMPLEMENTATION_DIGEST"] = previous_digest

    successful = [item for item in recovery["attempts"] if item.get("status") == "success"]
    recovery["successful_runs"] = len(successful)
    recovery["status"] = "PASS" if len(successful) >= REQUIRED_RECOVERY_SUCCESSES else "IN_PROGRESS"
    audit_report = {
        "stage": "stage4a",
        "recovery_id": RECOVERY_ID,
        "status": "PASS" if len(successful) >= REQUIRED_RECOVERY_SUCCESSES and all(item["output_audit"]["status"] == "PASS" for item in successful[-2:]) else "IN_PROGRESS",
        "successful_recovery_runs": len(successful),
        "required_successful_runs": REQUIRED_RECOVERY_SUCCESSES,
        "runs": [
            {
                "attempt": item["attempt"],
                "implementation_digest": item["implementation_digest"],
                "wall_seconds": item["wall_seconds"],
                "code_cells": item["code_cells"],
                "executed_code_cells": item["executed_code_cells"],
                "cells_with_outputs": item["cells_with_outputs"],
                "error_outputs": item["error_outputs"],
                "cache_reused": item["cache_reused"],
                "key_section_outputs": item["output_audit"]["key_section_outputs"],
            }
            for item in successful
        ],
    }
    atomic_json(OUTPUT_AUDIT, audit_report)
    atomic_json(REPORT, history)

successful = [item for item in recovery["attempts"] if item.get("status") == "success"]
recovery["successful_runs"] = len(successful)
recovery["status"] = "PASS" if len(successful) >= REQUIRED_RECOVERY_SUCCESSES else "FAIL"
recovery["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
history["status"] = recovery["status"]
atomic_json(REPORT, history)
print(json.dumps(history, indent=2))
if recovery["status"] != "PASS":
    raise SystemExit(1)
