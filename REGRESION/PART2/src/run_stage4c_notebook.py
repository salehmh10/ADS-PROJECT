"""Execute the Stage 4C notebook once fully and once in cache-only mode."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_CATBOOST.ipynb"
REPORT = ROOT / "artifacts/reports/stage4c_notebook_executions.json"
AUDIT_REPORT = ROOT / "artifacts/reports/stage4c_notebook_output_audit.json"
MAX_ATTEMPTS = 3


def _digest(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _heavy_files() -> list[Path]:
    p = c4.paths(ROOT)
    files = []
    for key in ("models", "candidate_models", "predictions", "checkpoints"):
        files.extend(path for path in p[key].rglob("*") if path.is_file())
    return sorted(files, key=lambda value: str(value).lower())


def _snapshot() -> dict:
    files = _heavy_files()
    return {
        str(path.relative_to(ROOT)): {"sha256": s4.sha256_file(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    }


def _audit(notebook) -> dict:
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.metadata.get("stage4c_owned")]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ") and cell.metadata.get("stage4c_owned")]
    checks = {
        "twenty_five_sections": len(headings) == 25 and len(set(headings)) == 25,
        "twenty_five_code_cells": len(code_cells) == 25,
        "all_code_cells_executed": all(cell.get("execution_count") is not None for cell in code_cells),
        "all_code_cells_have_outputs": all(bool(cell.get("outputs")) for cell in code_cells),
        "zero_error_outputs": len(errors) == 0,
        "no_fit_call_in_notebook_code": all(".fit(" not in cell.source and ".fit (" not in cell.source for cell in code_cells),
    }
    return {
        "checks": checks,
        "owned_sections": len(headings),
        "owned_code_cells": len(code_cells),
        "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
        "cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
        "error_outputs": len(errors),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _save_notebook(notebook, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.ipynb")
    nbformat.write(notebook, temporary)
    os.replace(temporary, path)


def run() -> dict:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(NOTEBOOK)
    history = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else {"attempts": []}
    successes = [item for item in history["attempts"] if item.get("status") == "success"]
    while len(history["attempts"]) < MAX_ATTEMPTS and len(successes) < 2:
        attempt_number = len(history["attempts"]) + 1
        mode = "complete" if not successes else "cache_only"
        before = _snapshot()
        registry_before = s4.sha256_file(c4.paths(ROOT)["registry"])
        implementation_digest = _digest([NOTEBOOK, ROOT / "stage4_catboost_utils.py", ROOT / "build_stage4c_notebook.py"])
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        previous = os.environ.get("STAGE4C_CACHE_ONLY")
        os.environ["STAGE4C_CACHE_ONLY"] = "1" if mode == "cache_only" else "0"
        started = time.perf_counter()
        try:
            client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
            client.execute()
            audit = _audit(notebook)
            if audit["status"] != "PASS":
                raise AssertionError(f"Notebook output audit failed: {audit}")
            after = _snapshot()
            if before != after:
                raise AssertionError("A heavy model, prediction, or checkpoint changed during notebook execution.")
            if registry_before != s4.sha256_file(c4.paths(ROOT)["registry"]):
                raise AssertionError("The Registry changed during notebook execution.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = ROOT / "artifacts/backups" / f"REGRESSION_PART4_CATBOOST_{mode}_run{attempt_number}_{timestamp}.ipynb"
            _save_notebook(notebook, backup)
            _save_notebook(notebook, NOTEBOOK)
            record = {
                "attempt": attempt_number,
                "run_mode": mode,
                "completed_at_utc": s4.utc_now(),
                "status": "success",
                "wall_seconds": time.perf_counter() - started,
                "implementation_digest": implementation_digest,
                "cache_only": mode == "cache_only",
                "model_fit_calls": 0,
                "heavy_artifacts_unchanged": True,
                "registry_unchanged": True,
                "backup": str(backup.relative_to(ROOT)),
                "output_audit": audit,
            }
        except Exception as exc:
            record = {
                "attempt": attempt_number,
                "run_mode": mode,
                "completed_at_utc": s4.utc_now(),
                "status": "failed",
                "wall_seconds": time.perf_counter() - started,
                "implementation_digest": implementation_digest,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if previous is None:
                os.environ.pop("STAGE4C_CACHE_ONLY", None)
            else:
                os.environ["STAGE4C_CACHE_ONLY"] = previous
        history["attempts"].append(record)
        successes = [item for item in history["attempts"] if item.get("status") == "success"]
        history.update({
            "stage": c4.STAGE_ID,
            "maximum_attempts": MAX_ATTEMPTS,
            "required_complete_runs": 1,
            "required_cache_only_runs": 1,
            "successful_runs": len(successes),
            "status": "PASS" if len(successes) >= 2 and successes[0]["run_mode"] == "complete" and successes[1]["run_mode"] == "cache_only" else "IN_PROGRESS",
        })
        s4.atomic_write_json(REPORT, history)
    successes = [item for item in history["attempts"] if item.get("status") == "success"]
    valid = len(successes) >= 2 and successes[0]["run_mode"] == "complete" and successes[1]["run_mode"] == "cache_only"
    history["status"] = "PASS" if valid else "FAIL"
    audit_report = {
        "stage": c4.STAGE_ID,
        "complete_runs": sum(item.get("status") == "success" and item.get("run_mode") == "complete" for item in history["attempts"]),
        "cache_only_runs": sum(item.get("status") == "success" and item.get("run_mode") == "cache_only" for item in history["attempts"]),
        "no_model_retraining": all(item.get("model_fit_calls") == 0 for item in successes),
        "heavy_artifacts_unchanged": all(item.get("heavy_artifacts_unchanged") is True for item in successes),
        "registry_unchanged": all(item.get("registry_unchanged") is True for item in successes),
        "last_output_audit": successes[-1]["output_audit"] if successes else {},
        "status": "PASS" if valid else "FAIL",
    }
    s4.atomic_write_json(AUDIT_REPORT, audit_report)
    s4.atomic_write_json(REPORT, history)
    if not valid:
        raise RuntimeError(f"Stage 4C notebook execution failed: {history}")
    return history


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
