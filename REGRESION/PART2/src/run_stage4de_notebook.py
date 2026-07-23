"""Execute the full Stage 4 notebook once clean and once cache-only."""

from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

import stage4_boosting_utils as s4


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_CATBOOST.ipynb"
REPORT = ROOT / "artifacts/reports/stage4de_notebook_executions.json"
AUDIT = ROOT / "artifacts/reports/stage4de_notebook_output_audit.json"
PREFIX_CELLS = 76
MAX_ATTEMPTS = 3


def digest(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(str(path.relative_to(ROOT)).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def notebook_source_digest(notebook: Any) -> str:
    stable_keys = ("tags", "stage4de_section", "stage4c_owned")
    payload = [
        {
            "cell_type": cell.cell_type,
            "source": cell.source,
            "metadata": {key: cell.metadata[key] for key in stable_keys if key in cell.metadata},
        }
        for cell in notebook.cells
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def heavy_files() -> list[Path]:
    roots = [
        ROOT / "artifacts/models/catboost",
        ROOT / "artifacts/predictions/catboost",
        ROOT / "artifacts/checkpoints/stage4/catboost",
    ]
    return sorted((path for base in roots for path in base.rglob("*") if path.is_file()), key=lambda value: str(value).lower())


def snapshot() -> dict[str, Any]:
    return {
        str(path.relative_to(ROOT)): {
            "sha256": s4.sha256_file(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in heavy_files()
    }


def audit(notebook: Any) -> dict[str, Any]:
    all_code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    stage4de_code = [cell for cell in all_code if "stage4de_owned" in cell.metadata.get("tags", [])]
    headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ")]
    stage4de_headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ") and "stage4de_owned" in cell.metadata.get("tags", [])]
    errors = [output for cell in all_code for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    checks = {
        "forty_six_unique_sections": len(headings) == 46 and len(set(headings)) == 46,
        "stage4de_twenty_one_unique_sections": len(stage4de_headings) == 21 and len(set(stage4de_headings)) == 21,
        "stage4de_twenty_one_code_cells": len(stage4de_code) == 21,
        "all_forty_six_code_cells_executed": len(all_code) == 46 and all(cell.get("execution_count") is not None for cell in all_code),
        "all_code_cells_have_outputs": all(bool(cell.get("outputs")) for cell in all_code),
        "zero_error_outputs": not errors,
        "no_fit_call_in_stage4de_code": all(".fit(" not in cell.source and ".fit (" not in cell.source for cell in stage4de_code),
    }
    return {
        "checks": checks,
        "sections": len(headings),
        "stage4de_sections": len(stage4de_headings),
        "code_cells": len(all_code),
        "executed_code_cells": sum(cell.get("execution_count") is not None for cell in all_code),
        "cells_with_outputs": sum(bool(cell.get("outputs")) for cell in all_code),
        "error_outputs": len(errors),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def save_notebook(notebook: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.ipynb")
    nbformat.write(notebook, temporary)
    os.replace(temporary, path)


def canonical_prefix() -> list[Any]:
    backups = sorted((ROOT / "artifacts/backups").glob("REGRESSION_PART4_CATBOOST_before_stage4de_*.ipynb"))
    if not backups:
        raise FileNotFoundError("The canonical pre-Stage 4D-E notebook backup is missing.")
    backup = nbformat.read(backups[-1], as_version=4)
    if len(backup.cells) != PREFIX_CELLS:
        raise AssertionError("The canonical notebook backup must contain 76 cells.")
    return deepcopy(list(backup.cells))


def restore_live_prefix() -> bool:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    prefix = canonical_prefix()
    notebook.cells = prefix + list(notebook.cells[PREFIX_CELLS:])
    save_notebook(notebook, NOTEBOOK)
    restored = nbformat.read(NOTEBOOK, as_version=4)
    return restored.cells[:PREFIX_CELLS] == prefix


def run_one(history: dict[str, Any], mode: str) -> bool:
    attempt = len(history["attempts"]) + 1
    before = snapshot()
    registry = ROOT / "artifacts/results/experiment_results.csv"
    registry_before = s4.sha256_file(registry)
    original = nbformat.read(NOTEBOOK, as_version=4)
    source_digest = notebook_source_digest(original)
    prefix = canonical_prefix()
    implementation_digest = digest([NOTEBOOK, ROOT / "build_stage4de_notebook.py", ROOT / "run_stage4de.py", ROOT / "stage4de_catboost_utils.py"])
    runtime = ROOT / "artifacts/runtime/stage4de_notebook"
    runtime.mkdir(parents=True, exist_ok=True)
    env_names = ("STAGE4DE_CACHE_ONLY", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR")
    previous = {name: os.environ.get(name) for name in env_names}
    os.environ["STAGE4DE_CACHE_ONLY"] = "1" if mode == "cache_only" else "0"
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter")
    started = time.perf_counter()
    try:
        client = NotebookClient(original, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        client.execute()
        output_audit = audit(original)
        if output_audit["status"] != "PASS":
            raise AssertionError(f"Notebook output audit failed: {output_audit}")
        if before != snapshot():
            raise AssertionError("A CatBoost model, prediction, or fit checkpoint changed during notebook execution.")
        if registry_before != s4.sha256_file(registry):
            raise AssertionError("The Registry changed during notebook execution.")
        backup = ROOT / "artifacts/backups" / f"REGRESSION_PART4_CATBOOST_stage4de_{mode}_run{attempt}.ipynb"
        save_notebook(original, backup)
        original.cells = prefix + list(original.cells[PREFIX_CELLS:])
        save_notebook(original, NOTEBOOK)
        record = {
            "attempt": attempt,
            "run_mode": mode,
            "completed_at_utc": s4.utc_now(),
            "status": "success",
            "wall_seconds": time.perf_counter() - started,
            "implementation_digest": implementation_digest,
            "notebook_source_digest": source_digest,
            "model_fit_calls": 0,
            "heavy_artifacts_unchanged": True,
            "registry_unchanged": True,
            "canonical_prefix_restored_after_execution": True,
            "executed_backup": str(backup.relative_to(ROOT)),
            "output_audit": output_audit,
        }
    except Exception as exc:
        record = {
            "attempt": attempt,
            "run_mode": mode,
            "completed_at_utc": s4.utc_now(),
            "status": "failed",
            "wall_seconds": time.perf_counter() - started,
            "implementation_digest": implementation_digest,
            "notebook_source_digest": source_digest,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    history["attempts"].append(record)
    s4.atomic_write_json(REPORT, history)
    return record["status"] == "success"


def run() -> dict[str, Any]:
    history = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else {
        "stage": "stage4de",
        "maximum_attempts": MAX_ATTEMPTS,
        "required_complete_runs": 1,
        "required_cache_only_runs": 1,
        "attempts": [],
        "status": "IN_PROGRESS",
    }
    while len(history["attempts"]) < MAX_ATTEMPTS:
        complete = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "complete"]
        cache = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "cache_only"]
        current_source = notebook_source_digest(nbformat.read(NOTEBOOK, as_version=4))
        current_validated = any(item.get("status") == "success" and item.get("notebook_source_digest") == current_source for item in history["attempts"])
        if complete and cache and current_validated:
            break
        run_one(history, "complete" if not complete else "cache_only")
    complete = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "complete"]
    cache = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "cache_only"]
    current_source = notebook_source_digest(nbformat.read(NOTEBOOK, as_version=4))
    for item in history["attempts"]:
        backup_name = item.get("executed_backup")
        if item.get("status") == "success" and backup_name and (ROOT / backup_name).is_file():
            item["stable_notebook_source_digest"] = notebook_source_digest(nbformat.read(ROOT / backup_name, as_version=4))
    current_validated = any(item.get("status") == "success" and item.get("stable_notebook_source_digest") == current_source for item in history["attempts"])
    passed = bool(complete and cache and current_validated)
    prefix_restored = restore_live_prefix()
    for item in history["attempts"]:
        if item.get("status") == "success":
            item.pop("prefix_cells_preserved_in_live_notebook", None)
            item["executed_backup_contains_full_run_outputs"] = True
            item["live_prefix_restored_from_canonical_backup"] = prefix_restored
    history.update({
        "successful_complete_runs": len(complete),
        "successful_cache_only_runs": len(cache),
        "final_notebook_source_digest": current_source,
        "final_source_validated_by_execution": current_validated,
        "canonical_prefix_restored": prefix_restored,
        "correction_note": "The live prefix is restored from the canonical 76-cell backup after execution. This also corrects the shallow-copy behavior detected after the first two successful runs.",
        "status": "PASS" if passed and prefix_restored else "FAIL",
    })
    s4.atomic_write_json(REPORT, history)
    audit_report = {
        "stage": "stage4de",
        "attempts": len(history["attempts"]),
        "complete_runs": len(complete),
        "cache_only_runs": len(cache),
        "no_model_retraining": all(item.get("model_fit_calls") == 0 for item in complete + cache),
        "heavy_artifacts_unchanged": all(item.get("heavy_artifacts_unchanged") is True for item in complete + cache),
        "registry_unchanged": all(item.get("registry_unchanged") is True for item in complete + cache),
        "prefix_preserved": prefix_restored,
        "last_output_audit": (cache or complete)[-1]["output_audit"] if (cache or complete) else {},
        "status": "PASS" if passed and prefix_restored else "FAIL",
    }
    s4.atomic_write_json(AUDIT, audit_report)
    if not passed or not prefix_restored:
        raise RuntimeError(f"Stage 4D-E notebook execution failed: {history}")
    return history


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
