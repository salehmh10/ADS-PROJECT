"""Run the authorized Stage 4C notebook recovery without repeating model work."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4
import run_stage4c_notebook as base


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_CATBOOST.ipynb"
HISTORICAL = ROOT / "artifacts/reports/stage4c_notebook_executions.json"
REPORT = ROOT / "artifacts/reports/stage4c_notebook_recovery.json"
AUDIT = ROOT / "artifacts/reports/stage4c_notebook_output_audit.json"
RECOVERY_ID = "stage4c-recovery-stage4de-20260714"
MAX_ATTEMPTS = 4


def _history() -> dict:
    prior = json.loads(HISTORICAL.read_text(encoding="utf-8")) if HISTORICAL.is_file() else {"attempts": []}
    if REPORT.is_file():
        value = json.loads(REPORT.read_text(encoding="utf-8"))
        if value.get("recovery_id") != RECOVERY_ID:
            raise RuntimeError("A different Stage 4C recovery record already exists.")
        return value
    return {
        "stage": c4.STAGE_ID,
        "recovery_id": RECOVERY_ID,
        "historical": {
            "attempts": prior.get("attempts", []),
            "attempt_count": len(prior.get("attempts", [])),
            "preserved": True,
        },
        "maximum_new_attempts": MAX_ATTEMPTS,
        "required_complete_runs": 1,
        "required_cache_only_runs": 1,
        "attempts": [],
        "status": "IN_PROGRESS",
    }


def _run_one(history: dict, mode: str) -> bool:
    attempt = len(history["attempts"]) + 1
    before = base._snapshot()
    registry_before = s4.sha256_file(c4.paths(ROOT)["registry"])
    implementation_digest = base._digest([NOTEBOOK, ROOT / "stage4_catboost_utils.py", ROOT / "build_stage4c_notebook.py"])
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    runtime = ROOT / "artifacts/runtime/stage4c_recovery"
    runtime.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in ("STAGE4C_CACHE_ONLY", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR")}
    os.environ["STAGE4C_CACHE_ONLY"] = "1" if mode == "cache_only" else "0"
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter")
    started = time.perf_counter()
    try:
        client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        client.execute()
        audit = base._audit(notebook)
        if audit["status"] != "PASS":
            raise AssertionError(f"Notebook audit failed: {audit}")
        if before != base._snapshot():
            raise AssertionError("A heavy Stage 4C artifact changed during recovery execution.")
        if registry_before != s4.sha256_file(c4.paths(ROOT)["registry"]):
            raise AssertionError("The Registry changed during Stage 4C recovery execution.")
        backup = ROOT / "artifacts/backups" / f"REGRESSION_PART4_CATBOOST_recovery_{mode}_run{attempt}.ipynb"
        base._save_notebook(notebook, backup)
        base._save_notebook(notebook, NOTEBOOK)
        record = {
            "attempt": attempt,
            "run_mode": mode,
            "completed_at_utc": s4.utc_now(),
            "status": "success",
            "wall_seconds": time.perf_counter() - started,
            "implementation_digest": implementation_digest,
            "model_fit_calls": 0,
            "heavy_artifacts_unchanged": True,
            "registry_unchanged": True,
            "backup": str(backup.relative_to(ROOT)),
            "output_audit": audit,
        }
    except Exception as exc:
        record = {
            "attempt": attempt,
            "run_mode": mode,
            "completed_at_utc": s4.utc_now(),
            "status": "failed",
            "wall_seconds": time.perf_counter() - started,
            "implementation_digest": implementation_digest,
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


def run() -> dict:
    history = _history()
    while len(history["attempts"]) < MAX_ATTEMPTS:
        complete = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "complete"]
        cache = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "cache_only"]
        if complete and cache:
            break
        mode = "complete" if not complete else "cache_only"
        if not _run_one(history, mode):
            break
    complete = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "complete"]
    cache = [item for item in history["attempts"] if item.get("status") == "success" and item.get("run_mode") == "cache_only"]
    passed = bool(complete and cache)
    history["successful_complete_runs"] = len(complete)
    history["successful_cache_only_runs"] = len(cache)
    history["status"] = "PASS" if passed else "FAIL"
    s4.atomic_write_json(REPORT, history)
    audit = {
        "stage": c4.STAGE_ID,
        "recovery_id": RECOVERY_ID,
        "historical_attempts_preserved": history["historical"]["attempt_count"],
        "new_attempts": len(history["attempts"]),
        "complete_runs": len(complete),
        "cache_only_runs": len(cache),
        "no_model_retraining": all(item.get("model_fit_calls") == 0 for item in complete + cache),
        "heavy_artifacts_unchanged": all(item.get("heavy_artifacts_unchanged") for item in complete + cache),
        "registry_unchanged": all(item.get("registry_unchanged") for item in complete + cache),
        "last_output_audit": (cache or complete)[-1]["output_audit"] if (cache or complete) else {},
        "status": "PASS" if passed else "FAIL",
    }
    s4.atomic_write_json(AUDIT, audit)
    if not passed:
        raise RuntimeError(f"Stage 4C recovery did not pass: {history}")
    return history


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
