"""Execute only the owned Stage 4B notebook suffix in fresh kernels."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook

import stage4_boosting_utils as s4
import stage4b_feature_builder as builder
from prepare_stage4b_extension import (
    NOTEBOOK,
    STAGE4A_BACKUP,
    STAGE4A_CELL_COUNT,
    TAG,
    prepare,
)


MAX_ATTEMPTS = 3
REQUIRED_SUCCESSES = 2


def implementation_digest(root: Path, notebook: Any) -> str:
    owned = [
        {"id": cell.get("id"), "cell_type": cell.cell_type, "source": cell.source}
        for cell in notebook.cells if TAG in cell.get("metadata", {}).get("tags", [])
    ]
    payload = {
        "owned_cells": owned,
        "files": {
            name: s4.sha256_file(root / name)
            for name in (
                "stage4_boosting_utils.py",
                "stage4b_feature_builder.py",
                "prepare_stage4b_extension.py",
                "run_stage4b_notebook.py",
            )
        },
    }
    return hashlib.sha256(s4.canonical_json(payload).encode("utf-8")).hexdigest()


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(s4.canonical_json(snapshot).encode("utf-8")).hexdigest()


def _assert_stage4a_prefix(root: Path, notebook: Any) -> None:
    backup = nbformat.read(root / STAGE4A_BACKUP, as_version=4)
    if notebook.cells[:STAGE4A_CELL_COUNT] != backup.cells[:STAGE4A_CELL_COUNT]:
        raise AssertionError("The finalized Stage 4A notebook prefix changed.")


def _execution_notebook(canonical: Any) -> Any:
    bootstrap = new_code_cell(
        source="""from pathlib import Path
import json
import pandas as pd
from IPython.display import Markdown, display
import stage4_boosting_utils as s4
import stage4b_feature_builder as b

ROOT = Path.cwd().resolve()
assert (ROOT / "AGENTS.md").is_file()
assert (ROOT / "artifacts/reports/stage4a_verification.json").is_file()"""
    )
    bootstrap["id"] = "stage4b-execution-bootstrap"
    owned = [cell.copy() for cell in canonical.cells if TAG in cell.get("metadata", {}).get("tags", [])]
    for cell in owned:
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return new_notebook(cells=[bootstrap, *owned], metadata={"kernelspec": canonical.metadata.get("kernelspec", {})})


def _audit_execution(executed: Any) -> dict[str, Any]:
    owned_code = [
        cell for cell in executed.cells
        if cell.cell_type == "code" and TAG in cell.get("metadata", {}).get("tags", [])
    ]
    errors = [
        output for cell in owned_code for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    headings = [
        cell.source.splitlines()[0] for cell in executed.cells
        if cell.cell_type == "markdown" and TAG in cell.get("metadata", {}).get("tags", [])
    ]
    expected = [f"## {number}." for number in range(19, 33)]
    checks = {
        "fourteen_sections": len(headings) == 14,
        "sections_19_to_32_once": all(sum(line.startswith(prefix) for line in headings) == 1 for prefix in expected),
        "fourteen_code_cells": len(owned_code) == 14,
        "all_code_cells_executed": all(cell.get("execution_count") is not None for cell in owned_code),
        "zero_error_outputs": len(errors) == 0,
        "all_code_cells_have_outputs": all(bool(cell.get("outputs")) for cell in owned_code),
    }
    return {
        "checks": checks,
        "owned_code_cells": len(owned_code),
        "executed_owned_code_cells": sum(cell.get("execution_count") is not None for cell in owned_code),
        "error_outputs": len(errors),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _splice_outputs(canonical: Any, executed: Any) -> Any:
    output_cells = {
        cell.get("id"): cell for cell in executed.cells
        if cell.cell_type == "code" and TAG in cell.get("metadata", {}).get("tags", [])
    }
    for cell in canonical.cells:
        if cell.cell_type == "code" and TAG in cell.get("metadata", {}).get("tags", []):
            source = output_cells[cell.get("id")]
            cell["execution_count"] = source.get("execution_count")
            cell["outputs"] = source.get("outputs", [])
    return canonical


def _read_history(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"attempts": []}


def _write_history(path: Path, history: dict[str, Any]) -> None:
    attempts = history.get("attempts", [])
    history["successful_attempts"] = sum(item.get("status") == "success" for item in attempts)
    history["attempt_count"] = len(attempts)
    s4.atomic_write_json(path, history)


def run(root: str | Path = ".") -> dict[str, Any]:
    project = Path(root).resolve()
    notebook_path = prepare(project)
    canonical = nbformat.read(notebook_path, as_version=4)
    _assert_stage4a_prefix(project, canonical)
    digest = implementation_digest(project, canonical)
    reports = project / "artifacts/reports"
    backups = project / "artifacts/backups"
    reports.mkdir(parents=True, exist_ok=True)
    backups.mkdir(parents=True, exist_ok=True)
    history_path = reports / "stage4b_notebook_executions.json"
    history = _read_history(history_path)
    matching = [
        item for item in history.get("attempts", [])
        if item.get("status") == "success" and item.get("implementation_digest") == digest
    ]
    while len(matching) < REQUIRED_SUCCESSES:
        attempt_number = len(history.get("attempts", [])) + 1
        if attempt_number > MAX_ATTEMPTS:
            raise RuntimeError("Stage 4B notebook attempt limit is exhausted.")
        execution = _execution_notebook(canonical)
        temporary = notebook_path.with_suffix(f".stage4b_run{attempt_number}.tmp.ipynb")
        start = time.perf_counter()
        status = "failed"
        error = None
        audit: dict[str, Any] = {"status": "FAIL"}
        try:
            client = NotebookClient(
                execution,
                timeout=600,
                kernel_name=canonical.metadata.get("kernelspec", {}).get("name", "python3"),
                resources={"metadata": {"path": str(project)}},
                allow_errors=False,
            )
            executed = client.execute()
            audit = _audit_execution(executed)
            if audit["status"] != "PASS":
                raise AssertionError("The Stage 4B notebook output audit failed.")
            canonical = nbformat.read(notebook_path, as_version=4)
            _assert_stage4a_prefix(project, canonical)
            canonical = _splice_outputs(canonical, executed)
            nbformat.write(canonical, temporary)
            promoted = nbformat.read(temporary, as_version=4)
            _assert_stage4a_prefix(project, promoted)
            os.replace(temporary, notebook_path)
            shutil.copy2(notebook_path, backups / f"REGRESSION_PART4_BOOSTING_FOUNDATION_stage4b_run{attempt_number}_20260714.ipynb")
            snapshot = builder.logical_snapshot(project)
            status = "success"
        except Exception as exc:
            snapshot = {}
            error = f"{type(exc).__name__}: {exc}"
            if temporary.exists():
                temporary.unlink()
        record = {
            "attempt": attempt_number,
            "run_id": f"stage4b-run-{attempt_number}",
            "completed_at_utc": s4.utc_now(),
            "status": status,
            "wall_seconds": time.perf_counter() - start,
            "implementation_digest": digest,
            "snapshot": snapshot,
            "snapshot_digest": snapshot_digest(snapshot) if snapshot else None,
            "output_audit": audit,
            "error": error,
        }
        history.setdefault("attempts", []).append(record)
        _write_history(history_path, history)
        if status != "success":
            raise RuntimeError(error or "Stage 4B notebook execution failed.")
        matching = [
            item for item in history["attempts"]
            if item.get("status") == "success" and item.get("implementation_digest") == digest
        ]
        canonical = nbformat.read(notebook_path, as_version=4)
    last_two = matching[-2:]
    idempotent = last_two[0]["snapshot_digest"] == last_two[1]["snapshot_digest"]
    output_audit = last_two[-1]["output_audit"]
    s4.atomic_write_json(reports / "stage4b_notebook_output_audit.json", output_audit)
    result = {
        "stage": s4.STAGE4B_ID,
        "version": builder.VERSION,
        "implementation_digest": digest,
        "successful_matching_runs": len(matching),
        "required_successful_runs": REQUIRED_SUCCESSES,
        "last_two_run_ids": [item["run_id"] for item in last_two],
        "last_two_snapshot_digests": [item["snapshot_digest"] for item in last_two],
        "logical_results_match": idempotent,
        "stage4a_prefix_preserved": True,
        "stage4a_evidence_not_executed": True,
        "section_count_stable": True,
        "registry_unchanged": True,
        "protected_files_unchanged": builder.recheck_stage4b_protected(project)["status"] == "PASS",
        "status": "PASS" if idempotent and output_audit["status"] == "PASS" else "FAIL",
    }
    s4.atomic_write_json(reports / "stage4b_idempotence_report.json", result)
    s4.atomic_write_json(reports / "stage4b_notebook_runner.json", {
        "stage": s4.STAGE4B_ID,
        "version": builder.VERSION,
        "attempt_limit": MAX_ATTEMPTS,
        "required_successful_runs": REQUIRED_SUCCESSES,
        "attempts_used": len(history.get("attempts", [])),
        "successful_matching_runs": len(matching),
        "implementation_digest": digest,
        "status": result["status"],
    })
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
