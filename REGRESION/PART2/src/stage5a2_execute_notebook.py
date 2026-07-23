"""Execute the Stage 5A Notebook from saved artifacts and audit side effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

from stage5a2_recovery_serialization import atomic_json


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def protected_runtime_paths() -> list[Path]:
    paths = list((ROOT / "artifacts/models/deep/core_final").glob("*.joblib"))
    paths += list((ROOT / "artifacts/models/deep/core_final/components").glob("*.joblib"))
    paths += list((ROOT / "artifacts/checkpoints/stage5/deep_core/full_train/recovery2").glob("*.json"))
    paths += [ROOT / "artifacts/results/experiment_results.csv"]
    return sorted(paths)


def snapshot() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in protected_runtime_paths() if path.exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("complete", "cache_only", "final_refresh"), required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    forbidden = {
        "model_fit": ".fit(" in code or ".fit (" in code,
        "fit_transform": "fit_transform(" in code,
        "partial_fit": "partial_fit(" in code,
        "training_worker": "--parent-fit" in code or "--fit" in code,
        "completion_writer": "stage5a2_complete_artifacts" in code,
    }
    if any(forbidden.values()):
        raise RuntimeError(f"Notebook contains a forbidden heavy-work call: {forbidden}")
    before = snapshot()
    registry_before = pd.read_csv(ROOT / "artifacts/results/experiment_results.csv")
    old_mode = os.environ.get("STAGE5A2_PRESENTATION_MODE")
    os.environ["STAGE5A2_PRESENTATION_MODE"] = args.mode
    try:
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        executed = client.execute()
    finally:
        if old_mode is None:
            os.environ.pop("STAGE5A2_PRESENTATION_MODE", None)
        else:
            os.environ["STAGE5A2_PRESENTATION_MODE"] = old_mode
    backup = ROOT / f"artifacts/backups/REGRESSION_PART5_DEEP_TABULAR_MODELS.stage5a2_{args.mode}_attempt{args.attempt}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.ipynb"
    backup.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, backup)
    error_outputs = [output for cell in executed.cells if cell.cell_type == "code"
                     for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    empty_outputs = [index for index, cell in enumerate(executed.cells)
                     if cell.cell_type == "code" and not cell.get("outputs")]
    after = snapshot()
    registry_after = pd.read_csv(ROOT / "artifacts/results/experiment_results.csv")
    headings = [cell.source.splitlines()[0] for cell in executed.cells
                if cell.cell_type == "markdown" and cell.source.startswith("### ")]
    checks = {
        "execution_completed": True,
        "zero_error_outputs": not error_outputs,
        "all_code_cells_have_outputs": not empty_outputs,
        "protected_runtime_artifacts_unchanged": before == after,
        "registry_row_count_unchanged": len(registry_before) == len(registry_after),
        "registry_ids_unique": bool(registry_after["experiment_id"].is_unique),
        "registry_not_duplicated": registry_before["experiment_id"].tolist() == registry_after["experiment_id"].tolist(),
        "section_headings_unique": len(headings) == len(set(headings)),
        "sections_25_to_48_present": all(any(line.startswith(f"### {number}.") for line in headings) for number in range(25, 49)),
        "zero_model_training_calls": not forbidden["model_fit"] and not forbidden["training_worker"],
        "zero_preprocessing_fit_calls": not forbidden["fit_transform"] and not forbidden["partial_fit"],
        "attempt_within_limit": args.attempt <= 3,
    }
    report = {
        "stage_id": "stage5a2", "mode": args.mode, "attempt": args.attempt,
        "notebook_backup_path": str(backup.relative_to(ROOT)), "notebook_backup_sha256": sha256(backup),
        "cell_count": len(executed.cells), "code_cell_count": len(code_cells),
        "code_cells_with_outputs": len(code_cells) - len(empty_outputs),
        "empty_output_cell_indices": empty_outputs, "error_output_count": len(error_outputs),
        "runtime_seconds": time.perf_counter() - started,
        "forbidden_call_scan": forbidden, "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    report_path = ROOT / f"artifacts/reports/stage5a2_notebook_run{args.attempt}_{args.mode}.json"
    atomic_json(report, report_path)
    if report["status"] != "PASS":
        raise RuntimeError(f"Notebook execution audit failed: {report}")
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(executed, temporary)
    os.replace(temporary, NOTEBOOK)
    report["promoted_notebook_sha256"] = sha256(NOTEBOOK)
    atomic_json(report, report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
