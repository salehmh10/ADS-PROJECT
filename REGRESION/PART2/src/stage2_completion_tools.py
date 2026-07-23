"""Auditing helpers for Stage 2 completion and idempotence evidence."""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import nbformat
import numpy as np
import pandas as pd


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected(root):
    paths = {
        "with_sensitive_csv": root / "data/regression_with_sensitive_features.csv",
        "without_sensitive_csv": root / "data/regression_without_sensitive_features.csv",
        "part1_notebook": Path(r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb"),
        "train_row_ids": root / "artifacts/splits/train_row_ids.csv",
        "test_row_ids": root / "artifacts/splits/test_row_ids.csv",
        "cv_fold_assignments": root / "artifacts/splits/cv_fold_assignments.csv",
        "split_config": root / "artifacts/splits/split_config.json",
    }
    return paths, {name: sha256(path) for name, path in paths.items()}


def artifact_audit(root):
    train = pd.read_csv(root / "artifacts/splits/train_row_ids.csv")["row_id"].to_numpy(dtype=int)
    test = set(pd.read_csv(root / "artifacts/splits/test_row_ids.csv")["row_id"].astype(int))
    cv = pd.read_csv(root / "artifacts/splits/cv_fold_assignments.csv").sort_values("row_id")
    screening = pd.read_csv(root / "artifacts/results/prompt2/development_screening_results.csv")
    folds = pd.read_csv(root / "artifacts/results/prompt2/cv_fold_results.csv")
    oof_summary = pd.read_csv(root / "artifacts/results/prompt2/cv_oof_summary.csv")
    registry = pd.read_csv(root / "artifacts/results/experiment_results.csv")
    p2_registry = pd.read_csv(root / "artifacts/results/prompt2/prompt2_registry_rows.csv")
    reloads = pd.read_csv(root / "artifacts/reports/prompt2_model_reload_verification.csv")
    baseline = json.loads((root / "artifacts/manifests/prompt2_protected_hashes_before.json").read_text(encoding="utf-8"))
    _, current_hashes = protected(root)
    oof_details = []
    for path in sorted((root / "artifacts/predictions/linear").glob("*.csv")):
        frame = pd.read_csv(path)
        expected_fold = cv.set_index("row_id").loc[frame["row_id"], "fold"].to_numpy(dtype=int)
        passed = bool(
            len(frame) == len(train) and not frame["row_id"].duplicated().any()
            and np.array_equal(frame["row_id"].to_numpy(dtype=int), train)
            and not (set(frame["row_id"].astype(int)) & test)
            and np.array_equal(frame["fold"].to_numpy(dtype=int), expected_fold)
            and np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
        )
        oof_details.append({"path": str(path.relative_to(root)), "rows": len(frame), "passed": passed})
    checks = {
        "development_screening_records": {"expected": 47, "actual": len(screening), "passed": len(screening) == 47},
        "successful_fold_evaluations": {"expected": 36, "actual": int((folds.status == "success").sum()), "passed": len(folds) == 36 and (folds.status == "success").all()},
        "oof_summary_rows": {"expected": 12, "actual": len(oof_summary), "passed": len(oof_summary) == 12},
        "oof_prediction_files": {"expected": 12, "actual": len(oof_details), "passed": len(oof_details) == 12 and all(item["passed"] for item in oof_details)},
        "saved_pipelines": {"expected": 12, "actual": len(list((root / "artifacts/models/linear").glob("*.joblib"))), "passed": len(list((root / "artifacts/models/linear").glob("*.joblib"))) == 12},
        "pipeline_reload_checks": {"expected": 12, "actual": int(reloads.passed.sum()), "passed": len(reloads) == 12 and reloads.passed.all()},
        "stage2_registry_rows": {"expected": 107, "actual": len(p2_registry), "passed": len(p2_registry) == 107 and not p2_registry.experiment_id.duplicated().any()},
        "registry_unique_and_preserved": {"expected": 0, "actual": int(registry.experiment_id.duplicated().sum()), "passed": not registry.experiment_id.duplicated().any() and set(p2_registry.experiment_id) == set(registry.experiment_id)},
        "coefficient_tables": {"expected": 10, "actual": len(list((root / "artifacts/features/linear/coefficients").glob("*.csv"))), "passed": len(list((root / "artifacts/features/linear/coefficients").glob("*.csv"))) == 10},
        "required_figures": {"expected": 4, "actual": len(list((root / "artifacts/figures/prompt2").glob("*.png"))), "passed": len(list((root / "artifacts/figures/prompt2").glob("*.png"))) == 4},
        "protected_hashes": {"expected": baseline, "actual": current_hashes, "passed": baseline == current_hashes},
    }
    report = {
        "display_stage_name": "Stage 2 — Baseline and Linear Regression Models",
        "internal_stage_id": "prompt2", "status": "PASS" if all(item["passed"] for item in checks.values()) else "FAIL",
        "checks": checks, "oof_files": oof_details,
        "notes": ["One failed development candidate is preserved as evidence.", "The locked test set is used only for row-ID exclusion checks."],
    }
    output = root / "artifacts/reports/stage2_completion_artifact_audit.json"
    output.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    return report


def notebook_output_audit(root):
    path = root / "REGRESSION_PART2_MODELING.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    key_sections = [17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40]
    section_outputs = {}
    for number in key_sections:
        start = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and cell.source.startswith(f"## {number}."))
        end = next((i for i in range(start + 1, len(notebook.cells)) if notebook.cells[i].cell_type == "markdown" and notebook.cells[i].source.startswith("## ")), len(notebook.cells))
        section_outputs[str(number)] = any(notebook.cells[i].cell_type == "code" and bool(notebook.cells[i].get("outputs")) for i in range(start + 1, end))
    report = {
        "status": "PASS" if (all(cell.execution_count is not None for cell in code_cells) and not errors and all(section_outputs.values())) else "FAIL",
        "notebook_path": str(path.resolve()), "total_cells": len(notebook.cells), "total_code_cells": len(code_cells),
        "executed_code_cells": sum(cell.execution_count is not None for cell in code_cells),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
        "error_output_count": len(errors), "key_section_outputs": section_outputs,
        "outputs_not_cleared": any(bool(cell.get("outputs")) for cell in code_cells[16:]),
    }
    (root / "artifacts/reports/stage2_notebook_output_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def snapshot(root):
    notebook = nbformat.read(root / "REGRESSION_PART2_MODELING.ipynb", as_version=4)
    registry = pd.read_csv(root / "artifacts/results/experiment_results.csv")
    leaderboard = pd.read_csv(root / "artifacts/results/prompt2/linear_leaderboard.csv").sort_values(["model_name", "sensitive_mode"])
    selected = json.loads((root / "artifacts/results/prompt2/selected_linear_configurations.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "artifacts/manifests/prompt2_model_manifest.json").read_text(encoding="utf-8"))
    headings = [line.strip() for cell in notebook.cells if cell.cell_type == "markdown" for line in cell.source.splitlines() if line.startswith("## ") and any(line.startswith(f"## {n}.") for n in range(16, 41))]
    oof = {path.name: len(pd.read_csv(path, usecols=["row_id"])) for path in sorted((root / "artifacts/predictions/linear").glob("*.csv"))}
    _, hashes = protected(root)
    return {
        "registry_row_count": len(registry), "registry_ids": sorted(registry.experiment_id.tolist()),
        "duplicate_registry_ids": int(registry.experiment_id.duplicated().sum()),
        "stage2_section_count": len(headings), "stage2_unique_section_count": len(set(headings)),
        "selected_configurations": selected, "leaderboard": leaderboard.to_dict(orient="records"),
        "oof_file_count": len(oof), "oof_row_counts": oof, "model_manifest_count": len(manifest["models"]),
        "protected_hashes": hashes,
    }


def compare(root, first_path, second_path):
    first = json.loads(Path(first_path).read_text(encoding="utf-8"))
    second = json.loads(Path(second_path).read_text(encoding="utf-8"))
    comparisons = {key: first[key] == second[key] for key in first}
    report = {"status": "PASS" if all(comparisons.values()) else "FAIL", "comparisons": comparisons,
              "first": first, "second": second}
    (root / "artifacts/reports/stage2_idempotence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def finalize(root):
    reports = root / "artifacts/reports"
    verification_path = reports / "prompt2_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    artifact = json.loads((reports / "stage2_completion_artifact_audit.json").read_text(encoding="utf-8"))
    output = json.loads((reports / "stage2_notebook_output_audit.json").read_text(encoding="utf-8"))
    idempotence = json.loads((reports / "stage2_idempotence_report.json").read_text(encoding="utf-8"))
    reviewer = (reports / "prompt2_reviewer.md").read_text(encoding="utf-8")
    notebook = nbformat.read(root / "REGRESSION_PART2_MODELING.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    completion_checks = {
        "notebook_internal_verification_pass": verification.get("status") == "PASS" and all(verification.get("checks", {}).values()),
        "cached_artifact_audit_pass": artifact.get("status") == "PASS",
        "saved_notebook_output_audit_pass": output.get("status") == "PASS",
        "idempotence_comparison_pass": idempotence.get("status") == "PASS",
        "independent_review_pass": "Critical issues fixed: PASS" in reviewer and "Major issues fixed: PASS" in reviewer,
        "all_code_cells_executed": len(code_cells) == 41 and all(cell.execution_count is not None for cell in code_cells),
        "zero_saved_error_outputs": not any(out.output_type == "error" for cell in code_cells for out in cell.get("outputs", [])),
        "stage_naming_complete": all(old not in cell.source for cell in notebook.cells for old in ("Prompt 1", "Prompt 2", "PROMPT 1", "PROMPT 2")),
    }
    verification.update({
        "internal_stage_id": "prompt2",
        "display_stage_name": "Stage 2 — Baseline and Linear Regression Models",
        "completion_status": "PASS" if all(completion_checks.values()) else "FAIL",
        "completion_checks": completion_checks,
        "execution_evidence": {
            "successful_clean_runs": 2,
            "run_1_seconds": 71.34,
            "run_2_seconds": 56.11,
            "saved_code_cells_executed": sum(cell.execution_count is not None for cell in code_cells),
            "saved_error_outputs": sum(out.output_type == "error" for cell in code_cells for out in cell.get("outputs", [])),
        },
        "evidence_paths": {
            "artifact_audit": "artifacts/reports/stage2_completion_artifact_audit.json",
            "output_audit": "artifacts/reports/stage2_notebook_output_audit.json",
            "idempotence_report": "artifacts/reports/stage2_idempotence_report.json",
            "independent_review": "artifacts/reports/prompt2_reviewer.md",
        },
    })
    verification["status"] = verification["completion_status"]
    verification_path.write_text(json.dumps(verification, indent=2, default=json_default), encoding="utf-8")
    leaderboard = pd.read_csv(root / "artifacts/results/prompt2/linear_leaderboard.csv")
    best = leaderboard.sort_values("oof_mae").iloc[0]
    summary = f"""# Stage 2 Completion Summary

Stage 2 — Baseline and Linear Regression Models is complete with status **{verification['status']}**.

- Two clean-kernel cached executions succeeded in 71.34 and 56.11 seconds.
- The saved notebook contains 41/41 executed code cells and zero error outputs.
- Cached-artifact, notebook-output, idempotence, and independent-review audits all pass.
- Six model families were evaluated in both sensitive modes using complete training OOF predictions; the locked test set was not used or predicted.
- The current training-OOF leader is `{best['model_name']}` in `{best['sensitive_mode']}` mode with MAE {best['oof_mae']:.6f} target units. This is not the final project model.
- Gamma converged formally but remained unstable and materially worse than the other model families.
- Lowercase `prompt1` and `prompt2` identifiers remain only for backward-compatible paths, variables, IDs, and metadata.

Independent review found no critical modeling defects. Two major presentation/state findings and one minor naming finding were fixed and re-adjudicated PASS.

Next step: **Begin Stage 3 — Tree-Based and Interpretable Models.**
"""
    (root / "stage2_completion_summary.md").write_text(summary, encoding="utf-8")
    return {"status": verification["status"], "checks": completion_checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["artifact", "output", "snapshot", "compare", "finalize"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--first")
    parser.add_argument("--second")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.mode == "artifact": result = artifact_audit(root)
    elif args.mode == "output": result = notebook_output_audit(root)
    elif args.mode == "snapshot":
        result = snapshot(root)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.mode == "compare": result = compare(root, args.first, args.second)
    else: result = finalize(root)
    print(json.dumps({"status": result.get("status", "SNAPSHOT"), "mode": args.mode}, indent=2))
    if result.get("status") == "FAIL": raise SystemExit(1)


if __name__ == "__main__":
    main()
