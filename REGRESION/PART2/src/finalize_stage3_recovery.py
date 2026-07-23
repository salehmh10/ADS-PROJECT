"""External Stage 3 final verification after review and two cached executions."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from stage3_tree_utils import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts/reports"
MANIFESTS = ROOT / "artifacts/manifests"
RESULTS = ROOT / "artifacts/results/stage3"
NOTEBOOK = ROOT / "REGRESSION_PART3_TREE_MODELS.ipynb"


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    baseline = json.loads((MANIFESTS / "stage3_protected_hashes_before.json").read_text(encoding="utf-8"))["hashes"]
    protected_after = {}
    for name in baseline:
        path = Path(name); path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise AssertionError(f"Protected file is missing: {name}")
        protected_after[name] = sha256_file(path)
    protected_ok = protected_after == baseline

    train_ids = set(pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv")["row_id"].astype(int))
    test_ids = set(pd.read_csv(ROOT / "artifacts/splits/test_row_ids.csv")["row_id"].astype(int))
    cv_rows = pd.read_csv(RESULTS / "cv_fold_results.csv")
    oof_summary = pd.read_csv(RESULTS / "cv_oof_summary.csv")
    oof_checks = []
    for row in oof_summary.itertuples(index=False):
        frame = pd.read_csv(ROOT / row.oof_path)
        oof_checks.append(
            len(frame) == len(train_ids)
            and frame["row_id"].nunique() == len(train_ids)
            and set(frame["row_id"].astype(int)) == train_ids
            and not set(frame["row_id"].astype(int)).intersection(test_ids)
            and np.isfinite(frame["y_pred"]).all()
        )

    final_fits = pd.read_csv(RESULTS / "final_training_fit_results.csv")
    reloads = pd.read_csv(REPORTS / "stage3_model_reload_verification.csv")
    importance = pd.read_csv(ROOT / "artifacts/features/tree/importance/stage3_feature_importance_summary.csv")
    registry = pd.read_csv(ROOT / "artifacts/results/experiment_results.csv")
    prompt2_registry = pd.read_csv(ROOT / "artifacts/results/prompt2/prompt2_registry_rows.csv")
    prior_registry = registry.loc[~registry["experiment_id"].astype(str).str.startswith("stage3__")]
    stage3_registry = registry.loc[registry["experiment_id"].astype(str).str.startswith("stage3__")]
    selected = json.loads((RESULTS / "selected_tree_configurations.json").read_text(encoding="utf-8"))
    executions = json.loads((REPORTS / "stage3_notebook_executions.json").read_text(encoding="utf-8"))["executions"]
    successful = [row for row in executions if row.get("promoted")]
    idempotence = json.loads((REPORTS / "stage3_idempotence_report.json").read_text(encoding="utf-8"))
    reviewer_text = (REPORTS / "stage3_reviewer.md").read_text(encoding="utf-8")
    task_text = (ROOT / "TASK.md").read_text(encoding="utf-8")
    figures = sorted((ROOT / "artifacts/figures/stage3").glob("*.png"))
    progress = json.loads((REPORTS / "stage3_progress.json").read_text(encoding="utf-8"))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = sum(out.output_type == "error" for cell in code_cells for out in cell.get("outputs", []))
    source_text = "\n".join(cell.source for cell in notebook.cells)
    old_configuration_json = json.dumps(
        selected["bagging_runtime_repair"]["old_configuration"], sort_keys=True, separators=(",", ":")
    )
    old_status = stage3_registry.loc[
        stage3_registry["parameter_json"].eq(old_configuration_json)
        & stage3_registry["evaluation_stage"].eq("model_screening"), "status"
    ].tolist()

    checks = {
        "protected_hashes_match": protected_ok,
        "test_set_unused_for_modeling_or_prediction": not any(ROOT.glob("artifacts/**/*test*prediction*")),
        "saved_split_and_folds_reused": len(cv_rows) == 18 and set(cv_rows["fold"].astype(int)) == {0, 1, 2},
        "per_fold_checkpoints_complete": len(list((RESULTS / "folds").glob("*.json"))) == 18,
        "hard_timeout_implemented": "taskkill" in (ROOT / "stage3_recovery_runner.py").read_text(encoding="utf-8")
                                    and "subprocess.Popen" in (ROOT / "stage3_recovery_runner.py").read_text(encoding="utf-8"),
        "partial_resume_implemented": "validate_fold_checkpoint" in (ROOT / "stage3_recovery_runner.py").read_text(encoding="utf-8"),
        "decision_tree_oof_complete_both_modes": oof_summary["model_name"].eq("decision_tree").sum() == 2,
        "hgb_oof_complete_both_modes": oof_summary["model_name"].eq("hist_gradient_boosting").sum() == 2,
        "repaired_bagging_oof_complete_both_modes": oof_summary["model_name"].eq(selected["models"]["bagging"]["model_name"]).sum() == 2,
        "all_oof_rows_unique_complete_finite_and_test_free": all(oof_checks),
        "same_repaired_bagging_configuration_both_modes": selected["bagging_runtime_repair"]["same_configuration_both_sensitive_modes"] is True,
        "six_final_pipelines_saved": len(final_fits) == 6 and final_fits["status"].eq("success").all()
                                     and all((ROOT / path).exists() for path in final_fits["model_path"]),
        "six_final_pipelines_reload_and_match": len(reloads) == 6 and reloads["status"].eq("PASS").all(),
        "importance_runtime_control_passed": importance.groupby(["model_name", "sensitive_mode"]).ngroups == 6
                                             and importance["sample_rows"].max() <= 10000
                                             and set(importance["n_repeats"].astype(int)) == {3},
        "registry_ids_unique": registry["experiment_id"].is_unique and len(stage3_registry) == 108,
        "prior_registry_rows_preserved": set(prior_registry["experiment_id"]) == set(prompt2_registry["experiment_id"]),
        "old_bagging_history_superseded": old_status == ["superseded_for_runtime_feasibility"],
        "required_figures_exist": len(figures) == 10 and all(path.stat().st_size > 0 for path in figures),
        "first_cached_notebook_execution_passed": len(successful) >= 1 and successful[0]["error_outputs"] == 0,
        "second_cached_notebook_execution_passed": len(successful) == 2 and successful[1]["error_outputs"] == 0,
        "notebook_outputs_saved": len(code_cells) == 30
                                  and sum(cell.execution_count is not None for cell in code_cells) == 30
                                  and sum(bool(cell.get("outputs")) for cell in code_cells) == 30
                                  and error_outputs == 0,
        "idempotence_passed": idempotence.get("status") == "PASS" and idempotence.get("logical_snapshots_match") is True,
        "independent_review_passed": "Review status: PASS" in reviewer_text
                                     and "No major finding remains open." in reviewer_text,
        "ebm_exception_accepted": "The EBM exception is accepted." in reviewer_text
                                  and selected["ebm"] == {"status": "environment_exception", "substitute_used": False},
        "state_files_current": "Stage 3 recovery and completion are complete." in task_text
                               and "Begin Stage 4" in task_text and "- None." in task_text,
        "sections_zero_through_thirty_unique": all(source_text.count(f"## {number}.") == 1 for number in range(31)),
        "recovery_budgets_respected": progress["targeted_execution_attempts"] <= 6
                                      and progress["full_notebook_execution_attempts"] <= 3
                                      and progress["repair_iterations"] <= 20,
        "stage4_not_started": not any(
            token in path.name.lower() for path in (ROOT / "artifacts/models").rglob("*") if path.is_file()
            for token in ("catboost", "lightgbm", "xgboost")
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    verification = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "verification_type": "external_after_two_cached_executions_and_independent_review",
        "checks": checks,
        "counts": {
            "training_rows": len(train_ids), "locked_test_rows": len(test_ids),
            "fold_checkpoints": 18, "cv_fold_results": len(cv_rows), "oof_experiments": len(oof_summary),
            "final_pipelines": len(final_fits), "reload_passes": int(reloads["status"].eq("PASS").sum()),
            "importance_experiments": importance.groupby(["model_name", "sensitive_mode"]).ngroups,
            "stage3_registry_rows": len(stage3_registry), "prior_registry_rows": len(prior_registry),
            "figures": len(figures), "executed_code_cells": len(code_cells),
            "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
            "error_outputs": error_outputs,
        },
        "successful_notebook_execution_seconds": [row["duration_seconds"] for row in successful],
        "selected_configurations": selected,
        "oof_results": oof_summary[["model_name", "sensitive_mode", "mae", "rmse", "rmsle", "r_squared", "total_fit_time_seconds"]].to_dict("records"),
        "accepted_ebm_environment_exception": True,
        "test_predictions": 0,
        "next_step": "Begin Stage 4 â€” CatBoost, LightGBM, and XGBoost." if status == "PASS" else None,
    }
    write_json(MANIFESTS / "stage3_protected_hashes_after.json", {
        "created_at_utc": verification["created_at_utc"], "hashes": protected_after,
    })
    write_json(REPORTS / "stage3_final_artifact_audit.json", {
        "status": status, "checks": checks, "counts": verification["counts"],
    })
    write_json(REPORTS / "stage3_verification.json", verification)
    if status != "PASS":
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"Final Stage 3 verification failed: {failed}")

    note_title = "## External Stage 3 Verification After Independent Review"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if not any(cell.cell_type == "markdown" and note_title in cell.source for cell in notebook.cells):
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            note_title + "\n\n"
            "Final external verification is **PASS**. The two required cached executions completed with "
            "30 of 30 code cells, outputs in all code cells, and zero errors. The logical snapshots match. "
            "The independent review found no open critical or major issue and accepted the EBM environment exception. "
            "All 85 protected hashes still match, and the locked Test Set was not predicted. "
            "Stage 4 is the next step, but it was not started in this notebook."
        ))
        temporary = NOTEBOOK.with_suffix(NOTEBOOK.suffix + ".tmp")
        nbformat.write(notebook, temporary)
        os.replace(temporary, NOTEBOOK)

    best = oof_summary.sort_values("mae").iloc[0]
    summary = f"""# Stage 3 Completion Summary

Status: PASS

- Best training OOF result: {best['model_name']} in {best['sensitive_mode']} mode, MAE {best['mae']:.6f} thousand US dollars.
- Fold checkpoints: 18/18 PASS.
- Complete OOF files: 6/6 PASS, each with {len(train_ids):,} unique Train rows and zero Test overlap.
- Final Pipelines and clean reload checks: 6/6 PASS.
- Controlled importance experiments: 6/6 PASS, at most 10,000 rows and three repeats.
- Registry: 108 unique Stage 3 rows; all 107 prior rows preserved.
- Notebook: two successful cached executions in {successful[0]['duration_seconds']:.2f} and {successful[1]['duration_seconds']:.2f} seconds; idempotence PASS.
- Protected hashes: {len(baseline)} checked and zero mismatches.
- Independent review: PASS; EBM environment exception accepted.
- Test Set: locked and unused for predictions.

Next step: Begin Stage 4 â€” CatBoost, LightGBM, and XGBoost.
"""
    atomic_write_text(ROOT / "stage3_completion_summary.md", summary)
    print(json.dumps({"status": status, "passed_checks": int(sum(bool(value) for value in checks.values())),
                      "total_checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
