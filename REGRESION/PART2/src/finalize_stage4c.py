"""Write the final Stage 4C verification after recovery and review."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4
from build_stage4c_notebook import internal_verification


ROOT = Path(__file__).resolve().parent


def finalize() -> dict:
    p = c4.paths(ROOT)
    internal = internal_verification()
    protected = c4.recheck_protected(ROOT, save=True)
    recovery = c4.read_json(p["reports"] / "stage4c_notebook_recovery.json")
    output_audit = c4.read_json(p["reports"] / "stage4c_notebook_output_audit.json")
    reviewer_path = p["reports"] / "stage4c_reviewer.md"
    reviewer = reviewer_path.read_text(encoding="utf-8")
    notebook = nbformat.read(ROOT / "REGRESSION_PART4_CATBOOST.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.metadata.get("stage4c_owned")]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]

    test_ids = set(pd.read_csv(ROOT / "artifacts/splits/test_row_ids.csv", dtype={"row_id": "int64"})["row_id"].astype(int))
    prediction_files = sorted(path for path in p["predictions"].glob("*.csv") if path.is_file())
    prediction_overlap: dict[str, int] = {}
    prediction_rows: dict[str, int] = {}
    predictions_finite: dict[str, bool] = {}
    for path in prediction_files:
        frame = pd.read_csv(path, usecols=lambda name: name in {"row_id", "y_pred"})
        prediction_overlap[str(path.relative_to(ROOT))] = len(set(frame["row_id"].astype(int)).intersection(test_ids))
        prediction_rows[str(path.relative_to(ROOT))] = len(frame)
        predictions_finite[str(path.relative_to(ROOT))] = bool(np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all())

    screening = pd.read_csv(p["results"] / "catboost_initial_screening.csv")
    controlled = pd.read_csv(p["results"] / "catboost_controlled_validation_results.csv")
    reloads = pd.read_csv(p["reports"] / "stage4c_reload_verification.csv")
    proposals = pd.read_csv(p["features"] / "catboost_round2_feature_candidates.csv")
    registry = pd.read_csv(p["registry"])
    stage4c_registry = registry.loc[registry["experiment_id"].astype(str).str.startswith("stage4c__")]
    historical = c4.read_json(p["reports"] / "stage4c_notebook_executions.json")

    checks = {
        "stage4a_pass": c4.read_json(p["reports"] / "stage4a_verification.json")["status"] == "PASS",
        "stage4b_pass": c4.read_json(p["reports"] / "stage4b_verification.json")["status"] == "PASS",
        "protected_hashes_unchanged": protected["status"] == "PASS" and protected["file_count"] == 325,
        "foundation_notebook_unchanged": "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb" not in protected["mismatches"],
        "internal_verification_pass": internal["status"] == "PASS",
        "discovery_and_screening_valid": c4.read_json(p["reports"] / "stage4c_screening_subset_verification.json")["status"] == "PASS",
        "zero_test_overlap_all_predictions": prediction_files and all(value == 0 for value in prediction_overlap.values()),
        "candidate_count_four": len(screening) == 4,
        "three_required_candidates_pass": set(c4.REQUIRED_CANDIDATES).issubset(set(screening.loc[screening["status"].eq("PASS"), "candidate_id"])),
        "raw_and_log_targets_evaluated": {"raw", "log1p"}.issubset(set(screening["target_mode"])),
        "candidate_predictions_finite": all(predictions_finite.values()),
        "preliminary_configuration_saved": (p["results"] / "catboost_preliminary_configuration.json").is_file(),
        "two_controlled_results": len(controlled) == 2 and controlled["status"].eq("PASS").all(),
        "same_frozen_configuration_both_modes": controlled["fixed_iteration_count"].nunique() == 1 and controlled["feature_pack"].nunique() == 1 and controlled["target_mode"].nunique() == 1,
        "two_preliminary_models_saved": len(list(p["models"].glob("catboost_preliminary_*.joblib"))) == 2,
        "two_clean_reload_tests_pass": len(reloads) == 2 and reloads["status"].eq("PASS").all(),
        "importance_both_modes": all((p["features"] / f"catboost_importance_{mode}.csv").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "shap_both_modes": all((p["features"] / f"catboost_shap_importance_{mode}.csv").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "shap_rows_bounded": len(pd.read_csv(p["manifests"] / "catboost_shap_sample_row_ids.csv")) == 300,
        "error_analysis_both_modes": all((p["results"] / f"catboost_error_by_decile_{mode}.csv").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "three_safe_feature_proposals": len(proposals) == 3 and (~proposals["target_derived"].astype(bool)).all() and (~proposals["sensitive_derived"].astype(bool)).all(),
        "registry_ids_unique": not registry["experiment_id"].duplicated().any() and len(stage4c_registry) == 12,
        "historical_failures_preserved": historical.get("status") == "FAIL" and len(historical.get("attempts", [])) == 3,
        "recovery_supersedes_historical_execution": recovery["status"] == "PASS" and recovery["historical"]["attempt_count"] == 3,
        "complete_notebook_execution_passes": recovery.get("successful_complete_runs") == 1,
        "cache_only_execution_passes": recovery.get("successful_cache_only_runs") == 1,
        "no_model_retrained_in_recovery": output_audit["no_model_retraining"] is True,
        "notebook_outputs_saved": len(code_cells) == 25 and all(cell.get("execution_count") is not None and cell.get("outputs") for cell in code_cells) and not errors,
        "independent_reviewer_pass": "PASS" in reviewer and "None." in reviewer,
        "accepted_critical_and_major_findings_fixed": "## Critical Issues\n\nNone." in reviewer and "## Major Issues\n\nNone." in reviewer,
        "state_files_present": all((ROOT / name).is_file() for name in ("AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md")),
    }
    result = {
        "stage": c4.STAGE_ID,
        "stage_name": c4.STAGE_NAME,
        "created_at_utc": s4.utc_now(),
        "execution_evidence": {
            "historical_report": "artifacts/reports/stage4c_notebook_executions.json",
            "historical_status": historical.get("status"),
            "superseding_recovery_report": "artifacts/reports/stage4c_notebook_recovery.json",
            "recovery_status": recovery.get("status"),
        },
        "prediction_row_id_audit": {
            "files": prediction_rows,
            "test_overlap_rows": prediction_overlap,
            "finite_predictions": predictions_finite,
        },
        "reviewer_report": str(reviewer_path.relative_to(ROOT)),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
        "limitations": [
            "Discovery validation is a single holdout and is not OOF or Test evaluation.",
            "The extreme target tail remains difficult and is often underestimated.",
            "High-cardinality and possible proxy variables may generalize unevenly.",
            "Native SHAP values are associative and use the log1p model-output scale.",
        ],
    }
    s4.atomic_write_json(p["reports"] / "stage4c_verification.json", result)
    if result["status"] != "PASS":
        failed = [key for key, value in result["checks"].items() if not value]
        raise AssertionError(f"Stage 4C final verification failed: {failed}")
    return result


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))
