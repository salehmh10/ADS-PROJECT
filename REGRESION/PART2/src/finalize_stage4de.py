"""Run final Stage 4D-E safety, artifact, notebook, and review verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd

import run_stage4de as runner
import stage4_boosting_utils as s4
import stage4de_catboost_utils as de


ROOT = Path(__file__).resolve().parent
MODES = ("without_sensitive", "with_sensitive")


def read_json(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return json.loads((value if value.is_absolute() else ROOT / value).read_text(encoding="utf-8"))


def prediction_audit() -> dict[str, Any]:
    train_ids = set(pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv", usecols=["row_id"], dtype={"row_id": "int64"})["row_id"].astype(int))
    test_ids = set(pd.read_csv(ROOT / "artifacts/splits/test_row_ids.csv", usecols=["row_id"], dtype={"row_id": "int64"})["row_id"].astype(int))
    candidates = set((ROOT / "artifacts/predictions/catboost/final").glob("*.csv"))
    candidates.update((ROOT / "artifacts/results/stage4/catboost/feature_confirmation").glob("*predictions*.csv"))
    candidates.update((ROOT / "artifacts/results/stage4/catboost/final").glob("*predictions*.csv"))
    files: dict[str, Any] = {}
    for path in sorted(candidates, key=lambda value: str(value).lower()):
        frame = pd.read_csv(path)
        if not {"row_id", "y_pred"}.issubset(frame.columns):
            continue
        ids = set(frame["row_id"].astype(int))
        files[str(path.relative_to(ROOT))] = {
            "rows": len(frame),
            "unique_rows": frame["row_id"].nunique(),
            "all_rows_in_saved_train": ids.issubset(train_ids),
            "test_overlap_rows": len(ids.intersection(test_ids)),
            "finite_predictions": bool(np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all()),
        }
    return {
        "file_count": len(files),
        "files": files,
        "all_in_saved_train": bool(files) and all(item["all_rows_in_saved_train"] for item in files.values()),
        "zero_test_overlap": bool(files) and all(item["test_overlap_rows"] == 0 for item in files.values()),
        "all_predictions_finite": bool(files) and all(item["finite_predictions"] for item in files.values()),
        "status": "PASS" if files and all(item["all_rows_in_saved_train"] and item["test_overlap_rows"] == 0 and item["finite_predictions"] for item in files.values()) else "FAIL",
    }


def artifact_manifest() -> dict[str, Any]:
    roots = [
        ROOT / "artifacts/results/stage4/catboost/feature_confirmation",
        ROOT / "artifacts/results/stage4/catboost/final",
        ROOT / "artifacts/models/catboost/final",
        ROOT / "artifacts/models/catboost/stage4de_candidates",
        ROOT / "artifacts/predictions/catboost/final",
        ROOT / "artifacts/checkpoints/stage4/catboost/stage4de",
        ROOT / "artifacts/figures/stage4/catboost/final",
    ]
    explicit = [
        ROOT / "artifacts/features/stage4/catboost/catboost_round2_proposal_review.csv",
        ROOT / "artifacts/features/stage4/catboost/catboost_selected_round2_features.csv",
        ROOT / "artifacts/features/stage4/catboost/catboost_rejected_round2_features.csv",
        ROOT / "artifacts/manifests/stage4/catboost/catboost_feature_engineer_v2_manifest.json",
        ROOT / "artifacts/reports/stage4de_preflight.json",
        ROOT / "artifacts/reports/stage4de_catboost_reload_verification.csv",
        ROOT / "artifacts/reports/stage4de_runtime_report.json",
        ROOT / "artifacts/reports/stage4de_analysis_summary.json",
        ROOT / "artifacts/reports/stage4de_internal_verification.json",
        ROOT / "artifacts/reports/stage4de_notebook_executions.json",
        ROOT / "artifacts/reports/stage4de_notebook_output_audit.json",
        ROOT / "artifacts/reports/stage4de_reviewer.md",
    ]
    files = {path for base in roots for path in base.rglob("*") if path.is_file()}
    files.update(path for path in explicit if path.is_file())
    records = {
        str(path.relative_to(ROOT)): {"sha256": s4.sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(files, key=lambda value: str(value).lower())
    }
    result = {"stage": de.STAGE_ID, "created_at_utc": s4.utc_now(), "file_count": len(records), "files": records, "status": "PASS" if records else "FAIL"}
    s4.atomic_write_json(ROOT / "artifacts/manifests/stage4/catboost/stage4de_artifact_manifest.json", result)
    return result


def finalize() -> dict[str, Any]:
    p = de.paths(ROOT)
    protected = de.recheck_protected(ROOT, save=True)
    samples = read_json("artifacts/splits/stage4/stage4_sample_verification.json")
    preflight = read_json("artifacts/reports/stage4de_preflight.json")
    confirmation = pd.read_csv(p["confirmation"] / "catboost_feature_confirmation_results.csv")
    impact = read_json(p["results"] / "catboost_feature_engineering_impact.json")
    feature_pack = read_json(p["results"] / "catboost_final_feature_pack.json")
    tuning = pd.read_csv(p["results"] / "catboost_final_tuning.csv")
    final_config = read_json(p["results"] / "catboost_final_configuration.json")
    controlled = pd.read_csv(p["results"] / "catboost_final_validation_results.csv")
    full = read_json(p["results"] / "catboost_full_train_manifest.json")
    interpretation = read_json(p["results"] / "catboost_final_interpretation_manifest.json")
    reload_table = pd.read_csv(p["reports"] / "stage4de_catboost_reload_verification.csv")
    notebook_runs = read_json(p["reports"] / "stage4de_notebook_executions.json")
    notebook_audit = read_json(p["reports"] / "stage4de_notebook_output_audit.json")
    runtime = read_json(p["reports"] / "stage4de_runtime_report.json")
    registry = pd.read_csv(p["registry"])
    stage_rows = registry.loc[registry["experiment_id"].astype(str).str.startswith("stage4de__")]
    reviewer_path = p["reports"] / "stage4de_reviewer.md"
    reviewer = reviewer_path.read_text(encoding="utf-8") if reviewer_path.is_file() else ""
    prediction = prediction_audit()

    confirmation_checkpoints = list(p["checkpoints"].glob("confirmation_*.json"))
    confirmation_fit_files = [path for path in confirmation_checkpoints if "__config" not in path.name]
    tuning_fit_files = [path for path in p["checkpoints"].glob("tuning_*.json") if "__config" not in path.name]
    controlled_meta = [read_json(p["checkpoints"] / f"final_controlled_{mode}.json") for mode in MODES]
    full_meta = [read_json(p["checkpoints"] / f"full_train_{mode}.json") for mode in MODES]
    fresh_reloads = [runner._reload_final(item["sensitive_mode"], item) for item in full_meta]
    s4.atomic_write_csv(pd.DataFrame(fresh_reloads), p["reports"] / "stage4de_final_reload_verification.csv")

    notebook = nbformat.read(ROOT / "REGRESSION_PART4_CATBOOST.ipynb", as_version=4)
    backups = sorted((ROOT / "artifacts/backups").glob("REGRESSION_PART4_CATBOOST_before_stage4de_*.ipynb"))
    backup = nbformat.read(backups[-1], as_version=4) if backups else None
    stage4de_code = [cell for cell in notebook.cells if cell.cell_type == "code" and "stage4de_owned" in cell.metadata.get("tags", [])]
    stage4de_headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ") and "stage4de_owned" in cell.metadata.get("tags", [])]
    errors = [output for cell in stage4de_code for output in cell.get("outputs", []) if output.get("output_type") == "error"]

    shap_ids = pd.read_csv(p["results"] / "catboost_final_shap_sample_ids.csv", dtype={"row_id": "int64"})
    importance = {mode: pd.read_csv(p["results"] / f"catboost_final_importance_{mode}.csv") for mode in MODES}
    shap = {mode: pd.read_csv(p["results"] / f"catboost_final_shap_{mode}.csv") for mode in MODES}
    previous_reference = pd.read_csv(p["results"] / "catboost_previous_stage_reference.csv")
    canonical_prior_paths = [
        ROOT / "artifacts/results/prompt2/prompt2_registry_rows.csv",
        ROOT / "artifacts/results/stage3/stage3_registry_rows.csv",
        ROOT / "artifacts/results/stage4/catboost/initial/stage4c_registry_rows.csv",
    ]
    main_text = pd.read_csv(p["registry"], dtype=str, keep_default_na=False).set_index("experiment_id")
    canonical_prior = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in canonical_prior_paths], ignore_index=True).set_index("experiment_id")
    prior_registry_text_equal = main_text.loc[canonical_prior.index].equals(canonical_prior)
    same_controlled = (
        controlled_meta[0]["sample_digest"] == controlled_meta[1]["sample_digest"]
        and controlled_meta[0]["parameters"] == controlled_meta[1]["parameters"]
        and controlled_meta[0]["feature_pack"] == controlled_meta[1]["feature_pack"]
        and controlled_meta[0]["target_mode"] == controlled_meta[1]["target_mode"]
        and controlled_meta[0]["fixed_iteration_count"] == controlled_meta[1]["fixed_iteration_count"]
    )
    model_files = [p["models"] / f"catboost_final_{mode}.{suffix}" for mode in MODES for suffix in ("joblib", "cbm")]
    checks = {
        "stage4a_pass": read_json("artifacts/reports/stage4a_verification.json")["status"] == "PASS",
        "stage4b_pass": read_json("artifacts/reports/stage4b_verification.json")["status"] == "PASS",
        "stage4c_pass": read_json("artifacts/reports/stage4c_verification.json")["status"] == "PASS",
        "protected_397_files_unchanged": protected["status"] == "PASS" and protected["file_count"] == 397,
        "preflight_pass": preflight["status"] == "PASS",
        "saved_samples_valid_and_test_free": samples["status"] == "PASS" and samples["test_overlap_rows"] == 0 and samples["all_sample_rows_unique"] is True,
        "feature_confirmation_sample_exact": samples["samples"]["feature_confirmation"]["train_rows"] == 80000 and samples["samples"]["feature_confirmation"]["validation_rows"] == 20000,
        "final_selection_sample_exact": samples["samples"]["final_selection"]["train_rows"] == 100000 and samples["samples"]["final_selection"]["validation_rows"] == 25000,
        "feature_confirmation_at_most_three_fits": len(confirmation) == len(confirmation_fit_files) and len(confirmation) <= 3,
        "required_final_confirmation_copy_present": (p["results"] / "catboost_feature_confirmation_results.csv").is_file(),
        "feature_selection_honest": impact["combined_accepted"] is False and impact["rescue_accepted"] is False and feature_pack["selected_proposals"] == [] and feature_pack["feature_pack"]["pack_id"] == "catboost_native_v1",
        "exactly_three_tuning_fits": len(tuning) == len(tuning_fit_files) == 3 and tuning["selected"].sum() == 1,
        "tuning_non_sensitive_only": tuning["sensitive_mode"].eq("without_sensitive").all(),
        "frozen_configuration_matches_selected_tuning": final_config["selected_fit_id"] == tuning.loc[tuning["selected"], "fit_id"].iloc[0] and final_config["fixed_iteration_count"] == 2000,
        "two_controlled_modes_pass": len(controlled) == 2 and controlled["status"].eq("PASS").all() and set(controlled["sensitive_mode"]) == set(MODES),
        "controlled_settings_identical_except_sensitive_sources": same_controlled,
        "two_full_train_modes_all_rows": full["status"] == "PASS" and all(item["training_row_count"] == 399788 and item["fixed_iteration_count"] == 2000 for item in full["models"].values()),
        "two_bundles_and_two_native_models": len(model_files) == 4 and all(path.is_file() and path.stat().st_size > 0 for path in model_files),
        "saved_model_hashes_match_manifest": all(s4.sha256_file(p["models"] / f"catboost_final_{mode}.joblib") == full["models"][mode]["model_sha256"] and s4.sha256_file(p["models"] / f"catboost_final_{mode}.cbm") == full["models"][mode]["native_model_sha256"] for mode in MODES),
        "bundle_metadata_contract_recorded": all(all(full["models"][mode]["bundle_metadata_contract"].values()) for mode in MODES),
        "original_and_fresh_reload_pass": len(reload_table) == 2 and reload_table["status"].eq("PASS").all() and all(item["status"] == "PASS" for item in fresh_reloads),
        "importance_complete_both_modes": all(len(importance[mode]) == full["models"][mode]["feature_count"] and np.isfinite(importance[mode]["importance"]).all() for mode in MODES),
        "shap_same_bounded_sample_both_modes": interpretation["status"] == "PASS" and interpretation["same_ids_both_modes"] is True and len(shap_ids) == shap_ids["row_id"].nunique() == 300 and all(len(shap[mode]) == full["models"][mode]["feature_count"] and shap[mode]["sample_rows"].eq(300).all() for mode in MODES),
        "all_stage4de_predictions_train_only_and_test_free": prediction["status"] == "PASS" and prediction["zero_test_overlap"] is True,
        "registry_unique_and_sixteen_stage_rows": registry["experiment_id"].is_unique and len(stage_rows) == 16,
        "prior_registry_rows_text_identical": prior_registry_text_equal and len(canonical_prior) == 227,
        "previous_stage_reference_complete": {"sensitive_mode", "runtime_seconds", "prediction_runtime_seconds", "locked_test_note"}.issubset(previous_reference.columns) and previous_reference["locked_test_note"].eq("The locked Test Set will later provide the common final comparison.").all(),
        "runtime_under_210_minutes": runtime["status"] == "PASS" and runtime["within_budget"] is True,
        "one_clean_and_cache_notebook_evidence_within_limit": notebook_runs["status"] == "PASS" and notebook_runs["successful_complete_runs"] >= 1 and notebook_runs["successful_cache_only_runs"] >= 1 and len(notebook_runs["attempts"]) <= 3 and notebook_runs["final_source_validated_by_execution"] is True,
        "notebook_idempotence_and_no_refit": notebook_audit["status"] == "PASS" and notebook_audit["no_model_retraining"] is True and notebook_audit["heavy_artifacts_unchanged"] is True and notebook_audit["registry_unchanged"] is True,
        "notebook_prefix_preserved": backup is not None and len(backup.cells) == 76 and notebook.cells[:76] == backup.cells,
        "notebook_sections_25_to_45_once": len(stage4de_headings) == 21 and len(set(stage4de_headings)) == 21 and [int(value.split(".")[0].split()[-1]) for value in stage4de_headings] == list(range(25, 46)),
        "notebook_reason_before_every_stage4de_code": all("**Reason.**" in cell.source for cell in notebook.cells if cell.cell_type == "markdown" and "stage4de_owned" in cell.metadata.get("tags", []) and cell.source.startswith("## ")),
        "notebook_outputs_saved": len(stage4de_code) == 21 and all(cell.get("execution_count") is not None and cell.get("outputs") for cell in stage4de_code) and not errors,
        "independent_reviewer_pass": "Recommendation: PASS" in reviewer and "Critical: 0" in reviewer and "Major: 0" in reviewer,
        "state_files_present": all((ROOT / name).is_file() for name in ("AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md")),
        "stage4f_not_started": not registry["experiment_id"].astype(str).str.startswith("stage4f__").any(),
    }
    manifest = artifact_manifest()
    checks["artifact_manifest_pass"] = manifest["status"] == "PASS" and manifest["file_count"] > 0
    result = {
        "stage": de.STAGE_ID,
        "stage_name": de.STAGE_NAME,
        "created_at_utc": s4.utc_now(),
        "protected_hash_report": "artifacts/manifests/stage4/catboost/stage4de_protected_hashes_after.json",
        "artifact_manifest": "artifacts/manifests/stage4/catboost/stage4de_artifact_manifest.json",
        "reviewer_report": str(reviewer_path.relative_to(ROOT)),
        "prediction_row_id_audit": prediction,
        "fresh_reload_verification": fresh_reloads,
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
        "limitations": [
            "Final Selection is a saved validation sample, not OOF or Test evaluation.",
            "The locked Test Set has not been evaluated in Stage 4D-E.",
            "Sensitive-mode differences are accuracy comparisons, not a complete fairness assessment.",
            "Native SHAP values are associative and use the log1p model-output scale.",
            "The extreme target tail remains difficult and may be underestimated.",
        ],
        "next_stage": "Stage 4F",
    }
    s4.atomic_write_json(p["reports"] / "stage4de_verification.json", result)
    if result["status"] != "PASS":
        failed = [key for key, value in result["checks"].items() if not value]
        raise AssertionError(f"Stage 4D-E final verification failed: {failed}")
    return result


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))
