"""Create final external Stage 4B verification after review and state handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd

import stage4_boosting_utils as s4
import stage4b_feature_builder as builder
from prepare_stage4b_extension import STAGE4A_BACKUP, STAGE4A_CELL_COUNT, TAG


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(root: str | Path = ".") -> dict[str, Any]:
    project = Path(root).resolve()
    features = project / "artifacts/features/stage4"
    reports = project / "artifacts/reports"
    manifests = project / "artifacts/manifests/stage4"
    notebook_path = project / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    backup = nbformat.read(project / STAGE4A_BACKUP, as_version=4)
    owned = [cell for cell in notebook.cells if TAG in cell.get("metadata", {}).get("tags", [])]
    owned_code = [cell for cell in owned if cell.cell_type == "code"]
    owned_markdown = "\n".join(cell.source for cell in owned if cell.cell_type == "markdown")
    stage4a = _read(reports / "stage4a_verification.json")
    start = _read(reports / "stage4b_start_validation.json")
    internal = builder.build_internal_verification(project)
    idempotence = _read(reports / "stage4b_idempotence_report.json")
    output_audit = _read(reports / "stage4b_notebook_output_audit.json")
    smoke = _read(reports / "stage4b_smoke_tests.json")
    packs = _read(features / "boosting_feature_packs.json")
    proposals = pd.read_csv(features / "initial_feature_proposals.csv")
    roundtrips = pd.read_csv(features / "transformer_roundtrip_results.csv")
    clean_models = _read(reports / "stage4b_clean_model_roundtrip.json")
    protected = builder.recheck_stage4b_protected(project)
    reviewer_path = reports / "stage4b_reviewer.md"
    reviewer = reviewer_path.read_text(encoding="utf-8") if reviewer_path.is_file() else ""
    task = (project / "TASK.md").read_text(encoding="utf-8")
    required_artifacts = [
        features / "feature_audit.csv",
        features / "initial_feature_proposals.csv",
        features / "boosting_feature_packs.json",
        features / "catboost_feature_schema.json",
        features / "lightgbm_feature_schema.json",
        features / "xgboost_feature_schema.json",
        features / "transformer_roundtrip_results.csv",
        features / "clean_model_roundtrip_results.csv",
        reports / "stage4b_leakage_review.md",
        reports / "stage4b_clean_model_roundtrip.json",
        reports / "stage4b_reviewer.md",
        reports / "stage4b_notebook_executions.json",
        reports / "stage4b_notebook_output_audit.json",
        reports / "stage4b_idempotence_report.json",
    ]
    model_roots = [
        project / "artifacts/models/catboost",
        project / "artifacts/models/lightgbm",
        project / "artifacts/models/xgboost",
    ]
    prediction_roots = [
        project / "artifacts/predictions/catboost",
        project / "artifacts/predictions/lightgbm",
        project / "artifacts/predictions/xgboost",
    ]
    saved_model_files = sum(sum(path.is_file() for path in root_path.rglob("*")) for root_path in model_roots if root_path.exists())
    saved_prediction_files = sum(sum(path.is_file() for path in root_path.rglob("*")) for root_path in prediction_roots if root_path.exists())
    selected = proposals.loc[proposals["selected"].astype(str).str.lower().isin({"true", "1"})]
    checks = {
        "stage4a_pass": stage4a.get("status") == "PASS",
        "starting_requirements_pass": start.get("status") == "PASS",
        "five_initial_feature_packs_exist": set(packs["packs"]) == {
            "boosting_base_v1", "boosting_engineered_v1", "catboost_native_v1",
            "lightgbm_encoded_v1", "xgboost_sparse_v1",
        },
        "proposal_limit_met": len(proposals) <= 12,
        "selected_features_target_independent": not selected["target_derived"].astype(bool).any(),
        "selected_features_not_sensitive_derived": not selected["sensitive_derived"].astype(bool).any(),
        "learned_transforms_pipeline_ready": internal["checks"]["learned_steps_pipeline_bound"],
        "catboost_native_categories_preserved": internal["checks"]["native_catboost_categories_preserved"],
        "lightgbm_memory_safe_design": "frequency" in packs["packs"]["lightgbm_encoded_v1"]["encoding"],
        "xgboost_sparse_memory_safe_design": internal["checks"]["xgboost_sparse_design"],
        "transformer_roundtrips_pass": roundtrips["status"].eq("PASS").all(),
        "clean_process_model_reload_predict_pass": clean_models.get("status") == "PASS" and all(clean_models.get("checks", {}).values()),
        "smoke_tests_pass": smoke.get("status") == "PASS",
        "smoke_limits_pass": all(smoke["checks"].values()),
        "no_real_screening": smoke["checks"]["no_screening"],
        "test_set_locked": smoke["checks"]["test_rows_zero"] and internal["checks"]["test_set_locked"],
        "notebook_sections_19_to_32_once": all(owned_markdown.count(f"## {number}.") == 1 for number in range(19, 33)),
        "notebook_owned_cells_exact": len(owned) == 28 and len(owned_code) == 14,
        "notebook_outputs_saved": all(cell.get("execution_count") is not None and cell.get("outputs") for cell in owned_code),
        "notebook_zero_errors": not any(output.get("output_type") == "error" for cell in owned_code for output in cell.get("outputs", [])),
        "notebook_output_audit_pass": output_audit.get("status") == "PASS",
        "notebook_executes_twice": idempotence.get("successful_matching_runs", 0) >= 2,
        "notebook_runs_idempotent": idempotence.get("status") == "PASS" and idempotence.get("logical_results_match"),
        "stage4a_prefix_preserved": notebook.cells[:STAGE4A_CELL_COUNT] == backup.cells[:STAGE4A_CELL_COUNT],
        "stage4a_evidence_not_reexecuted": idempotence.get("stage4a_evidence_not_executed") is True,
        "protected_hashes_pass": protected.get("status") == "PASS",
        "required_artifacts_complete": all(path.is_file() and path.stat().st_size > 0 for path in required_artifacts),
        "independent_review_complete": "Overall result: PASS" in reviewer,
        "open_critical_findings_zero": "Open critical findings: 0" in reviewer,
        "open_major_findings_zero": "Open major findings: 0" in reviewer,
        "state_files_updated": "Next Step: Begin Stage 4C — Initial CatBoost Model and Importance Analysis." in task,
        "no_saved_real_boosting_model": saved_model_files == 0,
        "no_stage4_prediction": saved_prediction_files == 0,
    }
    result = {
        "stage": s4.STAGE4B_ID,
        "official_name": "Stage 4B — Initial Boosting Feature Packs",
        "version": builder.VERSION,
        "created_at_utc": s4.utc_now(),
        "checks": checks,
        "notebook": {
            "path": notebook_path.name,
            "cells": len(notebook.cells),
            "stage4a_prefix_cells": STAGE4A_CELL_COUNT,
            "stage4b_owned_cells": len(owned),
            "stage4b_code_cells": len(owned_code),
            "stage4b_executed_code_cells": sum(cell.get("execution_count") is not None for cell in owned_code),
            "implementation_digest": idempotence.get("implementation_digest"),
            "last_two_run_ids": idempotence.get("last_two_run_ids"),
            "last_two_snapshot_digests": idempotence.get("last_two_snapshot_digests"),
        },
        "feature_pack_names": list(packs["packs"]),
        "proposal_count": len(proposals),
        "selected_feature_count": len(selected),
        "rejected_feature_count": len(proposals) - len(selected),
        "protected_file_count": protected["file_count"],
        "protected_mismatches": protected["mismatches"],
        "saved_real_boosting_models": saved_model_files,
        "stage4_prediction_files": saved_prediction_files,
        "next_step": "Begin Stage 4C — Initial CatBoost Model and Importance Analysis.",
    }
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    s4.atomic_write_json(reports / "stage4b_verification.json", result)
    return result


if __name__ == "__main__":
    result = finalize()
    print(json.dumps(result, indent=2, default=s4._json_default))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
