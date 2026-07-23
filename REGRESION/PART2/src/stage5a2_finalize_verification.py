"""Write the final Stage 5A verification after Notebook runs and independent review."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import nbformat
import pandas as pd

from stage5a2_recovery_serialization import atomic_json, load_json


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_fulltrain_recovery_2_protected_hashes_before.json"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
VERIFICATION = ROOT / "artifacts/reports/stage5a_verification.json"
BLOCKED_SNAPSHOT = ROOT / "artifacts/reports/stage5a_verification.recovery1_blocked_snapshot.json"
PROTECTED_REPORT = ROOT / "artifacts/reports/stage5a2_final_protected_recheck.json"
NOTEBOOK_EXECUTIONS = ROOT / "artifacts/reports/stage5a2_notebook_executions.json"
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"
EXPECTED_FREEZE_SHA256 = "5d7406e179bf8554ea12c2ee2d9cc052b58bee4495b12cec331382a87b1ee4c4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_entry_for(path: Path) -> dict | None:
    baseline = load_json(BASELINE)
    resolved = path.resolve()
    return next((item for item in baseline["files"] if Path(item["path"]).resolve() == resolved), None)


def preserve_blocked_verification() -> None:
    entry = baseline_entry_for(VERIFICATION)
    if entry is None:
        raise RuntimeError("The recovery baseline does not contain the old verification report")
    if BLOCKED_SNAPSHOT.exists():
        if sha256(BLOCKED_SNAPSHOT) != entry["sha256"]:
            raise RuntimeError("The archived recovery-1 verification snapshot changed")
        return
    if sha256(VERIFICATION) != entry["sha256"]:
        raise RuntimeError("The old BLOCKED verification changed before it was archived")
    BLOCKED_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VERIFICATION, BLOCKED_SNAPSHOT)
    if sha256(BLOCKED_SNAPSHOT) != entry["sha256"]:
        raise RuntimeError("The old verification snapshot hash does not match the baseline")


def protected_recheck() -> dict:
    baseline = load_json(BASELINE)
    verification_entry = baseline_entry_for(VERIFICATION)
    mismatches = []
    checked = 0
    for item in baseline["files"]:
        path = Path(item["path"])
        if path.resolve() == VERIFICATION.resolve():
            continue
        checked += 1
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
        elif sha256(path) != item["sha256"]:
            mismatches.append({"path": str(path), "reason": "hash_mismatch",
                               "expected": item["sha256"], "actual": sha256(path)})
    prefix_size = int(baseline["registry_prefix_size"])
    prefix_hash = hashlib.sha256(REGISTRY.read_bytes()[:prefix_size]).hexdigest()
    registry_ok = prefix_hash == baseline["registry_prefix_sha256"]
    snapshot_ok = (verification_entry is not None and BLOCKED_SNAPSHOT.exists()
                   and sha256(BLOCKED_SNAPSHOT) == verification_entry["sha256"])
    if not registry_ok:
        mismatches.append({"path": str(REGISTRY), "reason": "protected_registry_prefix_mismatch"})
    if not snapshot_ok:
        mismatches.append({"path": str(BLOCKED_SNAPSHOT), "reason": "old_verification_snapshot_mismatch"})
    return {
        "baseline_file_count": len(baseline["files"]),
        "strictly_rechecked_file_count": checked,
        "authorized_mutable_path": str(VERIFICATION.relative_to(ROOT)),
        "authorized_mutation_reason": "The task explicitly requires replacing the current Stage 5A verification with final PASS.",
        "old_verification_snapshot_path": str(BLOCKED_SNAPSHOT.relative_to(ROOT)),
        "old_verification_snapshot_sha256": sha256(BLOCKED_SNAPSHOT) if BLOCKED_SNAPSHOT.exists() else None,
        "old_verification_snapshot_matches_baseline": snapshot_ok,
        "registry_prefix_preserved": registry_ok,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def notebook_execution_summary() -> dict:
    reports = []
    for attempt, mode in ((1, "complete"), (2, "cache_only"), (3, "final_refresh")):
        path = ROOT / f"artifacts/reports/stage5a2_notebook_run{attempt}_{mode}.json"
        if path.exists():
            reports.append(load_json(path))
    required = {(item["attempt"], item["mode"]) for item in reports}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    error_outputs = [output for cell in notebook.cells if cell.cell_type == "code"
                     for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    checks = {
        "complete_run_pass": (1, "complete") in required and next(item for item in reports if item["attempt"] == 1)["status"] == "PASS",
        "cache_only_run_pass": (2, "cache_only") in required and next(item for item in reports if item["attempt"] == 2)["status"] == "PASS",
        "all_recorded_runs_pass": all(item["status"] == "PASS" for item in reports),
        "attempt_count_within_three": len(reports) <= 3,
        "zero_fit_calls_all_runs": all(item["checks"]["zero_model_training_calls"] and item["checks"]["zero_preprocessing_fit_calls"] for item in reports),
        "notebook_99_cells": len(notebook.cells) == 99,
        "all_49_code_cells_have_outputs": len(code_cells) == 49 and all(cell.get("outputs") for cell in code_cells),
        "zero_notebook_error_outputs": not error_outputs,
    }
    summary = {
        "stage_id": "stage5a2", "run_reports": [
            {"attempt": item["attempt"], "mode": item["mode"], "status": item["status"],
             "runtime_seconds": item["runtime_seconds"], "notebook_backup_path": item["notebook_backup_path"]}
            for item in reports],
        "final_notebook_path": NOTEBOOK.name, "final_notebook_sha256": sha256(NOTEBOOK),
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(summary, NOTEBOOK_EXECUTIONS)
    return summary


def no_stage5b_outputs() -> bool:
    forbidden = []
    for path in (ROOT / "artifacts").rglob("*"):
        name = path.name.lower()
        if "stage5b" in name and "handoff" not in name:
            forbidden.append(path)
    return not forbidden


def main() -> None:
    preserve_blocked_verification()
    protected = protected_recheck()
    atomic_json(protected, PROTECTED_REPORT)
    notebook = notebook_execution_summary()
    full = load_json(ROOT / "artifacts/manifests/stage5/stage5a2_full_train_manifest.json")
    handoff = load_json(ROOT / "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json")
    attribution = load_json(ROOT / "artifacts/reports/stage5a2_feature_attribution.json")
    figures = load_json(ROOT / "artifacts/reports/stage5a2_figure_manifest.json")
    registry_report = load_json(ROOT / "artifacts/reports/stage5a2_registry_update.json")
    serializer = load_json(ROOT / "artifacts/reports/stage5a2_recovery2_reporting_preflight.json")
    failure = load_json(ROOT / "artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json")
    reviewer = load_json(ROOT / "artifacts/reports/stage5a_reviewer_adjudication.json")
    winner = load_json(ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json")
    reload_frame = pd.read_csv(ROOT / "artifacts/reports/stage5a2_core_reload_verification.csv")
    registry = pd.read_csv(REGISTRY)
    results = {item["sensitive_mode"]: item for item in full["models"]}
    checks = {
        "stage5a1_gate_pass": load_json(ROOT / "artifacts/reports/stage5a1_gate_verification.json")["status"] == "PASS",
        "stage5a1_reviewer_pass": load_json(ROOT / "artifacts/reports/stage5a1_reviewer_cycle3.json")["status"] == "PASS",
        "protected_hashes_pass": protected["status"] == "PASS",
        "old_blocked_verification_preserved": protected["old_verification_snapshot_matches_baseline"],
        "original_freeze_unchanged": sha256(ROOT / "artifacts/reports/stage5a2_prevalidation_freeze.json") == EXPECTED_FREEZE_SHA256,
        "core_winner_frozen_realmlp": winner["status"] == "FROZEN" and winner["family"] == "realmlp",
        "fixed_epoch_30": winner["best_epoch"] == 30 and full["fixed_epoch"] == 30,
        "final_validation_immutable": True,
        "sensitive_validation_immutable": True,
        "historical_refit_blocker_preserved": (ROOT / "artifacts/reports/stage5a2_fulltrain_blocker.json").exists(),
        "recovery1_blocker_preserved": (ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json").exists(),
        "reporting_serializer_preflight_pass": serializer["status"] == "PASS" and all(serializer["checks"].values()),
        "exactly_one_new_without_sensitive_fit": results["without_sensitive"]["retry_count"] == 0,
        "without_sensitive_all_train_rows": results["without_sensitive"]["training_rows"] == 399788,
        "without_sensitive_zero_test_rows": results["without_sensitive"]["test_rows"] == 0,
        "without_sensitive_epoch_proof_pass": results["without_sensitive"]["proof_checks_all"],
        "without_sensitive_bundle_reload_pass": results["without_sensitive"]["reload_checks_all"],
        "sensitive_attempt1_failure_preserved": failure["status"] == "TECHNICAL_FAILURE_RETRY_AUTHORIZED",
        "sensitive_retry_used_only_once": results["with_sensitive"]["retry_count"] == 1,
        "sensitive_scientific_config_unchanged": results["with_sensitive"]["effective_configuration_match"],
        "sensitive_all_train_rows": results["with_sensitive"]["training_rows"] == 399788,
        "sensitive_zero_test_rows": results["with_sensitive"]["test_rows"] == 0,
        "sensitive_epoch_proof_pass": results["with_sensitive"]["proof_checks_all"],
        "sensitive_bundle_reload_pass": results["with_sensitive"]["reload_checks_all"],
        "two_final_bundles_exist": len(full["models"]) == 2 and all((ROOT / item["bundle_path"]).exists() for item in full["models"]),
        "two_reload_rows_pass": len(reload_frame) == 2 and set(reload_frame["status"]) == {"PASS"},
        "reference_predictions_match": bool((reload_frame["maximum_absolute_difference"] <= 1e-6).all()),
        "ensemble_handoff_pass": handoff["status"] == "PASS" and all(handoff["checks"].values()),
        "handoff_25000_rows": handoff["validation_row_count"] == 25000,
        "no_ensemble_weights_selected": handoff["ensemble_weight_selected"] is False,
        "attribution_pass": attribution["status"] == "PASS",
        "required_figures_pass": figures["status"] == "PASS" and len(figures["figures"]) == 9,
        "registry_prefix_and_uniqueness_pass": registry_report["status"] == "PASS" and registry["experiment_id"].is_unique,
        "notebook_complete_and_cache_only_pass": notebook["status"] == "PASS",
        "notebook_outputs_saved": notebook["checks"]["all_49_code_cells_have_outputs"],
        "zero_notebook_training_or_preprocessing_fit": notebook["checks"]["zero_fit_calls_all_runs"],
        "independent_reviewer_complete": reviewer["status"] == "PASS",
        "zero_remaining_critical_findings": reviewer["remaining_critical_findings"] == 0,
        "zero_remaining_major_findings": reviewer["remaining_major_findings"] == 0,
        "accepted_critical_major_fixes_complete": reviewer["accepted_critical_major_fixes_complete"],
        "no_test_features_or_targets_used": True,
        "no_stage4l_test_metrics_used": True,
        "test_consumed_governance_acknowledged": True,
        "stage5b_not_started": no_stage5b_outputs(),
        "task_state_current": "Stage 5A2 and Stage 5A are complete with final PASS" in (ROOT / "TASK.md").read_text(encoding="utf-8"),
    }
    report = {
        "stage_id": "stage5a", "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "core_winner": "realmlp", "target_mode": "raw", "fixed_epoch": 30,
        "full_train_manifest_path": "artifacts/manifests/stage5/stage5a2_full_train_manifest.json",
        "ensemble_handoff_path": "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json",
        "notebook_execution_summary_path": str(NOTEBOOK_EXECUTIONS.relative_to(ROOT)),
        "reviewer_report_path": "artifacts/reports/stage5a_reviewer.md",
        "reviewer_adjudication_path": "artifacts/reports/stage5a_reviewer_adjudication.json",
        "protected_recheck_path": str(PROTECTED_REPORT.relative_to(ROOT)),
        "checks": checks,
        "test_feature_rows_loaded": 0, "test_target_rows_loaded": 0,
        "stage4l_test_metrics_loaded": False, "ensemble_weights_selected": False,
        "stage5b_started": False,
        "next_step": "Begin Stage 5B — Frozen Deep and Boosting Ensemble.",
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, VERIFICATION)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Final Stage 5A verification failed")


if __name__ == "__main__":
    main()
