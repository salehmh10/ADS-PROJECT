"""Record the non-retriable Stage 5A2 recovery blocker without another fit."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage5a2_deep_utils import ROOT, atomic_json, sha256_file
from stage5a2_fulltrain_recovery import RECOVERY_BASELINE, WITH_ID, WITHOUT_ID, mode_paths


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def recheck_recovery_baseline() -> dict:
    baseline = json.loads(RECOVERY_BASELINE.read_text(encoding="utf-8"))
    mismatches = []
    registry_name = "experiment_results.csv"
    for item in baseline["files"]:
        path = Path(item["path"])
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        if path.name == registry_name and path.stat().st_size >= int(item["size"]):
            actual = hashlib.sha256(path.read_bytes()[: int(item["size"])]).hexdigest()
            reason = "prior_registry_prefix_mismatch"
        else:
            actual = sha256_file(path)
            reason = "hash_mismatch"
        if actual != item["sha256"]:
            mismatches.append({"path": str(path), "reason": reason,
                               "expected": item["sha256"], "actual": actual})
    notebook_backup = ROOT / baseline["notebook_backup_path"]
    backup_ok = notebook_backup.exists() and sha256_file(notebook_backup) == baseline["notebook_backup_sha256"]
    if not backup_ok:
        mismatches.append({"path": str(notebook_backup), "reason": "notebook_backup_hash_mismatch"})
    report = {
        "stage_id": "stage5a2", "recovery_id": "stage5a2_fulltrain_recovery_1",
        "checked_file_count": len(baseline["files"]), "mismatches": mismatches,
        "notebook_backup_unchanged": backup_ok,
        "status": "PASS" if not mismatches else "FAIL",
    }
    path = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_protected_recheck.json"
    atomic_json(report, path)
    return report


def main() -> None:
    p = mode_paths("without_sensitive")
    checkpoint = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
    history = pd.read_csv(p["history"])
    effective = json.loads(p["effective"].read_text(encoding="utf-8"))
    parent_before = json.loads(p["parent"].read_text(encoding="utf-8"))
    archive_dir = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_attempt"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_parent = archive_dir / "without_sensitive_parent_report_before_failure_finalization.json"
    if not archived_parent.exists():
        shutil.copy2(p["parent"], archived_parent)
    archived_checkpoint = archive_dir / "without_sensitive_postfit_reporting_failure_checkpoint.json"
    if not archived_checkpoint.exists():
        shutil.copy2(p["checkpoint"], archived_checkpoint)
    archived_history = archive_dir / "without_sensitive_completed_30_epoch_history.csv"
    if not archived_history.exists():
        shutil.copy2(p["history"], archived_history)
    for source, destination in ((p["parent"], archived_parent), (p["checkpoint"], archived_checkpoint),
                                (p["history"], archived_history)):
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Recovery failure archive mismatch: {source}")

    proof_checks = {
        "one_new_physical_fit_recorded": checkpoint.get("physical_fit_started") is True,
        "checkpoint_failed_after_fit": checkpoint.get("status") == "FAIL",
        "failure_is_json_proof_serialization": "not JSON serializable" in checkpoint.get("error", ""),
        "history_has_30_rows": len(history) == 30,
        "history_epoch_sequence_1_to_30": history["epoch"].tolist() == list(range(1, 31)),
        "last_epoch_is_30": int(history.iloc[-1]["epoch"]) == 30,
        "epoch_float_is_30": float(history.iloc[-1]["pytabkit_progress_epoch_float"]) == 30.0,
        "optimizer_steps_increase": bool(history["trainer_global_step"].is_monotonic_increasing
                                          and history["trainer_global_step"].is_unique),
        "effective_config_saved_before_fit": effective.get("saved_before_model_fit") is True,
        "effective_config_was_prefit_pass": all(bool(value) for value in effective.get("checks", {}).values()),
        "staging_bundle_absent": not p["staging_bundle"].exists(),
        "final_bundle_absent": not p["bundle"].exists(),
        "with_sensitive_checkpoint_absent": not mode_paths("with_sensitive")["checkpoint"].exists(),
        "with_sensitive_bundle_absent": not mode_paths("with_sensitive")["bundle"].exists(),
    }
    proof = {
        "stage_id": "stage5a2", "recovery_id": "stage5a2_fulltrain_recovery_1",
        "candidate_id": WITHOUT_ID, "sensitive_mode": "without_sensitive",
        "requested_epoch": 30, "completed_train_epoch": int(history.iloc[-1]["epoch"]),
        "training_history_row_count": len(history), "last_training_history_epoch": int(history.iloc[-1]["epoch"]),
        "pytabkit_progress_epoch": int(history.iloc[-1]["pytabkit_progress_epoch"]),
        "pytabkit_progress_epoch_float": float(history.iloc[-1]["pytabkit_progress_epoch_float"]),
        "trainer_global_step": int(history.iloc[-1]["trainer_global_step"]),
        "epoch_zero_interpretation": (
            "The no-validation PyTabKit counter remained 0 while the independent train-epoch callback reached 30. "
            "This training evidence cannot promote a model because serialization did not occur."
        ),
        "history_path": str(p["history"].relative_to(ROOT)), "history_sha256": sha256_file(p["history"]),
        "effective_config_path": str(p["effective"].relative_to(ROOT)),
        "effective_config_sha256": sha256_file(p["effective"]),
        "model_weight_hash": None, "serialized_estimator_path": None,
        "reference_predictions_created": False, "clean_process_reload_completed": False,
        "checks": proof_checks,
        "failure_phase": "post_fit_proof_json_serialization_before_model_serialization",
        "failure": checkpoint.get("error"), "status": "FAIL",
    }
    atomic_json(proof, p["proof"])

    parent = {
        **parent_before,
        "status": "FAIL", "checkpoint_status": "FAIL", "return_code": None,
        "terminated_after_confirmed_postfit_reporting_failure": True,
        "termination_reason": "Child completed epoch 30, failed proof JSON serialization, and hung with zero CPU during shutdown.",
        "physical_fit_retried": False, "finalized_at": utc_now(),
    }
    atomic_json(parent, p["parent"])
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    if not p["log"].exists():
        p["log"].write_text(
            "Stage 5A2 recovery fit completed 30 audited train epochs.\n"
            "Post-fit proof JSON serialization failed before any model bundle was saved.\n\n"
            + checkpoint.get("traceback", ""),
            encoding="utf-8",
        )

    protected = recheck_recovery_baseline()
    blocker = {
        "stage_id": "stage5a2", "recovery_id": "stage5a2_fulltrain_recovery_1",
        "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "status": "BLOCKED", "recorded_at": utc_now(),
        "blocker_code": "authorized_without_sensitive_replacement_failed_before_model_serialization_no_retry_allowed",
        "candidate_id": WITHOUT_ID, "candidate_family": "RealMLP", "sensitive_mode": "without_sensitive",
        "target_mode": "raw", "requested_fixed_epoch": 30,
        "new_physical_fits_completed": 1, "new_retries_used": 0, "new_retries_authorized": 0,
        "training_evidence": {
            "completed_train_epochs": 30, "history_rows": 30,
            "last_epoch": 30, "trainer_global_step": int(history.iloc[-1]["trainer_global_step"]),
            "pytabkit_progress_epoch": int(history.iloc[-1]["pytabkit_progress_epoch"]),
            "pytabkit_progress_epoch_float": float(history.iloc[-1]["pytabkit_progress_epoch_float"]),
        },
        "root_cause": (
            "A pandas/NumPy boolean in the post-fit proof checks was passed to the standard JSON encoder. "
            "The proof write raised TypeError before the trained estimator was serialized."
        ),
        "technical_repair_applied_for_future_authorized_resume": (
            "Proof check values are now normalized to native Python bool before JSON serialization. "
            "No model was refitted after this repair."
        ),
        "artifact_state": {
            "epoch_history_saved": True, "effective_configuration_saved": True,
            "weight_hash_saved": False, "staging_bundle_saved": False,
            "without_sensitive_final_bundle_saved": False,
            "without_sensitive_clean_reload_completed": False,
            "with_sensitive_full_train_started": False, "with_sensitive_bundle_saved": False,
            "ensemble_handoff_created": False, "stage5b_started": False,
            "test_feature_rows_loaded": 0, "test_target_rows_loaded": 0,
            "stage4l_test_metrics_used": False,
        },
        "preserved_evidence": {
            "checkpoint": {"path": str(p["checkpoint"].relative_to(ROOT)), "sha256": sha256_file(p["checkpoint"])},
            "history": {"path": str(p["history"].relative_to(ROOT)), "sha256": sha256_file(p["history"])},
            "effective_config": {"path": str(p["effective"].relative_to(ROOT)), "sha256": sha256_file(p["effective"])},
            "failed_epoch_proof": {"path": str(p["proof"].relative_to(ROOT)), "sha256": sha256_file(p["proof"])},
            "parent_report": {"path": str(p["parent"].relative_to(ROOT)), "sha256": sha256_file(p["parent"])},
            "log": {"path": str(p["log"].relative_to(ROOT)), "sha256": sha256_file(p["log"])},
            "archived_checkpoint": {"path": str(archived_checkpoint.relative_to(ROOT)), "sha256": sha256_file(archived_checkpoint)},
            "archived_history": {"path": str(archived_history.relative_to(ROOT)), "sha256": sha256_file(archived_history)},
            "archived_parent_before_finalization": {"path": str(archived_parent.relative_to(ROOT)), "sha256": sha256_file(archived_parent)},
        },
        "protected_recheck": protected,
        "required_human_action": (
            "A new explicit authorization is required for any further without-sensitive physical fit. "
            "Do not start the with-sensitive fit, Stage 5B, or another recovery fit under the exhausted authorization."
        ),
    }
    blocker_path = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json"
    atomic_json(blocker, blocker_path)
    print(json.dumps(blocker, indent=2))


if __name__ == "__main__":
    main()
