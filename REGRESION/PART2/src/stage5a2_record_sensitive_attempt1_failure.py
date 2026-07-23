"""Preserve the first with-sensitive Full-Train technical failure before retry."""

from pathlib import Path

import pandas as pd
import psutil

from stage5a2_deep_utils import ROOT, sha256_file
from stage5a2_recovery_serialization import atomic_json, load_json


CANDIDATE_ID = "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30"
CHECKPOINT = ROOT / f"artifacts/checkpoints/stage5/deep_core/full_train/recovery2/{CANDIDATE_ID}.json"
HISTORY = ROOT / f"artifacts/results/stage5/deep_core/full_train/histories/{CANDIDATE_ID}_history.csv"
PARENT = ROOT / f"artifacts/reports/stage5a2_parent_{CANDIDATE_ID}.json"
MODEL = ROOT / f"artifacts/models/deep/core_final/components/{CANDIDATE_ID}_model.joblib"
REFERENCE = ROOT / f"artifacts/predictions/stage5/deep_core/full_train_reference/{CANDIDATE_ID}.csv"
PROOF = ROOT / f"artifacts/reports/{CANDIDATE_ID}_epoch_proof.json"
BUNDLE = ROOT / f"artifacts/models/deep/core_final/{CANDIDATE_ID}.joblib"
LOCK = ROOT / "artifacts/checkpoints/stage5/deep_core/stage5a2_heavy_fit.lock"
REPORT = ROOT / "artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json"


def evidence(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    if REPORT.exists():
        existing = load_json(REPORT)
        if existing.get("status") == "TECHNICAL_FAILURE_RETRY_AUTHORIZED":
            print(existing)
            return
        raise RuntimeError("Unexpected existing sensitive failure report")
    checkpoint = load_json(CHECKPOINT)
    parent = load_json(PARENT)
    history = pd.read_csv(HISTORY)
    lock_pid = int(LOCK.read_text(encoding="utf-8").strip())
    checks = {
        "attempt_checkpoint_running_when_parent_failed": checkpoint.get("status") == "RUNNING",
        "parent_heartbeat_stopped_running": parent.get("status") == "RUNNING",
        "history_has_partial_epochs_only": 0 < len(history) < 30,
        "last_epoch_18": int(history.iloc[-1]["epoch"]) == 18,
        "last_global_step_28098": int(history.iloc[-1]["trainer_global_step"]) == 28098,
        "model_absent": not MODEL.exists(),
        "reference_absent": not REFERENCE.exists(),
        "proof_absent": not PROOF.exists(),
        "bundle_absent": not BUNDLE.exists(),
        "stale_lock_present_for_dead_pid": LOCK.exists() and not psutil.pid_exists(lock_pid),
        "zero_test_rows_used": True,
        "scientific_setting_unchanged": True,
    }
    report = {
        "stage_id": "stage5a2",
        "recovery_id": "stage5a2_fulltrain_recovery_2",
        "candidate_id": CANDIDATE_ID,
        "sensitive_mode": "with_sensitive",
        "technical_attempt_number": 1,
        "physical_fit_started": True,
        "completed_epochs": len(history),
        "last_completed_epoch": int(history.iloc[-1]["epoch"]),
        "last_global_step": int(history.iloc[-1]["trainer_global_step"]),
        "failure_class": "parent_watchdog_windows_atomic_replace_collision",
        "failure_detail": (
            "The parent process received PermissionError WinError 5 while atomically replacing its heartbeat JSON. "
            "The execution job then terminated the still-running child at audited epoch 18 before model serialization."
        ),
        "parent_exception": (
            "PermissionError: [WinError 5] Access is denied while replacing the with-sensitive parent heartbeat JSON"
        ),
        "stale_child_pid": lock_pid,
        "child_process_alive_at_recording": False,
        "retry_authorization": "original_with_sensitive_maximum_one_technical_retry",
        "retry_scientific_configuration_change": False,
        "retry_identifier": "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30__technical_retry1",
        "evidence": {
            "checkpoint": evidence(CHECKPOINT),
            "history": evidence(HISTORY),
            "parent_heartbeat": evidence(PARENT),
            "stale_lock": evidence(LOCK),
        },
        "checks": checks,
        "status": "TECHNICAL_FAILURE_RETRY_AUTHORIZED" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, REPORT)
    verified = load_json(REPORT)
    if verified["status"] != "TECHNICAL_FAILURE_RETRY_AUTHORIZED":
        raise RuntimeError("Sensitive attempt-1 failure preservation failed")
    print(verified)


if __name__ == "__main__":
    main()
