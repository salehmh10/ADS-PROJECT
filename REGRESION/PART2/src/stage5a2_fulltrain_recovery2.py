"""Model-first Stage 5A2 Full-Train recovery 2 and matched sensitive fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import torch

from stage5_deep_models import StrictFinalEpochRealMLPRegressor, seed_everything
from stage5_deep_preprocessing import RealMLPPreprocessor, TargetTransform
from stage5a2_deep_utils import (
    EXPECTED_FREEZE_SHA256,
    FREEZE,
    ROOT,
    SOURCE_WITH,
    SOURCE_WITHOUT,
    TEST_IDS,
    TRAIN_IDS,
    _load_source_rows,
    atomic_csv,
    digest_values,
    feature_lists,
    load_full_train,
    sha256_file,
    validate_freeze,
)
from stage5a2_fulltrain_recovery import (
    EpochAuditCallback,
    canonical_hash,
    frame_hash,
    model_weight_hash,
    preprocessor_contract,
    scientific_config_match,
)
from stage5a2_recovery_serialization import atomic_json, load_json


STAGE_NAME = "Stage 5A2 — Top-Two Deep Validation and Core Final Models"
RECOVERY_ID = "stage5a2_fulltrain_recovery_2"
WITHOUT_ID = "stage5a2__realmlp__full_train__without_sensitive__direct_no_refit_recovery2"
FAILED_WITH_ID = "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30"
WITH_ID = "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30__technical_retry1"
WITH_FAILURE_REPORT = ROOT / "artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json"
MODES = ("without_sensitive", "with_sensitive")
WINNER = ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json"
REPORTING_PREFLIGHT = ROOT / "artifacts/reports/stage5a2_recovery2_reporting_preflight.json"
RECOVERY1_BLOCKER = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json"
RECOVERY2_PREFLIGHT = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_2_preflight.json"
RECOVERY2_BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_fulltrain_recovery_2_protected_hashes_before.json"
RECOVERY1_BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_fulltrain_recovery_1_protected_hashes_before.json"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
FIXED_EPOCH = 30
TRAIN_ROWS = 399_788
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_id(mode: str) -> str:
    if mode == "without_sensitive":
        return WITHOUT_ID
    if mode == "with_sensitive":
        return WITH_ID
    raise ValueError(mode)


def paths(mode: str) -> dict[str, Path]:
    cid = candidate_id(mode)
    base = ROOT / "artifacts/checkpoints/stage5/deep_core/full_train/recovery2"
    return {
        "checkpoint": base / f"{cid}.json",
        "bundle_staging": base / f"{cid}.bundle_candidate.joblib",
        "model": ROOT / f"artifacts/models/deep/core_final/components/{cid}_model.joblib",
        "model_manifest": ROOT / f"artifacts/reports/{cid}_model_manifest.json",
        "bundle": ROOT / f"artifacts/models/deep/core_final/{cid}.joblib",
        "result": ROOT / f"artifacts/results/stage5/deep_core/full_train/{cid}.json",
        "history": ROOT / f"artifacts/results/stage5/deep_core/full_train/histories/{cid}_history.csv",
        "reference": ROOT / f"artifacts/predictions/stage5/deep_core/full_train_reference/{cid}.csv",
        "effective": ROOT / f"artifacts/reports/{cid}_effective_config.json",
        "proof": ROOT / f"artifacts/reports/{cid}_epoch_proof.json",
        "parent": ROOT / f"artifacts/reports/stage5a2_parent_{cid}.json",
        "log": ROOT / f"artifacts/reports/stage5a2_parent_{cid}.log",
        "staging_reload": ROOT / f"artifacts/reports/stage5a2_staging_reload_{cid}.json",
        "reload": ROOT / f"artifacts/reports/stage5a2_reload_{cid}.json",
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def resilient_parent_json(payload: dict[str, Any], path: Path) -> None:
    """Retry only transient Windows parent-heartbeat replacement collisions."""
    last_error = None
    for attempt in range(20):
        try:
            atomic_json(payload, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"Parent heartbeat JSON remained locked: {path}") from last_error


def atomic_model_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Save a trained model to a temporary file, validate it, then atomically promote it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"Recovery 2 model path already exists: {path}")
    try:
        joblib.dump(payload, temporary)
        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary trained-model file is missing or empty")
        reloaded = joblib.load(temporary)
        if not isinstance(reloaded, dict) or "model" not in reloaded or "preprocessor" not in reloaded:
            raise RuntimeError("Temporary trained-model payload is incomplete")
        temporary_size = temporary.stat().st_size
        temporary_hash = sha256_file(temporary)
        os.replace(temporary, path)
        if path.stat().st_size != temporary_size or sha256_file(path) != temporary_hash:
            raise RuntimeError("Atomically promoted trained-model file differs from its temporary file")
        return {"size_bytes": path.stat().st_size, "sha256": temporary_hash,
                "mtime_ns": path.stat().st_mtime_ns, "temporary_file_removed": not temporary.exists()}
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_bundle(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(payload, temporary)
        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary bundle is missing or empty")
        check = joblib.load(temporary)
        if not isinstance(check, dict) or "model" not in check or "preprocessor" not in check:
            raise RuntimeError("Temporary bundle is incomplete")
        size = temporary.stat().st_size
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError("Staged bundle differs after atomic promotion")
        return {"size_bytes": size, "sha256": digest, "mtime_ns": path.stat().st_mtime_ns,
                "temporary_file_removed": not temporary.exists()}
    finally:
        if temporary.exists():
            temporary.unlink()


def predict_payload(path: Path, raw: pd.DataFrame) -> np.ndarray:
    payload = joblib.load(path)
    features = payload["numerical_features"] + payload["categorical_features"]
    transformed = payload["preprocessor"].transform(raw.loc[:, features].copy())
    prediction = np.asarray(payload["model"].predict(transformed)).reshape(-1)
    return payload["target_transform"].inverse(prediction, standardized=True)


def check_baseline(baseline_path: Path) -> dict[str, Any]:
    baseline = load_json(baseline_path)
    mismatches = []
    for item in baseline["files"]:
        path = Path(item["path"])
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        if path.name == REGISTRY.name and path.stat().st_size >= int(item["size"]):
            actual = hashlib.sha256(path.read_bytes()[: int(item["size"])]).hexdigest()
            reason = "prior_registry_prefix_mismatch"
        else:
            actual = sha256_file(path)
            reason = "hash_mismatch"
        if actual != item["sha256"]:
            mismatches.append({"path": str(path), "reason": reason,
                               "expected": item["sha256"], "actual": actual})
    return {"checked_file_count": len(baseline["files"]), "mismatches": mismatches,
            "status": "PASS" if not mismatches else "FAIL"}


def recovery2_protected_paths() -> list[Path]:
    prior = load_json(RECOVERY1_BASELINE)
    protected = {Path(item["path"]) for item in prior["files"]}
    patterns = (
        "artifacts/reports/stage5a2_recovery2_reporting_preflight.json",
        "artifacts/checkpoints/stage5/deep_core/full_train/recovery2/reporting_preflight/**/*",
        "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json",
        "artifacts/reports/stage5a2_without_sensitive_recovery_epoch_proof.json",
        "artifacts/reports/stage5a2_without_sensitive_recovery_effective_config.json",
        "artifacts/reports/stage5a2_fulltrain_recovery_1_attempt/**/*",
        "artifacts/checkpoints/stage5/deep_core/full_train/recovery1/**/*",
        "artifacts/results/stage5/deep_core/full_train/histories/*recovery1_history.csv",
        "artifacts/reports/stage5a2_parent_*recovery1.*",
        "artifacts/reports/stage5a_verification.json",
    )
    for pattern in patterns:
        protected.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(protected, key=lambda item: str(item).lower())


def run_preflight() -> dict[str, Any]:
    if RECOVERY2_PREFLIGHT.exists():
        existing = load_json(RECOVERY2_PREFLIGHT)
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "REUSED", "path": relative(RECOVERY2_PREFLIGHT)}))
            return existing
    reporting = load_json(REPORTING_PREFLIGHT)
    blocker = load_json(RECOVERY1_BLOCKER)
    winner = load_json(WINNER)
    recovery1_recheck = check_baseline(RECOVERY1_BASELINE)
    blocker_hash_checks = {}
    for name, item in blocker["preserved_evidence"].items():
        artifact = ROOT / item["path"]
        blocker_hash_checks[name] = artifact.exists() and sha256_file(artifact) == item["sha256"]
    train_ids = pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    old_notebook = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"
    backup = ROOT / f"artifacts/backups/REGRESSION_PART5_DEEP_TABULAR_MODELS.stage5a2_recovery2_pre_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.ipynb"
    shutil.copy2(old_notebook, backup)
    if sha256_file(old_notebook) != sha256_file(backup):
        raise RuntimeError("Recovery 2 Notebook backup mismatch")
    entries = [{"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}
               for path in recovery2_protected_paths()]
    baseline = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "captured_at": utc_now(),
        "protected_file_count": len(entries), "files": entries,
        "registry_prefix_size": REGISTRY.stat().st_size, "registry_prefix_sha256": sha256_file(REGISTRY),
        "notebook_backup_path": relative(backup), "notebook_backup_sha256": sha256_file(backup),
        "status": "PASS",
    }
    atomic_json(baseline, RECOVERY2_BASELINE)
    checks = {
        "reporting_preflight_pass": reporting.get("status") == "PASS" and all(reporting["checks"].values()),
        "recovery1_blocker_valid": blocker.get("status") == "BLOCKED",
        "recovery1_evidence_hashes_match": all(blocker_hash_checks.values()),
        "recovery1_protected_recheck_pass": recovery1_recheck["status"] == "PASS",
        "freeze_unchanged": sha256_file(FREEZE) == EXPECTED_FREEZE_SHA256,
        "winner_frozen_realmlp": winner.get("status") == "FROZEN" and winner.get("family") == "realmlp",
        "winner_raw_epoch30": winner.get("target_mode") == "raw" and winner.get("best_epoch") == FIXED_EPOCH,
        "train_rows_exact": len(train_ids) == TRAIN_ROWS and len(np.unique(train_ids)) == TRAIN_ROWS,
        "train_test_overlap_zero": len(np.intersect1d(train_ids, test_ids)) == 0,
        "recovery2_without_model_absent": not paths("without_sensitive")["model"].exists(),
        "recovery2_without_bundle_absent": not paths("without_sensitive")["bundle"].exists(),
        "with_sensitive_not_started": not paths("with_sensitive")["checkpoint"].exists(),
        "notebook_backup_valid": sha256_file(old_notebook) == sha256_file(backup),
    }
    report = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "recorded_at": utc_now(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "reporting_preflight_path": relative(REPORTING_PREFLIGHT),
        "reporting_preflight_sha256": sha256_file(REPORTING_PREFLIGHT),
        "recovery1_blocker_sha256": sha256_file(RECOVERY1_BLOCKER),
        "recovery1_evidence_hash_checks": blocker_hash_checks,
        "recovery1_protected_recheck": recovery1_recheck,
        "train_rows": len(train_ids), "train_row_id_hash": digest_values(train_ids),
        "test_ids_used_for_overlap_audit_only": len(test_ids), "test_feature_rows_loaded": 0,
        "test_target_rows_loaded": 0, "stage4l_test_metrics_loaded": False,
        "protected_baseline_path": relative(RECOVERY2_BASELINE),
        "protected_file_count": len(entries), "notebook_backup_path": relative(backup),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, RECOVERY2_PREFLIGHT)
    verified = load_json(RECOVERY2_PREFLIGHT)
    print(json.dumps(verified, indent=2))
    if verified["status"] != "PASS":
        raise RuntimeError("Recovery 2 preflight failed")
    return verified


def build_model_payload(mode: str, model, preprocessor, target, numerical, categorical,
                        resolved, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_version": "stage5a2_recovery2_trained_model_v1",
        "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
        "candidate_id": candidate_id(mode), "family": "realmlp", "model_family": "realmlp",
        "sensitive_mode": mode, "target_mode": "raw", "numerical_features": numerical,
        "categorical_features": categorical, "preprocessor": preprocessor, "target_transform": target,
        "model": model, "resolved_config": resolved, "fixed_epoch": FIXED_EPOCH,
        "training_metadata": metadata,
    }


def finalize_reporting(mode: str) -> dict[str, Any]:
    """Resume all post-model reporting from the safely serialized model artifact."""
    p = paths(mode)
    payload = joblib.load(p["model"])
    metadata = payload["training_metadata"]
    control_checkpoint = load_json(p["checkpoint"])
    numerical, categorical = feature_lists(mode)
    source = SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH
    reference_ids = np.asarray(metadata["reference_row_ids"], dtype=np.int64)
    raw = _load_source_rows(source, reference_ids, ["loan_amount_000s", *numerical, *categorical])
    reference_raw = raw.loc[:, numerical + categorical].copy()
    prediction = predict_payload(p["model"], reference_raw)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Saved model produced non-finite reference predictions")
    reference = pd.DataFrame({
        "row_id": reference_ids,
        "y_true": raw["loan_amount_000s"].to_numpy(np.float64),
        "y_pred": prediction,
        "sensitive_mode": mode,
        "candidate_id": candidate_id(mode),
        "fixed_epoch": FIXED_EPOCH,
    })
    atomic_csv(reference, p["reference"])
    reference_saved_at_ns = time.time_ns()

    history = pd.read_csv(p["history"])
    atomic_csv(history.copy(), p["history"])
    history_finalized_at_ns = time.time_ns()

    effective = dict(metadata["effective_configuration"])
    effective.update({
        "fit_completed": True, "model_safely_serialized": True,
        "model_path": relative(p["model"]), "model_sha256": sha256_file(p["model"]),
        "reference_predictions_saved": True, "history_finalized": True,
        "status": "PASS",
    })
    atomic_json(effective, p["effective"])

    model_reloaded = joblib.load(p["model"])
    weight_hash_after = model_weight_hash(model_reloaded["model"])
    expected_steps = (TRAIN_ROWS // 256) * FIXED_EPOCH
    proof_checks = {
        "requested_epoch_30": metadata["requested_epoch"] == FIXED_EPOCH,
        "history_rows_30": len(history) == FIXED_EPOCH,
        "history_epochs_1_to_30": history["epoch"].tolist() == list(range(1, FIXED_EPOCH + 1)),
        "last_epoch_30": int(history.iloc[-1]["epoch"]) == FIXED_EPOCH,
        "global_step_consistent": int(history.iloc[-1]["trainer_global_step"]) == expected_steps,
        "early_stopping_disabled": metadata["internal_use_early_stopping"] is False,
        "restoration_disabled": metadata["internal_use_best_epoch"] is False and metadata["restoration_callback_created"] is False,
        "n_cv_1": metadata["resolved_config"]["n_cv"] == 1,
        "n_refit_0": metadata["resolved_config"]["n_refit"] == 0 and metadata["no_refit_interface"],
        "val_fraction_0": float(metadata["resolved_config"]["val_fraction"]) == 0.0,
        "no_external_validation": metadata["external_validation_rows"] == 0,
        "all_train_rows_internal": metadata["internal_training_membership"]["unique_train_index_count"] == TRAIN_ROWS,
        "all_saved_train_rows": metadata["training_rows"] == TRAIN_ROWS,
        "zero_test_rows": metadata["test_rows"] == 0,
        "model_serialized_after_epoch30": p["model"].stat().st_mtime_ns >= metadata["fit_completed_ns"],
        "model_sha256_valid": len(sha256_file(p["model"])) == 64,
        "reference_predictions_finite": bool(np.isfinite(prediction).all()),
        "source_dataframe_unchanged": metadata["source_frame_before_sha256"] == metadata["source_frame_after_sha256"],
        "effective_configuration_match": metadata["scientific_configuration_match"],
        "weight_hash_matches_serialization": metadata["weight_sha256_before_serialization"] == weight_hash_after,
        "no_previous_estimator_copied": metadata["model_origin"] == f"physical_fit_{candidate_id(mode)}",
        "model_temp_removed": control_checkpoint["model_atomic_save"]["temporary_file_removed"],
    }
    proof = {
        "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
        "candidate_id": candidate_id(mode), "sensitive_mode": mode, "target_mode": "raw",
        "requested_epoch": FIXED_EPOCH, "completed_epoch": int(history.iloc[-1]["epoch"]),
        "training_history_length": len(history), "serialized_artifact_epoch": FIXED_EPOCH,
        "final_global_step": int(history.iloc[-1]["trainer_global_step"]),
        "expected_global_step": expected_steps,
        "pytabkit_progress_epoch": metadata["pytabkit_progress_epoch"],
        "pytabkit_progress_epoch_float": metadata["pytabkit_progress_epoch_float"],
        "epoch_zero_interpretation": (
            "Installed PyTabKit increments progress.epoch only at validation end. The direct no-validation path "
            "therefore reports 0; the 30-row train-epoch audit, epoch_float, global steps, serialized weights, "
            "and clean reload are the controlling proof."
        ),
        "early_stopping": False, "best_checkpoint_restoration": False,
        "n_cv": 1, "n_refit": 0, "val_fraction": 0.0, "external_validation_rows": 0,
        "training_rows": TRAIN_ROWS, "test_rows": 0,
        "model_path": relative(p["model"]), "model_sha256": sha256_file(p["model"]),
        "model_size_bytes": p["model"].stat().st_size,
        "model_mtime_ns": p["model"].stat().st_mtime_ns,
        "fit_completed_ns": metadata["fit_completed_ns"],
        "weight_sha256_before_serialization": metadata["weight_sha256_before_serialization"],
        "weight_sha256_after_serialization": weight_hash_after,
        "reference_prediction_path": relative(p["reference"]),
        "reference_prediction_sha256": sha256_file(p["reference"]),
        "history_path": relative(p["history"]), "history_sha256": sha256_file(p["history"]),
        "effective_config_path": relative(p["effective"]), "effective_config_sha256": sha256_file(p["effective"]),
        "reference_saved_at_ns": reference_saved_at_ns,
        "history_finalized_at_ns": history_finalized_at_ns,
        "internal_training_membership": metadata["internal_training_membership"],
        "checks": proof_checks,
        "status": "PASS" if all(proof_checks.values()) else "FAIL",
    }
    atomic_json(proof, p["proof"])
    if load_json(p["proof"])["status"] != "PASS":
        raise RuntimeError("Recovery 2 epoch/model proof failed")

    bundle = {
        "bundle_version": "stage5a2_core_final_recovery2_v1",
        "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
        "candidate_id": candidate_id(mode), "family": "realmlp", "model_family": "realmlp",
        "sensitive_mode": mode,
        "feature_schema": "deep_core_v1" if mode == "without_sensitive" else "deep_core_v1_with_validated_sensitive_sources",
        "base_feature_schema": "deep_core_v1", "numerical_features": numerical,
        "categorical_features": categorical, "numeric_imputer": "training-fit medians",
        "missing_indicators": "official PyTabKit missing handling after the saved preprocessor",
        "category_vocabularies": "saved Train-only vocabularies in RealMLPPreprocessor and official encoder",
        "missing_token": payload["preprocessor"].unknown_token,
        "unknown_token": payload["preprocessor"].unknown_token,
        "rare_token": payload["preprocessor"].unknown_token,
        "preprocessor": payload["preprocessor"], "target_mode": "raw",
        "target_transform": payload["target_transform"], "model": payload["model"],
        "architecture": {"implementation": "pytabkit.RealMLP_TD_Regressor",
                         "resolved_config": payload["resolved_config"]},
        "effective_configuration_path": relative(p["effective"]),
        "effective_configuration_sha256": sha256_file(p["effective"]),
        "fixed_epoch": FIXED_EPOCH, "epoch_proof_path": relative(p["proof"]),
        "epoch_proof_sha256": sha256_file(p["proof"]),
        "model_component_path": relative(p["model"]), "model_component_sha256": sha256_file(p["model"]),
        "weight_sha256": weight_hash_after,
        "batch_size_policy": {"train": 256, "predict": 1024},
        "seed": SEED, "device": "cpu", "precision": "float32", "cpu_loadable": True,
        "package_versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "sklearn": sklearn.__version__, "joblib": joblib.__version__,
                             "pytabkit": importlib.metadata.version("pytabkit")},
        "source_path": relative(source), "source_sha256": sha256_file(source),
        "train_ids_path": relative(TRAIN_IDS), "train_ids_sha256": sha256_file(TRAIN_IDS),
        "test_ids_sha256_for_overlap_audit_only": sha256_file(TEST_IDS),
        "training_row_count": TRAIN_ROWS, "training_row_id_hash": metadata["training_row_id_hash"],
        "test_rows": 0, "training_history_path": relative(p["history"]),
        "training_history_sha256": sha256_file(p["history"]),
        "reference_row_ids": reference_ids.tolist(),
        "reference_prediction_path": relative(p["reference"]),
        "reference_prediction_sha256": sha256_file(p["reference"]),
        "test_or_stage4l_test_evidence_used": False,
    }
    bundle_save = atomic_bundle(bundle, p["bundle_staging"])
    model_manifest = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": candidate_id(mode),
        "sensitive_mode": mode, "model_path": relative(p["model"]),
        "model_sha256": sha256_file(p["model"]), "model_size_bytes": p["model"].stat().st_size,
        "bundle_staging_path": relative(p["bundle_staging"]),
        "bundle_staging_sha256": bundle_save["sha256"], "bundle_staging_size_bytes": bundle_save["size_bytes"],
        "reference_prediction_path": relative(p["reference"]),
        "reference_prediction_sha256": sha256_file(p["reference"]),
        "history_path": relative(p["history"]), "history_sha256": sha256_file(p["history"]),
        "epoch_proof_path": relative(p["proof"]), "epoch_proof_sha256": sha256_file(p["proof"]),
        "status": "FIT_COMPLETE_PENDING_CLEAN_RELOAD",
    }
    atomic_json(model_manifest, p["model_manifest"])
    result = {
        **model_manifest,
        "official_stage_name": STAGE_NAME, "model_family": "realmlp", "target_mode": "raw",
        "fixed_epoch": FIXED_EPOCH, "requested_epoch": FIXED_EPOCH, "completed_epoch": FIXED_EPOCH,
        "artifact_epoch": FIXED_EPOCH, "training_rows": TRAIN_ROWS, "validation_rows": 0, "test_rows": 0,
        "n_cv": 1, "n_refit": 0, "val_fraction": 0.0, "early_stopping": False,
        "best_checkpoint_restoration": False,
        "fit_time_seconds": metadata["fit_time_seconds"],
        "physical_fit_count_for_candidate": 1, "retry_count": metadata["retry_count"],
        "training_row_id_hash": metadata["training_row_id_hash"],
        "status": "FIT_COMPLETE_PENDING_CLEAN_RELOAD",
    }
    atomic_json(result, p["result"])
    checkpoint = load_json(p["checkpoint"])
    checkpoint.update(result)
    atomic_json(checkpoint, p["checkpoint"])
    return result


def run_fit(mode: str) -> dict[str, Any]:
    if load_json(REPORTING_PREFLIGHT).get("status") != "PASS":
        raise RuntimeError("Mandatory reporting serializer preflight is not PASS")
    if load_json(RECOVERY2_PREFLIGHT).get("status") != "PASS":
        raise RuntimeError("Recovery 2 preflight is not PASS")
    if mode == "with_sensitive":
        prior = paths("without_sensitive")["checkpoint"]
        if not prior.exists() or load_json(prior).get("status") != "PASS":
            raise RuntimeError("With-sensitive fit is blocked until recovery-2 non-sensitive promotion PASS")
        if not WITH_FAILURE_REPORT.exists() or load_json(WITH_FAILURE_REPORT).get("status") != "TECHNICAL_FAILURE_RETRY_AUTHORIZED":
            raise RuntimeError("Sensitive technical retry requires the preserved attempt-1 failure report")
    p = paths(mode)
    if p["checkpoint"].exists():
        existing = load_json(p["checkpoint"])
        if existing.get("status") in {"MODEL_SAVED_REPORTING_PENDING", "MODEL_SAVED_REPORTING_FAILED",
                                      "FIT_COMPLETE_PENDING_CLEAN_RELOAD", "PROMOTED_PENDING_FINAL_RELOAD", "PASS"}:
            if p["model"].exists() and existing.get("status") in {"MODEL_SAVED_REPORTING_PENDING", "MODEL_SAVED_REPORTING_FAILED"}:
                return finalize_reporting(mode)
            print(json.dumps({"status": "REUSED", "candidate_id": candidate_id(mode)}))
            return existing
        raise RuntimeError(f"Physical fit already recorded for {mode}: {existing.get('status')}")
    for protected_output in (p["model"], p["bundle"], p["bundle_staging"], p["history"], p["reference"]):
        if protected_output.exists():
            raise RuntimeError(f"Unexpected recovery-2 output exists before fit: {protected_output}")
    lock = ROOT / "artifacts/checkpoints/stage5/deep_core/stage5a2_heavy_fit.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError("Another heavy fit is active") from exc
    physical_fit_started = False
    model_saved = False
    model_save = None
    started = time.perf_counter()
    checkpoint = {"stage_id": "stage5a2", "recovery_id": RECOVERY_ID,
                  "candidate_id": candidate_id(mode), "sensitive_mode": mode,
                  "physical_fit_started": False, "status": "PREPARING", "started_at": utc_now()}
    atomic_json(checkpoint, p["checkpoint"])
    try:
        X_train, y_train, train_ids = load_full_train(mode)
        numerical, categorical = feature_lists(mode)
        source = SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH
        source_before = frame_hash(X_train)
        preprocessor = RealMLPPreprocessor(numerical_features=numerical, categorical_features=categorical).fit(X_train)
        transformed = preprocessor.transform(X_train)
        target = TargetTransform("raw").fit(y_train)
        preprocess = preprocessor_contract(preprocessor)
        model = StrictFinalEpochRealMLPRegressor(
            device="cpu", random_state=SEED, n_cv=1, n_refit=0, n_repeats=1,
            val_fraction=0.0, n_threads=4, verbosity=0, n_epochs=FIXED_EPOCH,
            batch_size=256, predict_batch_size=1024, train_metric_name="mae",
            val_metric_name="mae", use_early_stopping=False, p_drop=0.15,
        )
        resolved = model.get_config()
        winner = load_json(WINNER)
        config_match, config_differences = scientific_config_match(resolved, winner["architecture"]["resolved_config"])
        effective_checks = {
            "scientific_configuration_match": config_match,
            "n_cv_1": resolved.get("n_cv") == 1,
            "n_refit_0": resolved.get("n_refit") == 0,
            "val_fraction_0": float(resolved.get("val_fraction", -1)) == 0.0,
            "fixed_epoch_30": resolved.get("n_epochs") == FIXED_EPOCH,
            "use_best_epoch_false": resolved.get("use_best_epoch") is False,
            "early_stopping_false": resolved.get("use_early_stopping") is False,
            "seed_42": resolved.get("random_state") == SEED,
            "cpu": resolved.get("device") == "cpu",
            "batch_policy": resolved.get("batch_size") == 256 and resolved.get("predict_batch_size") == 1024,
            "p_drop_frozen": float(resolved.get("p_drop", -1)) == 0.15,
            "all_train_rows_loaded": len(train_ids) == TRAIN_ROWS and len(np.unique(train_ids)) == TRAIN_ROWS,
        }
        effective = {
            "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
            "candidate_id": candidate_id(mode), "sensitive_mode": mode, "target_mode": "raw",
            "pytabkit_version": importlib.metadata.version("pytabkit"),
            "official_estimator_class": model.__class__.__module__ + "." + model.__class__.__name__,
            "constructor_arguments": {"device": "cpu", "random_state": SEED, "n_cv": 1, "n_refit": 0,
                                      "n_repeats": 1, "val_fraction": 0.0, "n_threads": 4, "verbosity": 0,
                                      "n_epochs": FIXED_EPOCH, "batch_size": 256, "predict_batch_size": 1024,
                                      "train_metric_name": "mae", "val_metric_name": "mae",
                                      "use_early_stopping": False, "p_drop": 0.15},
            "resolved_config": resolved,
            "scientific_configuration_reference": winner["architecture"]["resolved_config"],
            "scientific_configuration_differences": config_differences,
            "early_stopping": False, "best_checkpoint_restoration": False,
            "seed": SEED, "device": "cpu", "precision": "float32",
            "preprocessing_contract": preprocess,
            "transformed_training_frame_sha256": frame_hash(transformed),
            "train_row_id_hash": digest_values(train_ids), "train_row_count": len(train_ids),
            "test_row_count": 0, "feature_schema_hash": canonical_hash({"numerical": numerical, "categorical": categorical}),
            "source_path": relative(source), "source_sha256": sha256_file(source),
            "train_ids_sha256": sha256_file(TRAIN_IDS),
            "test_ids_sha256_for_overlap_audit_only": sha256_file(TEST_IDS),
            "reporting_preflight_path": relative(REPORTING_PREFLIGHT),
            "reporting_preflight_sha256": sha256_file(REPORTING_PREFLIGHT),
            "checks": effective_checks, "fit_started": False,
            "saved_before_model_fit": True, "status": "PREFIT_PASS" if all(effective_checks.values()) else "FAIL",
        }
        if effective["status"] != "PREFIT_PASS":
            raise RuntimeError("Recovery 2 effective configuration failed")

        audit = EpochAuditCallback.create(p["history"], TRAIN_ROWS)
        from pytabkit.models.training.lightning_modules import TabNNModule
        original_callbacks = TabNNModule.create_callbacks

        def audited_callbacks(module_self):
            callbacks = original_callbacks(module_self)
            audit.original_callback_classes = [item.__class__.__module__ + "." + item.__class__.__name__ for item in callbacks]
            callbacks.append(audit)
            return callbacks

        TabNNModule.create_callbacks = audited_callbacks
        physical_fit_started = True
        checkpoint.update({"physical_fit_started": True, "physical_fit_started_at": utc_now(), "status": "RUNNING"})
        atomic_json(checkpoint, p["checkpoint"])
        seed_everything(SEED)
        fit_started_ns = time.time_ns()
        fit_start = time.perf_counter()
        try:
            model.fit(transformed, target.transform(y_train), cat_col_names=categorical, time_to_fit_in_seconds=7150)
        finally:
            TabNNModule.create_callbacks = original_callbacks
        fit_seconds = time.perf_counter() - fit_start
        fit_completed_ns = time.time_ns()
        module = model.cv_alg_interface_.model
        history = pd.read_csv(p["history"])
        callback_classes = audit.original_callback_classes
        restoration_callback_created = any(name.endswith("ModelCheckpointCallback") or name.endswith("StopAtEpochsCallback")
                                           for name in callback_classes)
        encoder = model.x_converter_.cat_tf.named_transformers_["categorical"]
        official_contract = {}
        for index, column in enumerate(categorical):
            expected = set(transformed[column].astype(str).unique())
            learned = {str(value) for value in encoder.categories_[index]}
            official_contract[column] = {
                "train_only_cardinality": len(expected), "official_encoder_cardinality": len(learned),
                "unexpected_official_categories": sorted(learned - expected),
                "missing_train_categories": sorted(expected - learned),
            }
        encoder_pass = not any(item["unexpected_official_categories"] or item["missing_train_categories"]
                               for item in official_contract.values())
        source_after = frame_hash(X_train)
        weight_before = model_weight_hash(model)
        model.to("cpu")
        reference_ids = train_ids[:128]
        metadata = {
            "model_origin": f"physical_fit_{candidate_id(mode)}",
            "requested_epoch": FIXED_EPOCH, "fit_started_ns": fit_started_ns,
            "fit_completed_ns": fit_completed_ns, "fit_completed_at": utc_now(),
            "fit_time_seconds": fit_seconds, "retry_count": 1 if mode == "with_sensitive" else 0,
            "technical_attempt_number": 2 if mode == "with_sensitive" else 1,
            "prior_technical_failure_path": relative(WITH_FAILURE_REPORT) if mode == "with_sensitive" else None,
            "resolved_config": resolved,
            "scientific_configuration_match": config_match,
            "scientific_configuration_differences": config_differences,
            "pytabkit_progress_epoch": int(module.progress.epoch),
            "pytabkit_progress_epoch_float": float(module.progress.epoch_float),
            "pytabkit_progress_max_epoch": int(module.progress.max_epochs),
            "official_stop_epoch": int(model.fit_params_["stop_epoch"]["mae"]),
            "internal_use_best_epoch": module.creator.config.get("use_best_epoch"),
            "internal_use_early_stopping": module.creator.config.get("use_early_stopping"),
            "restoration_callback_created": restoration_callback_created,
            "original_callback_classes": callback_classes,
            "no_refit_interface": not hasattr(model, "refit_alg_interface_"),
            "direct_interface_deployed": model.alg_interface_ is model.cv_alg_interface_,
            "external_validation_rows": 0,
            "internal_training_membership": audit.training_membership,
            "training_rows": TRAIN_ROWS, "test_rows": 0,
            "training_row_id_hash": digest_values(train_ids),
            "source_frame_before_sha256": source_before,
            "source_frame_after_sha256": source_after,
            "official_encoder_matches_train_only": encoder_pass,
            "official_encoder_contract": official_contract,
            "weight_sha256_before_serialization": weight_before,
            "history_path": relative(p["history"]), "history_sha256_at_fit_end": sha256_file(p["history"]),
            "effective_config_path": relative(p["effective"]), "reference_row_ids": reference_ids.tolist(),
            "effective_configuration": effective,
        }
        trained_estimator_exists = (
            getattr(model, "cv_alg_interface_", None) is not None
            and getattr(model, "alg_interface_", None) is not None
            and isinstance(weight_before, str)
            and len(weight_before) == 64
        )
        if not trained_estimator_exists:
            raise RuntimeError("The epoch-30 trained estimator could not be validated before serialization")
        model_payload = build_model_payload(mode, model, preprocessor, target, numerical, categorical, resolved, metadata)
        model_save = atomic_model_payload(model_payload, p["model"])
        model_saved = True
        if model_save["mtime_ns"] < fit_completed_ns:
            raise RuntimeError("Serialized model modification time does not follow epoch-30 completion")
        reloaded_payload = joblib.load(p["model"])
        if model_weight_hash(reloaded_payload["model"]) != weight_before:
            raise RuntimeError("Serialized model weight hash mismatch")
        checkpoint.update({
            "status": "MODEL_SAVED_REPORTING_PENDING", "model_safely_serialized": True,
            "model_path": relative(p["model"]), "model_sha256": sha256_file(p["model"]),
            "model_size_bytes": p["model"].stat().st_size, "fit_completed_ns": fit_completed_ns,
            "model_atomic_save": model_save,
        })
        atomic_json(checkpoint, p["checkpoint"])
        result = finalize_reporting(mode)
        print(json.dumps({"status": result["status"], "candidate_id": candidate_id(mode),
                          "fit_time_seconds": fit_seconds, "model_path": relative(p["model"])}))
        return result
    except Exception as exc:
        status = "MODEL_SAVED_REPORTING_FAILED" if model_saved else ("FAIL" if physical_fit_started else "PREFIT_FAIL")
        failure = {
            "stage_id": "stage5a2", "recovery_id": RECOVERY_ID,
            "candidate_id": candidate_id(mode), "sensitive_mode": mode,
            "physical_fit_started": physical_fit_started, "model_safely_serialized": model_saved,
            "model_path": relative(p["model"]) if model_saved else None,
            "model_sha256": sha256_file(p["model"]) if model_saved else None,
            "model_atomic_save": model_save,
            "status": status, "error": repr(exc), "traceback": traceback.format_exc(),
            "worker_elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(failure, p["checkpoint"])
        raise
    finally:
        if lock.exists():
            lock.unlink()


def parent_fit(mode: str) -> dict[str, Any]:
    p = paths(mode)
    if p["checkpoint"].exists():
        existing = load_json(p["checkpoint"])
        if existing.get("status") in {"MODEL_SAVED_REPORTING_PENDING", "MODEL_SAVED_REPORTING_FAILED"} and p["model"].exists():
            result = finalize_reporting(mode)
            report = {"stage_id": "stage5a2", "candidate_id": candidate_id(mode),
                      "status": "REPORTING_RESUMED", "result_status": result["status"]}
            resilient_parent_json(report, p["parent"])
            print(json.dumps(report))
            return report
        if existing.get("status") in {"FIT_COMPLETE_PENDING_CLEAN_RELOAD", "PROMOTED_PENDING_FINAL_RELOAD", "PASS"}:
            report = {"stage_id": "stage5a2", "candidate_id": candidate_id(mode), "status": "REUSED"}
            resilient_parent_json(report, p["parent"])
            print(json.dumps(report))
            return report
        raise RuntimeError(f"Physical fit already recorded for {mode}; refusing another attempt")
    command = [sys.executable, str(Path(__file__).resolve()), "--fit", mode]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.perf_counter()
    peak = 0.0
    timed_out = False
    while process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            peak = max(peak, sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)]) / 1024**2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        elapsed = time.perf_counter() - started
        resilient_parent_json({"stage_id": "stage5a2", "recovery_id": RECOVERY_ID,
                               "candidate_id": candidate_id(mode), "status": "RUNNING",
                               "elapsed_seconds": elapsed, "peak_process_tree_ram_mib": peak,
                               "timeout_seconds": 7200}, p["parent"])
        if elapsed > 7200:
            timed_out = True
            process.kill()
            break
        time.sleep(2)
    stdout, stderr = process.communicate()
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    p["log"].write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    checkpoint_status = load_json(p["checkpoint"]).get("status") if p["checkpoint"].exists() else "MISSING"
    accepted = checkpoint_status == "FIT_COMPLETE_PENDING_CLEAN_RELOAD"
    report = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": candidate_id(mode),
        "command": command, "elapsed_seconds": time.perf_counter() - started,
        "timeout_seconds": 7200, "timed_out": timed_out, "return_code": process.returncode,
        "peak_process_tree_ram_mib": peak, "checkpoint_status": checkpoint_status,
        "model_safely_serialized": p["model"].exists(),
        "model_sha256": sha256_file(p["model"]) if p["model"].exists() else None,
        "log_path": relative(p["log"]),
        "status": "PASS" if not timed_out and process.returncode == 0 and accepted else "FAIL",
    }
    resilient_parent_json(report, p["parent"])
    if report["status"] == "PASS":
        result = load_json(p["result"])
        result.update({"peak_process_tree_ram_mib": peak, "parent_elapsed_seconds": report["elapsed_seconds"]})
        atomic_json(result, p["result"])
        checkpoint = load_json(p["checkpoint"])
        checkpoint.update(result)
        atomic_json(checkpoint, p["checkpoint"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(f"Recovery 2 physical fit failed for {mode}")
    return report


def clean_verify(mode: str, scope: str) -> dict[str, Any]:
    p = paths(mode)
    bundle_path = p["bundle_staging"] if scope == "staging" else p["bundle"]
    report_path = p["staging_reload"] if scope == "staging" else p["reload"]
    reference = pd.read_csv(p["reference"])
    row_ids = reference["row_id"].to_numpy(np.int64)
    numerical, categorical = feature_lists(mode)
    source = SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH
    raw = _load_source_rows(source, row_ids, [*numerical, *categorical])
    before = frame_hash(raw)
    prediction = predict_payload(bundle_path, raw)
    after = frame_hash(raw)
    expected = reference["y_pred"].to_numpy(np.float64)
    probe = raw.iloc[:4].copy()
    probe.iloc[0, probe.columns.get_loc(categorical[0])] = "__STAGE5A2_RECOVERY2_UNSEEN__"
    probe.iloc[1, probe.columns.get_loc(numerical[0])] = np.nan
    probe_prediction = predict_payload(bundle_path, probe)
    bundle = joblib.load(bundle_path)
    proof = load_json(p["proof"])
    effective = load_json(p["effective"])
    weight_hash = model_weight_hash(bundle["model"])
    difference = np.abs(prediction - expected)
    expected_features = numerical + categorical
    checks = {
        "clean_process": True,
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "correct_output_length": len(prediction) == len(reference),
        "source_dataframe_unchanged": before == after,
        "unknown_category_handling": bool(np.isfinite(probe_prediction[0])),
        "missing_value_handling": bool(np.isfinite(probe_prediction[1])),
        "target_inverse_transformation": bundle["target_mode"] == "raw" and bundle["target_transform"].mode == "raw",
        "reference_predictions_match": bool(np.allclose(prediction, expected, rtol=1e-6, atol=1e-6)),
        "cpu_inference": bundle["device"] == "cpu" and bundle["precision"] == "float32",
        "no_preprocessing_refit": True,
        "fixed_epoch_metadata": bundle["fixed_epoch"] == proof["requested_epoch"] == FIXED_EPOCH,
        "effective_configuration_pass": effective.get("status") == "PASS" and all(effective["checks"].values()),
        "sensitive_mode_contract": bundle["sensitive_mode"] == mode and bundle["numerical_features"] + bundle["categorical_features"] == expected_features,
        "weight_hash_match": weight_hash == proof["weight_sha256_after_serialization"] == bundle["weight_sha256"],
        "bundle_hash_match": sha256_file(bundle_path) == (
            load_json(p["model_manifest"])["bundle_staging_sha256"] if scope == "staging"
            else load_json(p["result"])["bundle_sha256"]),
        "test_rows_zero": bundle["test_rows"] == 0,
    }
    report = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID,
        "candidate_id": candidate_id(mode), "sensitive_mode": mode, "scope": scope,
        "bundle_path": relative(bundle_path), "bundle_sha256": sha256_file(bundle_path),
        "prediction_count": len(prediction), "maximum_absolute_difference": float(difference.max()),
        "weight_sha256": weight_hash, "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, report_path)
    print(json.dumps(load_json(report_path), indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(f"Clean-process {scope} reload failed for {mode}")
    if scope == "final":
        checkpoint = load_json(p["checkpoint"])
        checkpoint.update({"final_reload_status": "PASS", "status": "PASS"})
        atomic_json(checkpoint, p["checkpoint"])
        result = load_json(p["result"])
        result.update({"final_reload_status": "PASS", "status": "PASS"})
        atomic_json(result, p["result"])
        manifest = load_json(p["model_manifest"])
        manifest.update({"final_reload_status": "PASS", "status": "PASS"})
        atomic_json(manifest, p["model_manifest"])
    return report


def promote(mode: str) -> dict[str, Any]:
    p = paths(mode)
    staging_reload = load_json(p["staging_reload"])
    checkpoint = load_json(p["checkpoint"])
    if staging_reload.get("status") != "PASS" or checkpoint.get("status") != "FIT_COMPLETE_PENDING_CLEAN_RELOAD":
        raise RuntimeError("Staging reload and fit proof must pass before bundle promotion")
    if sha256_file(p["bundle_staging"]) != staging_reload["bundle_sha256"]:
        raise RuntimeError("Staging bundle changed after clean reload")
    p["bundle"].parent.mkdir(parents=True, exist_ok=True)
    if p["bundle"].exists():
        raise RuntimeError("Final bundle already exists before recovery-2 promotion")
    os.replace(p["bundle_staging"], p["bundle"])
    digest = sha256_file(p["bundle"])
    result = load_json(p["result"])
    result.update({
        "bundle_path": relative(p["bundle"]), "bundle_sha256": digest,
        "model_path": relative(p["model"]), "model_sha256": sha256_file(p["model"]),
        "model_size_bytes": p["model"].stat().st_size, "bundle_size_bytes": p["bundle"].stat().st_size,
        "staging_reload_status": "PASS", "status": "PROMOTED_PENDING_FINAL_RELOAD",
    })
    atomic_json(result, p["result"])
    checkpoint.update(result)
    atomic_json(checkpoint, p["checkpoint"])
    manifest = load_json(p["model_manifest"])
    manifest.update({"bundle_path": relative(p["bundle"]), "bundle_sha256": digest,
                     "status": "PROMOTED_PENDING_FINAL_RELOAD"})
    atomic_json(manifest, p["model_manifest"])
    print(json.dumps({"status": "PROMOTED_PENDING_FINAL_RELOAD", "candidate_id": candidate_id(mode),
                      "bundle_path": relative(p["bundle"]), "bundle_sha256": digest}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--fit", choices=MODES)
    parser.add_argument("--parent-fit", choices=MODES)
    parser.add_argument("--resume-reporting", choices=MODES)
    parser.add_argument("--clean-verify", choices=MODES)
    parser.add_argument("--scope", choices=("staging", "final"), default="final")
    parser.add_argument("--promote", choices=MODES)
    args = parser.parse_args()
    if args.preflight:
        run_preflight()
    elif args.fit:
        run_fit(args.fit)
    elif args.parent_fit:
        parent_fit(args.parent_fit)
    elif args.resume_reporting:
        finalize_reporting(args.resume_reporting)
    elif args.clean_verify:
        clean_verify(args.clean_verify, args.scope)
    elif args.promote:
        promote(args.promote)
    else:
        parser.error("Choose a recovery-2 action")


if __name__ == "__main__":
    main()
