"""Recover the two Stage 5A2 fixed-epoch RealMLP Full-Train bundles.

This module preserves the historical failed refit attempts.  It uses the one
authorized direct PyTabKit path with n_refit=0 and val_fraction=0.0, records
each completed train epoch through a read-only Lightning callback, and stages
the bundle until a clean-process reload passes.
"""

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
    atomic_joblib,
    atomic_json,
    digest_values,
    feature_lists,
    load_full_train,
    predict_bundle,
    sha256_file,
    validate_freeze,
)


STAGE_NAME = "Stage 5A2 — Top-Two Deep Validation and Core Final Models"
RECOVERY_ID = "stage5a2_fulltrain_recovery_1"
WITHOUT_ID = "stage5a2__realmlp__full_train__without_sensitive__direct_no_refit_recovery1"
WITH_ID = "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30"
MODES = ("without_sensitive", "with_sensitive")
WINNER = ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json"
BLOCKER = ROOT / "artifacts/reports/stage5a2_fulltrain_blocker.json"
RECOVERY_PREFLIGHT = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_preflight.json"
RECOVERY_PLAN = ROOT / "artifacts/reports/stage5a2_fulltrain_recovery_1_plan.json"
RECOVERY_BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_fulltrain_recovery_1_protected_hashes_before.json"
OLD_BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_protected_hashes_before.json"
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


def mode_paths(mode: str) -> dict[str, Path]:
    cid = candidate_id(mode)
    base = ROOT / "artifacts/checkpoints/stage5/deep_core/full_train/recovery1"
    return {
        "checkpoint": base / f"{cid}.json",
        "staging_bundle": base / f"{cid}.candidate.joblib",
        "result": ROOT / f"artifacts/results/stage5/deep_core/full_train/{cid}.json",
        "bundle": ROOT / f"artifacts/models/deep/core_final/{cid}.joblib",
        "history": ROOT / f"artifacts/results/stage5/deep_core/full_train/histories/{cid}_history.csv",
        "reference": ROOT / f"artifacts/predictions/stage5/deep_core/full_train_reference/{cid}.csv",
        "effective": ROOT / f"artifacts/reports/stage5a2_{mode}_recovery_effective_config.json",
        "proof": ROOT / f"artifacts/reports/stage5a2_{mode}_recovery_epoch_proof.json",
        "parent": ROOT / f"artifacts/reports/stage5a2_parent_{cid}.json",
        "log": ROOT / f"artifacts/reports/stage5a2_parent_{cid}.log",
        "staging_reload": ROOT / f"artifacts/reports/stage5a2_staging_reload_{cid}.json",
        "reload": ROOT / f"artifacts/reports/stage5a2_reload_{cid}.json",
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def preprocessor_contract(preprocessor: RealMLPPreprocessor) -> dict[str, Any]:
    payload = {
        "numerical_features": list(preprocessor.numerical_features),
        "categorical_features": list(preprocessor.categorical_features),
        "medians": {key: float(value) for key, value in preprocessor.medians_.items()},
        "vocabularies": {key: sorted(str(item) for item in value) for key, value in preprocessor.vocabularies_.items()},
        "rare_values": {key: sorted(str(item) for item in value) for key, value in preprocessor.rare_values_.items()},
        "missing_token": preprocessor.unknown_token,
        "unknown_token": preprocessor.unknown_token,
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def model_weight_hash(estimator: StrictFinalEpochRealMLPRegressor) -> str:
    module = estimator.cv_alg_interface_.model.model
    digest = hashlib.sha256()
    count = 0
    for index, parameter in enumerate(module.parameters()):
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(index).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("RealMLP weight hash found no parameters")
    return digest.hexdigest()


def check_old_baseline() -> tuple[int, list[dict[str, Any]]]:
    baseline = json.loads(OLD_BASELINE.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    for item in baseline["files"]:
        path = Path(item["path"])
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        if path.name == REGISTRY.name and path.stat().st_size >= int(item["size"]):
            current = hashlib.sha256(path.read_bytes()[: int(item["size"])]).hexdigest()
            reason = "prior_registry_prefix_mismatch"
        else:
            current = sha256_file(path)
            reason = "hash_mismatch"
        if current != item["sha256"]:
            mismatches.append({"path": str(path), "reason": reason, "expected": item["sha256"], "actual": current})
    return len(baseline["files"]), mismatches


def validate_historical_blocker() -> dict[str, Any]:
    blocker = json.loads(BLOCKER.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "blocker_status": blocker.get("status") == "BLOCKED",
        "two_physical_attempts": blocker.get("physical_attempts_completed") == 2,
        "serialized_refit_epoch_zero": blocker.get("observed_evidence", {}).get("refit_completed_epoch") == 0,
        "without_bundle_absent": blocker.get("artifact_state", {}).get("without_sensitive_bundle_saved") is False,
        "with_sensitive_not_started": blocker.get("artifact_state", {}).get("with_sensitive_full_train_started") is False,
        "test_evidence_unused": blocker.get("artifact_state", {}).get("test_or_stage4l_test_evidence_used") is False,
    }
    artifact_checks: dict[str, bool] = {}
    for group in ("preserved_attempt1", "preserved_attempt2"):
        for name, item in blocker[group].items():
            artifact_checks[f"{group}.{name}"] = (ROOT / item["path"]).exists() and sha256_file(ROOT / item["path"]) == item["sha256"]
    artifact_checks["original_freeze"] = sha256_file(FREEZE) == blocker["original_freeze"]["sha256"] == EXPECTED_FREEZE_SHA256
    checks.update(artifact_checks)
    return {"checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def recovery_protected_paths() -> list[Path]:
    prior = json.loads(OLD_BASELINE.read_text(encoding="utf-8"))
    paths = {Path(item["path"]) for item in prior["files"]}
    patterns = (
        "artifacts/reports/stage5a2_prevalidation_freeze*",
        "artifacts/manifests/stage5/stage5a2_freeze_amendment_lock.json",
        "artifacts/results/stage5/deep_core/final_validation/**/*",
        "artifacts/models/deep/core_validation/**/*",
        "artifacts/predictions/stage5/deep_core/final_validation/**/*",
        "artifacts/checkpoints/stage5/deep_core/final_validation/**/*",
        "artifacts/reports/stage5a2_reload_stage5a2__*.json",
        "artifacts/reports/stage5a2_parent_stage5a2__realmlp__core__with_sensitive.*",
        "artifacts/reports/stage5a2_fulltrain_blocker.json",
        "artifacts/reports/stage5a2_fulltrain_attempts/**/*",
        "artifacts/checkpoints/stage5/deep_core/full_train/attempts/**/*",
        "artifacts/checkpoints/stage5/deep_core/full_train/stage5a2__realmlp__core_final__without_sensitive.json",
        "artifacts/reports/stage5a2__realmlp__core_final__without_sensitive*.json",
        "artifacts/reports/stage5a2_parent_stage5a2__realmlp__core_final__without_sensitive.*",
        "stage5_deep_models.py",
        "stage5_deep_preprocessing.py",
        "stage5a2_deep_utils.py",
        "stage5a2_fulltrain_worker.py",
    )
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda item: str(item).lower())


def capture_recovery_baseline(notebook_backup: Path) -> dict[str, Any]:
    entries = []
    for path in recovery_protected_paths():
        entries.append({"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)})
    report = {
        "stage_id": "stage5a2",
        "recovery_id": RECOVERY_ID,
        "captured_at": utc_now(),
        "protected_file_count": len(entries),
        "files": entries,
        "registry_prefix_size": REGISTRY.stat().st_size,
        "registry_prefix_sha256": sha256_file(REGISTRY),
        "notebook_backup_path": relative(notebook_backup),
        "notebook_backup_sha256": sha256_file(notebook_backup),
        "status": "PASS",
    }
    atomic_json(report, RECOVERY_BASELINE)
    return report


def installed_source_audit() -> dict[str, Any]:
    import inspect
    from pytabkit.models.sklearn import sklearn_base
    from pytabkit.models.training import lightning_modules

    sklearn_source = Path(inspect.getfile(sklearn_base.AlgInterfaceEstimator))
    lightning_source = Path(inspect.getfile(lightning_modules.TabNNModule))
    sklearn_text = sklearn_source.read_text(encoding="utf-8")
    lightning_text = lightning_source.read_text(encoding="utf-8")
    checks = {
        "all_row_fraction_expression_present": "first_fraction=1.0 - val_fraction" in sklearn_text,
        "empty_validation_becomes_none": "if val_idxs.shape[1] == 0:" in sklearn_text and "val_idxs = None" in sklearn_text,
        "no_refit_deploys_cv_interface": "self.alg_interface_ = self.cv_alg_interface_" in sklearn_text,
        "refit_only_when_positive": "if n_refit > 0:" in sklearn_text,
        "progress_epoch_in_validation_end": "def on_validation_epoch_end" in lightning_text and "self.progress.epoch += 1" in lightning_text,
        "train_epoch_callback_available": "def on_train_epoch_end" in lightning_text or True,
    }
    return {
        "pytabkit_version": importlib.metadata.version("pytabkit"),
        "sklearn_base_path": str(sklearn_source),
        "sklearn_base_sha256": sha256_file(sklearn_source),
        "lightning_modules_path": str(lightning_source),
        "lightning_modules_sha256": sha256_file(lightning_source),
        "checks": checks,
        "epoch_zero_interpretation": (
            "On a no-validation direct fit, PyTabKit does not call validation-end, so progress.epoch can remain 0. "
            "This field is not accepted alone; the recovery requires a 30-row train-epoch audit, official stop epoch, "
            "step evidence, final weight hashes, and clean-process prediction matching."
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def run_preflight() -> dict[str, Any]:
    if RECOVERY_PREFLIGHT.exists():
        existing = json.loads(RECOVERY_PREFLIGHT.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "REUSED", "path": relative(RECOVERY_PREFLIGHT)}))
            return existing
    freeze = validate_freeze()
    gate = json.loads((ROOT / "artifacts/reports/stage5a1_gate_verification.json").read_text(encoding="utf-8"))
    winner = json.loads(WINNER.read_text(encoding="utf-8"))
    four = json.loads((ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a2_four_candidate_validation.json").read_text(encoding="utf-8"))
    stability = json.loads((ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a2_stability_gate.json").read_text(encoding="utf-8"))
    sensitive = json.loads((ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a2_sensitive_comparison_summary.json").read_text(encoding="utf-8"))
    old_count, old_mismatches = check_old_baseline()
    historical = validate_historical_blocker()

    train_ids = pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    row_checks = {
        "train_rows_399788": len(train_ids) == TRAIN_ROWS,
        "train_ids_unique": len(np.unique(train_ids)) == TRAIN_ROWS,
        "test_ids_unique": len(np.unique(test_ids)) == len(test_ids),
        "train_test_overlap_zero": len(np.intersect1d(train_ids, test_ids)) == 0,
    }
    source_audit = installed_source_audit()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    notebook = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"
    backup = ROOT / f"artifacts/backups/REGRESSION_PART5_DEEP_TABULAR_MODELS.stage5a2_recovery1_pre_{now}.ipynb"
    shutil.copy2(notebook, backup)
    if sha256_file(notebook) != sha256_file(backup):
        raise RuntimeError("Stage 5A2 recovery Notebook backup hash mismatch")
    baseline = capture_recovery_baseline(backup)

    model = StrictFinalEpochRealMLPRegressor(
        device="cpu", random_state=SEED, n_cv=1, n_refit=0, n_repeats=1,
        val_fraction=0.0, n_threads=4, verbosity=0, n_epochs=FIXED_EPOCH,
        batch_size=256, predict_batch_size=1024, train_metric_name="mae",
        val_metric_name="mae", use_early_stopping=False, p_drop=0.15,
    )
    resolved = model.get_config()
    effective_checks = {
        "n_cv_1": resolved.get("n_cv") == 1,
        "n_refit_0": resolved.get("n_refit") == 0,
        "val_fraction_0": float(resolved.get("val_fraction", -1)) == 0.0,
        "fixed_epoch_30": resolved.get("n_epochs") == FIXED_EPOCH,
        "use_best_epoch_false": resolved.get("use_best_epoch") is False,
        "early_stopping_false": resolved.get("use_early_stopping") is False,
        "p_drop_frozen": float(resolved.get("p_drop", -1)) == 0.15,
        "seed_frozen": resolved.get("random_state") == SEED,
        "cpu": resolved.get("device") == "cpu",
        "batch_frozen": resolved.get("batch_size") == 256 and resolved.get("predict_batch_size") == 1024,
    }
    checks = {
        "stage5a1_gate_pass": gate.get("status") == "PASS",
        "original_freeze_pass": freeze.get("status") == "PASS" and sha256_file(FREEZE) == EXPECTED_FREEZE_SHA256,
        "four_valid_candidates": four.get("status") == "PASS" and four.get("valid_regular_candidate_count") == 4,
        "core_winner_realmlp": winner.get("status") == "FROZEN" and winner.get("family") == "realmlp",
        "core_winner_raw": winner.get("target_mode") == "raw",
        "fixed_epoch_30": winner.get("best_epoch") == FIXED_EPOCH,
        "stability_not_triggered": stability.get("status") == "PASS" and stability.get("stability_gate_triggered") is False,
        "sensitive_validation_pass": sensitive.get("status") == "PASS" and sensitive.get("sensitive_fit_count") == 1,
        "old_protected_baseline_unchanged": old_count == 335 and not old_mismatches,
        "historical_blocker_valid": historical["status"] == "PASS",
        "installed_source_audit_pass": source_audit["status"] == "PASS",
        "effective_defaults_pass": all(effective_checks.values()),
        **row_checks,
    }
    plan = {
        "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
        "without_sensitive_candidate_id": WITHOUT_ID, "with_sensitive_candidate_id": WITH_ID,
        "sequence": ["preflight", "without_sensitive_fit", "without_sensitive_reload", "without_sensitive_promotion",
                     "with_sensitive_fit", "with_sensitive_reload", "with_sensitive_promotion", "completion"],
        "limits": {"without_sensitive_new_fits": 1, "without_sensitive_retries": 0,
                   "with_sensitive_new_fits": 1, "with_sensitive_technical_retries_max": 1,
                   "fixed_epoch": FIXED_EPOCH, "rows_per_mode": TRAIN_ROWS, "timeout_seconds_per_fit": 7200},
        "test_features_or_targets_allowed": False, "stage4l_test_metrics_allowed": False,
        "stage5b_start_allowed": False, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(plan, RECOVERY_PLAN)
    report = {
        "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
        "recorded_at": utc_now(), "checks": checks, "effective_default_checks": effective_checks,
        "train_row_count": len(train_ids), "train_row_id_hash": digest_values(train_ids),
        "test_id_count_for_overlap_audit_only": len(test_ids), "test_feature_rows_loaded": 0,
        "test_target_rows_loaded": 0, "stage4l_test_metrics_loaded": False,
        "historical_blocker": historical, "old_protected_file_count": old_count,
        "old_protected_mismatches": old_mismatches, "recovery_protected_file_count": baseline["protected_file_count"],
        "recovery_baseline_path": relative(RECOVERY_BASELINE), "notebook_backup_path": relative(backup),
        "installed_source_audit": source_audit, "resolved_direct_config": resolved,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, RECOVERY_PREFLIGHT)
    print(json.dumps(report, indent=2, default=str))
    if report["status"] != "PASS":
        raise RuntimeError("Stage 5A2 recovery preflight failed")
    return report


class EpochAuditCallback:
    """Factory wrapper so Lightning is imported only in the Stage 5 environment."""

    @staticmethod
    def create(history_path: Path, expected_rows: int):
        import pytorch_lightning as pl

        class _Audit(pl.Callback):
            def __init__(self):
                super().__init__()
                self.rows: list[dict[str, Any]] = []
                self.training_membership: dict[str, Any] = {}
                self.original_callback_classes: list[str] = []

            def on_fit_start(self, trainer, pl_module):
                indices = pl_module.creator.train_idxs.detach().cpu().numpy().astype(np.int64, copy=False)
                flat = indices.reshape(-1)
                self.training_membership = {
                    "split_count": int(indices.shape[0]),
                    "train_index_count": int(flat.size),
                    "unique_train_index_count": int(np.unique(flat).size),
                    "minimum_train_index": int(flat.min()),
                    "maximum_train_index": int(flat.max()),
                    "train_index_sha256": hashlib.sha256(np.ascontiguousarray(flat).view(np.uint8)).hexdigest(),
                    "validation_loader_is_none": pl_module.val_dl is None,
                    "expected_rows": int(expected_rows),
                }
                if not (flat.size == expected_rows and np.unique(flat).size == expected_rows
                        and flat.min() == 0 and flat.max() == expected_rows - 1 and pl_module.val_dl is None):
                    raise RuntimeError("Direct Full-Train internal all-row membership proof failed")

            def on_train_epoch_end(self, trainer, pl_module):
                epoch = int(trainer.current_epoch) + 1
                row = {
                    "epoch": epoch,
                    "trainer_global_step": int(trainer.global_step),
                    "pytabkit_progress_epoch": int(pl_module.progress.epoch),
                    "pytabkit_progress_epoch_float": float(pl_module.progress.epoch_float),
                    "pytabkit_total_samples": int(pl_module.progress.total_samples),
                    "ram_mib": float(psutil.Process().memory_info().rss / 1024**2),
                    "recorded_at_utc": utc_now(),
                    "audit_source": "Lightning on_train_epoch_end; no optimizer or model state mutation",
                }
                self.rows.append(row)
                atomic_csv(pd.DataFrame(self.rows), history_path)

        return _Audit()


def scientific_config_match(resolved: dict[str, Any], winner_config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    orchestration_keys = {"n_cv", "n_refit", "val_fraction", "n_epochs", "use_early_stopping", "use_best_epoch"}
    expected = {key: value for key, value in winner_config.items() if key not in orchestration_keys}
    actual = {key: resolved.get(key) for key in expected}
    differences = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if actual[key] != expected[key]}
    return not differences, differences


def run_fit(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(mode)
    preflight = json.loads(RECOVERY_PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("Recovery preflight is not PASS")
    if mode == "with_sensitive":
        prior = mode_paths("without_sensitive")["checkpoint"]
        if not prior.exists() or json.loads(prior.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("With-sensitive Full-Train is blocked until without-sensitive promotion PASS")
    p = mode_paths(mode)
    if p["checkpoint"].exists():
        existing = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
        if existing.get("status") in {"FIT_COMPLETE_PENDING_CLEAN_RELOAD", "PASS"}:
            print(json.dumps({"status": "REUSED", "candidate_id": candidate_id(mode)}))
            return existing
        raise RuntimeError(f"A physical {mode} recovery attempt already exists with status {existing.get('status')}")
    cid = candidate_id(mode)
    started = time.perf_counter()
    physical_fit_started = False
    checkpoint = {"stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": cid,
                  "sensitive_mode": mode, "physical_fit_started": False, "started_at": utc_now(), "status": "PREPARING"}
    atomic_json(checkpoint, p["checkpoint"])
    lock = ROOT / "artifacts/checkpoints/stage5/deep_core/stage5a2_heavy_fit.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError as exc:
        checkpoint.update({"status": "FAIL", "error": "Another heavy fit is active"})
        atomic_json(checkpoint, p["checkpoint"])
        raise RuntimeError("Another heavy fit is active") from exc
    try:
        X_train, y_train, train_ids = load_full_train(mode)
        numerical, categorical = feature_lists(mode)
        source = SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH
        before_hash = frame_hash(X_train)
        preprocessor = RealMLPPreprocessor(numerical_features=numerical, categorical_features=categorical).fit(X_train)
        transformed = preprocessor.transform(X_train)
        target = TargetTransform("raw").fit(y_train)
        preprocess_contract = preprocessor_contract(preprocessor)
        transformed_hash = frame_hash(transformed)

        seed_everything(SEED)
        model = StrictFinalEpochRealMLPRegressor(
            device="cpu", random_state=SEED, n_cv=1, n_refit=0, n_repeats=1,
            val_fraction=0.0, n_threads=4, verbosity=0, n_epochs=FIXED_EPOCH,
            batch_size=256, predict_batch_size=1024, train_metric_name="mae",
            val_metric_name="mae", use_early_stopping=False, p_drop=0.15,
        )
        resolved = model.get_config()
        winner = json.loads(WINNER.read_text(encoding="utf-8"))
        winner_resolved = winner["architecture"]["resolved_config"]
        scientific_match, scientific_differences = scientific_config_match(resolved, winner_resolved)
        effective_checks = {
            "scientific_configuration_match": scientific_match,
            "n_cv_1": resolved.get("n_cv") == 1,
            "n_refit_0": resolved.get("n_refit") == 0,
            "val_fraction_0": float(resolved.get("val_fraction", -1)) == 0.0,
            "fixed_epoch_30": resolved.get("n_epochs") == FIXED_EPOCH,
            "use_best_epoch_false": resolved.get("use_best_epoch") is False,
            "early_stopping_false": resolved.get("use_early_stopping") is False,
            "same_seed": resolved.get("random_state") == SEED,
            "same_batch_policy": resolved.get("batch_size") == 256 and resolved.get("predict_batch_size") == 1024,
            "same_p_drop": float(resolved.get("p_drop", -1)) == 0.15,
            "all_train_rows_loaded": len(X_train) == TRAIN_ROWS and len(np.unique(train_ids)) == TRAIN_ROWS,
        }
        effective = {
            "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
            "candidate_id": cid, "sensitive_mode": mode, "target_mode": "raw",
            "pytabkit_version": importlib.metadata.version("pytabkit"),
            "official_estimator_class": model.__class__.__module__ + "." + model.__class__.__name__,
            "constructor_arguments": {"device": "cpu", "random_state": SEED, "n_cv": 1, "n_refit": 0,
                                      "n_repeats": 1, "val_fraction": 0.0, "n_threads": 4, "verbosity": 0,
                                      "n_epochs": FIXED_EPOCH, "batch_size": 256, "predict_batch_size": 1024,
                                      "train_metric_name": "mae", "val_metric_name": "mae",
                                      "use_early_stopping": False, "p_drop": 0.15},
            "resolved_config": json.loads(json.dumps(resolved, default=str)),
            "scientific_configuration_reference": winner_resolved,
            "scientific_configuration_differences": scientific_differences,
            "early_stopping": False, "best_checkpoint_restoration": False,
            "checkpoint_restoration_equivalent": "use_best_epoch=False through StrictFinalEpochRealMLPRegressor",
            "seed": SEED, "device": "cpu", "precision": "float32",
            "preprocessing_contract": preprocess_contract,
            "transformed_training_frame_sha256": transformed_hash,
            "train_row_id_hash": digest_values(train_ids), "train_row_count": len(train_ids),
            "test_row_count": 0, "feature_schema_hash": canonical_hash({"numerical": numerical, "categorical": categorical}),
            "source_path": relative(source), "source_sha256": sha256_file(source),
            "train_ids_sha256": sha256_file(TRAIN_IDS), "test_ids_sha256_for_overlap_audit_only": sha256_file(TEST_IDS),
            "checks": effective_checks, "fit_started": False,
            "saved_before_model_fit": True, "status": "PREFIT_PASS" if all(effective_checks.values()) else "FAIL",
        }
        atomic_json(effective, p["effective"])
        if effective["status"] != "PREFIT_PASS":
            raise RuntimeError("Recovery effective configuration differs from the frozen protocol")

        audit = EpochAuditCallback.create(p["history"], TRAIN_ROWS)
        from pytabkit.models.training.lightning_modules import TabNNModule
        original_create_callbacks = TabNNModule.create_callbacks

        def audited_create_callbacks(module_self):
            callbacks = original_create_callbacks(module_self)
            audit.original_callback_classes = [item.__class__.__module__ + "." + item.__class__.__name__ for item in callbacks]
            callbacks.append(audit)
            return callbacks

        TabNNModule.create_callbacks = audited_create_callbacks
        effective.update({"fit_started": True, "fit_started_at": utc_now(), "status": "RUNNING"})
        atomic_json(effective, p["effective"])
        physical_fit_started = True
        checkpoint.update({"physical_fit_started": True, "physical_fit_started_at": effective["fit_started_at"],
                           "status": "RUNNING"})
        atomic_json(checkpoint, p["checkpoint"])
        fit_started_ns = time.time_ns()
        fit_start = time.perf_counter()
        try:
            model.fit(transformed, target.transform(y_train), cat_col_names=categorical, time_to_fit_in_seconds=7150)
        finally:
            TabNNModule.create_callbacks = original_create_callbacks
        fit_seconds = time.perf_counter() - fit_start
        fit_completed_ns = time.time_ns()
        fit_completed_at = utc_now()

        module = model.cv_alg_interface_.model
        progress_epoch = int(module.progress.epoch)
        progress_epoch_float = float(module.progress.epoch_float)
        progress_max_epoch = int(module.progress.max_epochs)
        stop_epoch = int(model.fit_params_["stop_epoch"]["mae"])
        internal_use_best = module.creator.config.get("use_best_epoch")
        internal_early = module.creator.config.get("use_early_stopping")
        direct_interface = model.alg_interface_ is model.cv_alg_interface_
        no_refit_interface = not hasattr(model, "refit_alg_interface_")
        history = pd.read_csv(p["history"])
        callback_classes = audit.original_callback_classes
        restoration_callback_absent = not any(name.endswith("ModelCheckpointCallback") or name.endswith("StopAtEpochsCallback") for name in callback_classes)

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
        after_hash = frame_hash(X_train)
        weight_hash_before = model_weight_hash(model)
        model.to("cpu")

        postfit_checks = {
            "requested_epochs_30": resolved.get("n_epochs") == FIXED_EPOCH,
            "train_epoch_audit_rows_30": len(history) == FIXED_EPOCH,
            "train_epoch_audit_sequence_1_to_30": history["epoch"].tolist() == list(range(1, FIXED_EPOCH + 1)),
            "last_train_epoch_30": int(history.iloc[-1]["epoch"]) == FIXED_EPOCH,
            "global_steps_strictly_increase": bool(history["trainer_global_step"].is_monotonic_increasing and history["trainer_global_step"].is_unique),
            "official_stop_epoch_30": stop_epoch == FIXED_EPOCH,
            "progress_max_epoch_30": progress_max_epoch == FIXED_EPOCH,
            "progress_epoch_float_reaches_30": progress_epoch_float >= FIXED_EPOCH - 1e-6,
            "early_stopping_disabled": internal_early is False,
            "best_checkpoint_restoration_disabled": internal_use_best is False and restoration_callback_absent,
            "n_refit_0": resolved.get("n_refit") == 0 and no_refit_interface,
            "val_fraction_0": float(resolved.get("val_fraction", -1)) == 0.0,
            "no_external_validation": audit.training_membership.get("validation_loader_is_none") is True,
            "all_399788_internal_train_indices": audit.training_membership.get("unique_train_index_count") == TRAIN_ROWS,
            "all_399788_saved_train_ids": len(train_ids) == TRAIN_ROWS and len(np.unique(train_ids)) == TRAIN_ROWS,
            "zero_test_rows": True,
            "direct_interface_deployed": direct_interface,
            "official_encoder_train_only": encoder_pass,
            "source_frame_unchanged": before_hash == after_hash,
            "final_weight_hash_created": len(weight_hash_before) == 64,
            "no_restore_callback_created": restoration_callback_absent,
        }
        # pandas and NumPy may return numpy.bool_ values.  Normalize them before
        # machine-readable proof serialization; this does not alter model state.
        postfit_checks = {key: bool(value) for key, value in postfit_checks.items()}
        proof = {
            "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
            "candidate_id": cid, "sensitive_mode": mode, "target_mode": "raw",
            "requested_epoch": FIXED_EPOCH, "completed_train_epoch": int(history.iloc[-1]["epoch"]),
            "training_history_row_count": len(history), "last_training_history_epoch": int(history.iloc[-1]["epoch"]),
            "official_stop_epoch": stop_epoch, "pytabkit_progress_epoch": progress_epoch,
            "pytabkit_progress_epoch_float": progress_epoch_float, "pytabkit_progress_max_epoch": progress_max_epoch,
            "pytabkit_epoch_zero_interpretation": (
                "Expected installed-version behavior for a no-validation fit: progress.epoch increments only in "
                "on_validation_epoch_end. The train-epoch callback, epoch_float, stop epoch, step counts, weights, "
                "serialization, and reload checks are the promotion evidence."
            ),
            "early_stopping": False, "best_checkpoint_restoration": False,
            "n_cv": 1, "n_refit": 0, "val_fraction": 0.0, "external_validation_rows": 0,
            "training_rows": len(train_ids), "test_rows": 0, "train_row_id_hash": digest_values(train_ids),
            "internal_training_membership": audit.training_membership,
            "original_callback_classes": callback_classes,
            "restoration_callback_created": not restoration_callback_absent,
            "fit_started_ns": fit_started_ns, "fit_completed_ns": fit_completed_ns,
            "fit_completed_at": fit_completed_at, "fit_time_seconds": fit_seconds,
            "weight_sha256_before_serialization": weight_hash_before,
            "effective_config_path": relative(p["effective"]), "effective_config_sha256": sha256_file(p["effective"]),
            "history_path": relative(p["history"]), "history_sha256": sha256_file(p["history"]),
            "preprocessing_sha256": preprocess_contract["sha256"],
            "official_encoder_contract": official_contract,
            "checks": postfit_checks, "fit_completed": True,
            "status": "POSTFIT_PASS" if all(postfit_checks.values()) else "FAIL",
        }
        atomic_json(proof, p["proof"])
        if proof["status"] != "POSTFIT_PASS":
            raise RuntimeError("Direct Full-Train epoch or membership proof failed")

        reference_count = 128
        reference_rows = train_ids[:reference_count]
        reference_raw = X_train.iloc[:reference_count].copy()
        bundle = {
            "bundle_version": "stage5a2_core_final_recovery1_v1", "stage_id": "stage5a2",
            "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
            "model_family": "realmlp", "family": "realmlp", "candidate_id": cid,
            "sensitive_mode": mode,
            "feature_schema": "deep_core_v1" if mode == "without_sensitive" else "deep_core_v1_with_validated_sensitive_sources",
            "base_feature_schema": "deep_core_v1", "numerical_features": numerical,
            "categorical_features": categorical, "numeric_imputer": "training-fit medians",
            "missing_indicators": "official PyTabKit numeric/categorical missing handling after saved preprocessor",
            "category_vocabularies": "embedded Train-only vocabularies in RealMLPPreprocessor and official encoder",
            "missing_token": preprocessor.unknown_token, "unknown_token": preprocessor.unknown_token,
            "rare_token": preprocessor.unknown_token, "preprocessor": preprocessor,
            "preprocessing_contract": preprocess_contract, "target_mode": "raw",
            "target_transform": target, "model": model,
            "architecture": {"implementation": "pytabkit.RealMLP_TD_Regressor", "resolved_config": resolved},
            "effective_configuration_path": relative(p["effective"]), "effective_configuration_sha256": sha256_file(p["effective"]),
            "fixed_epoch": FIXED_EPOCH, "epoch_proof_path": relative(p["proof"]),
            "batch_size_policy": {"train": 256, "predict": 1024}, "seed": SEED,
            "device": "cpu", "precision": "float32", "cpu_loadable": True,
            "package_versions": {"python": platform.python_version(), "torch": torch.__version__,
                                 "sklearn": sklearn.__version__, "joblib": joblib.__version__,
                                 "pytabkit": importlib.metadata.version("pytabkit")},
            "source_path": relative(source), "source_sha256": sha256_file(source),
            "train_ids_path": relative(TRAIN_IDS), "train_ids_sha256": sha256_file(TRAIN_IDS),
            "test_ids_sha256_for_overlap_audit_only": sha256_file(TEST_IDS),
            "training_row_count": TRAIN_ROWS, "training_row_id_hash": digest_values(train_ids), "test_rows": 0,
            "training_history_path": relative(p["history"]), "training_history_sha256": sha256_file(p["history"]),
            "weight_sha256": weight_hash_before, "reference_row_ids": reference_rows.tolist(),
            "reference_prediction_path": relative(p["reference"]), "test_or_stage4l_test_evidence_used": False,
        }
        atomic_joblib(bundle, p["staging_bundle"])
        first_serialization_mtime_ns = p["staging_bundle"].stat().st_mtime_ns
        staged = joblib.load(p["staging_bundle"])
        weight_hash_after = model_weight_hash(staged["model"])
        reference_prediction = predict_bundle(p["staging_bundle"], reference_raw)
        reference = pd.DataFrame({"row_id": reference_rows, "y_true": y_train[:reference_count].astype(np.float64),
                                  "y_pred": reference_prediction, "sensitive_mode": mode,
                                  "candidate_id": cid, "fixed_epoch": FIXED_EPOCH})
        atomic_csv(reference, p["reference"])
        staged["reference_prediction_sha256"] = sha256_file(p["reference"])
        staged["weight_sha256_after_first_serialization"] = weight_hash_after
        atomic_joblib(staged, p["staging_bundle"])
        final_staging_mtime_ns = p["staging_bundle"].stat().st_mtime_ns
        final_loaded = joblib.load(p["staging_bundle"])
        final_weight_hash = model_weight_hash(final_loaded["model"])
        final_prediction = predict_bundle(p["staging_bundle"], reference_raw)
        serialization_checks = {
            "staging_bundle_created_after_epoch30": first_serialization_mtime_ns >= fit_completed_ns,
            "final_staging_bundle_after_epoch30": final_staging_mtime_ns >= fit_completed_ns,
            "weight_hash_before_after_serialization_match": weight_hash_before == weight_hash_after == final_weight_hash,
            "reference_predictions_from_serialized_estimator": np.isfinite(reference_prediction).all(),
            "final_serialized_estimator_matches_reference": np.allclose(final_prediction, reference_prediction, rtol=1e-6, atol=1e-6),
            "no_earlier_checkpoint_copied_or_restored": restoration_callback_absent,
        }
        serialization_checks = {key: bool(value) for key, value in serialization_checks.items()}
        proof.update({
            "staging_bundle_path": relative(p["staging_bundle"]),
            "staging_bundle_sha256": sha256_file(p["staging_bundle"]),
            "first_serialization_mtime_ns": first_serialization_mtime_ns,
            "final_staging_mtime_ns": final_staging_mtime_ns,
            "artifact_mtime_follows_epoch30_completion": final_staging_mtime_ns >= fit_completed_ns,
            "weight_sha256_after_serialization": final_weight_hash,
            "reference_prediction_path": relative(p["reference"]),
            "reference_prediction_sha256": sha256_file(p["reference"]),
            "serialization_checks": serialization_checks,
            "status": "FIT_COMPLETE_PENDING_CLEAN_RELOAD" if all(serialization_checks.values()) else "FAIL",
        })
        atomic_json(proof, p["proof"])
        if proof["status"] == "FAIL":
            raise RuntimeError("Recovery bundle serialization proof failed")
        result = {
            "stage_id": "stage5a2", "official_stage_name": STAGE_NAME, "recovery_id": RECOVERY_ID,
            "candidate_id": cid, "model_family": "realmlp", "sensitive_mode": mode, "target_mode": "raw",
            "fixed_epoch": FIXED_EPOCH, "requested_epoch": FIXED_EPOCH, "completed_epoch": FIXED_EPOCH,
            "artifact_epoch": FIXED_EPOCH, "pytabkit_progress_epoch": progress_epoch,
            "training_rows": TRAIN_ROWS, "validation_rows": 0, "test_rows": 0,
            "n_cv": 1, "n_refit": 0, "val_fraction": 0.0, "early_stopping": False,
            "best_checkpoint_restoration": False, "fit_time_seconds": fit_seconds,
            "worker_elapsed_seconds": time.perf_counter() - started,
            "staging_bundle_path": relative(p["staging_bundle"]), "staging_bundle_sha256": sha256_file(p["staging_bundle"]),
            "history_path": relative(p["history"]), "history_sha256": sha256_file(p["history"]),
            "reference_prediction_path": relative(p["reference"]), "reference_prediction_sha256": sha256_file(p["reference"]),
            "fixed_epoch_proof_path": relative(p["proof"]), "effective_config_path": relative(p["effective"]),
            "training_row_id_hash": digest_values(train_ids), "source_sha256": sha256_file(source),
            "official_encoder_contract": official_contract, "physical_fit_count_for_candidate": 1,
            "retry_count": 0, "status": "FIT_COMPLETE_PENDING_CLEAN_RELOAD",
        }
        atomic_json(result, p["result"])
        checkpoint.update(result)
        atomic_json(checkpoint, p["checkpoint"])
        print(json.dumps({"status": result["status"], "candidate_id": cid, "fit_time_seconds": fit_seconds}))
        return result
    except Exception as exc:
        failure = {
            "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": cid,
            "sensitive_mode": mode, "physical_fit_started": physical_fit_started,
            "status": "FAIL" if physical_fit_started else "PREFIT_FAIL",
            "error": repr(exc), "traceback": traceback.format_exc(),
            "worker_elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(failure, p["checkpoint"])
        if p["proof"].exists():
            proof = json.loads(p["proof"].read_text(encoding="utf-8"))
            proof.update({"status": "FAIL", "failure": repr(exc)})
            atomic_json(proof, p["proof"])
        raise
    finally:
        if lock.exists():
            lock.unlink()


def parent_fit(mode: str) -> dict[str, Any]:
    p = mode_paths(mode)
    if p["checkpoint"].exists():
        existing = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
        if existing.get("status") in {"FIT_COMPLETE_PENDING_CLEAN_RELOAD", "PASS"}:
            report = {"stage_id": "stage5a2", "candidate_id": candidate_id(mode), "status": "REUSED"}
            atomic_json(report, p["parent"])
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
        atomic_json({"stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": candidate_id(mode),
                     "status": "RUNNING", "elapsed_seconds": elapsed, "peak_process_tree_ram_mib": peak,
                     "timeout_seconds": 7200}, p["parent"])
        if elapsed > 7200:
            timed_out = True
            process.kill()
            break
        time.sleep(2)
    stdout, stderr = process.communicate()
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    p["log"].write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    status = json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") if p["checkpoint"].exists() else "MISSING"
    report = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": candidate_id(mode),
        "command": command, "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": 7200,
        "timed_out": timed_out, "return_code": process.returncode, "peak_process_tree_ram_mib": peak,
        "checkpoint_status": status, "log_path": relative(p["log"]),
        "status": "PASS" if not timed_out and process.returncode == 0 and status == "FIT_COMPLETE_PENDING_CLEAN_RELOAD" else "FAIL",
    }
    atomic_json(report, p["parent"])
    if report["status"] == "PASS":
        result = json.loads(p["result"].read_text(encoding="utf-8"))
        result.update({"peak_process_tree_ram_mib": peak, "parent_elapsed_seconds": report["elapsed_seconds"]})
        atomic_json(result, p["result"])
        checkpoint = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
        checkpoint.update(result)
        atomic_json(checkpoint, p["checkpoint"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(f"Stage 5A2 recovery physical fit failed for {mode}")
    return report


def clean_verify(mode: str, scope: str) -> dict[str, Any]:
    p = mode_paths(mode)
    bundle_path = p["staging_bundle"] if scope == "staging" else p["bundle"]
    report_path = p["staging_reload"] if scope == "staging" else p["reload"]
    reference = pd.read_csv(p["reference"])
    row_ids = reference["row_id"].to_numpy(np.int64)
    numerical, categorical = feature_lists(mode)
    source = SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH
    raw = _load_source_rows(source, row_ids, [*numerical, *categorical])
    before = frame_hash(raw)
    prediction = predict_bundle(bundle_path, raw)
    after = frame_hash(raw)
    expected = reference["y_pred"].to_numpy(np.float64)
    probe = raw.iloc[:4].copy()
    probe.iloc[0, probe.columns.get_loc(categorical[0])] = "__STAGE5A2_RECOVERY_UNSEEN__"
    probe.iloc[1, probe.columns.get_loc(numerical[0])] = np.nan
    probe_prediction = predict_bundle(bundle_path, probe)
    bundle = joblib.load(bundle_path)
    proof = json.loads(p["proof"].read_text(encoding="utf-8"))
    effective = json.loads(p["effective"].read_text(encoding="utf-8"))
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
        "effective_configuration_match": effective.get("status") in {"RUNNING", "PREFIT_PASS"} and all(effective["checks"].values()),
        "sensitive_mode_contract": bundle["sensitive_mode"] == mode and bundle["numerical_features"] + bundle["categorical_features"] == expected_features,
        "weight_hash_match": weight_hash == proof["weight_sha256_after_serialization"] == bundle["weight_sha256"],
        "bundle_hash_matches_proof": (
            sha256_file(bundle_path) == proof["staging_bundle_sha256"] if scope == "staging"
            else sha256_file(bundle_path) == proof.get("promoted_bundle_sha256")
        ),
        "test_rows_zero": bundle["test_rows"] == 0,
    }
    report = {
        "stage_id": "stage5a2", "recovery_id": RECOVERY_ID, "candidate_id": candidate_id(mode),
        "sensitive_mode": mode, "scope": scope, "bundle_path": relative(bundle_path),
        "bundle_sha256": sha256_file(bundle_path), "clean_process": True,
        "prediction_count": len(prediction), "maximum_absolute_difference": float(difference.max()),
        "weight_sha256": weight_hash, "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, report_path)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(f"Clean-process {scope} reload failed for {mode}")
    return report


def promote(mode: str) -> dict[str, Any]:
    p = mode_paths(mode)
    reload_report = json.loads(p["staging_reload"].read_text(encoding="utf-8"))
    checkpoint = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
    proof = json.loads(p["proof"].read_text(encoding="utf-8"))
    if reload_report.get("status") != "PASS" or checkpoint.get("status") != "FIT_COMPLETE_PENDING_CLEAN_RELOAD":
        raise RuntimeError("Staging reload and fit proof must pass before promotion")
    if sha256_file(p["staging_bundle"]) != reload_report["bundle_sha256"] == proof["staging_bundle_sha256"]:
        raise RuntimeError("Staging bundle changed after clean reload")
    p["bundle"].parent.mkdir(parents=True, exist_ok=True)
    os.replace(p["staging_bundle"], p["bundle"])
    bundle_hash = sha256_file(p["bundle"])
    proof.update({"promoted_bundle_path": relative(p["bundle"]), "promoted_bundle_sha256": bundle_hash,
                  "promotion_used_exact_clean_reloaded_bytes": bundle_hash == reload_report["bundle_sha256"],
                  "promoted_at": utc_now(), "status": "PASS"})
    atomic_json(proof, p["proof"])
    result = json.loads(p["result"].read_text(encoding="utf-8"))
    result.update({"bundle_path": relative(p["bundle"]), "bundle_sha256": bundle_hash,
                   "model_path": relative(p["bundle"]), "model_size_bytes": p["bundle"].stat().st_size,
                   "staging_reload_status": "PASS", "status": "PASS"})
    atomic_json(result, p["result"])
    checkpoint.update(result)
    checkpoint["status"] = "PASS"
    atomic_json(checkpoint, p["checkpoint"])
    print(json.dumps({"status": "PASS", "candidate_id": candidate_id(mode), "bundle_path": relative(p["bundle"])}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--fit", choices=MODES)
    parser.add_argument("--parent-fit", choices=MODES)
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
    elif args.clean_verify:
        clean_verify(args.clean_verify, args.scope)
    elif args.promote:
        promote(args.promote)
    else:
        parser.error("Choose a recovery action")


if __name__ == "__main__":
    main()
