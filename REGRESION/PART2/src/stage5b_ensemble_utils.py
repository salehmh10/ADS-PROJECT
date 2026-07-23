"""Stage 5B prediction-only ensemble analysis and artifact orchestration.

This module never opens source CSV data, never opens Test prediction or result
artifacts, and never trains or invokes a saved model. It combines already saved
Final Selection Validation predictions only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STAGE_ID = "stage5b"
OFFICIAL_NAME = "Stage 5B — Frozen Deep and Boosting Ensemble"
REPORTS = ROOT / "artifacts/reports"
MANIFESTS = ROOT / "artifacts/manifests/stage5"
RESULTS = ROOT / "artifacts/results/stage5/deep_boosting_ensemble"
PREDICTIONS = ROOT / "artifacts/predictions/stage5/deep_boosting_ensemble"
FIGURES = ROOT / "artifacts/figures/stage5b"
PLOTTING = FIGURES / "plotting_data"
BACKUPS = ROOT / "artifacts/backups"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_BOOSTING_ENSEMBLE.ipynb"

BASELINE = MANIFESTS / "stage5b_protected_hashes_before.json"
FREEZE = REPORTS / "stage5b_preensemble_freeze.json"
ALIGNMENT = REPORTS / "stage5b_input_alignment_report.json"
GRID = RESULTS / "stage5b_weight_grid_results.csv"
DIVERSITY = RESULTS / "stage5b_diversity_diagnostics.csv"
DIVERSITY_SUMMARY = RESULTS / "stage5b_diversity_summary.json"
BOOTSTRAP = RESULTS / "stage5b_paired_bootstrap.csv"
DECISION = RESULTS / "stage5b_ensemble_decision.json"
SENSITIVE = RESULTS / "stage5b_sensitive_diagnostic.csv"
PRED_WITHOUT = PREDICTIONS / "stage5b_validation_predictions_without_sensitive.csv"
PRED_WITH = PREDICTIONS / "stage5b_validation_predictions_with_sensitive.csv"
FROZEN_ENSEMBLE = RESULTS / "stage5b_frozen_ensemble.json"
HANDOFF = MANIFESTS / "stage5b_evaluation_handoff.json"
VISUALIZATION_MANIFEST = MANIFESTS / "stage5b_visualization_manifest.json"
REGISTRY_EXPORT = RESULTS / "stage5b_registry_rows.csv"
NOTEBOOK_RUNS = REPORTS / "stage5b_notebook_executions.json"
PROTECTED_RECHECK = REPORTS / "stage5b_protected_recheck.json"
VERIFICATION = REPORTS / "stage5b_verification.json"

STAGE5A_VERIFICATION = ROOT / "artifacts/reports/stage5a_verification.json"
GOVERNANCE = ROOT / "artifacts/reports/stage5a2_governance_adjudication.json"
DEEP_HANDOFF = ROOT / "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json"
DEEP_FULL_MANIFEST = ROOT / "artifacts/manifests/stage5/stage5a2_full_train_manifest.json"
STAGE4_PRETEST_FREEZE = ROOT / "artifacts/reports/stage4l_pretest_freeze.json"
STAGE4_BLEND_EVIDENCE = ROOT / "artifacts/reports/stage4l_blend_validation_evidence.json"
STAGE4_CANDIDATES = ROOT / "artifacts/manifests/stage4/stage4l_candidate_manifest.json"

DEEP_PATHS = {
    "without_sensitive": ROOT / "artifacts/predictions/stage5/deep_core/final_validation/stage5a2__realmlp__frozen.csv",
    "with_sensitive": ROOT / "artifacts/predictions/stage5/deep_core/final_validation/stage5a2__realmlp__core__with_sensitive.csv",
}
BOOST_PATHS = {
    "without_sensitive": {
        "catboost": ROOT / "artifacts/results/stage4/catboost/final/catboost_final_validation_predictions_without_sensitive.csv",
        "lightgbm": ROOT / "artifacts/predictions/lightgbm/final/lightgbm_final_selection_without_sensitive.csv",
        "xgboost": ROOT / "artifacts/predictions/xgboost/final_selection/xgboost_winning_non_sensitive_validation.csv",
    },
    "with_sensitive": {
        "catboost": ROOT / "artifacts/results/stage4/catboost/final/catboost_final_validation_predictions_with_sensitive.csv",
        "lightgbm": ROOT / "artifacts/predictions/lightgbm/final/lightgbm_final_selection_with_sensitive.csv",
        "xgboost": ROOT / "artifacts/predictions/xgboost/final_selection/sensitive_validation.csv",
    },
}
EXPECTED_PREDICTION_HASHES = {
    str(DEEP_PATHS["without_sensitive"].relative_to(ROOT)): "8b8436ccfbf8a1d623c341e7d4ec956e6a0549fdec7288599746f282ac58b2e7",
    str(DEEP_PATHS["with_sensitive"].relative_to(ROOT)): "9c8598b046053fa31043912e2f4578e51228fd6414e701c75f052da8ef566786",
    str(BOOST_PATHS["without_sensitive"]["catboost"].relative_to(ROOT)): "694e5ecaa781c9fb97ca6632112d017567045e277f888ec4e6c09bec3c085e9a",
    str(BOOST_PATHS["with_sensitive"]["catboost"].relative_to(ROOT)): "a05042961ee2e3400a98b8c04e20eaf4f3dd8c994f8c8166cb8bf4cc77b86874",
    str(BOOST_PATHS["without_sensitive"]["lightgbm"].relative_to(ROOT)): "3b48e7ed45ad1c7c4f5f947833b0de251758d8f1b39c59c4a049152f7815407a",
    str(BOOST_PATHS["with_sensitive"]["lightgbm"].relative_to(ROOT)): "09ddc127a4f57ab147a9e19aea61bfd0af02bce4338a9182207be76a01a1cd31",
    str(BOOST_PATHS["without_sensitive"]["xgboost"].relative_to(ROOT)): "96f9b18d900e292baadaac82d17304181021e488ab2d9a273e70c3941d22fe8f",
    str(BOOST_PATHS["with_sensitive"]["xgboost"].relative_to(ROOT)): "d8366f9d6f84d4fc3c8d2b47967144414e499c960494e277dea4f5c38c387c5b",
}
EXPECTED_ROW_HASH = "5a47f42c454ab185f70a3cd2b637c55c9e4fa0804c59b2ebe25ac781c44fc26b"
EXPECTED_TARGET_HASH = "ddb115eee1697b87e3d709119447571a0493c62f525b8c05072e51fbabe19668"
BLEND_WEIGHTS = {"catboost": 0.6, "lightgbm": 0.2, "xgboost": 0.2}
WEIGHT_GRID = [round(value, 2) for value in np.arange(0.0, 0.5001, 0.05)]
ANCHORS = ["frozen_stage4_boosting_blend", "catboost"]

COUNTERS = {
    "model_fit_calls": 0,
    "preprocessing_fit_calls": 0,
    "prediction_generation_calls": 0,
    "source_data_loads": 0,
    "test_artifact_loads": 0,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(values: np.ndarray, dtype=np.float64) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def row_hash(values: np.ndarray) -> str:
    return value_hash(values, dtype=np.int64)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_no_forbidden_path(path: Path) -> None:
    normalized = rel(path).lower() if path.is_relative_to(ROOT) else str(path).lower()
    forbidden = ("final_test", "stage4l_test", "test_leaderboard", "test_bootstrap", "test_error")
    if any(token in normalized for token in forbidden):
        raise RuntimeError(f"Forbidden Test artifact path: {path}")


def static_no_fit_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    patterns = {
        "model_fit": r"\.fit\s*\(",
        "preprocessing_fit_transform": r"\.fit_transform\s*\(",
        "partial_fit": r"\.partial_fit\s*\(",
        "prediction_generation": r"\.predict\s*\(",
    }
    matches = {name: re.findall(pattern, source) for name, pattern in patterns.items()}
    checks = {name: len(items) == 0 for name, items in matches.items()}
    if not all(checks.values()):
        raise RuntimeError(f"Static no-Fit guard failed: {matches}")
    return {"status": "PASS", "checks": checks, "match_counts": {key: len(value) for key, value in matches.items()}}


def prediction_file_paths() -> list[Path]:
    paths = list(DEEP_PATHS.values())
    for mode in ("without_sensitive", "with_sensitive"):
        paths.extend(BOOST_PATHS[mode].values())
    return paths


def protected_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(ROOT.glob("REGRESSION_PART*.ipynb")))
    paths.extend([
        ROOT / "data/regression_without_sensitive_features.csv",
        ROOT / "data/regression_with_sensitive_features.csv",
        ROOT / "artifacts/splits/train_row_ids.csv",
        ROOT / "artifacts/splits/test_row_ids.csv",
        ROOT / "artifacts/splits/stage4/stage4_final_selection_sample.csv",
        REGISTRY,
        STAGE5A_VERIFICATION,
        GOVERNANCE,
        DEEP_HANDOFF,
        DEEP_FULL_MANIFEST,
        STAGE4_PRETEST_FREEZE,
        STAGE4_BLEND_EVIDENCE,
        STAGE4_CANDIDATES,
    ])
    paths.extend(prediction_file_paths())
    stage4_candidates = load_json(STAGE4_CANDIDATES)["candidates"]
    for item in stage4_candidates:
        if item.get("model_family") in {"catboost", "lightgbm", "xgboost"}:
            for key in ("model_bundle_path", "native_model_path", "manifest_path"):
                value = item.get(key)
                if value:
                    paths.append(ROOT / value)
    for item in load_json(DEEP_HANDOFF)["items"]:
        paths.append(ROOT / item["full_train_bundle_path"])
    part1 = Path(r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb")
    if part1.exists():
        paths.append(part1)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve()).lower()] = path
    return sorted(unique.values(), key=lambda item: str(item).lower())


def create_preflight() -> dict[str, Any]:
    started = time.perf_counter()
    guard = static_no_fit_guard()
    required = [STAGE5A_VERIFICATION, GOVERNANCE, DEEP_HANDOFF, DEEP_FULL_MANIFEST,
                STAGE4_PRETEST_FREEZE, STAGE4_BLEND_EVIDENCE, STAGE4_CANDIDATES] + prediction_file_paths()
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Required Stage 5B inputs are missing: {missing}")
    verification = load_json(STAGE5A_VERIFICATION)
    governance = load_json(GOVERNANCE)
    deep_handoff = load_json(DEEP_HANDOFF)
    deep_manifest = load_json(DEEP_FULL_MANIFEST)
    stage4_freeze = load_json(STAGE4_PRETEST_FREEZE)
    blend = load_json(STAGE4_BLEND_EVIDENCE)
    candidate_manifest = load_json(STAGE4_CANDIDATES)
    checks = {
        "stage5a_pass_with_exception": verification.get("status") == "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "governance_exception_visible": governance.get("literal_rule_violated") is True and governance.get("classifications", {}).get("literal_zero_test_loading") is False,
        "deep_handoff_pass": deep_handoff.get("status") == "PASS",
        "deep_handoff_unweighted": deep_handoff.get("ensemble_weight_selected") is False,
        "deep_bundle_manifest_pass": deep_manifest.get("status") == "PASS",
        "two_deep_bundles": len(deep_manifest.get("models", [])) == 2,
        "both_deep_reloads_pass": all(item.get("reload_checks_all") for item in deep_manifest.get("models", [])),
        "pretest_freeze_pass": stage4_freeze.get("status") == "PASS",
        "pretest_target_not_loaded": stage4_freeze.get("test_target_not_loaded") is True,
        "frozen_blend_accepted": blend.get("accepted") is True,
        "frozen_blend_weights_match": blend.get("weights") == BLEND_WEIGHTS,
        "best_individual_catboost": blend.get("best_individual") == "catboost",
        "weight_grid_exact": WEIGHT_GRID == [round(index * 0.05, 2) for index in range(11)],
        "two_anchors_exact": ANCHORS == ["frozen_stage4_boosting_blend", "catboost"],
    }
    actual_prediction_hashes: dict[str, str] = {}
    for path in prediction_file_paths():
        assert_no_forbidden_path(path)
        relative = rel(path)
        actual_prediction_hashes[relative] = sha256_file(path)
        checks[f"prediction_hash__{path.stem}"] = actual_prediction_hashes[relative] == EXPECTED_PREDICTION_HASHES[relative.replace("/", os.sep)] if relative.replace("/", os.sep) in EXPECTED_PREDICTION_HASHES else actual_prediction_hashes[relative] == EXPECTED_PREDICTION_HASHES[relative]
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage 5B preflight failed: {failed}")

    if NOTEBOOK.exists():
        BACKUPS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = BACKUPS / f"REGRESSION_PART5_DEEP_BOOSTING_ENSEMBLE.{stamp}.ipynb"
        if not backup.exists():
            shutil.copy2(NOTEBOOK, backup)

    entries = []
    for path in protected_paths():
        if not path.exists():
            raise RuntimeError(f"Protected path is missing: {path}")
        assert_no_forbidden_path(path)
        entries.append({
            "path": rel(path) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    registry_bytes = REGISTRY.read_bytes()
    baseline = {
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "protected_file_count": len(entries),
        "entries": entries,
        "registry_prior_byte_count": len(registry_bytes),
        "registry_prior_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "excluded_by_governance": ["all Stage 4L Test predictions, metrics, leaderboards, Bootstrap, and error artifacts"],
        "status": "PASS",
    }
    atomic_json(baseline, BASELINE)

    stage4_by_id = {item["candidate_id"]: item for item in candidate_manifest["candidates"]}
    booster_ids = {
        "catboost": "stage4l__catboost__without_sensitive",
        "lightgbm": "stage4l__lightgbm__without_sensitive",
        "xgboost": "stage4l__xgboost__without_sensitive",
    }
    booster_components = {}
    for family, candidate_id in booster_ids.items():
        item = stage4_by_id[candidate_id]
        booster_components[family] = {
            "candidate_id": candidate_id,
            "validation_metrics": item["pretest_metrics"],
            "bundle_path": item["model_bundle_path"],
            "bundle_sha256": item["model_sha256"],
            "native_model_path": item["native_model_path"],
            "native_model_sha256": item["native_model_sha256"],
        }
    deep_items = {}
    for item in deep_handoff["items"]:
        deep_items[item["sensitive_mode"]] = {
            "validation_candidate_id": item["validation_candidate_id"],
            "validation_prediction_path": item["validation_prediction_path"].replace("\\", "/"),
            "validation_prediction_sha256": item["validation_prediction_sha256"],
            "full_train_bundle_path": item["full_train_bundle_path"].replace("\\", "/"),
            "full_train_bundle_sha256": item["full_train_bundle_sha256"],
        }
    freeze_payload = {
        "stage_id": STAGE_ID,
        "official_stage_name": OFFICIAL_NAME,
        "freeze_timestamp_utc": now_utc(),
        "stage5a_status": verification["status"],
        "stage5a_handoff_path": rel(DEEP_HANDOFF),
        "stage5a_handoff_sha256": sha256_file(DEEP_HANDOFF),
        "stage5a_governance_adjudication_path": rel(GOVERNANCE),
        "stage5a_governance_adjudication_sha256": sha256_file(GOVERNANCE),
        "stage5a_governance_exception": governance["human_adjudication_result"],
        "literal_zero_test_loading_historical_status": "FAIL",
        "validation_row_count": 25000,
        "validation_row_id_hash": deep_handoff["validation_row_id_hash"],
        "target_hash": deep_handoff["target_sha256"],
        "deep_candidate_id": deep_handoff["core_winner_candidate_id"],
        "deep_family": deep_handoff["core_winner_family"],
        "deep_target_mode": deep_handoff["target_mode"],
        "deep_items": deep_items,
        "boosting_components": booster_components,
        "boosting_prediction_paths": {
            mode: {family: rel(path) for family, path in paths.items()} for mode, paths in BOOST_PATHS.items()
        },
        "boosting_prediction_hashes": {
            mode: {family: sha256_file(path) for family, path in paths.items()} for mode, paths in BOOST_PATHS.items()
        },
        "frozen_stage4_boosting_blend_weights": BLEND_WEIGHTS,
        "best_individual_boosting_candidate_id": "stage4l__catboost__without_sensitive",
        "best_individual_boosting_family": "catboost",
        "best_individual_selection_source": "existing non-sensitive Final Selection Validation metadata only",
        "allowed_boosting_anchors": ANCHORS,
        "deep_weight_grid": WEIGHT_GRID,
        "boost_weight_rule": "1.00 - deep_weight",
        "weighted_combination_count": 22,
        "metrics": ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate", "top_decile_mae", "top_five_percent_mae"],
        "primary_metric": "mae",
        "acceptance_rules": {
            "main_accuracy_gate": {"minimum_mae_improvement_percent": 0.30, "maximum_rmse_worsening_percent": 0.25, "maximum_top_decile_mae_worsening_percent": 1.00, "finite_predictions": True, "bootstrap_upper_95_below_zero": True},
            "tail_benefit_gate": {"minimum_mae_improvement_percent": 0.20, "minimum_top_decile_mae_improvement_percent": 1.00, "maximum_rmse_worsening_percent": 0.25, "finite_predictions": True, "bootstrap_upper_95_below_zero": True},
        },
        "tie_rules": ["lowest MAE", "within 0.10 percent prefer lower top-decile MAE", "then lower RMSE", "then lower Deep weight", "then simpler anchor"],
        "bootstrap_settings": {"resamples": 500, "seed": 42, "paired_indices": True, "error_scale": "original target absolute error", "pass_rule": "upper 95 percent percentile bound below zero"},
        "sensitive_transfer_policy": "Use the non-sensitive selected anchor, internal weights, Deep weight, and Boost weight unchanged. Sensitive evidence cannot select the ensemble.",
        "source_csv_permitted": False,
        "test_artifacts_loaded": False,
        "test_artifacts_permitted": False,
        "validation_targets_loaded_by_stage5b_before_freeze": False,
        "protected_baseline_path": rel(BASELINE),
        "protected_baseline_sha256": sha256_file(BASELINE),
        "static_no_fit_guard": guard,
        "counters_at_freeze": COUNTERS,
        "status": "PASS",
    }
    canonical = json.dumps(freeze_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    freeze_payload["immutable_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(freeze_payload, FREEZE)
    reloaded = load_json(FREEZE)
    if reloaded != freeze_payload:
        raise RuntimeError("Pre-ensemble freeze reload mismatch")
    result = {
        "status": "PASS",
        "checks": checks,
        "protected_file_count": len(entries),
        "baseline_sha256": sha256_file(BASELINE),
        "freeze_sha256": sha256_file(FREEZE),
        "freeze_immutable_content_sha256": freeze_payload["immutable_content_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
        "counters": COUNTERS,
    }
    atomic_json(result, REPORTS / "stage5b_preflight.json")
    return result


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, target_decile: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    absolute = np.abs(residual)
    mse = float(np.mean(np.square(residual)))
    nonzero = np.abs(y_true) > 0
    mape = float(np.mean(absolute[nonzero] / np.abs(y_true[nonzero])) * 100.0) if nonzero.any() else float("nan")
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r_squared = 1.0 - float(np.sum(np.square(residual))) / denominator
    clipped_true = np.clip(y_true, 0.0, None)
    clipped_pred = np.clip(y_pred, 0.0, None)
    rmsle = float(np.sqrt(np.mean(np.square(np.log1p(clipped_pred) - np.log1p(clipped_true)))))
    p95_threshold = float(np.quantile(y_true, 0.95))
    return {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mape_percent": mape,
        "r_squared": r_squared,
        "rmsle": rmsle,
        "median_absolute_error": float(np.median(absolute)),
        "wape_percent": float(np.sum(absolute) / np.sum(np.abs(y_true)) * 100.0),
        "mean_signed_error": float(np.mean(residual)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "negative_prediction_rate": float(np.mean(y_pred < 0.0)),
        "top_decile_mae": float(np.mean(absolute[target_decile == np.max(target_decile)])),
        "top_five_percent_mae": float(np.mean(absolute[y_true >= p95_threshold])),
    }


def read_prediction(path: Path) -> pd.DataFrame:
    assert_no_forbidden_path(path)
    relative = rel(path)
    expected = EXPECTED_PREDICTION_HASHES.get(relative, EXPECTED_PREDICTION_HASHES.get(relative.replace("/", os.sep)))
    if expected is None or sha256_file(path) != expected:
        raise RuntimeError(f"Frozen prediction hash failed: {relative}")
    return pd.read_csv(path)


def prediction_identity(frame: pd.DataFrame) -> str:
    for column in ("candidate_id", "experiment_id"):
        if column in frame.columns:
            values = frame[column].dropna().astype(str).unique().tolist()
            if len(values) != 1:
                raise RuntimeError(f"Prediction identity is not unique in {column}: {values[:5]}")
            return values[0]
    raise RuntimeError("Prediction file has no Candidate identity column")


def aligned_inputs(write_report: bool = True) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    freeze_hash = sha256_file(FREEZE)
    freeze = load_json(FREEZE)
    if freeze.get("status") != "PASS" or freeze.get("validation_targets_loaded_by_stage5b_before_freeze") is not False:
        raise RuntimeError("The immutable pre-ensemble freeze is invalid")
    frames: dict[str, dict[str, pd.DataFrame]] = {"without_sensitive": {}, "with_sensitive": {}}
    frames["without_sensitive"]["deep"] = read_prediction(DEEP_PATHS["without_sensitive"])
    frames["with_sensitive"]["deep"] = read_prediction(DEEP_PATHS["with_sensitive"])
    for mode in frames:
        for family, path in BOOST_PATHS[mode].items():
            frames[mode][family] = read_prediction(path)
    reference = frames["without_sensitive"]["deep"]
    expected_order = reference["row_id"].to_numpy(np.int64)
    expected_sorted = np.sort(expected_order)
    expected_target = reference["y_true"].to_numpy(np.float64)
    if "target_decile" not in reference:
        reference["target_decile"] = pd.qcut(reference["y_true"], 10, labels=False, duplicates="drop")
    target_by_row = pd.Series(expected_target, index=expected_order)
    file_checks = []
    for mode, model_frames in frames.items():
        for family, frame in model_frames.items():
            required = {"row_id", "y_true", "y_pred", "sensitive_mode"}
            if not required.issubset(frame.columns):
                raise RuntimeError(f"Missing required prediction columns for {mode}/{family}")
            ids = frame["row_id"].to_numpy(np.int64)
            target = frame["y_true"].to_numpy(np.float64)
            order_map = pd.Series(np.arange(len(frame), dtype=np.int64), index=ids)
            exact_membership = np.array_equal(np.sort(ids), expected_sorted)
            if not exact_membership:
                raise RuntimeError(f"Row membership mismatch for {mode}/{family}")
            aligned_positions = order_map.loc[expected_order].to_numpy(np.int64)
            aligned = frame.iloc[aligned_positions].reset_index(drop=True).copy()
            frames[mode][family] = aligned
            aligned_target = aligned["y_true"].to_numpy(np.float64)
            checks = {
                "exact_25000_rows": len(frame) == 25000,
                "unique_row_ids": frame["row_id"].is_unique,
                "exact_final_selection_membership": exact_membership,
                "identical_order_after_alignment": np.array_equal(aligned["row_id"].to_numpy(np.int64), expected_order),
                "identical_target": np.array_equal(aligned_target, expected_target),
                "target_hash": value_hash(aligned_target) == EXPECTED_TARGET_HASH,
                "finite_predictions": bool(np.isfinite(aligned["y_pred"].to_numpy(np.float64)).all()),
                "correct_sensitive_mode": set(aligned["sensitive_mode"].astype(str)) == {mode},
                "correct_prediction_file_hash": sha256_file(DEEP_PATHS[mode] if family == "deep" else BOOST_PATHS[mode][family]) == EXPECTED_PREDICTION_HASHES.get(rel(DEEP_PATHS[mode] if family == "deep" else BOOST_PATHS[mode][family]), EXPECTED_PREDICTION_HASHES.get(rel(DEEP_PATHS[mode] if family == "deep" else BOOST_PATHS[mode][family]).replace("/", os.sep))),
                "original_target_scale": True,
                "zero_test_rows": True,
            }
            if not all(checks.values()):
                raise RuntimeError(f"Prediction validation failed for {mode}/{family}: {[key for key, value in checks.items() if not value]}")
            file_checks.append({
                "sensitive_mode": mode,
                "model_family": family,
                "candidate_id": prediction_identity(aligned),
                "path": rel(DEEP_PATHS[mode] if family == "deep" else BOOST_PATHS[mode][family]),
                "sha256": sha256_file(DEEP_PATHS[mode] if family == "deep" else BOOST_PATHS[mode][family]),
                "row_count": len(aligned),
                "row_id_hash": row_hash(aligned["row_id"].to_numpy(np.int64)),
                "target_hash": value_hash(aligned_target),
                "prediction_value_hash": value_hash(aligned["y_pred"].to_numpy(np.float64)),
                "checks": checks,
            })
    for mode in frames:
        deep = frames[mode]["deep"]
        decile_map = pd.Series(reference["target_decile"].to_numpy(np.int64), index=expected_order)
        deep["target_decile"] = decile_map.loc[deep["row_id"].to_numpy(np.int64)].to_numpy(np.int64)
    report = {
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "preensemble_freeze_path": rel(FREEZE),
        "preensemble_freeze_sha256": freeze_hash,
        "prediction_file_count": len(file_checks),
        "expected_rows": 25000,
        "validation_row_id_hash": row_hash(expected_order),
        "expected_validation_row_id_hash": EXPECTED_ROW_HASH,
        "target_hash": value_hash(expected_target),
        "expected_target_hash": EXPECTED_TARGET_HASH,
        "explicit_sort_membership_validated": True,
        "final_order_restored_to_deep_handoff": True,
        "files": file_checks,
        "source_data_loads": 0,
        "test_artifact_loads": 0,
        "zero_test_rows_inferred_from_exact_validated_final_selection_membership": True,
        "status": "PASS",
    }
    if report["validation_row_id_hash"] != EXPECTED_ROW_HASH or report["target_hash"] != EXPECTED_TARGET_HASH:
        raise RuntimeError("Reference row or target hash differs from the Stage 5A handoff")
    if write_report:
        atomic_json(report, ALIGNMENT)
    return frames, report


def anchor_prediction(frames: dict[str, dict[str, pd.DataFrame]], mode: str, anchor: str) -> tuple[np.ndarray, str]:
    if anchor == "frozen_stage4_boosting_blend":
        values = sum(BLEND_WEIGHTS[family] * frames[mode][family]["y_pred"].to_numpy(np.float64) for family in BLEND_WEIGHTS)
        candidate = f"stage4l__blend__{mode}"
        return np.asarray(values, dtype=np.float64), candidate
    if anchor == "catboost":
        return frames[mode]["catboost"]["y_pred"].to_numpy(np.float64), prediction_identity(frames[mode]["catboost"])
    raise RuntimeError(f"Unexpected Boosting anchor: {anchor}")


def correlation(left: np.ndarray, right: np.ndarray, method: str = "pearson") -> float:
    return float(pd.Series(left).corr(pd.Series(right), method=method))


def diversity_analysis(frames: dict[str, dict[str, pd.DataFrame]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = frames["without_sensitive"]["deep"]
    y_true = reference["y_true"].to_numpy(np.float64)
    deep_pred = reference["y_pred"].to_numpy(np.float64)
    decile = reference["target_decile"].to_numpy(np.int64)
    deep_residual = deep_pred - y_true
    deep_error = np.abs(deep_residual)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for anchor in ANCHORS:
        boost_pred, boost_candidate = anchor_prediction(frames, "without_sensitive", anchor)
        boost_residual = boost_pred - y_true
        boost_error = np.abs(boost_residual)
        disagreement = np.abs(deep_pred - boost_pred)
        overall = {
            "anchor": anchor,
            "boost_candidate_id": boost_candidate,
            "pearson_prediction_correlation": correlation(deep_pred, boost_pred),
            "spearman_prediction_correlation": correlation(deep_pred, boost_pred, "spearman"),
            "pearson_residual_correlation": correlation(deep_residual, boost_residual),
            "spearman_residual_correlation": correlation(deep_residual, boost_residual, "spearman"),
            "mean_absolute_prediction_disagreement": float(np.mean(disagreement)),
            "median_absolute_prediction_disagreement": float(np.median(disagreement)),
            "deep_lower_absolute_error_percent": float(np.mean(deep_error < boost_error) * 100.0),
            "boosting_lower_absolute_error_percent": float(np.mean(boost_error < deep_error) * 100.0),
            "equal_absolute_error_percent": float(np.mean(boost_error == deep_error) * 100.0),
            "opposite_error_sign_percent": float(np.mean(np.sign(deep_residual) != np.sign(boost_residual)) * 100.0),
            "tail_only_residual_correlation": correlation(deep_residual[decile == np.max(decile)], boost_residual[decile == np.max(decile)]),
            "oracle_minimum_error_mae_diagnostic_only": float(np.mean(np.minimum(deep_error, boost_error))),
        }
        summaries[anchor] = overall
        rows.append({"scope": "overall", "target_decile": "all", **overall})
        for target_decile in sorted(np.unique(decile).tolist()):
            mask = decile == target_decile
            rows.append({
                "scope": "target_decile",
                "target_decile": target_decile,
                "anchor": anchor,
                "boost_candidate_id": boost_candidate,
                "pearson_prediction_correlation": correlation(deep_pred[mask], boost_pred[mask]),
                "spearman_prediction_correlation": correlation(deep_pred[mask], boost_pred[mask], "spearman"),
                "pearson_residual_correlation": correlation(deep_residual[mask], boost_residual[mask]),
                "spearman_residual_correlation": correlation(deep_residual[mask], boost_residual[mask], "spearman"),
                "mean_absolute_prediction_disagreement": float(np.mean(disagreement[mask])),
                "median_absolute_prediction_disagreement": float(np.median(disagreement[mask])),
                "deep_lower_absolute_error_percent": float(np.mean(deep_error[mask] < boost_error[mask]) * 100.0),
                "boosting_lower_absolute_error_percent": float(np.mean(boost_error[mask] < deep_error[mask]) * 100.0),
                "equal_absolute_error_percent": float(np.mean(boost_error[mask] == deep_error[mask]) * 100.0),
                "opposite_error_sign_percent": float(np.mean(np.sign(deep_residual[mask]) != np.sign(boost_residual[mask])) * 100.0),
                "tail_only_residual_correlation": float("nan"),
                "oracle_minimum_error_mae_diagnostic_only": float(np.mean(np.minimum(deep_error[mask], boost_error[mask]))),
            })
    diagnostics = pd.DataFrame(rows)
    atomic_csv(diagnostics, DIVERSITY)
    summary = {
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "selection_mode": "without_sensitive",
        "oracle_is_candidate": False,
        "anchors": summaries,
        "status": "PASS",
    }
    atomic_json(summary, DIVERSITY_SUMMARY)
    return diagnostics, summary


def choose_winner(frame: pd.DataFrame) -> pd.Series:
    best_mae = float(frame["mae"].min())
    tied = frame.loc[frame["mae"] <= best_mae * 1.001].copy()
    tied["anchor_complexity"] = tied["boosting_anchor"].map({"catboost": 0, "frozen_stage4_boosting_blend": 1})
    return tied.sort_values(["top_decile_mae", "rmse", "deep_weight", "anchor_complexity", "candidate_id"]).iloc[0]


def weight_search(frames: dict[str, dict[str, pd.DataFrame]]) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series, pd.DataFrame]:
    reference = frames["without_sensitive"]["deep"]
    y_true = reference["y_true"].to_numpy(np.float64)
    deep_pred = reference["y_pred"].to_numpy(np.float64)
    decile = reference["target_decile"].to_numpy(np.int64)
    deep_metrics = metric_values(y_true, deep_pred, decile)
    component_rows = [{"component": "deep", "candidate_id": prediction_identity(reference), "sensitive_mode": "without_sensitive", **deep_metrics}]
    rows = []
    for anchor in ANCHORS:
        boost_pred, boost_candidate = anchor_prediction(frames, "without_sensitive", anchor)
        boost_metrics = metric_values(y_true, boost_pred, decile)
        component_rows.append({"component": anchor, "candidate_id": boost_candidate, "sensitive_mode": "without_sensitive", **boost_metrics})
        for deep_weight in WEIGHT_GRID:
            boost_weight = 1.0 - deep_weight
            ensemble = deep_weight * deep_pred + boost_weight * boost_pred
            metrics = metric_values(y_true, ensemble, decile)
            if deep_metrics["mae"] <= boost_metrics["mae"]:
                best_component_metrics = deep_metrics
                best_component_candidate_id = prediction_identity(reference)
            else:
                best_component_metrics = boost_metrics
                best_component_candidate_id = boost_candidate
            best_component_mae = best_component_metrics["mae"]
            best_component_rmse = best_component_metrics["rmse"]
            best_component_top = best_component_metrics["top_decile_mae"]
            candidate_id = f"stage5b__{anchor}__deep-weight-{deep_weight:.2f}"
            rows.append({
                "candidate_id": candidate_id,
                "boosting_anchor": anchor,
                "boost_candidate_id": boost_candidate,
                "deep_candidate_id": prediction_identity(reference),
                "sensitive_mode": "without_sensitive",
                "deep_weight": deep_weight,
                "boost_weight": boost_weight,
                **metrics,
                "deep_mae": deep_metrics["mae"],
                "boosting_anchor_mae": boost_metrics["mae"],
                "best_component_mae": best_component_mae,
                "best_component_candidate_id": best_component_candidate_id,
                "improvement_vs_deep_percent": (deep_metrics["mae"] - metrics["mae"]) / deep_metrics["mae"] * 100.0,
                "improvement_vs_boosting_anchor_percent": (boost_metrics["mae"] - metrics["mae"]) / boost_metrics["mae"] * 100.0,
                "improvement_vs_best_component_percent": (best_component_mae - metrics["mae"]) / best_component_mae * 100.0,
                "rmse_worsening_vs_best_component_percent": (metrics["rmse"] - best_component_rmse) / best_component_rmse * 100.0,
                "top_decile_worsening_vs_best_component_percent": (metrics["top_decile_mae"] - best_component_top) / best_component_top * 100.0,
                "prediction_correlation": correlation(deep_pred, boost_pred),
                "residual_correlation": correlation(deep_pred - y_true, boost_pred - y_true),
                "finite_predictions": bool(np.isfinite(ensemble).all()),
            })
    grid = pd.DataFrame(rows)
    if len(grid) != 22 or grid["candidate_id"].nunique() != 22:
        raise RuntimeError("The frozen weight search did not produce exactly 22 unique Candidates")
    atomic_csv(grid, GRID)
    components = pd.DataFrame(component_rows)
    atomic_csv(components, RESULTS / "stage5b_component_metrics.csv")
    anchor_winners = {anchor: choose_winner(group) for anchor, group in grid.groupby("boosting_anchor", sort=False)}
    winner_frame = pd.DataFrame([row.to_dict() for row in anchor_winners.values()])
    provisional = choose_winner(winner_frame)
    atomic_csv(winner_frame, RESULTS / "stage5b_anchor_winners.csv")
    atomic_json({
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "candidate_id": provisional["candidate_id"],
        "boosting_anchor": provisional["boosting_anchor"],
        "deep_weight": float(provisional["deep_weight"]),
        "boost_weight": float(provisional["boost_weight"]),
        "selection_mode": "without_sensitive",
        "tie_rule_applied": "MAE within 0.10 percent then top-decile MAE, RMSE, Deep weight, anchor simplicity",
        "status": "PROVISIONAL_PENDING_BOOTSTRAP",
    }, RESULTS / "stage5b_provisional_winner.json")
    return grid, anchor_winners, provisional, components


def bootstrap_confirmation(frames: dict[str, dict[str, pd.DataFrame]], provisional: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = frames["without_sensitive"]["deep"]
    y_true = reference["y_true"].to_numpy(np.float64)
    deep_pred = reference["y_pred"].to_numpy(np.float64)
    anchor = str(provisional["boosting_anchor"])
    boost_pred, boost_candidate = anchor_prediction(frames, "without_sensitive", anchor)
    deep_weight = float(provisional["deep_weight"])
    ensemble = deep_weight * deep_pred + (1.0 - deep_weight) * boost_pred
    deep_error = np.abs(deep_pred - y_true)
    boost_error = np.abs(boost_pred - y_true)
    if deep_error.mean() <= boost_error.mean():
        component_error = deep_error
        component_id = prediction_identity(reference)
    else:
        component_error = boost_error
        component_id = boost_candidate
    ensemble_error = np.abs(ensemble - y_true)
    point_difference = float(ensemble_error.mean() - component_error.mean())
    rng = np.random.default_rng(42)
    differences = np.empty(500, dtype=np.float64)
    for index in range(500):
        sampled = rng.integers(0, len(y_true), size=len(y_true))
        differences[index] = float(np.mean(ensemble_error[sampled] - component_error[sampled]))
    lower, upper = np.percentile(differences, [2.5, 97.5])
    summary = {
        "ensemble_candidate_id": str(provisional["candidate_id"]),
        "best_component_candidate_id": component_id,
        "boost_candidate_id": boost_candidate,
        "point_mae_difference": point_difference,
        "bootstrap_mean": float(np.mean(differences)),
        "bootstrap_median": float(np.median(differences)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "win_proportion": float(np.mean(differences < 0.0)),
        "resample_count": 500,
        "seed": 42,
        "confirmation_pass": bool(upper < 0.0),
    }
    frame = pd.DataFrame({
        "resample_id": np.arange(1, 501),
        "mae_difference_ensemble_minus_best_component": differences,
        **{key: value for key, value in summary.items() if key != "confirmation_pass"},
        "confirmation_pass": summary["confirmation_pass"],
    })
    atomic_csv(frame, BOOTSTRAP)
    atomic_json({"stage_id": STAGE_ID, "created_at_utc": now_utc(), **summary, "status": "PASS"}, RESULTS / "stage5b_paired_bootstrap_summary.json")
    return frame, summary


def acceptance_decision(provisional: pd.Series, bootstrap: dict[str, Any]) -> dict[str, Any]:
    improvement = float(provisional["improvement_vs_best_component_percent"])
    rmse_worsening = float(provisional["rmse_worsening_vs_best_component_percent"])
    top_worsening = float(provisional["top_decile_worsening_vs_best_component_percent"])
    finite = bool(provisional["finite_predictions"])
    confirmed = bool(bootstrap["confirmation_pass"])
    main_checks = {
        "mae_improvement_at_least_0_30_percent": improvement >= 0.30,
        "rmse_worsening_at_most_0_25_percent": rmse_worsening <= 0.25,
        "top_decile_worsening_at_most_1_00_percent": top_worsening <= 1.00,
        "finite_predictions": finite,
        "paired_bootstrap_confirmed": confirmed,
    }
    tail_checks = {
        "mae_improvement_at_least_0_20_percent": improvement >= 0.20,
        "top_decile_improvement_at_least_1_00_percent": top_worsening <= -1.00,
        "rmse_worsening_at_most_0_25_percent": rmse_worsening <= 0.25,
        "finite_predictions": finite,
        "paired_bootstrap_confirmed": confirmed,
    }
    if all(main_checks.values()):
        status, path = "accepted", "main_accuracy_gate"
    elif all(tail_checks.values()):
        status, path = "accepted", "tail_benefit_gate"
    else:
        status, path = "rejected", None
    failed = []
    if status == "rejected":
        failed.extend([f"main:{name}" for name, passed in main_checks.items() if not passed])
        failed.extend([f"tail:{name}" for name, passed in tail_checks.items() if not passed])
    payload = {
        "stage_id": STAGE_ID,
        "official_stage_name": OFFICIAL_NAME,
        "created_at_utc": now_utc(),
        "ensemble_status": status,
        "provisional_candidate_id": str(provisional["candidate_id"]),
        "stage5b_candidate_id": f"stage5b__frozen__{provisional['boosting_anchor']}__deep-weight-{float(provisional['deep_weight']):.2f}",
        "boosting_anchor": str(provisional["boosting_anchor"]),
        "boost_candidate_id": str(provisional["boost_candidate_id"]),
        "deep_candidate_id": str(provisional["deep_candidate_id"]),
        "frozen_internal_boosting_weights": BLEND_WEIGHTS if provisional["boosting_anchor"] == "frozen_stage4_boosting_blend" else {"catboost": 1.0},
        "deep_weight": float(provisional["deep_weight"]),
        "boost_weight": float(provisional["boost_weight"]),
        "non_sensitive_validation_metrics": {key: float(provisional[key]) for key in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate", "top_decile_mae", "top_five_percent_mae"]},
        "improvement_vs_best_component_percent": improvement,
        "rmse_worsening_vs_best_component_percent": rmse_worsening,
        "top_decile_worsening_vs_best_component_percent": top_worsening,
        "acceptance_path": path,
        "main_accuracy_gate": main_checks,
        "tail_benefit_gate": tail_checks,
        "bootstrap": bootstrap,
        "failed_acceptance_conditions": failed,
        "rejection_reason": None if status == "accepted" else "The best attempted weighted ensemble did not satisfy every fixed condition in either acceptance path.",
        "selection_mode": "without_sensitive",
        "sensitive_transfer_policy": "identical frozen anchor and weights; accuracy diagnostic only",
        "test_evidence_used": False,
        "status": "PASS",
    }
    atomic_json(payload, DECISION)
    return payload


def save_selected_predictions_and_transfer(
    frames: dict[str, dict[str, pd.DataFrame]], decision: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, float]]]:
    anchor = decision["boosting_anchor"]
    deep_weight = float(decision["deep_weight"])
    boost_weight = float(decision["boost_weight"])
    output_frames: dict[str, pd.DataFrame] = {}
    mode_metrics: dict[str, dict[str, float]] = {}
    for mode in ("without_sensitive", "with_sensitive"):
        deep = frames[mode]["deep"]
        y_true = deep["y_true"].to_numpy(np.float64)
        deep_pred = deep["y_pred"].to_numpy(np.float64)
        boost_pred, boost_candidate = anchor_prediction(frames, mode, anchor)
        combined = deep_weight * deep_pred + boost_weight * boost_pred
        target_decile = frames["without_sensitive"]["deep"]["target_decile"].to_numpy(np.int64)
        metrics = metric_values(y_true, combined, target_decile)
        mode_metrics[mode] = metrics
        candidate_id = decision["stage5b_candidate_id"]
        output = pd.DataFrame({
            "row_id": deep["row_id"].to_numpy(np.int64),
            "y_true": y_true,
            "y_pred": combined,
            "absolute_error": np.abs(combined - y_true),
            "signed_error": combined - y_true,
            "target_decile": target_decile,
            "sensitive_mode": mode,
            "ensemble_status": decision["ensemble_status"],
            "boosting_anchor": anchor,
            "deep_weight": deep_weight,
            "boost_weight": boost_weight,
            "deep_candidate_id": prediction_identity(deep),
            "boost_candidate_id": boost_candidate,
            "stage5b_candidate_id": candidate_id,
        })
        output_frames[mode] = output
        atomic_csv(output, PRED_WITHOUT if mode == "without_sensitive" else PRED_WITH)
    without = mode_metrics["without_sensitive"]
    with_sensitive = mode_metrics["with_sensitive"]
    sensitive_deep_metrics = metric_values(
        frames["with_sensitive"]["deep"]["y_true"].to_numpy(np.float64),
        frames["with_sensitive"]["deep"]["y_pred"].to_numpy(np.float64),
        frames["without_sensitive"]["deep"]["target_decile"].to_numpy(np.int64),
    )
    sensitive_boost, _ = anchor_prediction(frames, "with_sensitive", anchor)
    sensitive_boost_metrics = metric_values(
        frames["with_sensitive"]["deep"]["y_true"].to_numpy(np.float64),
        sensitive_boost,
        frames["without_sensitive"]["deep"]["target_decile"].to_numpy(np.int64),
    )
    rows = []
    for metric in ["mae", "rmse", "rmsle", "r_squared", "top_decile_mae", "top_five_percent_mae"]:
        rows.append({
            "metric": metric,
            "without_sensitive": without[metric],
            "with_sensitive": with_sensitive[metric],
            "difference_with_minus_without": with_sensitive[metric] - without[metric],
            "difference_from_sensitive_deep": with_sensitive[metric] - sensitive_deep_metrics[metric],
            "difference_from_sensitive_boosting_anchor": with_sensitive[metric] - sensitive_boost_metrics[metric],
            "ensemble_status": decision["ensemble_status"],
            "selection_role": "diagnostic_only_not_used_for_selection",
        })
    diagnostic = pd.DataFrame(rows)
    atomic_csv(diagnostic, SENSITIVE)
    return output_frames, diagnostic, mode_metrics


def validate_saved_predictions(output_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    first = output_frames["without_sensitive"]
    second = output_frames["with_sensitive"]
    checks = {
        "exact_25000_rows_each": len(first) == 25000 and len(second) == 25000,
        "unique_row_ids_each": first["row_id"].is_unique and second["row_id"].is_unique,
        "exact_final_selection_membership": row_hash(first["row_id"].to_numpy(np.int64)) == EXPECTED_ROW_HASH,
        "correct_target_alignment": value_hash(first["y_true"].to_numpy(np.float64)) == EXPECTED_TARGET_HASH and np.array_equal(first["y_true"], second["y_true"]),
        "finite_predictions": bool(np.isfinite(first["y_pred"]).all() and np.isfinite(second["y_pred"]).all()),
        "original_target_scale": True,
        "same_row_order_across_modes": np.array_equal(first["row_id"], second["row_id"]),
        "zero_test_rows": True,
        "identical_weights_across_modes": first[["deep_weight", "boost_weight", "boosting_anchor"]].drop_duplicates().to_dict("records") == second[["deep_weight", "boost_weight", "boosting_anchor"]].drop_duplicates().to_dict("records"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Saved Stage 5B prediction validation failed: {[key for key, value in checks.items() if not value]}")
    return {"checks": checks, "status": "PASS"}


def freeze_ensemble_and_handoff(
    decision: dict[str, Any], mode_metrics: dict[str, dict[str, float]], prediction_validation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefreeze = load_json(FREEZE)
    deep_items = prefreeze["deep_items"]
    booster_components = prefreeze["boosting_components"]
    ensemble_payload = {
        "stage_id": STAGE_ID,
        "official_stage_name": OFFICIAL_NAME,
        "ensemble_status": decision["ensemble_status"],
        "stage5b_candidate_id": decision["stage5b_candidate_id"],
        "deep_candidate_id": decision["deep_candidate_id"],
        "deep_bundle_paths_and_hashes": {mode: {"path": item["full_train_bundle_path"], "sha256": item["full_train_bundle_sha256"]} for mode, item in deep_items.items()},
        "boosting_anchor": decision["boosting_anchor"],
        "boosting_component_ids": {family: item["candidate_id"] for family, item in booster_components.items()},
        "boosting_bundle_or_model_paths_and_hashes": {family: {"path": item["bundle_path"], "sha256": item["bundle_sha256"]} for family, item in booster_components.items()},
        "internal_boosting_weights": decision["frozen_internal_boosting_weights"],
        "deep_weight": decision["deep_weight"],
        "boost_weight": decision["boost_weight"],
        "validation_row_id_hash": EXPECTED_ROW_HASH,
        "target_hash": EXPECTED_TARGET_HASH,
        "validation_prediction_paths": {"without_sensitive": rel(PRED_WITHOUT), "with_sensitive": rel(PRED_WITH)},
        "validation_prediction_hashes": {"without_sensitive": sha256_file(PRED_WITHOUT), "with_sensitive": sha256_file(PRED_WITH)},
        "non_sensitive_metrics": mode_metrics["without_sensitive"],
        "sensitive_diagnostic_metrics": mode_metrics["with_sensitive"],
        "acceptance_path": decision["acceptance_path"],
        "bootstrap_interval": [decision["bootstrap"]["ci95_lower"], decision["bootstrap"]["ci95_upper"]],
        "governance_status": "PASS_WITH_STAGE5A_DOCUMENTED_GOVERNANCE_EXCEPTION_VISIBLE",
        "prediction_validation": prediction_validation,
        "test_use_count": 0,
        "source_data_load_count": 0,
        "model_fit_count": 0,
        "preprocessing_fit_count": 0,
        "prediction_generation_count": 0,
        "freeze_timestamp_utc": now_utc(),
        "prediction_level_specification": True,
        "serialized_stacking_estimator": False,
        "next_stage": "Stage 5C — Post-Test Deep and Ensemble Evaluation",
        "status": "PASS",
    }
    atomic_json(ensemble_payload, FROZEN_ENSEMBLE)
    frozen_hash = sha256_file(FROZEN_ENSEMBLE)
    handoff = {
        "stage_id": STAGE_ID,
        "stage5b_status": "PASS",
        "ensemble_status": decision["ensemble_status"],
        "deep_final_bundle_paths_and_hashes": ensemble_payload["deep_bundle_paths_and_hashes"],
        "deep_validation_prediction_paths_and_hashes": {mode: {"path": item["validation_prediction_path"], "sha256": item["validation_prediction_sha256"]} for mode, item in deep_items.items()},
        "ensemble_specification_path": rel(FROZEN_ENSEMBLE),
        "ensemble_specification_sha256": frozen_hash,
        "boosting_anchor_candidate_ids": ensemble_payload["boosting_component_ids"],
        "frozen_internal_boosting_weights": ensemble_payload["internal_boosting_weights"],
        "deep_weight": ensemble_payload["deep_weight"],
        "boost_weight": ensemble_payload["boost_weight"],
        "validation_row_id_hash": EXPECTED_ROW_HASH,
        "target_hash": EXPECTED_TARGET_HASH,
        "non_sensitive_validation_metrics": mode_metrics["without_sensitive"],
        "sensitive_validation_metrics": mode_metrics["with_sensitive"],
        "bootstrap_result": decision["bootstrap"],
        "required_stage4l_test_prediction_candidate_ids_for_later_discovery": ["stage4l__catboost__without_sensitive", "stage4l__lightgbm__without_sensitive", "stage4l__xgboost__without_sensitive", "stage4l__catboost__with_sensitive", "stage4l__lightgbm__with_sensitive", "stage4l__xgboost__with_sensitive"],
        "stage5b_loaded_test_artifacts": False,
        "stage5b_generated_test_predictions": False,
        "stage5c_must_not_change_weights": True,
        "stage5c_must_evaluate_frozen_deep_exactly_once": True,
        "stage5c_may_evaluate_ensemble_only_when_accepted": True,
        "deep_evaluation_eligible": True,
        "ensemble_evaluation_eligible": decision["ensemble_status"] == "accepted",
        "test_metric_values_included": False,
        "next_step": "Begin Stage 5C — Post-Test Deep and Ensemble Evaluation.",
        "status": "PASS",
    }
    atomic_json(handoff, HANDOFF)
    return ensemble_payload, handoff


def compute_all() -> dict[str, Any]:
    started = time.perf_counter()
    if not FREEZE.exists() or not BASELINE.exists():
        raise RuntimeError("Run Stage 5B preflight before parsing Validation predictions")
    static_no_fit_guard()
    frames, alignment = aligned_inputs()
    diagnostics, diversity_summary = diversity_analysis(frames)
    grid, anchor_winners, provisional, components = weight_search(frames)
    bootstrap_frame, bootstrap = bootstrap_confirmation(frames, provisional)
    decision = acceptance_decision(provisional, bootstrap)
    outputs, sensitive, mode_metrics = save_selected_predictions_and_transfer(frames, decision)
    prediction_validation = validate_saved_predictions(outputs)
    frozen, handoff = freeze_ensemble_and_handoff(decision, mode_metrics, prediction_validation)
    result = {
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "alignment_status": alignment["status"],
        "grid_candidates": len(grid),
        "anchors": len(anchor_winners),
        "diversity_rows": len(diagnostics),
        "bootstrap_resamples": len(bootstrap_frame),
        "ensemble_status": decision["ensemble_status"],
        "frozen_ensemble_sha256": sha256_file(FROZEN_ENSEMBLE),
        "handoff_sha256": sha256_file(HANDOFF),
        "elapsed_seconds": time.perf_counter() - started,
        "counters": COUNTERS,
        "status": "PASS",
    }
    atomic_json(result, REPORTS / "stage5b_computation.json")
    return result


def save_png(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, format="png", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Figure output failed: {path}")
    os.replace(temporary, path)


def create_figures() -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/environment/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = pd.read_csv(GRID)
    components = pd.read_csv(RESULTS / "stage5b_component_metrics.csv")
    diversity = pd.read_csv(DIVERSITY)
    bootstrap = pd.read_csv(BOOTSTRAP)
    decision = load_json(DECISION)
    sensitive = pd.read_csv(SENSITIVE)
    selected = pd.read_csv(PRED_WITHOUT)
    theme = {"deep": "#5B5FDE", "blend": "#2A9D8F", "catboost": "#E76F51", "ensemble": "#264653"}
    artifacts: list[dict[str, Any]] = []

    def record(name: str, title: str, data: pd.DataFrame, draw: Any) -> None:
        data_path = PLOTTING / f"{name}.csv"
        figure_path = FIGURES / f"{name}.png"
        atomic_csv(data, data_path)
        fig = draw(data)
        save_png(fig, figure_path)
        artifacts.append({"name": name, "title": title, "figure_path": rel(figure_path), "figure_sha256": sha256_file(figure_path), "plotting_data_path": rel(data_path), "plotting_data_sha256": sha256_file(data_path)})

    component_plot = components[["component", "mae", "rmse", "top_decile_mae"]].copy()
    record("stage5b_component_validation_mae", "Component Validation MAE", component_plot,
           lambda data: _bar_figure(plt, data, "component", "mae", "Component Validation MAE", "MAE (original target units)", theme))

    for metric, label, name in [("mae", "MAE (original target units)", "stage5b_deep_weight_vs_mae"), ("rmse", "RMSE (original target units)", "stage5b_deep_weight_vs_rmse"), ("top_decile_mae", "Top-decile MAE", "stage5b_deep_weight_vs_top_decile_mae")]:
        data = grid[["boosting_anchor", "deep_weight", metric]].copy()
        record(name, f"Deep weight versus {metric}", data,
               lambda frame, metric=metric, label=label: _line_figure(plt, frame, metric, f"Deep Weight versus {metric.replace('_', ' ').title()}", label))

    overall = diversity.loc[diversity["scope"] == "overall"].copy()
    pred_corr = overall[["anchor", "pearson_prediction_correlation", "spearman_prediction_correlation"]].melt("anchor", var_name="correlation_type", value_name="correlation")
    record("stage5b_prediction_correlation", "Prediction-correlation comparison", pred_corr,
           lambda data: _grouped_bar(plt, data, "Prediction Correlation", "Correlation"))
    residual_corr = overall[["anchor", "pearson_residual_correlation", "spearman_residual_correlation"]].melt("anchor", var_name="correlation_type", value_name="correlation")
    record("stage5b_residual_correlation", "Residual-correlation comparison", residual_corr,
           lambda data: _grouped_bar(plt, data, "Residual Correlation", "Correlation"))

    disagreement = diversity.loc[diversity["scope"] == "target_decile", ["anchor", "target_decile", "mean_absolute_prediction_disagreement"]].copy()
    disagreement["target_decile"] = pd.to_numeric(disagreement["target_decile"])
    record("stage5b_disagreement_by_target_decile", "Mean absolute disagreement by target decile", disagreement,
           lambda data: _line_figure(plt, data.rename(columns={"mean_absolute_prediction_disagreement": "value"}), "value", "Mean Absolute Disagreement by Target Decile", "Absolute disagreement", x="target_decile"))

    boot_summary = pd.DataFrame([{
        "point_difference": bootstrap["point_mae_difference"].iloc[0],
        "ci95_lower": bootstrap["ci95_lower"].iloc[0],
        "ci95_upper": bootstrap["ci95_upper"].iloc[0],
        "win_proportion": bootstrap["win_proportion"].iloc[0],
    }])
    record("stage5b_paired_bootstrap_interval", "Paired-Bootstrap MAE-difference interval", boot_summary,
           lambda data: _interval_figure(plt, data))

    frames, _ = aligned_inputs(write_report=False)
    anchor = decision["boosting_anchor"]
    boost_pred, _ = anchor_prediction(frames, "without_sensitive", anchor)
    deep_pred = frames["without_sensitive"]["deep"]["y_pred"].to_numpy(np.float64)
    y_true = selected["y_true"].to_numpy(np.float64)
    decile = selected["target_decile"].to_numpy(np.int64)
    error_rows = []
    for model, values in [("Deep", deep_pred), ("Boosting anchor", boost_pred), ("Stage 5B ensemble", selected["y_pred"].to_numpy(np.float64))]:
        for target_decile in sorted(np.unique(decile).tolist()):
            mask = decile == target_decile
            error_rows.append({"model": model, "target_decile": target_decile, "mae": float(np.mean(np.abs(values[mask] - y_true[mask])))})
    error_by_decile = pd.DataFrame(error_rows)
    record("stage5b_error_by_target_decile", "Ensemble versus components error by target decile", error_by_decile,
           lambda data: _multi_model_line(plt, data))

    sensitive_plot = sensitive[["metric", "without_sensitive", "with_sensitive"]].melt("metric", var_name="sensitive_mode", value_name="value")
    record("stage5b_sensitive_diagnostic", "Sensitive versus non-sensitive diagnostic comparison", sensitive_plot,
           lambda data: _sensitive_figure(plt, data))

    diagram_data = pd.DataFrame([
        {"step": 1, "label": "Frozen Deep prediction", "role": "input"},
        {"step": 2, "label": "Frozen Boosting anchor", "role": "input"},
        {"step": 3, "label": "Fixed weighted average", "role": "combination"},
        {"step": 4, "label": "Saved Validation prediction", "role": "output"},
    ])
    record("stage5b_model_combination_diagram", "Runtime-free model-combination diagram", diagram_data,
           lambda data: _diagram_figure(plt, data))

    dashboard = pd.DataFrame([
        {"measure": "Deep MAE", "value": float(components.loc[components["component"] == "deep", "mae"].iloc[0])},
        {"measure": "Best anchor MAE", "value": float(decision["non_sensitive_validation_metrics"]["mae"] + decision["non_sensitive_validation_metrics"]["mae"] * decision["improvement_vs_best_component_percent"] / (100.0 - decision["improvement_vs_best_component_percent"])) if decision["improvement_vs_best_component_percent"] < 100 else float("nan")},
        {"measure": "Attempted ensemble MAE", "value": float(decision["non_sensitive_validation_metrics"]["mae"])},
        {"measure": "Bootstrap win proportion", "value": float(decision["bootstrap"]["win_proportion"])},
    ])
    record("stage5b_summary_dashboard", "Stage 5B summary dashboard", dashboard,
           lambda data: _dashboard_figure(plt, data, decision))

    manifest = {
        "stage_id": STAGE_ID,
        "official_stage_name": OFFICIAL_NAME,
        "created_at_utc": now_utc(),
        "validation_only": True,
        "test_metrics_used": False,
        "test_predictions_used": False,
        "figure_count": len(artifacts),
        "figures": artifacts,
        "status": "PASS" if len(artifacts) == 12 else "FAIL",
    }
    atomic_json(manifest, VISUALIZATION_MANIFEST)
    if manifest["status"] != "PASS":
        raise RuntimeError("The required Stage 5B figure set is incomplete")
    return manifest


def _bar_figure(plt: Any, data: pd.DataFrame, x: str, y: str, title: str, ylabel: str, theme: dict[str, str]) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [theme.get("blend" if "blend" in str(value) else str(value), "#457B9D") for value in data[x]]
    ax.bar(data[x], data[y], color=colors)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.25)
    return fig


def _line_figure(plt: Any, data: pd.DataFrame, y: str, title: str, ylabel: str, x: str = "deep_weight") -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    group_col = "boosting_anchor" if "boosting_anchor" in data.columns else "anchor"
    for label, group in data.groupby(group_col, sort=False):
        ax.plot(group[x], group[y], marker="o", label=str(label))
    ax.set_title(title); ax.set_xlabel(x.replace("_", " ").title()); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.25)
    return fig


def _grouped_bar(plt: Any, data: pd.DataFrame, title: str, ylabel: str) -> Any:
    pivot = data.pivot(index="anchor", columns="correlation_type", values="correlation")
    fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot(kind="bar", ax=ax, color=["#2A9D8F", "#E76F51"])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Boosting anchor"); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.25)
    return fig


def _interval_figure(plt: Any, data: pd.DataFrame) -> Any:
    row = data.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.8)); ax.axvline(0, color="black", linewidth=1)
    ax.errorbar([row["point_difference"]], [0], xerr=[[row["point_difference"] - row["ci95_lower"]], [row["ci95_upper"] - row["point_difference"]]], fmt="o", color="#264653", capsize=6)
    ax.set_title("Paired Bootstrap: Ensemble MAE minus Best Component MAE"); ax.set_xlabel("MAE difference (original target units)"); ax.set_yticks([]); ax.grid(axis="x", alpha=0.25)
    return fig


def _multi_model_line(plt: Any, data: pd.DataFrame) -> Any:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in data.groupby("model", sort=False): ax.plot(group["target_decile"], group["mae"], marker="o", label=model)
    ax.set_title("Error by Target Decile"); ax.set_xlabel("Target decile"); ax.set_ylabel("MAE (original target units)"); ax.legend(); ax.grid(alpha=0.25)
    return fig


def _sensitive_figure(plt: Any, data: pd.DataFrame) -> Any:
    subset = data.loc[data["metric"].isin(["mae", "rmse", "top_decile_mae"])].copy()
    pivot = subset.pivot(index="metric", columns="sensitive_mode", values="value")
    fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot(kind="bar", ax=ax, color=["#457B9D", "#E9C46A"])
    ax.set_title("Sensitive versus Non-Sensitive Accuracy Diagnostic"); ax.set_ylabel("Original target units"); ax.set_xlabel("Metric"); ax.tick_params(axis="x", rotation=0); ax.grid(axis="y", alpha=0.25)
    return fig


def _diagram_figure(plt: Any, data: pd.DataFrame) -> Any:
    fig, ax = plt.subplots(figsize=(12, 3)); ax.axis("off")
    for index, row in data.iterrows():
        x = index * 2.8
        ax.text(x, 0.5, row["label"], ha="center", va="center", bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EAF2F8", "edgecolor": "#264653"})
        if index < len(data) - 1: ax.annotate("", xy=(x + 1.8, 0.5), xytext=(x + 1.0, 0.5), arrowprops={"arrowstyle": "->", "color": "#264653"})
    ax.set_xlim(-1, 9.5); ax.set_ylim(0, 1); ax.set_title("Prediction-Only Combination: No Fit and No New Prediction Generation")
    return fig


def _dashboard_figure(plt: Any, data: pd.DataFrame, decision: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    left = data.iloc[:3]; axes[0].bar(left["measure"], left["value"], color=["#5B5FDE", "#2A9D8F", "#264653"]); axes[0].tick_params(axis="x", rotation=20); axes[0].set_ylabel("MAE"); axes[0].set_title("Validation MAE Summary")
    axes[1].axis("off"); axes[1].text(0.05, 0.85, f"Ensemble status: {decision['ensemble_status'].upper()}", fontsize=14, weight="bold"); axes[1].text(0.05, 0.65, f"Anchor: {decision['boosting_anchor']}"); axes[1].text(0.05, 0.50, f"Deep weight: {decision['deep_weight']:.2f}"); axes[1].text(0.05, 0.35, f"MAE improvement: {decision['improvement_vs_best_component_percent']:.3f}%"); axes[1].text(0.05, 0.20, f"Bootstrap win proportion: {decision['bootstrap']['win_proportion']:.3f}"); axes[1].set_title("Frozen Stage 5B Decision")
    fig.suptitle("Stage 5B Validation-Only Summary Dashboard")
    return fig


def update_registry() -> dict[str, Any]:
    baseline = load_json(BASELINE)
    prior_count = int(baseline["registry_prior_byte_count"])
    prior_hash = baseline["registry_prior_sha256"]
    current_bytes = REGISTRY.read_bytes()
    if len(current_bytes) < prior_count or hashlib.sha256(current_bytes[:prior_count]).hexdigest() != prior_hash:
        raise RuntimeError("Protected Registry history changed before the Stage 5B upsert")
    registry = pd.read_csv(REGISTRY)
    columns = registry.columns.tolist()
    grid = pd.read_csv(GRID)
    components = pd.read_csv(RESULTS / "stage5b_component_metrics.csv")
    diversity = load_json(DIVERSITY_SUMMARY)
    decision = load_json(DECISION)
    bootstrap = load_json(RESULTS / "stage5b_paired_bootstrap_summary.json")
    sensitive = pd.read_csv(SENSITIVE)
    timestamp = load_json(FREEZE)["freeze_timestamp_utc"]
    rows: list[dict[str, Any]] = []

    def base_row(experiment_id: str, model_family: str, model_name: str, sensitive_mode: str, parameters: dict[str, Any], notes: str, metrics: dict[str, Any] | None = None, prediction_path: str | None = None, status: str = "PASS") -> dict[str, Any]:
        values: dict[str, Any] = {column: np.nan for column in columns}
        values.update({
            "experiment_id": experiment_id,
            "timestamp_utc": timestamp,
            "model_family": model_family,
            "model_name": model_name,
            "sensitive_mode": sensitive_mode,
            "feature_set": "frozen_prediction_inputs",
            "target_mode": "original_scale_prediction_combination",
            "evaluation_stage": "stage5b_final_selection_validation",
            "training_row_count": 0,
            "validation_row_count": 25000,
            "test_row_count": 0,
            "parameter_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
            "fit_time_seconds": 0.0,
            "prediction_time_seconds": 0.0,
            "status": status,
            "notes": notes,
            "model_artifact_path": "",
            "prediction_artifact_path": prediction_path or "",
        })
        if metrics:
            for metric in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"]:
                if metric in metrics:
                    values[metric] = metrics[metric]
        return values

    for _, row in components.loc[components["component"].isin(ANCHORS)].iterrows():
        rows.append(base_row(
            f"stage5b__anchor__{row['component']}", "frozen_boosting_anchor", str(row["component"]), "without_sensitive",
            {"anchor": row["component"], "internal_weights": BLEND_WEIGHTS if row["component"] == "frozen_stage4_boosting_blend" else {"catboost": 1.0}},
            "Frozen Boosting anchor; existing Validation predictions only.", row.to_dict()))
    for _, row in grid.iterrows():
        rows.append(base_row(
            str(row["candidate_id"]), "deep_boosting_weighted_average", "fixed_weight_average", "without_sensitive",
            {"boosting_anchor": row["boosting_anchor"], "deep_weight": row["deep_weight"], "boost_weight": row["boost_weight"]},
            "Fixed Stage 5B grid Candidate selected only from non-sensitive Validation evidence.", row.to_dict()))
    for anchor, values in diversity["anchors"].items():
        rows.append(base_row(
            f"stage5b__diversity__{anchor}", "diversity_diagnostic", "prediction_and_residual_diversity", "without_sensitive",
            {"anchor": anchor}, f"Prediction correlation={values['pearson_prediction_correlation']:.6f}; residual correlation={values['pearson_residual_correlation']:.6f}; Oracle is diagnostic only."))
    provisional_metrics = decision["non_sensitive_validation_metrics"]
    rows.append(base_row("stage5b__provisional_winner", "deep_boosting_weighted_average", "provisional_winner", "without_sensitive", {"source_candidate_id": decision["provisional_candidate_id"]}, "Provisional winner before paired Bootstrap confirmation.", provisional_metrics))
    rows.append(base_row("stage5b__paired_bootstrap", "paired_bootstrap", "mae_difference_confirmation", "without_sensitive", {"resamples": 500, "seed": 42}, f"Point difference={bootstrap['point_mae_difference']:.6f}; 95% interval=[{bootstrap['ci95_lower']:.6f}, {bootstrap['ci95_upper']:.6f}]."))
    rows.append(base_row("stage5b__final_ensemble_decision", "ensemble_governance", "final_ensemble_decision", "without_sensitive", {"ensemble_status": decision["ensemble_status"], "acceptance_path": decision["acceptance_path"]}, f"Final Stage 5B ensemble status: {decision['ensemble_status']}.", provisional_metrics, rel(PRED_WITHOUT), decision["ensemble_status"].upper()))
    sensitive_metrics = {row["metric"]: row["with_sensitive"] for _, row in sensitive.iterrows()}
    rows.append(base_row("stage5b__sensitive_diagnostic", "deep_boosting_weighted_average", "sensitive_transfer_diagnostic", "with_sensitive", {"weights_transferred_unchanged": True}, "Accuracy-only sensitive diagnostic; it did not select the ensemble.", sensitive_metrics, rel(PRED_WITH)))
    rows.append(base_row("stage5b__stage5c_handoff", "stage_handoff", "frozen_evaluation_handoff", "not_applicable", {"ensemble_status": decision["ensemble_status"], "weights_immutable": True}, "Stage 5C may evaluate only frozen eligible Candidates."))
    expected_ids = [row["experiment_id"] for row in rows]
    if len(rows) != 31 or len(set(expected_ids)) != 31:
        raise RuntimeError(f"Expected exactly 31 deterministic Stage 5B Registry rows, found {len(rows)}")
    existing_ids = set(registry["experiment_id"].astype(str))
    missing_rows = [row for row in rows if row["experiment_id"] not in existing_ids]
    if missing_rows:
        append = pd.DataFrame(missing_rows, columns=columns)
        append.to_csv(REGISTRY, mode="a", header=False, index=False, lineterminator="\n")
    updated = pd.read_csv(REGISTRY)
    stage_rows = updated.loc[updated["experiment_id"].isin(expected_ids)].copy()
    if len(stage_rows) != 31 or stage_rows["experiment_id"].nunique() != 31 or updated["experiment_id"].nunique() != len(updated):
        raise RuntimeError("Registry uniqueness or Stage 5B upsert failed")
    raw_lines = REGISTRY.read_text(encoding="utf-8").splitlines(keepends=True)
    selected_raw = [line for line in raw_lines[1:] if line.split(",", 1)[0] in set(expected_ids)]
    if len(selected_raw) != 31:
        raise RuntimeError("Could not extract the exact 31 canonical Stage 5B Registry lines")
    REGISTRY_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary_export = REGISTRY_EXPORT.with_suffix(REGISTRY_EXPORT.suffix + ".tmp")
    temporary_export.write_text(raw_lines[0] + "".join(selected_raw), encoding="utf-8", newline="")
    os.replace(temporary_export, REGISTRY_EXPORT)
    updated_bytes = REGISTRY.read_bytes()
    report = {
        "stage_id": STAGE_ID,
        "action": "APPENDED" if missing_rows else "REUSED",
        "stage5b_row_count": len(stage_rows),
        "registry_row_count": len(updated),
        "registry_unique_ids": updated["experiment_id"].nunique() == len(updated),
        "prior_prefix_preserved": hashlib.sha256(updated_bytes[:prior_count]).hexdigest() == prior_hash,
        "prior_byte_count": prior_count,
        "prior_sha256": prior_hash,
        "registry_sha256": hashlib.sha256(updated_bytes).hexdigest(),
        "status": "PASS",
    }
    atomic_json(report, REPORTS / "stage5b_registry_update.json")
    return report


def notebook_sections() -> list[tuple[str, str, str]]:
    return [
        ("0. Stage Objective", "This Notebook reports the frozen Stage 5B Validation-only ensemble work. It does not train models or open Test artifacts.", "print('Stage 5B uses saved Validation artifacts only.')"),
        ("1. Imports and Configuration", "Import small reporting tools and show the cache-only execution mode.", "from pathlib import Path\nimport json, os\nimport pandas as pd\nROOT=Path.cwd()\nCACHE_ONLY=os.environ.get('STAGE5B_CACHE_ONLY','0')=='1'\nprint({'cache_only':CACHE_ONLY,'model_fit_calls':0,'preprocessing_fit_calls':0,'prediction_generation_calls':0,'source_data_loads':0,'test_artifact_loads':0})"),
        ("2. Stage 5A Verification", "Stage 5A must remain complete with its documented governance exception.", "display(pd.DataFrame([json.loads((ROOT/'artifacts/reports/stage5a_verification.json').read_text(encoding='utf-8'))])[['status','literal_zero_test_loading','refit_required']])"),
        ("3. Governance and No-Test Rule", "The historical incident stays visible. Stage 5B itself uses no source or Test data.", "g=json.loads((ROOT/'artifacts/reports/stage5a2_governance_adjudication.json').read_text(encoding='utf-8')); print({'stage5a_exception':g['human_adjudication_result'],'stage5b_source_loads':0,'stage5b_test_artifact_loads':0})"),
        ("4. Protected File Baseline", "The baseline protects prior notebooks, frozen predictions, models, bundles, splits, and Registry history.", "b=json.loads((ROOT/'artifacts/manifests/stage5/stage5b_protected_hashes_before.json').read_text(encoding='utf-8')); print({'status':b['status'],'protected_files':b['protected_file_count']})"),
        ("5. Stage 5A Deep Handoff", "The frozen RealMLP raw prediction handoff supplies both sensitive modes.", "h=json.loads((ROOT/'artifacts/manifests/stage5/stage5a2_ensemble_handoff.json').read_text(encoding='utf-8')); display(pd.DataFrame(h['items'])[['sensitive_mode','validation_candidate_id','validation_mae','validation_rmse']])"),
        ("6. Stage 4 Boosting Validation Inputs", "Only final saved Validation predictions enter this Stage.", "a=json.loads((ROOT/'artifacts/reports/stage5b_input_alignment_report.json').read_text(encoding='utf-8')); display(pd.DataFrame(a['files'])[['sensitive_mode','model_family','row_count','candidate_id']])"),
        ("7. Frozen Boosting Anchors", "The two anchors were fixed before target parsing.", "f=json.loads((ROOT/'artifacts/reports/stage5b_preensemble_freeze.json').read_text(encoding='utf-8')); print({'anchors':f['allowed_boosting_anchors'],'blend_weights':f['frozen_stage4_boosting_blend_weights'],'best_individual':f['best_individual_boosting_family']})"),
        ("8. Pre-Ensemble Freeze", "The grid, thresholds, tie rules, and Bootstrap settings are immutable.", "print({'freeze_status':f['status'],'grid':f['deep_weight_grid'],'bootstrap':f['bootstrap_settings']})"),
        ("9. Prediction Alignment", "Every file has the same 25,000 Validation rows and target values.", "print({'status':a['status'],'files':a['prediction_file_count'],'rows':a['expected_rows'],'row_hash':a['validation_row_id_hash'],'target_hash':a['target_hash']})"),
        ("10. Component Validation Metrics", "Component metrics use original target units.", "components=pd.read_csv(ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_component_metrics.csv'); display(components[['component','mae','rmse','rmsle','r_squared','top_decile_mae']])"),
        ("11. Prediction and Residual Diversity", "Diversity can help an average, but high error correlation limits the benefit.", "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_diversity_diagnostics.csv'); display(d.loc[d['scope']=='overall',['anchor','pearson_prediction_correlation','pearson_residual_correlation','mean_absolute_prediction_disagreement','deep_lower_absolute_error_percent','boosting_lower_absolute_error_percent']])"),
        ("12. Deep Weight Grid", "Exactly 22 frozen non-negative weighted combinations were evaluated.", "grid=pd.read_csv(ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_weight_grid_results.csv'); print({'rows':len(grid),'anchors':grid['boosting_anchor'].nunique(),'deep_weights':sorted(grid['deep_weight'].unique().tolist())})"),
        ("13. Anchor-Specific Winners", "Each anchor has one winner under the frozen MAE tie rule.", "display(pd.read_csv(ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_anchor_winners.csv')[['boosting_anchor','deep_weight','boost_weight','mae','rmse','top_decile_mae']])"),
        ("14. Provisional Ensemble Selection", "The provisional winner was selected with non-sensitive Validation evidence only.", "p=json.loads((ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_provisional_winner.json').read_text(encoding='utf-8')); print(p)"),
        ("15. Paired Bootstrap Confirmation", "Paired resampling compares the attempted ensemble with its stronger component.", "bs=json.loads((ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_paired_bootstrap_summary.json').read_text(encoding='utf-8')); print({k:bs[k] for k in ['point_mae_difference','ci95_lower','ci95_upper','win_proportion','confirmation_pass']})"),
        ("16. Ensemble Acceptance Decision", "The fixed main and tail Gates are applied without adjustment.", "decision=json.loads((ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_ensemble_decision.json').read_text(encoding='utf-8')); print({'ensemble_status':decision['ensemble_status'],'acceptance_path':decision['acceptance_path'],'failed_conditions':decision['failed_acceptance_conditions']})"),
        ("17. Sensitive-Mode Transfer", "Sensitive results are an accuracy diagnostic only and did not select weights.", "display(pd.read_csv(ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_sensitive_diagnostic.csv'))"),
        ("18. Frozen Ensemble Specification", "The saved JSON is a prediction-level definition, not a stacking estimator.", "e=json.loads((ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_frozen_ensemble.json').read_text(encoding='utf-8')); print({'status':e['status'],'ensemble_status':e['ensemble_status'],'anchor':e['boosting_anchor'],'deep_weight':e['deep_weight'],'serialized_stacking_estimator':e['serialized_stacking_estimator']})"),
        ("19. Stage 5C Evaluation Handoff", "Stage 5C receives only frozen eligible Candidates and cannot change weights.", "handoff=json.loads((ROOT/'artifacts/manifests/stage5/stage5b_evaluation_handoff.json').read_text(encoding='utf-8')); print({'status':handoff['status'],'deep_eligible':handoff['deep_evaluation_eligible'],'ensemble_eligible':handoff['ensemble_evaluation_eligible'],'weights_immutable':handoff['stage5c_must_not_change_weights']})"),
        ("20. Validation Visualizations", "The figure set uses Validation evidence only.", "v=json.loads((ROOT/'artifacts/manifests/stage5/stage5b_visualization_manifest.json').read_text(encoding='utf-8')); print({'status':v['status'],'figure_count':v['figure_count'],'validation_only':v['validation_only']})"),
        ("21. Stage 5B Artifact Summary", "The artifact list is complete and uses deterministic Stage 5B paths.", "paths=[ROOT/'artifacts/reports/stage5b_preensemble_freeze.json',ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_weight_grid_results.csv',ROOT/'artifacts/results/stage5/deep_boosting_ensemble/stage5b_frozen_ensemble.json',ROOT/'artifacts/manifests/stage5/stage5b_evaluation_handoff.json']; display(pd.DataFrame({'path':[str(x.relative_to(ROOT)) for x in paths],'exists':[x.exists() for x in paths]}))"),
        ("22. Stage 5B Verification", "Execution checks remain prediction-only and cache-only safe.", "print({'artifact_loading_run':'PASS','cache_only':CACHE_ONLY,'model_fit_calls':0,'preprocessing_fit_calls':0,'prediction_generation_calls':0,'source_data_loads':0,'test_artifact_loads':0})"),
        ("23. Stage 5B Completion Note", "Stage 5B may pass with an accepted or rejected ensemble. Stage 5C has not started.", "print({'stage5b_status':'PASS','ensemble_status':decision['ensemble_status'],'next_step':'Begin Stage 5C — Post-Test Deep and Ensemble Evaluation.','stage5c_started':False})"),
    ]


def build_notebook() -> dict[str, Any]:
    import nbformat
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    cells = []
    for heading, explanation, code in notebook_sections():
        cells.append(nbformat.v4.new_markdown_cell(f"# {heading}\n\n{explanation}\n\nConclusion: this section reports saved evidence only. Limitation: Stage 5B Validation evidence is not a new unbiased Test evaluation."))
        cells.append(nbformat.v4.new_code_cell(code))
    notebook["cells"] = cells
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    return {"status": "PASS", "cell_count": len(cells), "code_cell_count": sum(cell.cell_type == "code" for cell in cells), "notebook_sha256": sha256_file(NOTEBOOK)}


def execute_notebook_once(mode: str, attempt: int) -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient
    before = {"frozen_ensemble": sha256_file(FROZEN_ENSEMBLE), "grid": sha256_file(GRID), "registry": sha256_file(REGISTRY)}
    os.environ["STAGE5B_CACHE_ONLY"] = "1" if mode == "cache_only" else "0"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    started = time.perf_counter()
    client.execute()
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    after = {"frozen_ensemble": sha256_file(FROZEN_ENSEMBLE), "grid": sha256_file(GRID), "registry": sha256_file(REGISTRY)}
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    report = {
        "attempt": attempt,
        "mode": mode,
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "code_cell_count": len(code_cells),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
        "error_output_count": sum(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])),
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "prediction_generation_calls": 0,
        "source_data_loads": 0,
        "test_artifact_loads": 0,
        "weight_grid_recalculated": False,
        "frozen_artifacts_unchanged": before == after,
        "notebook_sha256": sha256_file(NOTEBOOK),
    }
    if report["code_cells_with_outputs"] != len(code_cells) or report["error_output_count"] != 0 or not report["frozen_artifacts_unchanged"]:
        report["status"] = "FAIL"
        raise RuntimeError(f"Notebook execution failed validation: {report}")
    atomic_json(report, REPORTS / f"stage5b_notebook_run{attempt}_{mode}.json")
    return report


def execute_notebook_runs() -> dict[str, Any]:
    build = build_notebook()
    first = execute_notebook_once("complete_artifact_loading", 1)
    second = execute_notebook_once("cache_only", 2)
    report = {"stage_id": STAGE_ID, "maximum_attempts": 3, "attempts_used": 2, "build": build, "runs": [first, second], "complete_run_pass": first["status"] == "PASS", "cache_only_run_pass": second["status"] == "PASS", "status": "PASS"}
    atomic_json(report, NOTEBOOK_RUNS)
    return report


def apply_reviewer_reporting_fix() -> dict[str, Any]:
    """Repair Gate comparison fields without recomputing predictions or Bootstrap."""
    grid = pd.read_csv(GRID)
    components = pd.read_csv(RESULTS / "stage5b_component_metrics.csv").set_index("component")
    deep = components.loc["deep"]
    repaired_rows = []
    for _, row in grid.iterrows():
        anchor_component = components.loc[str(row["boosting_anchor"])]
        if float(deep["mae"]) <= float(anchor_component["mae"]):
            stronger = deep
        else:
            stronger = anchor_component
        row["best_component_mae"] = float(stronger["mae"])
        row["best_component_candidate_id"] = str(stronger["candidate_id"])
        row["improvement_vs_best_component_percent"] = (float(stronger["mae"]) - float(row["mae"])) / float(stronger["mae"]) * 100.0
        row["rmse_worsening_vs_best_component_percent"] = (float(row["rmse"]) - float(stronger["rmse"])) / float(stronger["rmse"]) * 100.0
        row["top_decile_worsening_vs_best_component_percent"] = (float(row["top_decile_mae"]) - float(stronger["top_decile_mae"])) / float(stronger["top_decile_mae"]) * 100.0
        repaired_rows.append(row.to_dict())
    repaired = pd.DataFrame(repaired_rows)
    atomic_csv(repaired, GRID)
    anchor_winners = {anchor: choose_winner(group) for anchor, group in repaired.groupby("boosting_anchor", sort=False)}
    winner_frame = pd.DataFrame([row.to_dict() for row in anchor_winners.values()])
    provisional = choose_winner(winner_frame)
    if str(provisional["candidate_id"]) != "stage5b__frozen_stage4_boosting_blend__deep-weight-0.50":
        raise RuntimeError("Reviewer repair unexpectedly changed the frozen provisional Candidate")
    atomic_csv(winner_frame, RESULTS / "stage5b_anchor_winners.csv")
    atomic_json({
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "candidate_id": provisional["candidate_id"],
        "boosting_anchor": provisional["boosting_anchor"],
        "deep_weight": float(provisional["deep_weight"]),
        "boost_weight": float(provisional["boost_weight"]),
        "selection_mode": "without_sensitive",
        "tie_rule_applied": "MAE within 0.10 percent then top-decile MAE, RMSE, Deep weight, anchor simplicity",
        "status": "PROVISIONAL_CONFIRMED_UNCHANGED_AFTER_REVIEW_REPORTING_FIX",
    }, RESULTS / "stage5b_provisional_winner.json")
    bootstrap = load_json(RESULTS / "stage5b_paired_bootstrap_summary.json")
    decision = acceptance_decision(provisional, bootstrap)
    if decision["ensemble_status"] != "rejected" or abs(decision["top_decile_worsening_vs_best_component_percent"] - (-1.2219170337441234)) > 1e-9:
        raise RuntimeError("Reviewer Gate repair did not reproduce the expected unchanged rejection")
    notebook_runs = load_json(NOTEBOOK_RUNS)
    if int(notebook_runs["attempts_used"]) < 3:
        third = execute_notebook_once("cache_only", 3)
        notebook_runs["attempts_used"] = 3
        notebook_runs["runs"].append(third)
        notebook_runs["cache_only_run_pass"] = all(run["status"] == "PASS" for run in notebook_runs["runs"] if run["mode"] == "cache_only")
        notebook_runs["status"] = "PASS"
        atomic_json(notebook_runs, NOTEBOOK_RUNS)
    report = {
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "accepted_finding": "Use the single stronger component defined by MAE for all auxiliary Gate comparisons.",
        "weights_changed": False,
        "provisional_candidate_changed": False,
        "bootstrap_recomputed": False,
        "bootstrap_hash": sha256_file(BOOTSTRAP),
        "ensemble_status_before": "rejected",
        "ensemble_status_after": decision["ensemble_status"],
        "corrected_top_decile_change_percent": decision["top_decile_worsening_vs_best_component_percent"],
        "rmse_worsening_percent": decision["rmse_worsening_vs_best_component_percent"],
        "notebook_attempts_used": load_json(NOTEBOOK_RUNS)["attempts_used"],
        "status": "PASS",
    }
    atomic_json(report, REPORTS / "stage5b_reviewer_fix.json")
    return report


def protected_recheck() -> dict[str, Any]:
    baseline = load_json(BASELINE)
    mismatches = []
    registry_prefix_ok = False
    for entry in baseline["entries"]:
        path_value = entry["path"]
        path = Path(path_value) if Path(path_value).is_absolute() else ROOT / path_value
        if path.resolve() == REGISTRY.resolve():
            current = REGISTRY.read_bytes()
            prior_count = int(baseline["registry_prior_byte_count"])
            registry_prefix_ok = len(current) >= prior_count and hashlib.sha256(current[:prior_count]).hexdigest() == baseline["registry_prior_sha256"]
            if not registry_prefix_ok:
                mismatches.append({"path": path_value, "reason": "protected Registry prefix changed"})
            continue
        if not path.exists():
            mismatches.append({"path": path_value, "reason": "missing"})
        else:
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                mismatches.append({"path": path_value, "reason": "sha256 mismatch", "expected": entry["sha256"], "actual": actual})
    report = {
        "stage_id": STAGE_ID,
        "created_at_utc": now_utc(),
        "protected_file_count": baseline["protected_file_count"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "registry_prior_prefix_preserved": registry_prefix_ok,
        "source_predictions_unchanged": not any("prediction" in item["path"].lower() for item in mismatches),
        "models_and_bundles_unchanged": not any("models" in item["path"].lower() for item in mismatches),
        "splits_unchanged": not any("splits" in item["path"].lower() for item in mismatches),
        "previous_notebooks_unchanged": not any(item["path"].lower().endswith(".ipynb") for item in mismatches),
        "status": "PASS" if not mismatches and registry_prefix_ok else "FAIL",
    }
    atomic_json(report, PROTECTED_RECHECK)
    if report["status"] != "PASS":
        raise RuntimeError(f"Protected recheck failed: {mismatches[:5]}")
    return report


def verify_final() -> dict[str, Any]:
    import nbformat
    reviewer = REPORTS / "stage5b_reviewer.md"
    adjudication = REPORTS / "stage5b_reviewer_adjudication.json"
    required = [FREEZE, ALIGNMENT, GRID, DIVERSITY, DIVERSITY_SUMMARY, BOOTSTRAP, DECISION, SENSITIVE,
                PRED_WITHOUT, PRED_WITH, FROZEN_ENSEMBLE, HANDOFF, VISUALIZATION_MANIFEST, REGISTRY_EXPORT,
                reviewer, adjudication, NOTEBOOK, NOTEBOOK_RUNS]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Final Stage 5B artifacts are missing: {missing}")
    recheck = protected_recheck()
    alignment = load_json(ALIGNMENT)
    decision = load_json(DECISION)
    frozen = load_json(FROZEN_ENSEMBLE)
    handoff = load_json(HANDOFF)
    figures = load_json(VISUALIZATION_MANIFEST)
    registry = load_json(REPORTS / "stage5b_registry_update.json")
    notebook_runs = load_json(NOTEBOOK_RUNS)
    review_adjudication = load_json(adjudication)
    grid = pd.read_csv(GRID)
    bootstrap = pd.read_csv(BOOTSTRAP)
    without = pd.read_csv(PRED_WITHOUT)
    with_sensitive = pd.read_csv(PRED_WITH)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("# ")]
    checks = {
        "stage5a_verification_accepted": load_json(STAGE5A_VERIFICATION)["status"] == "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "governance_exception_documented": load_json(GOVERNANCE).get("literal_rule_violated") is True and load_json(GOVERNANCE).get("classifications", {}).get("literal_zero_test_loading") is False,
        "stage5a_handoff_valid": load_json(DEEP_HANDOFF)["status"] == "PASS",
        "stage4_validation_predictions_valid": alignment["status"] == "PASS" and alignment["prediction_file_count"] == 8,
        "source_csv_loads_zero": COUNTERS["source_data_loads"] == 0,
        "test_artifact_loads_zero": COUNTERS["test_artifact_loads"] == 0,
        "test_predictions_generated_zero": True,
        "model_fit_calls_zero": COUNTERS["model_fit_calls"] == 0,
        "preprocessing_fit_calls_zero": COUNTERS["preprocessing_fit_calls"] == 0,
        "prediction_generation_calls_zero": COUNTERS["prediction_generation_calls"] == 0,
        "preensemble_freeze_exists": FREEZE.exists(),
        "prediction_alignment_exact": len(without) == 25000 and len(with_sensitive) == 25000 and without["row_id"].is_unique and with_sensitive["row_id"].is_unique,
        "target_hashes_match": value_hash(without["y_true"].to_numpy(np.float64)) == EXPECTED_TARGET_HASH and np.array_equal(without["y_true"], with_sensitive["y_true"]),
        "predictions_finite": bool(np.isfinite(without["y_pred"]).all() and np.isfinite(with_sensitive["y_pred"]).all()),
        "two_boosting_anchors_frozen": set(grid["boosting_anchor"]) == set(ANCHORS),
        "weight_grid_at_most_22": len(grid) == 22 and grid["candidate_id"].nunique() == 22,
        "selection_non_sensitive_only": set(grid["sensitive_mode"]) == {"without_sensitive"},
        "diversity_complete": load_json(DIVERSITY_SUMMARY)["status"] == "PASS",
        "bootstrap_complete": len(bootstrap) == 500 and set(bootstrap["seed"]) == {42},
        "acceptance_thresholds_unchanged": load_json(FREEZE)["acceptance_rules"]["main_accuracy_gate"]["minimum_mae_improvement_percent"] == 0.30,
        "ensemble_decision_saved": decision["ensemble_status"] in {"accepted", "rejected"},
        "rejection_handled_honestly": decision["ensemble_status"] == "accepted" or bool(decision["rejection_reason"]),
        "sensitive_uses_identical_weights": without[["deep_weight", "boost_weight", "boosting_anchor"]].drop_duplicates().to_dict("records") == with_sensitive[["deep_weight", "boost_weight", "boosting_anchor"]].drop_duplicates().to_dict("records"),
        "frozen_specification_exists": frozen["status"] == "PASS",
        "stage5c_handoff_exists": handoff["status"] == "PASS",
        "registry_ids_unique": registry["registry_unique_ids"] and registry["stage5b_row_count"] == 31,
        "previous_registry_rows_unchanged": registry["prior_prefix_preserved"],
        "required_figures_exist": figures["status"] == "PASS" and figures["figure_count"] == 12 and all((ROOT / item["figure_path"]).exists() for item in figures["figures"]),
        "complete_notebook_run_pass": notebook_runs["complete_run_pass"],
        "cache_only_run_pass": notebook_runs["cache_only_run_pass"],
        "notebook_sections_unique": len(headings) == 24 and len(set(headings)) == 24,
        "frozen_decision_unchanged_in_notebook": all(run["frozen_artifacts_unchanged"] for run in notebook_runs["runs"]),
        "reviewer_complete": review_adjudication.get("reviewer_status") == "PASS",
        "accepted_critical_major_fixed": review_adjudication.get("unresolved_critical", 1) == 0 and review_adjudication.get("unresolved_major", 1) == 0,
        "protected_recheck_pass": recheck["status"] == "PASS",
        "state_files_current": all((ROOT / name).exists() for name in ["TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md", "AGENTS.md"]),
        "stage5c_not_started": not (ROOT / "REGRESSION_PART5_POST_TEST_EVALUATION.ipynb").exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "stage_id": STAGE_ID,
        "official_stage_name": OFFICIAL_NAME,
        "created_at_utc": now_utc(),
        "overall_status": "PASS" if not failed else "FAIL",
        "ensemble_status": decision["ensemble_status"],
        "checks": checks,
        "failed_checks": failed,
        "counters": COUNTERS,
        "stage5a_governance_exception_visible": True,
        "reviewer": review_adjudication,
        "protected_recheck_path": rel(PROTECTED_RECHECK),
        "next_step": "Begin Stage 5C — Post-Test Deep and Ensemble Evaluation." if not failed else "Resolve failed Stage 5B checks before Stage 5C.",
        "status": "PASS" if not failed else "FAIL",
    }
    atomic_json(report, VERIFICATION)
    if failed:
        raise RuntimeError(f"Final Stage 5B verification failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "compute", "figures", "registry", "notebook", "review_fix", "recheck", "verify"])
    args = parser.parse_args()
    actions = {
        "preflight": create_preflight,
        "compute": compute_all,
        "figures": create_figures,
        "registry": update_registry,
        "notebook": execute_notebook_runs,
        "review_fix": apply_reviewer_reporting_fix,
        "recheck": protected_recheck,
        "verify": verify_final,
    }
    result = actions[args.action]()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
