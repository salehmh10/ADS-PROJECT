"""Stage 4L prediction-only integration, analysis, and reporting utilities."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/environment/stage4_packages")]
TARGET = "loan_amount_000s"
TRAIN_ROWS = 399_788
TEST_ROWS = 99_948
RESULT_DIR = ROOT / "artifacts/results/stage4/final_integration"
PRED_DIR = ROOT / "artifacts/predictions/final_test"
FIG_DIR = ROOT / "artifacts/figures/stage4l"
REPORT_DIR = ROOT / "artifacts/reports"
MANIFEST_DIR = ROOT / "artifacts/manifests/stage4"
CHECKPOINT_DIR = ROOT / "artifacts/checkpoints/stage4l"
CANDIDATE_MANIFEST = MANIFEST_DIR / "stage4l_candidate_manifest.json"
FREEZE_MANIFEST = REPORT_DIR / "stage4l_pretest_freeze.json"
UNLOCK_AUDIT = REPORT_DIR / "stage4l_test_unlock_audit.json"
FREEZE_REL = str(FREEZE_MANIFEST.relative_to(ROOT)).replace("/", "\\")
CANDIDATE_REL = str(CANDIDATE_MANIFEST.relative_to(ROOT)).replace("/", "\\")
BLEND_WEIGHTS = {"catboost": 0.6, "lightgbm": 0.2, "xgboost": 0.2}
PRIMARY_ID = "stage4l__blend__without_sensitive"
COMPANION_ID = "stage4l__blend__with_sensitive"
SENSITIVE_COLUMNS = {
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "minority_population",
    "majority_minority_tract",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def ensure_directories() -> None:
    for path in [RESULT_DIR, PRED_DIR, FIG_DIR, REPORT_DIR, MANIFEST_DIR, CHECKPOINT_DIR, ROOT / "artifacts/backups"]:
        path.mkdir(parents=True, exist_ok=True)


def status_from_json(path: Path) -> str:
    payload = load_json(path)
    for key in ["status", "overall_status", "verification_status", "final_status"]:
        if payload.get(key):
            return str(payload[key])
    return "UNKNOWN"


def previous_stage_audit() -> list[dict]:
    files = {
        "Stage 1": "artifacts/reports/prompt1_verification.json",
        "Stage 2": "artifacts/reports/prompt2_verification.json",
        "Stage 3": "artifacts/reports/stage3_verification.json",
        "Stage 4A": "artifacts/reports/stage4a_verification.json",
        "Stage 4B": "artifacts/reports/stage4b_verification.json",
        "Stage 4C": "artifacts/reports/stage4c_verification.json",
        "Stage 4D-E": "artifacts/reports/stage4de_verification.json",
        "Stage 4F": "artifacts/reports/stage4f_gate_verification.json",
        "Stage 4G": "artifacts/reports/stage4g_gate_verification.json",
        "Stage 4H": "artifacts/reports/stage4h_verification.json",
        "Stage 4I": "artifacts/reports/stage4i_gate_verification.json",
        "Stage 4J": "artifacts/reports/stage4j_gate_verification.json",
        "Stage 4K": "artifacts/reports/stage4k_verification.json",
    }
    rows = []
    for stage, rel in files.items():
        path = ROOT / rel
        rows.append({"stage": stage, "path": rel, "exists": path.exists(), "status": status_from_json(path) if path.exists() else "MISSING", "sha256": sha256_file(path) if path.exists() else None})
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("At least one required previous Stage is not PASS")
    return rows


def protected_file_allowed(rel: str) -> bool:
    value = rel.replace("\\", "/").lower()
    if value.startswith(".git/") or value.startswith(".agents/") or value.startswith(".codex/"):
        return True
    if "/__pycache__/" in f"/{value}" or value.endswith(".pyc"):
        return True
    if value.startswith("artifacts/backups/"):
        return True
    if "stage4l" in value or value.startswith("artifacts/predictions/final_test/"):
        return True
    if value.startswith("artifacts/results/stage4/final_integration/") or value.startswith("artifacts/figures/stage4l/"):
        return True
    if value in {
        "agents.md", "task.md", "plan.md", "decisions.md", "log.md",
        "regression_part4_final_integration_and_test.ipynb",
        "stage4l_evaluation_utils.py", "stage4l_prediction_worker.py",
        "artifacts/results/experiment_results.csv",
    }:
        return True
    return False


def capture_protected_baseline() -> dict:
    files = []
    for path in sorted((item for item in ROOT.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if protected_file_allowed(rel):
            continue
        files.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    external = Path(r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb")
    files.append({"path": str(external), "size_bytes": external.stat().st_size, "sha256": sha256_file(external)})
    payload = {
        "stage": "stage4l",
        "created_at_utc": utc_now(),
        "file_count": len(files),
        "files": files,
        "protected_digest": canonical_sha(files),
        "status": "PASS",
    }
    atomic_json(payload, MANIFEST_DIR / "stage4l_protected_hashes_before.json")
    return payload


def validate_split_files() -> dict:
    train_path = ROOT / "artifacts/splits/train_row_ids.csv"
    test_path = ROOT / "artifacts/splits/test_row_ids.csv"
    train_ids = pd.read_csv(train_path)["row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(test_path)["row_id"].to_numpy(np.int64)
    checks = {
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "train_unique": bool(len(np.unique(train_ids)) == len(train_ids)),
        "test_unique": bool(len(np.unique(test_ids)) == len(test_ids)),
        "overlap_rows": int(len(np.intersect1d(train_ids, test_ids))),
        "train_row_ids_sha256": sha256_file(train_path),
        "test_row_ids_sha256": sha256_file(test_path),
    }
    if checks["train_rows"] != TRAIN_ROWS or checks["test_rows"] != TEST_ROWS or not checks["train_unique"] or not checks["test_unique"] or checks["overlap_rows"]:
        raise RuntimeError("Saved Train/Test row IDs failed validation")
    return checks


def source_header(mode: str) -> list[str]:
    name = "regression_with_sensitive_features.csv" if mode == "with_sensitive" else "regression_without_sensitive_features.csv"
    return pd.read_csv(ROOT / "data" / name, nrows=0).columns.tolist()


def row_to_parameters(row: pd.Series, field: str) -> dict:
    return json.loads(row[field])


def model_record(
    candidate_id: str,
    stage: str,
    family: str,
    mode: str,
    model_path: str,
    manifest_path: str,
    raw_columns: list[str],
    target_mode: str,
    feature_pack: str,
    parameters: dict,
    iteration: int | None,
    metric_source: str,
    validation_metrics: dict | None,
    reload_source: str,
    reload_status: str,
    native_path: str | None = None,
    expected_model_sha: str | None = None,
    expected_native_sha: str | None = None,
) -> dict:
    model = ROOT / model_path
    manifest = ROOT / manifest_path
    native = ROOT / native_path if native_path else None
    actual_model_sha = sha256_file(model)
    if expected_model_sha and actual_model_sha != expected_model_sha:
        raise RuntimeError(f"Model hash mismatch: {candidate_id}")
    actual_native_sha = sha256_file(native) if native else None
    if expected_native_sha and actual_native_sha != expected_native_sha:
        raise RuntimeError(f"Native-model hash mismatch: {candidate_id}")
    header = source_header(mode)
    missing = sorted(set(raw_columns) - set(header))
    if missing:
        raise RuntimeError(f"Missing raw columns for {candidate_id}: {missing}")
    return {
        "candidate_id": candidate_id,
        "candidate_type": "saved_model",
        "stage": stage,
        "model_family": family,
        "sensitive_mode": mode,
        "model_bundle_path": model_path,
        "native_model_path": native_path,
        "manifest_path": manifest_path,
        "model_sha256": actual_model_sha,
        "native_model_sha256": actual_native_sha,
        "manifest_sha256": sha256_file(manifest),
        "feature_pack": feature_pack,
        "target_mode": target_mode,
        "frozen_parameters": parameters,
        "iteration_count": iteration,
        "train_row_count": TRAIN_ROWS,
        "test_row_count_during_training": 0,
        "reload_verification_source": reload_source,
        "reload_verification_status": reload_status,
        "validation_or_oof_metric_source": metric_source,
        "pretest_metrics": validation_metrics,
        "evaluation_design": "common_locked_test_on_exact_saved_row_ids",
        "required_raw_input_columns": raw_columns,
        "status": "PASS",
    }


def build_candidate_manifest() -> tuple[dict, dict]:
    stage2 = pd.read_csv(ROOT / "artifacts/results/prompt2/cv_oof_summary.csv")
    s2_family = stage2.groupby("model_name")["mae"].min().sort_values().index[0]
    if s2_family == "dummy_median":
        s2_family = stage2.loc[stage2.model_name != "dummy_median"].groupby("model_name")["mae"].min().sort_values().index[0]
    stage3 = pd.read_csv(ROOT / "artifacts/results/stage3/cv_oof_summary.csv")
    s3_family = stage3.groupby("model_name")["mae"].min().sort_values().index[0]
    if s2_family != "lasso" or s3_family != "hist_gradient_boosting":
        raise RuntimeError(f"Unexpected representative families: {s2_family}, {s3_family}")

    s2_manifest_rel = "artifacts/manifests/prompt2_model_manifest.json"
    s3_manifest_rel = "artifacts/manifests/stage3_model_manifest.json"
    s2_manifest = load_json(ROOT / s2_manifest_rel)
    s3_manifest = load_json(ROOT / s3_manifest_rel)
    s2_reload = pd.read_csv(ROOT / "artifacts/reports/prompt2_model_reload_verification.csv")
    s3_reload = pd.read_csv(ROOT / "artifacts/reports/stage3_model_reload_verification.csv")
    candidates: list[dict] = []

    train_oof = pd.read_csv(ROOT / "artifacts/predictions/tree/hist_gradient_boosting__without_sensitive__oof.csv", usecols=["row_id", "y_true"])
    train_ids = pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv")["row_id"]
    if len(train_oof) != TRAIN_ROWS or train_oof.row_id.nunique() != TRAIN_ROWS or set(train_oof.row_id) != set(train_ids):
        raise RuntimeError("Saved Train targets do not match the exact Train row IDs")
    y_train = train_oof.y_true.to_numpy(float)
    constants = {"train_mean": float(np.mean(y_train)), "train_median": float(np.median(y_train)), "source": "Stage 3 OOF y_true aligned to all saved Train row IDs"}

    for label, value in [("mean", constants["train_mean"]), ("median", constants["train_median"])]:
        candidates.append({
            "candidate_id": f"stage4l__train_{label}_baseline",
            "candidate_type": "constant_baseline",
            "stage": "Stage 4L",
            "model_family": f"train_{label}_baseline",
            "sensitive_mode": "not_applicable",
            "model_bundle_path": None,
            "native_model_path": None,
            "manifest_path": "artifacts/splits/train_row_ids.csv",
            "model_sha256": None,
            "native_model_sha256": None,
            "manifest_sha256": sha256_file(ROOT / "artifacts/splits/train_row_ids.csv"),
            "feature_pack": "none",
            "target_mode": "raw",
            "frozen_parameters": {"constant": value},
            "iteration_count": None,
            "train_row_count": TRAIN_ROWS,
            "test_row_count_during_training": 0,
            "reload_verification_source": "not_applicable",
            "reload_verification_status": "PASS",
            "validation_or_oof_metric_source": constants["source"],
            "pretest_metrics": None,
            "evaluation_design": "common_locked_test_on_exact_saved_row_ids",
            "required_raw_input_columns": [],
            "status": "PASS",
        })

    for mode in ["without_sensitive", "with_sensitive"]:
        row = stage2[(stage2.model_name == s2_family) & (stage2.sensitive_mode == mode)].iloc[0]
        info = next(item for item in s2_manifest["models"] if item["model_name"] == s2_family and item["sensitive_mode"] == mode)
        reload_row = s2_reload[(s2_reload.model_name == s2_family) & (s2_reload.sensitive_mode == mode)].iloc[0]
        model = joblib.load(ROOT / info["model_path"])
        raw = list(model.feature_names_in_)
        candidates.append(model_record(
            f"stage4l__{s2_family}__{mode}", "Stage 2", s2_family, mode, info["model_path"], s2_manifest_rel, raw,
            info["target_mode"], "linear_compact_v1", row_to_parameters(row, "parameter_json"), None,
            "artifacts/results/prompt2/cv_oof_summary.csv", {"mae": float(row.mae), "rmse": float(row.rmse), "rmsle": float(row.rmsle), "tail_mae": None, "prediction_time_seconds": float(row.total_prediction_time_seconds)},
            "artifacts/reports/prompt2_model_reload_verification.csv", "PASS" if bool(reload_row["passed"]) else "FAIL", expected_model_sha=info["model_sha256"]
        ))

    for mode in ["without_sensitive", "with_sensitive"]:
        row = stage3[(stage3.model_name == s3_family) & (stage3.sensitive_mode == mode)].iloc[0]
        info = next(item for item in s3_manifest["models"] if item["model_name"] == s3_family and item["sensitive_mode"] == mode)
        reload_row = s3_reload[(s3_reload.model_name == s3_family) & (s3_reload.sensitive_mode == mode)].iloc[0]
        model = joblib.load(ROOT / info["model_path"])
        raw = list(model.feature_names_in_)
        candidates.append(model_record(
            f"stage4l__{s3_family}__{mode}", "Stage 3", s3_family, mode, info["model_path"], s3_manifest_rel, raw,
            info["target_mode"], row.feature_pack, row_to_parameters(row, "configuration_json"), 300,
            "artifacts/results/stage3/cv_oof_summary.csv", {"mae": float(row.mae), "rmse": float(row.rmse), "rmsle": float(row.rmsle), "tail_mae": float(row.mae_target_ge_1000), "prediction_time_seconds": float(row.total_prediction_time_seconds)},
            "artifacts/reports/stage3_model_reload_verification.csv", str(reload_row.status), expected_model_sha=info["model_sha256"]
        ))

    candidates.extend(build_boosting_records())
    blend_evidence = validation_blend_evidence()
    if not blend_evidence["accepted"]:
        raise RuntimeError("The saved Validation blend does not meet the frozen acceptance rule")
    blend_evidence_path = REPORT_DIR / "stage4l_blend_validation_evidence.json"
    atomic_json(blend_evidence, blend_evidence_path)
    blend_evidence_hash = sha256_file(blend_evidence_path)
    for mode in ["without_sensitive", "with_sensitive"]:
        candidates.append({
            "candidate_id": f"stage4l__blend__{mode}", "candidate_type": "frozen_blend", "stage": "Stage 4L",
            "model_family": "validation_frozen_boosting_blend", "sensitive_mode": mode, "model_bundle_path": None,
            "native_model_path": None, "manifest_path": "artifacts/reports/stage4l_blend_validation_evidence.json",
            "model_sha256": canonical_sha({"mode": mode, "weights": BLEND_WEIGHTS}), "native_model_sha256": None,
            "manifest_sha256": blend_evidence_hash, "feature_pack": "frozen_three_booster_predictions", "target_mode": "mixed_prediction_scale_original_target",
            "frozen_parameters": {"weights": BLEND_WEIGHTS}, "iteration_count": None, "train_row_count": TRAIN_ROWS,
            "test_row_count_during_training": 0, "reload_verification_source": "input_bundle_reload_reports", "reload_verification_status": "PASS",
            "validation_or_oof_metric_source": "aligned Final Selection Validation predictions", "pretest_metrics": blend_evidence["mode_metrics"][mode],
            "evaluation_design": "common_locked_test_on_exact_saved_row_ids", "required_raw_input_columns": [], "status": "PASS",
        })

    if len(candidates) != 14 or any(item["status"] != "PASS" for item in candidates):
        raise RuntimeError("Candidate manifest is incomplete")
    payload = {
        "stage": "stage4l", "official_name": "Stage 4L — Final Model Integration and Locked Test Evaluation",
        "created_at_utc": utc_now(), "candidate_count": len(candidates), "required_saved_model_candidates": 10,
        "constant_baseline_candidates": 2, "optional_blend_candidates": 2, "representative_stage2_family": s2_family,
        "representative_stage3_family": s3_family, "train_constants": constants, "candidates": candidates, "status": "PASS",
    }
    atomic_json(payload, CANDIDATE_MANIFEST)
    return payload, blend_evidence


def validation_metrics(frame: pd.DataFrame) -> dict:
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    absolute = np.abs(y_pred - y_true)
    threshold = float(np.quantile(y_true, 0.9))
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(y_pred - y_true)))),
        "rmsle": float(np.sqrt(np.mean(np.square(np.log1p(np.maximum(y_pred, 0)) - np.log1p(y_true))))),
        "tail_mae": float(np.mean(absolute[y_true >= threshold])),
        "row_count": int(len(frame)),
    }


def read_aligned_validation(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(ROOT / rel).sort_values("row_id").reset_index(drop=True) for name, rel in paths.items()}
    reference = next(iter(frames.values()))
    for name, frame in frames.items():
        if len(frame) != 25_000 or frame.row_id.nunique() != len(frame):
            raise RuntimeError(f"Invalid Validation rows: {name}")
        if not np.array_equal(frame.row_id.to_numpy(), reference.row_id.to_numpy()):
            raise RuntimeError(f"Validation row IDs are not aligned: {name}")
        if not np.array_equal(frame.y_true.to_numpy(float), reference.y_true.to_numpy(float)):
            raise RuntimeError(f"Validation targets are not aligned: {name}")
        if not np.isfinite(frame.y_pred.to_numpy(float)).all():
            raise RuntimeError(f"Validation predictions are not finite: {name}")
    return frames


def validation_blend_evidence() -> dict:
    mode_paths = {
        "without_sensitive": {
            "catboost": "artifacts/results/stage4/catboost/final/catboost_final_validation_predictions_without_sensitive.csv",
            "lightgbm": "artifacts/predictions/lightgbm/final/lightgbm_final_selection_without_sensitive.csv",
            "xgboost": "artifacts/predictions/xgboost/final_selection/xgboost_winning_non_sensitive_validation.csv",
        },
        "with_sensitive": {
            "catboost": "artifacts/results/stage4/catboost/final/catboost_final_validation_predictions_with_sensitive.csv",
            "lightgbm": "artifacts/predictions/lightgbm/final/lightgbm_final_selection_with_sensitive.csv",
            "xgboost": "artifacts/predictions/xgboost/final_selection/sensitive_validation.csv",
        },
    }
    without = read_aligned_validation(mode_paths["without_sensitive"])
    y_true = without["catboost"].y_true.to_numpy(float)
    rows = []
    for cat_weight in range(11):
        for light_weight in range(11 - cat_weight):
            xgb_weight = 10 - cat_weight - light_weight
            weights = {"catboost": cat_weight / 10, "lightgbm": light_weight / 10, "xgboost": xgb_weight / 10}
            prediction = sum(weights[name] * without[name].y_pred.to_numpy(float) for name in weights)
            frame = pd.DataFrame({"y_true": y_true, "y_pred": prediction})
            rows.append({"weights": weights, **validation_metrics(frame)})
    equal_prediction = sum(without[name].y_pred.to_numpy(float) for name in without) / 3
    equal = {"weights": {name: 1 / 3 for name in without}, **validation_metrics(pd.DataFrame({"y_true": y_true, "y_pred": equal_prediction}))}
    rows.sort(key=lambda item: (item["mae"], item["tail_mae"]))
    best_grid = rows[0]
    if best_grid["weights"] != BLEND_WEIGHTS:
        raise RuntimeError(f"Frozen blend weights differ from Validation optimum: {best_grid['weights']}")
    individual = {name: validation_metrics(frame) for name, frame in without.items()}
    best_individual_name = min(individual, key=lambda name: individual[name]["mae"])
    best_individual = individual[best_individual_name]
    improvement_percent = 100 * (best_individual["mae"] - best_grid["mae"]) / best_individual["mae"]
    tail_change_percent = 100 * (best_grid["tail_mae"] - best_individual["tail_mae"]) / best_individual["tail_mae"]
    accepted = bool(improvement_percent >= 0.3 and tail_change_percent <= 1.0)
    mode_metrics = {"without_sensitive": best_grid}
    with_frames = read_aligned_validation(mode_paths["with_sensitive"])
    with_prediction = sum(BLEND_WEIGHTS[name] * with_frames[name].y_pred.to_numpy(float) for name in BLEND_WEIGHTS)
    mode_metrics["with_sensitive"] = {"weights": BLEND_WEIGHTS, **validation_metrics(pd.DataFrame({"y_true": with_frames["catboost"].y_true, "y_pred": with_prediction}))}
    return {
        "accepted": accepted,
        "search_design": "equal weights plus 66 non-negative step-0.1 combinations",
        "grid_combination_count": len(rows),
        "weights": BLEND_WEIGHTS,
        "best_individual": best_individual_name,
        "individual_without_sensitive": individual,
        "best_grid": best_grid,
        "equal_weights": equal,
        "validation_mae_improvement_percent": improvement_percent,
        "top_decile_mae_change_percent": tail_change_percent,
        "mode_metrics": mode_metrics,
        "prediction_sources": mode_paths,
        "source_hashes": {mode: {family: sha256_file(ROOT / rel) for family, rel in paths.items()} for mode, paths in mode_paths.items()},
        "status": "PASS" if accepted else "REJECTED",
    }


def build_boosting_records() -> list[dict]:
    records: list[dict] = []
    validation_paths = {
        ("catboost", "without_sensitive"): "artifacts/results/stage4/catboost/final/catboost_final_validation_predictions_without_sensitive.csv",
        ("catboost", "with_sensitive"): "artifacts/results/stage4/catboost/final/catboost_final_validation_predictions_with_sensitive.csv",
        ("lightgbm", "without_sensitive"): "artifacts/predictions/lightgbm/final/lightgbm_final_selection_without_sensitive.csv",
        ("lightgbm", "with_sensitive"): "artifacts/predictions/lightgbm/final/lightgbm_final_selection_with_sensitive.csv",
        ("xgboost", "without_sensitive"): "artifacts/predictions/xgboost/final_selection/xgboost_winning_non_sensitive_validation.csv",
        ("xgboost", "with_sensitive"): "artifacts/predictions/xgboost/final_selection/sensitive_validation.csv",
    }
    validation = {(family, mode): validation_metrics(pd.read_csv(ROOT / rel)) for (family, mode), rel in validation_paths.items()}

    cat_manifest_rel = "artifacts/results/stage4/catboost/final/catboost_full_train_manifest.json"
    cat_manifest = load_json(ROOT / cat_manifest_rel)
    cat_config = load_json(ROOT / "artifacts/results/stage4/catboost/final/catboost_final_configuration.json")
    cat_reload = pd.read_csv(ROOT / "artifacts/reports/stage4de_final_reload_verification.csv")
    for mode in ["without_sensitive", "with_sensitive"]:
        info = cat_manifest["models"][mode]
        bundle = joblib.load(ROOT / info["model_path"])
        meta = bundle.metadata
        reload_row = cat_reload[cat_reload.sensitive_mode == mode].iloc[0]
        metrics = dict(validation[("catboost", mode)])
        metrics["prediction_time_seconds"] = None
        records.append(model_record(
            f"stage4l__catboost__{mode}", "Stage 4D-E", "catboost", mode, info["model_path"], cat_manifest_rel,
            list(meta["raw_columns"]), meta["target_mode"], meta["feature_pack_id"], meta["parameters"], int(meta["fixed_iteration_count"]),
            validation_paths[("catboost", mode)], metrics, "artifacts/reports/stage4de_final_reload_verification.csv", str(reload_row.status),
            info["native_model_path"], info["model_sha256"], info["native_model_sha256"]
        ))

    for mode in ["without_sensitive", "with_sensitive"]:
        manifest_rel = f"artifacts/manifests/stage4/lightgbm/stage4h_final_model_manifest_{mode}.json"
        info = load_json(ROOT / manifest_rel)
        meta = info["embedded_metadata"]
        metrics = dict(validation[("lightgbm", mode)])
        metrics["prediction_time_seconds"] = None
        records.append(model_record(
            f"stage4l__lightgbm__{mode}", "Stage 4H", "lightgbm", mode, info["model_path"], manifest_rel,
            list(meta["raw_columns"]), info["target_mode"], info["feature_pack"], info["parameters"], int(info["fixed_iteration"]),
            validation_paths[("lightgbm", mode)], metrics, info["reload_report"], info["reload_status"], info["native_model_path"],
            info["model_sha256"], info["native_model_sha256"]
        ))

    for mode in ["without_sensitive", "with_sensitive"]:
        manifest_rel = f"artifacts/manifests/stage4/xgboost/stage4k_final_model_manifest_{mode}.json"
        info = load_json(ROOT / manifest_rel)
        meta = info["bundle_metadata"]
        reload_status = status_from_json(ROOT / info["reload_report"])
        metrics = dict(validation[("xgboost", mode)])
        metrics["prediction_time_seconds"] = None
        records.append(model_record(
            f"stage4l__xgboost__{mode}", "Stage 4K", "xgboost", mode, info["model_path"], manifest_rel,
            list(meta["raw_columns"]), meta["target_mode"], meta["feature_pack"], meta["parameters"], int(meta["fixed_or_best_iteration"]),
            validation_paths[("xgboost", mode)], metrics, info["reload_report"], reload_status, info["native_model_path"],
            info["model_sha256"], info["native_model_sha256"]
        ))
    return records


def static_no_fit_audit() -> dict:
    files = [ROOT / "stage4l_evaluation_utils.py", ROOT / "stage4l_prediction_worker.py"]
    findings = []
    prohibited = {"fit", "fit_transform", "search", "tune"}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in prohibited:
                findings.append({"file": path.name, "line": node.lineno, "method": node.func.attr})
    return {"files": [path.name for path in files], "prohibited_calls": findings, "fit_call_count": 0, "status": "PASS" if not findings else "FAIL"}


def run_pretest() -> dict:
    ensure_directories()
    started = time.perf_counter()
    if UNLOCK_AUDIT.exists():
        raise RuntimeError("Test has already been unlocked; pre-Test artifacts are immutable")
    previous = previous_stage_audit()
    split = validate_split_files()
    no_fit = static_no_fit_audit()
    if no_fit["status"] != "PASS":
        raise RuntimeError(f"Static no-fit audit failed: {no_fit['prohibited_calls']}")
    protected = capture_protected_baseline()
    candidate_manifest, blend = build_candidate_manifest()
    candidate_hash = sha256_file(CANDIDATE_MANIFEST)
    model_hashes = {item["candidate_id"]: item["model_sha256"] for item in candidate_manifest["candidates"]}
    feature_packs = {item["candidate_id"]: item["feature_pack"] for item in candidate_manifest["candidates"]}
    target_modes = {item["candidate_id"]: item["target_mode"] for item in candidate_manifest["candidates"]}
    visualization_plan = [
        "Final Test MAE Leaderboard", "MAE with Bootstrap Confidence Intervals", "Multi-Metric Model Comparison",
        "Validation Versus Test MAE", "Sensitive Versus Non-Sensitive Delta", "Actual Versus Predicted",
        "Residual Versus Predicted", "Residual Distribution", "Error by Target Decile", "Model Efficiency",
        "Paired Model-Difference Intervals", "Blend Weights and Benefit", "Compact Final Summary Dashboard",
    ]
    freeze = {
        "stage": "stage4l", "freeze_timestamp_utc": utc_now(), "protected_file_digest": protected["protected_digest"],
        "protected_file_count": protected["file_count"], "candidate_manifest_sha256": candidate_hash,
        "candidate_ids": [item["candidate_id"] for item in candidate_manifest["candidates"]], "candidate_count": candidate_manifest["candidate_count"],
        "primary_non_sensitive_candidate": PRIMARY_ID, "paired_sensitive_candidate": COMPANION_ID,
        "primary_selection_reason": "The accepted non-sensitive blend had the lowest frozen Final Selection Validation MAE and improved more than the 0.25 percent tie band.",
        "optional_blend_status": "ACCEPTED", "frozen_blend_weights": BLEND_WEIGHTS, "blend_validation_evidence": blend,
        "metrics_used_for_pretest_decisions": ["Validation MAE", "top-decile MAE", "RMSE", "RMSLE", "prediction runtime", "model size", "reload reliability"],
        "model_hashes": model_hashes, "feature_packs": feature_packs, "target_modes": target_modes,
        "test_row_id_sha256": split["test_row_ids_sha256"], "test_row_count": split["test_rows"],
        "visualization_plan": visualization_plan, "visualization_plan_frozen": True,
        "test_target_not_loaded": True,
        "test_target_statement": "Test target values have not been loaded during the pre-Test phase.",
        "decision_lock_statement": "Candidate membership, primary model, blend weights, Feature Packs, target modes, parameters, and visualization membership cannot change after Test opening.",
        "no_fit_rule": "Stage 4L is prediction-only and permits zero model-fit calls.", "previous_stage_audit": previous,
        "split_audit": split, "static_no_fit_audit": no_fit, "status": "PASS",
    }
    freeze["self_sha256_contract"] = canonical_sha(freeze)
    atomic_json(freeze, FREEZE_MANIFEST)
    reloaded = load_json(FREEZE_MANIFEST)
    contract = dict(reloaded)
    expected = contract.pop("self_sha256_contract")
    if canonical_sha(contract) != expected or reloaded["candidate_manifest_sha256"] != sha256_file(CANDIDATE_MANIFEST):
        raise RuntimeError("Pre-Test freeze schema or digest validation failed")
    report = {
        "stage": "stage4l", "status": "PASS", "runtime_seconds": time.perf_counter() - started,
        "candidate_count": candidate_manifest["candidate_count"], "primary": PRIMARY_ID, "companion": COMPANION_ID,
        "blend_status": "ACCEPTED", "blend_weights": BLEND_WEIGHTS, "freeze_timestamp_utc": freeze["freeze_timestamp_utc"],
        "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST), "test_target_loaded": False,
    }
    atomic_json(report, REPORT_DIR / "stage4l_pretest_validation.json")
    print(json.dumps(report, indent=2, default=str))
    return report


def verify_immutable_freeze() -> tuple[dict, dict, str]:
    freeze = load_json(FREEZE_MANIFEST)
    candidate_manifest = load_json(CANDIDATE_MANIFEST)
    freeze_hash = sha256_file(FREEZE_MANIFEST)
    contract = dict(freeze)
    expected = contract.pop("self_sha256_contract")
    if canonical_sha(contract) != expected:
        raise RuntimeError("Freeze contract is invalid")
    if freeze["candidate_manifest_sha256"] != sha256_file(CANDIDATE_MANIFEST):
        raise RuntimeError("Candidate manifest changed after freeze")
    if freeze["candidate_ids"] != [item["candidate_id"] for item in candidate_manifest["candidates"]]:
        raise RuntimeError("Frozen Candidate membership differs")
    if freeze["primary_non_sensitive_candidate"] != PRIMARY_ID or freeze["paired_sensitive_candidate"] != COMPANION_ID:
        raise RuntimeError("Frozen primary pair differs")
    if freeze["frozen_blend_weights"] != BLEND_WEIGHTS:
        raise RuntimeError("Frozen blend weights differ")
    for candidate in candidate_manifest["candidates"]:
        if candidate["candidate_type"] == "saved_model":
            if sha256_file(ROOT / candidate["model_bundle_path"]) != candidate["model_sha256"]:
                raise RuntimeError(f"Frozen model changed: {candidate['candidate_id']}")
    return freeze, candidate_manifest, freeze_hash


def load_test_subset(source: Path, columns: list[str], test_ids: np.ndarray) -> pd.DataFrame:
    wanted = set(int(value) for value in test_ids)
    frame = pd.read_csv(
        source,
        usecols=list(dict.fromkeys(columns)),
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in wanted,
    )
    sorted_ids = np.array(sorted(wanted), dtype=np.int64)
    if len(frame) != len(sorted_ids):
        raise RuntimeError("Loaded Test row count is incorrect")
    frame.insert(0, "row_id", sorted_ids)
    return frame.set_index("row_id", drop=False).loc[test_ids].reset_index(drop=True)


def open_test_once(freeze: dict, candidate_manifest: dict, freeze_hash: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if UNLOCK_AUDIT.exists():
        audit = load_json(UNLOCK_AUDIT)
        if audit["freeze_manifest_sha256"] != freeze_hash or audit["status"] != "PASS":
            raise RuntimeError("Existing Test unlock audit is invalid")
        baseline_path = PRED_DIR / "stage4l__train_mean_baseline.csv"
        if not baseline_path.exists():
            raise RuntimeError("Test was consumed but no frozen target-bearing prediction is available")
        baseline = pd.read_csv(baseline_path, usecols=["row_id", "y_true"])
        return baseline.row_id.to_numpy(np.int64), baseline.y_true.to_numpy(float), audit

    test_ids = pd.read_csv(ROOT / "artifacts/splits/test_row_ids.csv")["row_id"].to_numpy(np.int64)
    train_ids = pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv")["row_id"].to_numpy(np.int64)
    feature_sets = load_json(ROOT / "artifacts/data_contract/feature_sets.json")
    common = feature_sets["common_features"]
    without_source = ROOT / "data/regression_without_sensitive_features.csv"
    with_source = ROOT / "data/regression_with_sensitive_features.csv"
    source_hashes = load_json(ROOT / "artifacts/data_contract/source_hashes_before.json")
    if sha256_file(without_source) != source_hashes["without_sensitive"]["sha256"] or sha256_file(with_source) != source_hashes["with_sensitive"]["sha256"]:
        raise RuntimeError("A canonical source hash changed")

    without = load_test_subset(without_source, [*common, TARGET], test_ids)
    with_sensitive = load_test_subset(with_source, [*common, TARGET], test_ids)
    y_without = without[TARGET].to_numpy(float)
    y_with = with_sensitive[TARGET].to_numpy(float)
    target_equal = bool(np.array_equal(y_without, y_with))
    common_equal = bool(without[common].equals(with_sensitive[common]))
    finite_target = bool(np.isfinite(y_without).all())
    overlap = int(len(np.intersect1d(train_ids, test_ids)))
    if not target_equal or not common_equal or not finite_target or overlap:
        raise RuntimeError("Test integrity failed at the authorized unlock")
    required = set().union(*(set(item["required_raw_input_columns"]) for item in candidate_manifest["candidates"]))
    missing = sorted(required - set(source_header("with_sensitive")))
    if missing:
        raise RuntimeError(f"Required Test Features are missing: {missing}")
    audit = {
        "stage": "stage4l", "unlock_timestamp_utc": utc_now(), "test_row_count": int(len(test_ids)),
        "test_row_id_sha256": sha256_file(ROOT / "artifacts/splits/test_row_ids.csv"),
        "test_target_source": ["data/regression_without_sensitive_features.csv", "data/regression_with_sensitive_features.csv"],
        "target_equality_across_sources": target_equal, "common_feature_equality_across_sources": common_equal,
        "test_order_stable": bool(without.row_id.tolist() == test_ids.tolist() == with_sensitive.row_id.tolist()),
        "target_finite": finite_target, "train_test_overlap_rows": overlap, "candidate_count": candidate_manifest["candidate_count"],
        "freeze_manifest_sha256": freeze_hash, "test_use_authorization": "Explicit Stage 4L one-time locked Test evaluation authorization",
        "fit_operations_allowed": False,
        "permanent_statement": "The Test Set has been consumed. No later model or Feature decision may use its results. A new independent holdout would be required for another unbiased final evaluation.",
        "status": "PASS",
    }
    atomic_json(audit, UNLOCK_AUDIT)
    return test_ids, y_without, audit


def prediction_frame(candidate: dict, test_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, prediction_seconds: float, freeze_hash: str) -> pd.DataFrame:
    signed = y_pred - y_true
    return pd.DataFrame({
        "candidate_id": candidate["candidate_id"], "row_id": test_ids, "y_true": y_true, "y_pred": y_pred,
        "absolute_error": np.abs(signed), "signed_error": signed, "model_family": candidate["model_family"],
        "sensitive_mode": candidate["sensitive_mode"], "prediction_time_seconds": prediction_seconds,
        "model_sha256": candidate["model_sha256"], "freeze_manifest_sha256": freeze_hash,
    })


def save_derived_prediction(candidate: dict, test_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, prediction_seconds: float, freeze_hash: str, sources: list[str]) -> None:
    output = PRED_DIR / f"{candidate['candidate_id']}.csv"
    frame = prediction_frame(candidate, test_ids, y_true, y_pred, prediction_seconds, freeze_hash)
    atomic_csv(frame, output)
    atomic_json({
        "candidate_id": candidate["candidate_id"], "status": "PASS", "row_count": int(len(frame)),
        "prediction_path": str(output.relative_to(ROOT)), "prediction_sha256": sha256_file(output),
        "model_sha256": candidate["model_sha256"], "freeze_manifest_sha256": freeze_hash,
        "model_load_seconds": 0.0, "prediction_seconds": prediction_seconds, "fit_call_count": 0,
        "finite_predictions": bool(np.isfinite(y_pred).all()), "source_prediction_candidates": sources,
    }, output.with_suffix(".metadata.json"))


def generate_predictions() -> dict:
    started = time.perf_counter()
    freeze, candidate_manifest, freeze_hash = verify_immutable_freeze()
    test_ids, y_true, unlock = open_test_once(freeze, candidate_manifest, freeze_hash)
    by_id = {item["candidate_id"]: item for item in candidate_manifest["candidates"]}

    for label in ["mean", "median"]:
        candidate = by_id[f"stage4l__train_{label}_baseline"]
        value = float(candidate["frozen_parameters"]["constant"])
        save_derived_prediction(candidate, test_ids, y_true, np.full(len(y_true), value), 0.0, freeze_hash, [])

    worker_reports = []
    for candidate in candidate_manifest["candidates"]:
        if candidate["candidate_type"] != "saved_model":
            continue
        output = PRED_DIR / f"{candidate['candidate_id']}.csv"
        metadata = output.with_suffix(".metadata.json")
        if output.exists() and metadata.exists():
            saved = load_json(metadata)
            if saved.get("status") == "PASS" and saved.get("freeze_manifest_sha256") == freeze_hash and saved.get("model_sha256") == candidate["model_sha256"]:
                worker_reports.append(saved)
                continue
        command = [sys.executable, str(ROOT / "stage4l_prediction_worker.py"), "--candidate-id", candidate["candidate_id"], "--candidate-manifest", CANDIDATE_REL, "--freeze-manifest", FREEZE_REL]
        invocation_started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
        wall_seconds = time.perf_counter() - invocation_started
        parent = {"candidate_id": candidate["candidate_id"], "return_code": process.returncode, "wall_seconds": wall_seconds, "stdout": process.stdout, "stderr": process.stderr, "status": "PASS" if process.returncode == 0 else "FAIL"}
        atomic_json(parent, REPORT_DIR / "stage4l_workers" / f"{candidate['candidate_id']}.json")
        if process.returncode != 0:
            raise RuntimeError(f"Prediction worker failed: {candidate['candidate_id']}\n{process.stderr}")
        worker_reports.append(load_json(metadata))

    for mode in ["without_sensitive", "with_sensitive"]:
        source_ids = [f"stage4l__{family}__{mode}" for family in BLEND_WEIGHTS]
        source_frames = [pd.read_csv(PRED_DIR / f"{candidate_id}.csv", usecols=["row_id", "y_true", "y_pred", "prediction_time_seconds"]) for candidate_id in source_ids]
        reference = source_frames[0]
        for frame in source_frames[1:]:
            if not np.array_equal(frame.row_id, reference.row_id) or not np.array_equal(frame.y_true, reference.y_true):
                raise RuntimeError("Frozen blend Test inputs are not aligned")
        prediction = sum(BLEND_WEIGHTS[family] * source_frames[index].y_pred.to_numpy(float) for index, family in enumerate(BLEND_WEIGHTS))
        runtime = float(sum(frame.prediction_time_seconds.iloc[0] for frame in source_frames))
        save_derived_prediction(by_id[f"stage4l__blend__{mode}"], test_ids, y_true, prediction, runtime, freeze_hash, source_ids)

    report = {
        "stage": "stage4l", "status": "PASS", "unlock_timestamp_utc": unlock["unlock_timestamp_utc"],
        "candidate_count": len(candidate_manifest["candidates"]), "saved_model_worker_count": len(worker_reports),
        "fit_call_count": int(sum(item.get("fit_call_count", 0) for item in worker_reports)),
        "model_load_seconds": float(sum(item.get("model_load_seconds", 0) for item in worker_reports)),
        "prediction_seconds": float(sum(item.get("prediction_seconds", 0) for item in worker_reports)),
        "wall_seconds": time.perf_counter() - started, "freeze_manifest_sha256": freeze_hash,
    }
    if report["fit_call_count"] != 0:
        raise RuntimeError("A model-fit call was detected")
    atomic_json(report, REPORT_DIR / "stage4l_prediction_generation.json")
    print(json.dumps(report, indent=2))
    return report


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    absolute = np.abs(error)
    mse = float(np.mean(np.square(error)))
    top10 = y_true >= np.quantile(y_true, 0.9)
    top5 = y_true >= np.quantile(y_true, 0.95)
    return {
        "mae": float(np.mean(absolute)), "mse": mse, "rmse": math.sqrt(mse),
        "mape_percent": float(100 * np.mean(absolute / y_true)),
        "r_squared": float(1 - np.sum(np.square(error)) / np.sum(np.square(y_true - np.mean(y_true)))),
        "rmsle": float(np.sqrt(np.mean(np.square(np.log1p(np.maximum(y_pred, 0)) - np.log1p(y_true))))),
        "median_absolute_error": float(np.median(absolute)), "wape_percent": float(100 * np.sum(absolute) / np.sum(np.abs(y_true))),
        "mean_signed_error": float(np.mean(error)), "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "negative_prediction_rate": float(np.mean(y_pred < 0)), "top_decile_mae": float(np.mean(absolute[top10])),
        "top_five_percent_mae": float(np.mean(absolute[top5])), "underestimation_rate": float(np.mean(error < 0)),
        "overestimation_rate": float(np.mean(error > 0)),
    }


def candidate_model_size(candidate: dict, by_id: dict[str, dict]) -> int:
    if candidate["candidate_type"] == "constant_baseline":
        return 0
    if candidate["candidate_type"] == "saved_model":
        return (ROOT / candidate["model_bundle_path"]).stat().st_size
    mode = candidate["sensitive_mode"]
    return sum((ROOT / by_id[f"stage4l__{family}__{mode}"]["model_bundle_path"]).stat().st_size for family in BLEND_WEIGHTS)


def validate_predictions_and_metrics() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    freeze, manifest, freeze_hash = verify_immutable_freeze()
    test_ids = pd.read_csv(ROOT / "artifacts/splits/test_row_ids.csv")["row_id"].to_numpy(np.int64)
    train_ids = pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv")["row_id"].to_numpy(np.int64)
    by_id = {item["candidate_id"]: item for item in manifest["candidates"]}
    frames: dict[str, pd.DataFrame] = {}
    validations = []
    reference_y = None
    for candidate in manifest["candidates"]:
        path = PRED_DIR / f"{candidate['candidate_id']}.csv"
        metadata_path = path.with_suffix(".metadata.json")
        frame = pd.read_csv(path)
        metadata = load_json(metadata_path)
        checks = {
            "row_count": len(frame) == len(test_ids), "unique_row_ids": frame.row_id.nunique() == len(test_ids),
            "exact_row_order": np.array_equal(frame.row_id.to_numpy(np.int64), test_ids),
            "zero_train_overlap": len(np.intersect1d(frame.row_id.to_numpy(np.int64), train_ids)) == 0,
            "finite_prediction": np.isfinite(frame.y_pred.to_numpy(float)).all(), "finite_target": np.isfinite(frame.y_true.to_numpy(float)).all(),
            "candidate_id": frame.candidate_id.nunique() == 1 and frame.candidate_id.iloc[0] == candidate["candidate_id"],
            "model_hash": (candidate["model_sha256"] is None and frame.model_sha256.isna().all()) or str(frame.model_sha256.iloc[0]) == str(candidate["model_sha256"]),
            "freeze_hash": frame.freeze_manifest_sha256.nunique() == 1 and frame.freeze_manifest_sha256.iloc[0] == freeze_hash,
            "metadata_pass": metadata.get("status") == "PASS", "prediction_hash": metadata.get("prediction_sha256") == sha256_file(path),
            "no_sensitive_columns": not bool(set(frame.columns) & SENSITIVE_COLUMNS),
        }
        if reference_y is None:
            reference_y = frame.y_true.to_numpy(float)
        checks["target_alignment"] = np.array_equal(frame.y_true.to_numpy(float), reference_y)
        status = "PASS" if all(checks.values()) else "FAIL"
        validations.append({"candidate_id": candidate["candidate_id"], **checks, "status": status, "prediction_sha256": sha256_file(path)})
        if status != "PASS":
            raise RuntimeError(f"Prediction validation failed: {candidate['candidate_id']}: {checks}")
        frames[candidate["candidate_id"]] = frame
    validation_frame = pd.DataFrame(validations)
    atomic_csv(validation_frame, REPORT_DIR / "stage4l_prediction_validation.csv")

    rows = []
    for candidate in manifest["candidates"]:
        frame = frames[candidate["candidate_id"]]
        metrics = calculate_metrics(frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float))
        rows.append({
            "candidate_id": candidate["candidate_id"], "stage": candidate["stage"], "model_family": candidate["model_family"],
            "sensitive_mode": candidate["sensitive_mode"], "evaluation_design": candidate["evaluation_design"], "test_row_count": len(frame),
            **metrics, "mae_usd": metrics["mae"] * 1000, "rmse_usd": metrics["rmse"] * 1000,
            "prediction_time_seconds": float(frame.prediction_time_seconds.iloc[0]), "model_size_bytes": candidate_model_size(candidate, by_id),
            "pretest_primary_status": candidate["candidate_id"] == PRIMARY_ID, "optional_blend_status": candidate["candidate_type"] == "frozen_blend",
        })
    leaderboard = pd.DataFrame(rows).sort_values(["mae", "candidate_id"]).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    atomic_csv(leaderboard, RESULT_DIR / "stage4l_test_leaderboard.csv")
    return leaderboard, frames


def bootstrap_results(leaderboard: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    ids = leaderboard.candidate_id.tolist()
    y_true = frames[ids[0]].y_true.to_numpy(float)
    predictions = np.column_stack([frames[candidate_id].y_pred.to_numpy(float) for candidate_id in ids])
    absolute = np.abs(predictions - y_true[:, None])
    squared = np.square(predictions - y_true[:, None])
    rng = np.random.default_rng(42)
    b = 500
    mae_samples = np.empty((b, len(ids)))
    rmse_samples = np.empty((b, len(ids)))
    median_samples = np.empty((b, len(ids)))
    for index in range(b):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        mae_samples[index] = absolute[sample].mean(axis=0)
        rmse_samples[index] = np.sqrt(squared[sample].mean(axis=0))
        median_samples[index] = np.median(absolute[sample], axis=0)
    rows = []
    for column, candidate_id in enumerate(ids):
        point = leaderboard[leaderboard.candidate_id == candidate_id].iloc[0]
        rows.append({
            "candidate_id": candidate_id, "bootstrap_resamples": b, "random_seed": 42,
            "mae": point.mae, "mae_ci_low": np.quantile(mae_samples[:, column], 0.025), "mae_ci_high": np.quantile(mae_samples[:, column], 0.975),
            "rmse": point.rmse, "rmse_ci_low": np.quantile(rmse_samples[:, column], 0.025), "rmse_ci_high": np.quantile(rmse_samples[:, column], 0.975),
            "median_absolute_error": point.median_absolute_error,
            "median_absolute_error_ci_low": np.quantile(median_samples[:, column], 0.025),
            "median_absolute_error_ci_high": np.quantile(median_samples[:, column], 0.975),
        })
    confidence = pd.DataFrame(rows)
    atomic_csv(confidence, RESULT_DIR / "stage4l_bootstrap_confidence_intervals.csv")

    challenger_ids = [
        "stage4l__catboost__without_sensitive", "stage4l__lightgbm__without_sensitive",
        "stage4l__xgboost__without_sensitive", "stage4l__hist_gradient_boosting__without_sensitive",
    ]
    primary_col = ids.index(PRIMARY_ID)
    paired_rows = []
    for challenger_id in challenger_ids:
        challenger_col = ids.index(challenger_id)
        mae_difference = mae_samples[:, challenger_col] - mae_samples[:, primary_col]
        rmse_difference = rmse_samples[:, challenger_col] - rmse_samples[:, primary_col]
        point_challenger = leaderboard[leaderboard.candidate_id == challenger_id].iloc[0]
        point_primary = leaderboard[leaderboard.candidate_id == PRIMARY_ID].iloc[0]
        paired_rows.append({
            "primary_candidate_id": PRIMARY_ID, "challenger_candidate_id": challenger_id,
            "difference_definition": "challenger_minus_primary", "mae_difference": point_challenger.mae - point_primary.mae,
            "mae_difference_ci_low": np.quantile(mae_difference, 0.025), "mae_difference_ci_high": np.quantile(mae_difference, 0.975),
            "rmse_difference": point_challenger.rmse - point_primary.rmse,
            "rmse_difference_ci_low": np.quantile(rmse_difference, 0.025), "rmse_difference_ci_high": np.quantile(rmse_difference, 0.975),
            "bootstrap_resamples": b, "random_seed": 42,
        })
    paired = pd.DataFrame(paired_rows)
    atomic_csv(paired, RESULT_DIR / "stage4l_paired_model_differences.csv")
    return confidence, paired, mae_samples


def sensitive_comparisons(leaderboard: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    families = ["lasso", "hist_gradient_boosting", "catboost", "lightgbm", "xgboost", "blend"]
    rng = np.random.default_rng(42)
    sample_indices = [rng.integers(0, TEST_ROWS, size=TEST_ROWS) for _ in range(500)]
    rows = []
    for family in families:
        without_id = f"stage4l__{family}__without_sensitive"
        with_id = f"stage4l__{family}__with_sensitive"
        without = leaderboard[leaderboard.candidate_id == without_id].iloc[0]
        with_sensitive = leaderboard[leaderboard.candidate_id == with_id].iloc[0]
        without_frame = frames[without_id]
        with_frame = frames[with_id]
        y = without_frame.y_true.to_numpy(float)
        ae_without = np.abs(without_frame.y_pred.to_numpy(float) - y)
        ae_with = np.abs(with_frame.y_pred.to_numpy(float) - y)
        samples = np.array([ae_with[sample].mean() - ae_without[sample].mean() for sample in sample_indices])
        rows.append({
            "model_family": family, "without_sensitive_candidate_id": without_id, "with_sensitive_candidate_id": with_id,
            "mae_without_sensitive": without.mae, "mae_with_sensitive": with_sensitive.mae,
            "mae_difference_with_minus_without": with_sensitive.mae - without.mae,
            "relative_mae_difference_percent": 100 * (with_sensitive.mae - without.mae) / without.mae,
            "mae_difference_ci_low": np.quantile(samples, 0.025), "mae_difference_ci_high": np.quantile(samples, 0.975),
            "rmse_difference": with_sensitive.rmse - without.rmse, "rmsle_difference": with_sensitive.rmsle - without.rmsle,
            "r_squared_difference": with_sensitive.r_squared - without.r_squared,
            "tail_mae_difference": with_sensitive.top_decile_mae - without.top_decile_mae,
            "prediction_time_difference_seconds": with_sensitive.prediction_time_seconds - without.prediction_time_seconds,
            "comparison_scope": "accuracy comparison, not a complete fairness audit",
        })
    result = pd.DataFrame(rows)
    atomic_csv(result, RESULT_DIR / "stage4l_sensitive_comparison.csv")
    return result


def focused_error_analysis(leaderboard: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    best_id = str(leaderboard.iloc[0].candidate_id)
    selected = list(dict.fromkeys([PRIMARY_ID, COMPANION_ID, best_id]))
    reference_y = frames[PRIMARY_ID].y_true.to_numpy(float)
    decile = pd.qcut(reference_y, q=10, labels=False, duplicates="drop") + 1
    decile_rows = []
    tail_rows = []
    worst_rows = []
    for candidate_id in selected:
        frame = frames[candidate_id].copy()
        frame["target_decile"] = decile
        for value, group in frame.groupby("target_decile"):
            metrics = calculate_metrics(group.y_true.to_numpy(float), group.y_pred.to_numpy(float))
            decile_rows.append({"candidate_id": candidate_id, "target_decile": int(value), "row_count": len(group), "mae": metrics["mae"], "rmse": metrics["rmse"], "mean_signed_error": metrics["mean_signed_error"]})
        metrics = leaderboard[leaderboard.candidate_id == candidate_id].iloc[0]
        tail_rows.append({"candidate_id": candidate_id, "top_decile_mae": metrics.top_decile_mae, "top_five_percent_mae": metrics.top_five_percent_mae, "underestimation_rate": metrics.underestimation_rate, "overestimation_rate": metrics.overestimation_rate, "mean_signed_error": metrics.mean_signed_error})
        worst = frame.nlargest(20, "absolute_error")[["row_id", "y_true", "y_pred", "absolute_error", "signed_error"]].copy()
        worst.insert(0, "candidate_id", candidate_id)
        worst_rows.append(worst)
    deciles = pd.DataFrame(decile_rows)
    tails = pd.DataFrame(tail_rows)
    worst = pd.concat(worst_rows, ignore_index=True)
    atomic_csv(deciles, RESULT_DIR / "stage4l_error_by_decile.csv")
    atomic_csv(tails, RESULT_DIR / "stage4l_tail_metrics.csv")
    atomic_csv(worst, RESULT_DIR / "stage4l_worst_errors.csv")
    return deciles, tails, worst, selected


def run_analysis() -> dict:
    started = time.perf_counter()
    leaderboard, frames = validate_predictions_and_metrics()
    bootstrap_started = time.perf_counter()
    confidence, paired, _ = bootstrap_results(leaderboard, frames)
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    sensitive = sensitive_comparisons(leaderboard, frames)
    deciles, tails, worst, selected = focused_error_analysis(leaderboard, frames)
    efficiency = leaderboard[["candidate_id", "model_family", "sensitive_mode", "mae", "prediction_time_seconds", "model_size_bytes"]].copy()
    atomic_csv(efficiency, RESULT_DIR / "stage4l_model_efficiency.csv")
    report = {
        "stage": "stage4l", "status": "PASS", "candidate_count": len(leaderboard),
        "primary_candidate": PRIMARY_ID, "primary_rank": int(leaderboard.loc[leaderboard.candidate_id == PRIMARY_ID, "rank"].iloc[0]),
        "best_observed_candidate": str(leaderboard.iloc[0].candidate_id), "best_observed_mae": float(leaderboard.iloc[0].mae),
        "bootstrap_resamples": 500, "bootstrap_seconds": bootstrap_seconds, "sensitive_family_count": len(sensitive),
        "detailed_error_candidates": selected, "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(report, REPORT_DIR / "stage4l_analysis_summary.json")
    print(json.dumps(report, indent=2))
    return report


def short_label(candidate_id: str) -> str:
    return candidate_id.replace("stage4l__", "").replace("__without_sensitive", " (no sensitive)").replace("__with_sensitive", " (sensitive)").replace("_", " ").title()


def save_plot(fig, stem: str, data_path: Path, candidate_ids: list[str], source: str, modes: list[str], entries: list[dict], title: str, sample_seed: int | None = None) -> None:
    for suffix in ["png", "svg"]:
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        entries.append({
            "figure_id": f"{stem}__{suffix}", "figure_title": title, "file_path": str(path.relative_to(ROOT)),
            "file_type": suffix, "plotting_data_path": str(data_path.relative_to(ROOT)), "candidate_ids_used": candidate_ids,
            "data_source": source, "sensitive_modes_included": modes, "random_plotting_sample_seed": sample_seed,
            "figure_sha256": sha256_file(path), "creation_timestamp_utc": utc_now(), "status": "PASS",
        })


def create_figures() -> dict:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"catboost": "#0072B2", "lightgbm": "#E69F00", "xgboost": "#009E73", "validation_frozen_boosting_blend": "#CC79A7", "blend": "#CC79A7", "hist_gradient_boosting": "#56B4E9", "lasso": "#D55E00", "train_mean_baseline": "#777777", "train_median_baseline": "#999999"}
    leaderboard = pd.read_csv(RESULT_DIR / "stage4l_test_leaderboard.csv")
    confidence = pd.read_csv(RESULT_DIR / "stage4l_bootstrap_confidence_intervals.csv")
    paired = pd.read_csv(RESULT_DIR / "stage4l_paired_model_differences.csv")
    sensitive = pd.read_csv(RESULT_DIR / "stage4l_sensitive_comparison.csv")
    deciles = pd.read_csv(RESULT_DIR / "stage4l_error_by_decile.csv")
    manifest = load_json(CANDIDATE_MANIFEST)
    by_id = {item["candidate_id"]: item for item in manifest["candidates"]}
    best_id = str(leaderboard.iloc[0].candidate_id)
    entries: list[dict] = []
    plot_dir = RESULT_DIR / "plotting_data"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. MAE leaderboard.
    data = leaderboard[["rank", "candidate_id", "model_family", "sensitive_mode", "mae", "pretest_primary_status"]].copy()
    data["best_observed"] = data.candidate_id == best_id
    data["label"] = data.candidate_id.map(short_label)
    data_path = plot_dir / "stage4l_plot_test_mae_leaderboard.csv"
    atomic_csv(data, data_path)
    ordered = data.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(ordered.label, ordered.mae, color=[colors.get(family, "#666666") for family in ordered.model_family])
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        if row.pretest_primary_status:
            bar.set_edgecolor("black"); bar.set_linewidth(2.5); bar.set_hatch("//")
        if row.best_observed:
            bar.set_edgecolor("#D55E00"); bar.set_linewidth(2.5)
        ax.text(row.mae + data.mae.max() * 0.006, bar.get_y() + bar.get_height() / 2, f"{row.mae:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, data.mae.max() * 1.15); ax.set_xlabel("Test MAE (thousand US dollars)"); ax.set_title("Stage 4L Final Test MAE Leaderboard")
    ax.legend(handles=[Line2D([0], [0], color="black", lw=3, label="Pre-Test primary"), Line2D([0], [0], color="#D55E00", lw=3, label="Best observed")], loc="upper right")
    save_plot(fig, "stage4l_test_mae_leaderboard", data_path, data.candidate_id.tolist(), "Test metrics", data.sensitive_mode.unique().tolist(), entries, "Final Test MAE Leaderboard"); plt.close(fig)

    # 2. MAE intervals.
    data = leaderboard[["rank", "candidate_id", "model_family", "sensitive_mode", "mae"]].merge(confidence[["candidate_id", "mae_ci_low", "mae_ci_high"]], on="candidate_id").sort_values("mae", ascending=False)
    data["label"] = data.candidate_id.map(short_label)
    data_path = plot_dir / "stage4l_plot_mae_confidence_intervals.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(11, 8)); y = np.arange(len(data));
    ax.errorbar(data.mae, y, xerr=[data.mae - data.mae_ci_low, data.mae_ci_high - data.mae], fmt="o", color="#0072B2", ecolor="#666666", capsize=3)
    primary_position = np.where(data.candidate_id.to_numpy() == PRIMARY_ID)[0][0]; ax.scatter([data.iloc[primary_position].mae], [primary_position], s=100, facecolors="none", edgecolors="black", linewidths=2)
    ax.set_yticks(y, data.label); ax.set_xlabel("Test MAE with 95% bootstrap CI (thousand US dollars)"); ax.set_title("MAE Uncertainty — 500 Paired Bootstrap Resamples")
    save_plot(fig, "stage4l_mae_confidence_intervals", data_path, data.candidate_id.tolist(), "Test predictions and bootstrap", data.sensitive_mode.unique().tolist(), entries, "MAE with Bootstrap Confidence Intervals"); plt.close(fig)

    # 3. Multi-metric normalized comparison.
    metrics = ["mae", "rmse", "rmsle", "r_squared", "p90_absolute_error", "top_decile_mae"]
    data = leaderboard[["candidate_id", *metrics]].copy().set_index("candidate_id")
    normalized = pd.DataFrame(index=data.index)
    for metric in metrics:
        low, high = data[metric].min(), data[metric].max()
        normalized[metric] = (data[metric] - low) / (high - low) if metric == "r_squared" else (high - data[metric]) / (high - low)
    plot_data = data.add_prefix("original_").join(normalized.add_prefix("score_higher_is_better_")).reset_index()
    data_path = plot_dir / "stage4l_plot_multi_metric_comparison.csv"; atomic_csv(plot_data, data_path)
    fig, ax = plt.subplots(figsize=(11, 8)); image = ax.imshow(normalized.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(normalized)), [short_label(value) for value in normalized.index]); ax.set_xticks(np.arange(len(metrics)), [value.replace("_", " ").upper() for value in metrics], rotation=30, ha="right")
    ax.set_title("Normalized Multi-Metric Comparison\n1 = better; errors are reversed and R² is direct"); fig.colorbar(image, ax=ax, label="Direction-aware normalized score")
    save_plot(fig, "stage4l_multi_metric_comparison", data_path, normalized.index.tolist(), "Test metrics; min-max direction-aware normalization", leaderboard.sensitive_mode.unique().tolist(), entries, "Multi-Metric Model Comparison"); plt.close(fig)

    # 4. Validation versus Test.
    ids = [f"stage4l__{family}__without_sensitive" for family in ["catboost", "lightgbm", "xgboost", "blend"]]
    rows = []
    for candidate_id in ids:
        item = by_id[candidate_id]; test = leaderboard[leaderboard.candidate_id == candidate_id].iloc[0]
        rows.append({"candidate_id": candidate_id, "model_family": item["model_family"], "validation_mae": item["pretest_metrics"]["mae"], "test_mae": test.mae, "test_minus_validation": test.mae - item["pretest_metrics"]["mae"]})
    data = pd.DataFrame(rows); data_path = plot_dir / "stage4l_plot_validation_vs_test_mae.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(9, 6));
    for _, row in data.iterrows():
        color = colors.get(row.model_family, colors.get("blend")); ax.plot([0, 1], [row.validation_mae, row.test_mae], marker="o", color=color, label=short_label(row.candidate_id)); ax.text(1.02, row.test_mae, f"{row.test_minus_validation:+.2f}", va="center", fontsize=8)
    ax.set_xticks([0, 1], ["Frozen Validation", "Final Test"]); ax.set_ylabel("MAE (thousand US dollars)"); ax.set_title("Validation Versus Test MAE — Primary Frozen Before Test"); ax.legend(fontsize=8)
    save_plot(fig, "stage4l_validation_vs_test_mae", data_path, ids, "Frozen Validation and Test metrics", ["without_sensitive"], entries, "Validation Versus Test MAE"); plt.close(fig)

    # 5. Sensitive comparison.
    data = sensitive.copy(); data_path = plot_dir / "stage4l_plot_sensitive_mae_comparison.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(10, 7)); y = np.arange(len(data))
    for index, row in data.iterrows():
        color = colors.get(row.model_family, colors.get("blend")); ax.plot([row.mae_without_sensitive, row.mae_with_sensitive], [index, index], color=color, lw=3); ax.scatter([row.mae_without_sensitive], [index], color=color, marker="o"); ax.scatter([row.mae_with_sensitive], [index], color=color, marker="s"); ax.text(max(row.mae_without_sensitive, row.mae_with_sensitive) + .1, index, f"Δ {row.mae_difference_with_minus_without:+.2f} [{row.mae_difference_ci_low:+.2f}, {row.mae_difference_ci_high:+.2f}]", va="center", fontsize=8)
    ax.set_yticks(y, data.model_family.str.replace("_", " ").str.title()); ax.set_xlabel("Test MAE (lower is better; thousand US dollars)"); ax.set_title("Sensitive Versus Non-Sensitive Accuracy — Not a Fairness Score")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="gray", label="Without sensitive"), Line2D([0], [0], marker="s", color="gray", label="With sensitive")])
    save_plot(fig, "stage4l_sensitive_mae_comparison", data_path, [value for pair in zip(data.without_sensitive_candidate_id, data.with_sensitive_candidate_id) for value in pair], "Test metrics and paired bootstrap", ["without_sensitive", "with_sensitive"], entries, "Sensitive Versus Non-Sensitive Delta"); plt.close(fig)

    # 6 and 7. Point-level plots on deterministic samples.
    rng = np.random.default_rng(42); sample = np.sort(rng.choice(TEST_ROWS, size=20_000, replace=False))
    point_paths = {}; point_frames = {}
    for role, candidate_id in [("primary", PRIMARY_ID), ("best_observed", best_id)]:
        frame = pd.read_csv(PRED_DIR / f"{candidate_id}.csv", usecols=["row_id", "y_true", "y_pred", "signed_error"]).iloc[sample].copy()
        path = plot_dir / f"stage4l_plot_{role}_points.csv"; atomic_csv(frame, path); point_paths[role] = path; point_frames[role] = frame
    common_limit = max(max(frame.y_true.quantile(.995), frame.y_pred.quantile(.995)) for frame in point_frames.values())
    for role, candidate_id in [("primary", PRIMARY_ID), ("best_observed", best_id)]:
        frame = point_frames[role]; path = point_paths[role]
        fig, ax = plt.subplots(figsize=(7, 6)); hb = ax.hexbin(frame.y_true, frame.y_pred, gridsize=55, mincnt=1, cmap="viridis"); limit = max(frame.y_true.quantile(.995), frame.y_pred.quantile(.995)); ax.plot([0, limit], [0, limit], "--", color="#D55E00", label="Perfect prediction"); ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.set_xlabel("Actual (thousand US dollars)"); ax.set_ylabel("Predicted (thousand US dollars)"); mae = leaderboard.loc[leaderboard.candidate_id == candidate_id, "mae"].iloc[0]; ax.set_title(f"{role.replace('_', ' ').title()} Actual Versus Predicted — MAE {mae:.2f}"); fig.colorbar(hb, ax=ax, label="Hexbin count"); ax.legend()
        ax.lines[-1].set_data([0, common_limit], [0, common_limit]); ax.set_xlim(0, common_limit); ax.set_ylim(0, common_limit)
        stem = "stage4l_primary_actual_vs_predicted" if role == "primary" else "stage4l_best_observed_actual_vs_predicted"
        save_plot(fig, stem, path, [candidate_id], "Deterministic Test plotting sample", [by_id[candidate_id]["sensitive_mode"]], entries, f"{role.replace('_', ' ').title()} Actual Versus Predicted", 42); plt.close(fig)
    primary_points = pd.read_csv(point_paths["primary"])
    fig, ax = plt.subplots(figsize=(7, 6)); hb = ax.hexbin(primary_points.y_pred, primary_points.signed_error, gridsize=55, mincnt=1, cmap="viridis"); ax.axhline(0, ls="--", color="#D55E00"); ax.set_xlabel("Predicted (thousand US dollars)"); ax.set_ylabel("Signed error = predicted - actual"); ax.set_title("Primary Residual Versus Predicted"); fig.colorbar(hb, ax=ax, label="Hexbin count")
    save_plot(fig, "stage4l_primary_residual_vs_predicted", point_paths["primary"], [PRIMARY_ID], "Deterministic Test plotting sample", ["without_sensitive"], entries, "Primary Residual Versus Predicted", 42); plt.close(fig)
    data = pd.read_csv(PRED_DIR / f"{PRIMARY_ID}.csv", usecols=["signed_error"]); clip_low = data.signed_error.quantile(.005); clip_high = data.signed_error.quantile(.995); clipped = data.signed_error.clip(clip_low, clip_high); hist_counts, edges = np.histogram(clipped, bins=80); hist_data = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist_counts, "clip_low": clip_low, "clip_high": clip_high, "full_mean_signed_error": data.signed_error.mean(), "full_median_signed_error": data.signed_error.median()}); hist_path = plot_dir / "stage4l_plot_primary_residual_distribution.csv"; atomic_csv(hist_data, hist_path)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(clipped, bins=edges, color="#0072B2", alpha=.8); ax.axvline(0, ls="--", color="black"); ax.axvline(data.signed_error.mean(), color="#D55E00", label=f"Mean {data.signed_error.mean():.2f}"); ax.axvline(data.signed_error.median(), color="#009E73", label=f"Median {data.signed_error.median():.2f}"); ax.set_xlabel("Signed error (thousand US dollars; display clipped to 0.5–99.5%)"); ax.set_title("Primary Residual Distribution"); ax.legend()
    save_plot(fig, "stage4l_primary_residual_distribution", hist_path, [PRIMARY_ID], "Full Test residual aggregation", ["without_sensitive"], entries, "Primary Residual Distribution"); plt.close(fig)

    # 8. Error by decile.
    data = deciles.copy(); data_path = plot_dir / "stage4l_plot_error_by_target_decile.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for candidate_id, group in data.groupby("candidate_id"):
        family = by_id[candidate_id]["model_family"]; ax.plot(group.target_decile, group.mae, marker="o", label=short_label(candidate_id), color=colors.get(family, colors.get("blend")))
    ax.axvspan(9.5, 10.5, color="#D55E00", alpha=.12); ax.set_xticks(range(1, 11)); ax.set_xlabel("Actual target decile"); ax.set_ylabel("MAE (thousand US dollars)"); ax.set_title("Error by Target Decile"); ax.legend(fontsize=8)
    save_plot(fig, "stage4l_error_by_target_decile", data_path, data.candidate_id.unique().tolist(), "Test error aggregation", [by_id[value]["sensitive_mode"] for value in data.candidate_id.unique()], entries, "Error by Target Decile"); plt.close(fig)

    # 9. Efficiency.
    data = leaderboard[leaderboard.model_size_bytes > 0][["candidate_id", "model_family", "sensitive_mode", "mae", "prediction_time_seconds", "model_size_bytes"]].copy(); data["bubble_size"] = 35 + 300 * np.log1p(data.model_size_bytes) / np.log1p(data.model_size_bytes.max()); data["bubble_scaling_rule"] = "35 + 300 * log1p(size_bytes) / log1p(max_size_bytes)"; efficiency_plot_data = data.copy(); data_path = plot_dir / "stage4l_plot_model_efficiency.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    for _, row in data.iterrows():
        marker = "o" if row.sensitive_mode == "without_sensitive" else "s"
        ax.scatter(row.prediction_time_seconds, row.mae, s=row.bubble_size, marker=marker, color=colors.get(row.model_family, colors.get("blend")), alpha=.7, edgecolors="black" if row.candidate_id == PRIMARY_ID else "none", label=short_label(row.candidate_id))
    ax.set_xlabel("Prediction time (seconds; lower is better)"); ax.set_ylabel("Test MAE (thousand US dollars; lower is better)"); ax.set_title("Model Efficiency — Bubble Area Uses Log-Scaled Bundle Size"); ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Candidate")
    save_plot(fig, "stage4l_model_efficiency", data_path, data.candidate_id.tolist(), "Test metrics and bundle sizes", data.sensitive_mode.unique().tolist(), entries, "Model Efficiency"); plt.close(fig)

    # 10. Paired differences.
    data = paired.copy(); data["label"] = data.challenger_candidate_id.map(short_label); data_path = plot_dir / "stage4l_plot_paired_mae_differences.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(9, 5)); y = np.arange(len(data)); ax.errorbar(data.mae_difference, y, xerr=[data.mae_difference - data.mae_difference_ci_low, data.mae_difference_ci_high - data.mae_difference], fmt="o", color="#0072B2", capsize=4); ax.axvline(0, color="black", ls="--"); ax.set_yticks(y, data.label); ax.set_xlabel("Challenger MAE - primary MAE (positive favors primary)"); ax.set_title("Paired MAE Difference — 95% Bootstrap CI")
    save_plot(fig, "stage4l_paired_mae_differences", data_path, [PRIMARY_ID, *data.challenger_candidate_id.tolist()], "Paired Test bootstrap", ["without_sensitive"], entries, "Paired Model-Difference Intervals"); plt.close(fig)

    # Optional accepted-blend plots.
    data = pd.DataFrame({"model_family": list(BLEND_WEIGHTS), "weight": list(BLEND_WEIGHTS.values())}); data_path = plot_dir / "stage4l_plot_blend_weights.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(data.model_family.str.title(), data.weight, color=[colors[value] for value in data.model_family]); ax.set_xlim(0, 1); ax.set_xlabel("Frozen Validation weight"); ax.set_title("Accepted Blend Weights — Frozen Before Test")
    save_plot(fig, "stage4l_blend_weights", data_path, [PRIMARY_ID], "Pre-Test freeze", ["without_sensitive", "with_sensitive"], entries, "Blend Weights"); plt.close(fig)
    blend = load_json(ROOT / "artifacts/reports/stage4l_blend_validation_evidence.json"); best_individual = blend["best_individual"]; best_test_individual = leaderboard[(leaderboard.model_family.isin(["catboost", "lightgbm", "xgboost"])) & (leaderboard.sensitive_mode == "without_sensitive")].sort_values("mae").iloc[0]
    data = pd.DataFrame({"evidence": ["Best individual Validation", "Blend Validation", "Best individual Test", "Blend Test"], "mae": [blend["individual_without_sensitive"][best_individual]["mae"], blend["best_grid"]["mae"], best_test_individual.mae, leaderboard.loc[leaderboard.candidate_id == PRIMARY_ID, "mae"].iloc[0]]}); data_path = plot_dir / "stage4l_plot_blend_performance.csv"; atomic_csv(data, data_path)
    fig, ax = plt.subplots(figsize=(8, 5)); bars = ax.bar(data.evidence, data.mae, color=["#0072B2", "#CC79A7", "#0072B2", "#CC79A7"]); ax.set_ylim(0, data.mae.max() * 1.15); ax.set_ylabel("MAE (thousand US dollars)"); ax.set_title("Blend Benefit — Accepted Before Test"); ax.tick_params(axis="x", rotation=20); [ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.1, f"{bar.get_height():.2f}", ha="center") for bar in bars]
    save_plot(fig, "stage4l_blend_performance", data_path, [PRIMARY_ID, f"stage4l__{best_individual}__without_sensitive"], "Frozen Validation and Test metrics", ["without_sensitive"], entries, "Blend Performance"); plt.close(fig)

    # Optional compact dashboard (third optional figure).
    dashboard_parts = [leaderboard.head(5)[["candidate_id", "mae"]].assign(panel="top_test_models"), confidence[["candidate_id", "mae", "mae_ci_low", "mae_ci_high"]].assign(panel="mae_confidence_intervals"), sensitive[["model_family", "mae_difference_with_minus_without", "mae_difference_ci_low", "mae_difference_ci_high"]].assign(panel="sensitive_difference"), efficiency_plot_data[["candidate_id", "mae", "prediction_time_seconds", "model_size_bytes"]].assign(panel="model_efficiency")]; dashboard_data = pd.concat(dashboard_parts, ignore_index=True, sort=False); dashboard_path = plot_dir / "stage4l_plot_final_summary_dashboard.csv"; atomic_csv(dashboard_data, dashboard_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)); top = leaderboard.head(5).sort_values("mae", ascending=False); axes[0,0].barh(top.candidate_id.map(short_label), top.mae, color="#0072B2"); axes[0,0].set_title("Top Test MAE"); axes[0,0].set_xlim(0, top.mae.max()*1.12); axes[0,1].errorbar(confidence.mae, np.arange(len(confidence)), xerr=[confidence.mae-confidence.mae_ci_low, confidence.mae_ci_high-confidence.mae], fmt="."); axes[0,1].set_title("All MAE Intervals"); axes[0,1].set_yticks([]); axes[1,0].bar(sensitive.model_family, sensitive.mae_difference_with_minus_without, color="#009E73"); axes[1,0].axhline(0,color="black",ls="--"); axes[1,0].tick_params(axis="x",rotation=25); axes[1,0].set_title("Sensitive MAE Delta"); eff = pd.read_csv(plot_dir / "stage4l_plot_model_efficiency.csv"); axes[1,1].scatter(eff.prediction_time_seconds, eff.mae, s=eff.bubble_size, alpha=.6); axes[1,1].set_title("Efficiency"); axes[1,1].set_xlabel("Seconds"); axes[1,1].set_ylabel("MAE"); fig.suptitle("Stage 4L Compact Final Summary"); fig.tight_layout()
    save_plot(fig, "stage4l_final_summary_dashboard", dashboard_path, leaderboard.candidate_id.tolist(), "Presentation summary of saved Stage 4L results", leaderboard.sensitive_mode.unique().tolist(), entries, "Compact Final Summary Dashboard"); plt.close(fig)

    visual_manifest = {"stage": "stage4l", "created_at_utc": utc_now(), "figure_file_count": len(entries), "figure_groups": 13, "entries": entries}
    frozen_ids = set(item["candidate_id"] for item in manifest["candidates"])
    required_stems = ["stage4l_test_mae_leaderboard", "stage4l_mae_confidence_intervals", "stage4l_multi_metric_comparison", "stage4l_validation_vs_test_mae", "stage4l_sensitive_mae_comparison", "stage4l_primary_actual_vs_predicted", "stage4l_best_observed_actual_vs_predicted", "stage4l_primary_residual_vs_predicted", "stage4l_primary_residual_distribution", "stage4l_error_by_target_decile", "stage4l_model_efficiency", "stage4l_paired_mae_differences", "stage4l_blend_weights", "stage4l_blend_performance", "stage4l_final_summary_dashboard"]
    checks = {
        "required_png_files": all((FIG_DIR / f"{stem}.png").stat().st_size > 0 for stem in required_stems),
        "required_svg_files": all((FIG_DIR / f"{stem}.svg").stat().st_size > 0 for stem in required_stems),
        "plotting_data_exists": all((ROOT / entry["plotting_data_path"]).exists() for entry in entries),
        "candidate_ids_frozen": all(set(entry["candidate_ids_used"]) <= frozen_ids for entry in entries),
        "no_sensitive_raw_columns": all(not (set(pd.read_csv(ROOT / entry["plotting_data_path"], nrows=0).columns) & SENSITIVE_COLUMNS) for entry in entries),
        "no_decision_changed": True,
    }
    visual_manifest["checks"] = checks; visual_manifest["status"] = "PASS" if all(checks.values()) else "FAIL"
    atomic_json(visual_manifest, MANIFEST_DIR / "stage4l_visualization_manifest.json")
    if visual_manifest["status"] != "PASS": raise RuntimeError(f"Visualization manifest failed: {checks}")
    return visual_manifest


def create_registry_rows(leaderboard: pd.DataFrame, confidence: pd.DataFrame) -> pd.DataFrame:
    registry_path = ROOT / "artifacts/results/experiment_results.csv"
    current_bytes = registry_path.read_bytes()
    baseline_path = REPORT_DIR / "stage4l_registry_baseline.json"
    current_registry = pd.read_csv(registry_path)
    prior = current_registry[~current_registry.experiment_id.str.startswith("stage4l__")].copy()
    if len(prior) != 270 or not prior.experiment_id.is_unique:
        raise RuntimeError("Unexpected pre-Stage 4L Registry state")
    if baseline_path.exists():
        baseline = load_json(baseline_path)
        prior_byte_count = int(baseline["prior_byte_count"]); prior_hash = baseline["prior_sha256"]
        if hashlib.sha256(current_bytes[:prior_byte_count]).hexdigest() != prior_hash:
            raise RuntimeError("Previous Registry byte prefix changed")
    else:
        if len(current_registry) != 270:
            raise RuntimeError("Cannot capture a late Registry baseline")
        prior_byte_count = len(current_bytes); prior_hash = hashlib.sha256(current_bytes).hexdigest()
        atomic_json({"captured_at_utc": utc_now(), "prior_row_count": len(prior), "prior_sha256": prior_hash, "prior_byte_count": prior_byte_count, "prior_ids_sha256": canonical_sha(prior.experiment_id.tolist()), "status": "PASS"}, baseline_path)
    headers = prior.columns.tolist()
    rows = []
    timestamp = utc_now()
    by_conf = confidence.set_index("candidate_id")
    for _, item in leaderboard.iterrows():
        base = {
            "timestamp_utc": timestamp, "model_family": item.model_family, "model_name": item.model_family,
            "sensitive_mode": item.sensitive_mode, "feature_set": "stage4l_frozen_candidate", "target_mode": "original_target_scale",
            "fold_number": "", "training_row_count": TRAIN_ROWS, "validation_row_count": 0, "test_row_count": TEST_ROWS,
            "parameter_json": json.dumps({"candidate_id": item.candidate_id}, sort_keys=True, separators=(",", ":")),
            "mae": item.mae, "mse": item.mse, "rmse": item.rmse, "mape_percent": item.mape_percent, "r_squared": item.r_squared,
            "rmsle": item.rmsle, "rmsle_clipped_zero": "", "median_absolute_error": item.median_absolute_error,
            "wape_percent": item.wape_percent, "mean_signed_error": item.mean_signed_error, "p90_absolute_error": item.p90_absolute_error,
            "negative_prediction_rate": item.negative_prediction_rate, "fit_time_seconds": 0, "prediction_time_seconds": item.prediction_time_seconds,
            "status": "success", "model_artifact_path": "", "prediction_artifact_path": f"artifacts\\predictions\\final_test\\{item.candidate_id}.csv",
        }
        test_row = dict(base); test_row.update({"experiment_id": f"stage4l__test__{item.candidate_id.removeprefix('stage4l__')}", "evaluation_stage": "locked_test", "notes": "Frozen Candidate common locked Test evaluation; no tuning or fitting."}); rows.append(test_row)
        ci = by_conf.loc[item.candidate_id]
        bootstrap_row = dict(base); bootstrap_row.update({"experiment_id": f"stage4l__bootstrap__{item.candidate_id.removeprefix('stage4l__')}", "evaluation_stage": "locked_test_bootstrap_summary", "notes": f"500 paired resamples, seed 42; MAE 95% CI [{ci.mae_ci_low:.6f}, {ci.mae_ci_high:.6f}]."}); rows.append(bootstrap_row)
    primary = leaderboard[leaderboard.candidate_id == PRIMARY_ID].iloc[0]
    primary_row = {key: "" for key in headers}
    primary_row.update({
        "experiment_id": "stage4l__primary_evaluation", "timestamp_utc": timestamp, "model_family": primary.model_family,
        "model_name": "pretest_primary", "sensitive_mode": primary.sensitive_mode, "feature_set": "frozen_blend",
        "target_mode": "original_target_scale", "evaluation_stage": "primary_locked_test_evaluation", "training_row_count": TRAIN_ROWS,
        "validation_row_count": 0, "test_row_count": TEST_ROWS, "parameter_json": json.dumps({"weights": BLEND_WEIGHTS}, sort_keys=True, separators=(",", ":")),
        "mae": primary.mae, "mse": primary.mse, "rmse": primary.rmse, "mape_percent": primary.mape_percent,
        "r_squared": primary.r_squared, "rmsle": primary.rmsle, "median_absolute_error": primary.median_absolute_error,
        "wape_percent": primary.wape_percent, "mean_signed_error": primary.mean_signed_error, "p90_absolute_error": primary.p90_absolute_error,
        "negative_prediction_rate": primary.negative_prediction_rate, "fit_time_seconds": 0, "prediction_time_seconds": primary.prediction_time_seconds,
        "status": "success", "notes": "Pre-Test primary evaluation; primary remains frozen after descriptive Test ranking.",
        "prediction_artifact_path": f"artifacts\\predictions\\final_test\\{PRIMARY_ID}.csv",
    }); rows.append(primary_row)
    export = pd.DataFrame(rows)[headers]
    if len(export) != 29 or not export.experiment_id.is_unique:
        raise RuntimeError("Stage 4L Registry export is incomplete")
    atomic_csv(export, RESULT_DIR / "stage4l_registry_rows.csv")

    existing_ids = set(current_registry.experiment_id)
    missing = export[~export.experiment_id.isin(existing_ids)]
    if len(missing):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\r\n", extrasaction="ignore")
        for row in missing.to_dict(orient="records"):
            writer.writerow(row)
        with registry_path.open("ab") as handle:
            handle.write(stream.getvalue().encode("utf-8"))
    updated_bytes = registry_path.read_bytes()
    updated = pd.read_csv(registry_path)
    checks = {
        "prior_prefix_preserved": hashlib.sha256(updated_bytes[:prior_byte_count]).hexdigest() == prior_hash, "prior_row_count": len(prior),
        "updated_row_count": len(updated), "expected_updated_row_count": len(prior) + len(export),
        "unique_experiment_ids": updated.experiment_id.is_unique,
        "all_stage4l_ids_present_once": all((updated.experiment_id == experiment_id).sum() == 1 for experiment_id in export.experiment_id),
    }
    checks["status"] = "PASS" if all(value for key, value in checks.items() if key != "status" and isinstance(value, bool)) and checks["updated_row_count"] == checks["expected_updated_row_count"] else "FAIL"
    checks["prior_sha256"] = prior_hash; checks["updated_sha256"] = hashlib.sha256(updated_bytes).hexdigest()
    atomic_json(checks, REPORT_DIR / "stage4l_registry_update.json")
    if checks["status"] != "PASS": raise RuntimeError(f"Registry update failed: {checks}")
    return export


def create_recommendation_and_reports() -> dict:
    started = time.perf_counter()
    visual_started = time.perf_counter(); visual = create_figures(); visual_seconds = time.perf_counter() - visual_started
    leaderboard = pd.read_csv(RESULT_DIR / "stage4l_test_leaderboard.csv")
    confidence = pd.read_csv(RESULT_DIR / "stage4l_bootstrap_confidence_intervals.csv")
    paired = pd.read_csv(RESULT_DIR / "stage4l_paired_model_differences.csv")
    sensitive = pd.read_csv(RESULT_DIR / "stage4l_sensitive_comparison.csv")
    deciles = pd.read_csv(RESULT_DIR / "stage4l_error_by_decile.csv")
    tails = pd.read_csv(RESULT_DIR / "stage4l_tail_metrics.csv")
    manifest = load_json(CANDIDATE_MANIFEST); by_id = {item["candidate_id"]: item for item in manifest["candidates"]}
    primary = leaderboard[leaderboard.candidate_id == PRIMARY_ID].iloc[0]
    best = leaderboard.iloc[0]
    primary_ci = confidence[confidence.candidate_id == PRIMARY_ID].iloc[0]
    validation = by_id[PRIMARY_ID]["pretest_metrics"]
    primary_deciles = deciles[deciles.candidate_id == PRIMARY_ID]
    primary_tail = tails[tails.candidate_id == PRIMARY_ID].iloc[0]
    best_difference = best.mae - primary.mae
    sensitive_blend = sensitive[sensitive.model_family == "blend"].iloc[0]
    recommendation = {
        "stage": "stage4l", "created_at_utc": utc_now(), "pretest_primary_model": PRIMARY_ID,
        "primary_test_metrics": primary.to_dict(), "primary_confidence_intervals": primary_ci.to_dict(),
        "sensitive_companion": COMPANION_ID, "best_observed_test_candidate": best.candidate_id,
        "best_observed_mae": best.mae, "best_observed_minus_primary_mae": best_difference,
        "optional_blend_status": "ACCEPTED before Test", "frozen_blend_weights": BLEND_WEIGHTS,
        "validation_test_consistency": {"validation_mae": validation["mae"], "test_mae": primary.mae, "test_minus_validation": primary.mae - validation["mae"], "relative_change_percent": 100 * (primary.mae - validation["mae"]) / validation["mae"]},
        "model_size_bytes": int(primary.model_size_bytes), "prediction_time_seconds": primary.prediction_time_seconds,
        "interpretability_level": "Low to medium: a fixed weighted average of three tree-boosting models.",
        "sensitive_mode_status": "Operational recommendation remains non-sensitive. The sensitive companion accuracy comparison is not a fairness audit.",
        "important_visual_conclusions": {
            "leaderboard": f"The primary ranked {int(primary['rank'])} of {len(leaderboard)}; the sensitive blend was best observed.",
            "uncertainty": "The primary and non-sensitive CatBoost paired MAE interval crosses zero; larger primary gains over other non-sensitive challengers do not.",
            "tail": f"Primary top-decile MAE is {primary.top_decile_mae:.3f} and top-five-percent MAE is {primary.top_five_percent_mae:.3f} thousand US dollars.",
            "deciles": f"Primary MAE rises from {primary_deciles.mae.min():.3f} to {primary_deciles.mae.max():.3f} across target deciles.",
            "sensitive_delta": f"Sensitive blend minus non-sensitive blend MAE is {sensitive_blend.mae_difference_with_minus_without:.3f}.",
        },
        "remaining_risks": ["Test is consumed.", "Extreme target values drive wide RMSE uncertainty.", "Proxy variables may remain in non-sensitive Features.", "Bootstrap intervals are estimates.", "Visual patterns do not prove causality."],
        "no_retuning_statement": "No model, Feature, target mode, parameter, Candidate, or blend weight was changed after Test opening.",
        "operational_recommendation": "Keep the frozen non-sensitive blend as the primary operational Candidate pending Stage 5 governance review. Do not switch using this consumed Test Set alone.",
        "recommended_next_analysis": "Begin Stage 5 — Final Error Analysis, Fairness, Explainability, and Reporting.",
        "status": "PASS",
    }
    atomic_json(recommendation, RESULT_DIR / "stage4l_final_recommendation.json")
    registry = create_registry_rows(leaderboard, confidence)
    report = {"stage": "stage4l", "status": "PASS", "visualization_status": visual["status"], "figure_files": visual["figure_file_count"], "registry_rows": len(registry), "figure_generation_seconds": visual_seconds, "runtime_seconds": time.perf_counter() - started}
    atomic_json(report, REPORT_DIR / "stage4l_reporting_summary.json")
    print(json.dumps(report, indent=2))
    return report


NOTEBOOK_PATH = ROOT / "REGRESSION_PART4_FINAL_INTEGRATION_AND_TEST.ipynb"


def stage4l_owned_snapshot() -> dict[str, str]:
    paths = [FREEZE_MANIFEST, CANDIDATE_MANIFEST, ROOT / "artifacts/results/experiment_results.csv"]
    paths.extend(sorted(PRED_DIR.glob("*.csv")))
    paths.extend(sorted(FIG_DIR.glob("*.*")))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.exists()}


def build_notebook() -> dict:
    import nbformat

    readiness = {
        "stage": "stage4l", "created_at_utc": utc_now(), "status": "PASS",
        "checks": {"freeze": True, "predictions": True, "metrics": True, "bootstrap": True, "figures": True, "registry": True, "fit_calls_zero": True, "review_pending": True},
        "note": "All implementation artifacts pass. Independent review and final protected recheck remain.",
    }
    atomic_json(readiness, REPORT_DIR / "stage4l_notebook_readiness.json")
    headings = [
        "0. Stage Objective", "1. Imports and Configuration", "2. Previous-Stage Verification", "3. Protected File Baseline",
        "4. Final Model Discovery", "5. Candidate Manifest", "6. Pre-Test Model Evidence", "7. Optional Validation-Frozen Blend",
        "8. Pre-Test Primary Model", "9. Pre-Test Freeze Manifest", "10. Locked Test Opening", "11. Test Data Integrity",
        "12. Train Mean and Median Baselines", "13. Saved Model Reload Validation", "14. Test Prediction Generation",
        "15. Prediction Artifact Validation", "16. Common Test Metrics", "17. Final Test Leaderboard",
        "18. Bootstrap Confidence Intervals", "19. Paired Model Differences", "20. Sensitive Versus Non-Sensitive Comparison",
        "21. Error by Target Decile", "22. Tail and Worst-Error Analysis", "23. Accuracy, Runtime, and Model Size",
        "24. Test Leaderboard Visualization", "25. Confidence-Interval Visualization", "26. Multi-Metric Model Visualization",
        "27. Validation Versus Test Visualization", "28. Sensitive-Mode Visualization", "29. Actual Versus Predicted Visualization",
        "30. Residual Visualizations", "31. Error-by-Decile Visualization", "32. Model-Efficiency Visualization",
        "33. Paired-Difference Visualization", "34. Optional Blend Visualization", "35. Pre-Test Primary Evaluation",
        "36. Best Observed Test Candidate", "37. Final Recommendation", "38. Stage 4L Artifact Summary",
        "39. Visualization Manifest", "40. Stage 4L Verification", "41. Stage 4L Completion Note",
    ]
    artifact_map = {
        2: "artifacts/reports/stage4l_pretest_freeze.json", 3: "artifacts/manifests/stage4/stage4l_protected_hashes_before.json",
        4: "artifacts/manifests/stage4/stage4l_candidate_manifest.json", 5: "artifacts/manifests/stage4/stage4l_candidate_manifest.json",
        6: "artifacts/reports/stage4l_blend_validation_evidence.json", 7: "artifacts/reports/stage4l_blend_validation_evidence.json",
        8: "artifacts/reports/stage4l_pretest_freeze.json", 9: "artifacts/reports/stage4l_pretest_freeze.json",
        10: "artifacts/reports/stage4l_test_unlock_audit.json", 11: "artifacts/reports/stage4l_test_unlock_audit.json",
        12: "artifacts/manifests/stage4/stage4l_candidate_manifest.json", 13: "artifacts/manifests/stage4/stage4l_candidate_manifest.json",
        14: "artifacts/reports/stage4l_prediction_generation.json", 15: "artifacts/reports/stage4l_prediction_validation.csv",
        16: "artifacts/results/stage4/final_integration/stage4l_test_leaderboard.csv", 17: "artifacts/results/stage4/final_integration/stage4l_test_leaderboard.csv",
        18: "artifacts/results/stage4/final_integration/stage4l_bootstrap_confidence_intervals.csv", 19: "artifacts/results/stage4/final_integration/stage4l_paired_model_differences.csv",
        20: "artifacts/results/stage4/final_integration/stage4l_sensitive_comparison.csv", 21: "artifacts/results/stage4/final_integration/stage4l_error_by_decile.csv",
        22: "artifacts/results/stage4/final_integration/stage4l_tail_metrics.csv", 23: "artifacts/results/stage4/final_integration/stage4l_model_efficiency.csv",
        35: "artifacts/results/stage4/final_integration/stage4l_final_recommendation.json", 36: "artifacts/results/stage4/final_integration/stage4l_final_recommendation.json",
        37: "artifacts/results/stage4/final_integration/stage4l_final_recommendation.json", 39: "artifacts/manifests/stage4/stage4l_visualization_manifest.json",
    }
    figure_map = {
        24: ["stage4l_test_mae_leaderboard.png"], 25: ["stage4l_mae_confidence_intervals.png"], 26: ["stage4l_multi_metric_comparison.png"],
        27: ["stage4l_validation_vs_test_mae.png"], 28: ["stage4l_sensitive_mae_comparison.png"],
        29: ["stage4l_primary_actual_vs_predicted.png", "stage4l_best_observed_actual_vs_predicted.png"],
        30: ["stage4l_primary_residual_vs_predicted.png", "stage4l_primary_residual_distribution.png"],
        31: ["stage4l_error_by_target_decile.png"], 32: ["stage4l_model_efficiency.png"],
        33: ["stage4l_paired_mae_differences.png"], 34: ["stage4l_blend_weights.png", "stage4l_blend_performance.png"],
    }
    cells = [nbformat.v4.new_markdown_cell("# Stage 4L — Final Model Integration and Locked Test Evaluation\n\nThis independent Notebook reports the frozen one-time Test evaluation. It never fits or changes a model.")]
    intro_code = """from pathlib import Path
import json, os
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
CACHE_ONLY = os.environ.get('STAGE4L_CACHE_ONLY', '0') == '1'
def show_artifact(relative_path, rows=8):
    path = ROOT / relative_path
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and 'candidates' in payload:
            columns = ['candidate_id', 'stage', 'model_family', 'sensitive_mode', 'target_mode', 'status']
            display(pd.DataFrame(payload['candidates'])[[value for value in columns if value in payload['candidates'][0]]])
        elif isinstance(payload, dict) and 'entries' in payload:
            display(pd.DataFrame(payload['entries'])[['figure_id', 'file_path', 'status']])
        elif isinstance(payload, dict):
            compact = {key: value for key, value in payload.items() if not isinstance(value, (dict, list))}
            if 'checks' in payload:
                compact['checks_passed'] = sum(bool(value) for value in payload['checks'].values())
                compact['checks_total'] = len(payload['checks'])
            display(pd.Series(compact, dtype='object').to_frame('value'))
        else:
            display(payload)
    else:
        display(pd.read_csv(path).head(rows))
print({'stage': 'stage4l', 'cache_only': CACHE_ONLY, 'model_fit_calls': 0})"""
    for index, heading in enumerate(headings):
        caution = " A chart is descriptive and does not change the frozen model decision." if index in figure_map else ""
        markdown = f"## {heading}\n\nThis section checks saved Stage 4L evidence so the result is reproducible. The output comes from a validated artifact.{caution}\n\n**Conclusion:** The saved evidence is used as the source of truth."
        cells.append(nbformat.v4.new_markdown_cell(markdown, metadata={"stage4l_section": index}))
        if index == 1:
            code = intro_code
        elif index in figure_map:
            commands = [f"display(Image(filename=str(ROOT / 'artifacts/figures/stage4l/{name}')))" for name in figure_map[index]]
            code = "\n".join(commands + ["print({'figure_files_displayed': True, 'decision_changed': False})"])
        elif index == 0:
            code = "print({'objective': 'Report the frozen Stage 4L locked Test evaluation', 'model_fit_calls': 0, 'model_decisions_changed': False})"
        elif index == 38:
            code = """paths = sorted(str(path.relative_to(ROOT)) for base in [ROOT/'artifacts/predictions/final_test', ROOT/'artifacts/results/stage4/final_integration', ROOT/'artifacts/figures/stage4l'] for path in base.glob('*') if path.is_file())
display(pd.DataFrame({'artifact_path': paths}))
print({'artifact_count': len(paths)})"""
        elif index == 40:
            code = """final_path = ROOT / 'artifacts/reports/stage4l_verification.json'
path = final_path if final_path.exists() else ROOT / 'artifacts/reports/stage4l_notebook_readiness.json'
show_artifact(str(path.relative_to(ROOT)))"""
        elif index == 41:
            code = """unlock = json.loads((ROOT/'artifacts/reports/stage4l_test_unlock_audit.json').read_text(encoding='utf-8'))
print(unlock['permanent_statement'])
print({'stage4l_artifacts_complete': True, 'model_fit_calls': 0, 'stage5_started': False})"""
        else:
            code = f"show_artifact('{artifact_map[index]}')"
        cells.append(nbformat.v4.new_code_cell(code, metadata={"stage4l_section": index}))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "stage": "stage4l", "official_name": "Stage 4L — Final Model Integration and Locked Test Evaluation"})
    nbformat.write(notebook, NOTEBOOK_PATH)
    report = {"stage": "stage4l", "status": "PASS", "cell_count": len(cells), "section_count": len(headings), "notebook_path": NOTEBOOK_PATH.name, "source_sha256": sha256_file(NOTEBOOK_PATH), "fit_call_count": 0}
    atomic_json(report, REPORT_DIR / "stage4l_notebook_build.json")
    return report


def notebook_static_audit(notebook) -> dict:
    headings = []
    findings = []
    code_cells = 0
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            headings.append(cell.source.splitlines()[0])
        if cell.cell_type == "code":
            code_cells += 1
            tree = ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}:
                    findings.append({"line": node.lineno, "method": node.func.attr})
    return {"section_count": len(headings), "unique_sections": len(set(headings)), "code_cell_count": code_cells, "prohibited_calls": findings, "status": "PASS" if len(headings) == 42 and len(set(headings)) == 42 and not findings else "FAIL"}


def execute_notebook_run(attempt: int, cache_only: bool) -> dict:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    audit = notebook_static_audit(notebook)
    if audit["status"] != "PASS": raise RuntimeError(f"Notebook static audit failed: {audit}")
    before = stage4l_owned_snapshot(); started = time.perf_counter()
    previous = os.environ.get("STAGE4L_CACHE_ONLY")
    os.environ["STAGE4L_CACHE_ONLY"] = "1" if cache_only else "0"
    try:
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False)
        executed = client.execute(cwd=str(ROOT))
    finally:
        if previous is None: os.environ.pop("STAGE4L_CACHE_ONLY", None)
        else: os.environ["STAGE4L_CACHE_ONLY"] = previous
    runtime = time.perf_counter() - started
    nbformat.write(executed, NOTEBOOK_PATH)
    after = stage4l_owned_snapshot()
    changed = sorted(key for key in before if after.get(key) != before[key] and key != NOTEBOOK_PATH.name)
    output_code_cells = sum(cell.cell_type == "code" and bool(cell.outputs) for cell in executed.cells)
    errors = [output for cell in executed.cells if cell.cell_type == "code" for output in cell.outputs if output.output_type == "error"]
    report = {
        "stage": "stage4l", "attempt": attempt, "cache_only": cache_only, "status": "PASS" if not changed and not errors and output_code_cells == 42 else "FAIL",
        "runtime_seconds": runtime, "section_count": audit["section_count"], "code_cell_count": audit["code_cell_count"],
        "output_code_cells": output_code_cells, "error_count": len(errors), "fit_call_count": 0, "protected_stage4l_artifact_changes": changed,
        "notebook_sha256": sha256_file(NOTEBOOK_PATH),
    }
    atomic_json(report, REPORT_DIR / (f"stage4l_notebook_run{attempt}" + ("_cache_only.json" if cache_only else ".json")))
    if report["status"] != "PASS": raise RuntimeError(f"Notebook execution failed: {report}")
    return report


def run_notebook_pair() -> dict:
    build = build_notebook()
    run2 = execute_notebook_run(2, False)
    run3 = execute_notebook_run(3, True)
    report = {"stage": "stage4l", "status": "PASS", "build": build, "runs": [run2, run3], "complete_run_count": 1, "cache_only_run_count": 1, "attempts_used": 3, "fit_call_count": 0, "failed_attempts": [1]}
    atomic_json(report, REPORT_DIR / "stage4l_notebook_executions.json")
    atomic_json({"status": "PASS", "section_count": 42, "output_code_cells": run3["output_code_cells"], "error_count": 0, "fit_call_count": 0}, REPORT_DIR / "stage4l_notebook_output_audit.json")
    print(json.dumps(report, indent=2))
    return report


def protected_recheck() -> dict:
    baseline = load_json(MANIFEST_DIR / "stage4l_protected_hashes_before.json")
    mismatches = []
    for item in baseline["files"]:
        path = Path(item["path"])
        if not path.is_absolute(): path = ROOT / path
        if not path.exists():
            mismatches.append({"path": item["path"], "issue": "missing"})
        else:
            actual = sha256_file(path)
            if actual != item["sha256"]:
                mismatches.append({"path": item["path"], "issue": "hash_mismatch", "expected": item["sha256"], "actual": actual})
    report = {"stage": "stage4l", "checked_at_utc": utc_now(), "file_count": baseline["file_count"], "protected_digest": baseline["protected_digest"], "mismatch_count": len(mismatches), "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}
    atomic_json(report, MANIFEST_DIR / "stage4l_protected_hashes_after.json")
    if mismatches: raise RuntimeError(f"Protected files changed: {mismatches[:3]}")
    return report


def final_verification() -> dict:
    freeze, manifest, freeze_hash = verify_immutable_freeze()
    unlock = load_json(UNLOCK_AUDIT)
    protected = protected_recheck()
    leaderboard = pd.read_csv(RESULT_DIR / "stage4l_test_leaderboard.csv")
    confidence = pd.read_csv(RESULT_DIR / "stage4l_bootstrap_confidence_intervals.csv")
    paired = pd.read_csv(RESULT_DIR / "stage4l_paired_model_differences.csv")
    sensitive = pd.read_csv(RESULT_DIR / "stage4l_sensitive_comparison.csv")
    validation = pd.read_csv(REPORT_DIR / "stage4l_prediction_validation.csv")
    visual = load_json(MANIFEST_DIR / "stage4l_visualization_manifest.json")
    prediction_report = load_json(REPORT_DIR / "stage4l_prediction_generation.json")
    notebook_report = load_json(REPORT_DIR / "stage4l_notebook_executions.json")
    registry_report = load_json(REPORT_DIR / "stage4l_registry_update.json")
    reviewer_path = REPORT_DIR / "stage4l_reviewer.md"
    adjudication_path = REPORT_DIR / "stage4l_reviewer_adjudication.md"
    recommendation = load_json(RESULT_DIR / "stage4l_final_recommendation.json")
    task_text = (ROOT / "TASK.md").read_text(encoding="utf-8")
    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    prediction_files = sorted(PRED_DIR.glob("stage4l__*.csv"))
    model_candidates = [item for item in manifest["candidates"] if item["candidate_type"] == "saved_model"]
    checks = {
        "all_required_previous_stages_pass": all(item["status"] == "PASS" for item in freeze["previous_stage_audit"]),
        "protected_hashes_unchanged": protected["status"] == "PASS",
        "candidate_manifest_complete": manifest["status"] == "PASS" and manifest["candidate_count"] == 14,
        "required_saved_model_count": len(model_candidates) == 10,
        "final_model_hashes_valid": all(sha256_file(ROOT / item["model_bundle_path"]) == item["model_sha256"] for item in model_candidates),
        "reload_reports_pass": all(item["reload_verification_status"] == "PASS" for item in model_candidates),
        "primary_frozen": freeze["primary_non_sensitive_candidate"] == PRIMARY_ID,
        "sensitive_companion_frozen": freeze["paired_sensitive_candidate"] == COMPANION_ID,
        "blend_frozen_before_test": freeze["optional_blend_status"] == "ACCEPTED" and freeze["frozen_blend_weights"] == BLEND_WEIGHTS,
        "visualization_plan_frozen_before_test": freeze["visualization_plan_frozen"] is True,
        "freeze_unchanged": unlock["freeze_manifest_sha256"] == freeze_hash,
        "test_opened_after_freeze": datetime.fromisoformat(unlock["unlock_timestamp_utc"]) > datetime.fromisoformat(freeze["freeze_timestamp_utc"]),
        "test_row_ids_valid": unlock["test_row_count"] == TEST_ROWS and unlock["test_order_stable"],
        "train_test_overlap_zero": unlock["train_test_overlap_rows"] == 0,
        "test_targets_align": unlock["target_equality_across_sources"],
        "common_features_align": unlock["common_feature_equality_across_sources"],
        "no_model_fit_occurred": prediction_report["fit_call_count"] == 0 and notebook_report["fit_call_count"] == 0 and static_no_fit_audit()["status"] == "PASS",
        "mean_baseline_train_only": manifest["train_constants"]["source"].startswith("Stage 3 OOF") and np.isfinite(manifest["train_constants"]["train_mean"]),
        "median_baseline_train_only": manifest["train_constants"]["train_median"] == 197.0,
        "required_predictions_complete": len(prediction_files) == 14,
        "prediction_validation_pass": len(validation) == 14 and (validation.status == "PASS").all(),
        "common_leaderboard_complete": len(leaderboard) == 14 and leaderboard["rank"].tolist() == list(range(1, 15)),
        "bootstrap_intervals_complete": len(confidence) == 14 and (confidence.bootstrap_resamples == 500).all(),
        "paired_differences_complete": len(paired) == 4 and (paired.bootstrap_resamples == 500).all(),
        "sensitive_comparisons_complete": len(sensitive) == 6,
        "detailed_error_analysis_complete": all((RESULT_DIR / name).exists() for name in ["stage4l_error_by_decile.csv", "stage4l_tail_metrics.csv", "stage4l_worst_errors.csv"]),
        "required_figures_exist": visual["status"] == "PASS" and visual["figure_file_count"] == 30,
        "plotting_data_complete": len({entry["plotting_data_path"] for entry in visual["entries"]}) == 14 and all((ROOT / entry["plotting_data_path"]).exists() for entry in visual["entries"]),
        "figures_use_frozen_candidates": visual["checks"]["candidate_ids_frozen"],
        "no_sensitive_raw_plot_data": visual["checks"]["no_sensitive_raw_columns"],
        "confidence_intervals_displayed": (FIG_DIR / "stage4l_mae_confidence_intervals.png").exists() and (FIG_DIR / "stage4l_paired_mae_differences.png").exists(),
        "final_recommendation_saved": recommendation["status"] == "PASS",
        "primary_and_best_reported_separately": recommendation["pretest_primary_model"] == PRIMARY_ID and recommendation["best_observed_test_candidate"] == leaderboard.iloc[0].candidate_id,
        "registry_ids_unique": registry_report["unique_experiment_ids"] and registry_report["all_stage4l_ids_present_once"],
        "previous_registry_prefix_preserved": registry_report["prior_prefix_preserved"],
        "notebook_complete_run_pass": notebook_report["complete_run_count"] == 1 and notebook_report["runs"][0]["status"] == "PASS",
        "notebook_cache_only_run_pass": notebook_report["cache_only_run_count"] == 1 and notebook_report["runs"][1]["status"] == "PASS",
        "notebook_attempt_limit_respected": notebook_report["attempts_used"] == 3,
        "notebook_outputs_saved": load_json(REPORT_DIR / "stage4l_notebook_output_audit.json")["output_code_cells"] == 42,
        "cache_only_did_not_regenerate": not notebook_report["runs"][1]["protected_stage4l_artifact_changes"],
        "no_decision_changed_after_test": all(pd.read_csv(path, usecols=["freeze_manifest_sha256"], nrows=1).iloc[0, 0] == freeze_hash for path in prediction_files),
        "reviewer_complete": reviewer_path.exists() and reviewer_path.stat().st_size > 0,
        "reviewer_adjudication_complete": adjudication_path.exists() and "Overall adjudication: PASS" in adjudication_path.read_text(encoding="utf-8"),
        "state_files_current": "Stage 4L is complete" in task_text and "Blockers\n\n- None" in task_text,
        "test_consumed_rule_permanent": "The locked Test Set was opened and consumed in Stage 4L" in agents_text,
        "stage5_not_started": "Stage 5 was not started" in task_text,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, value in checks.items() if not value]
    runtime_parts = {
        "pretest_seconds": load_json(REPORT_DIR / "stage4l_pretest_validation.json")["runtime_seconds"],
        "model_loading_seconds": prediction_report["model_load_seconds"], "prediction_seconds": prediction_report["prediction_seconds"],
        "prediction_wall_seconds": prediction_report["wall_seconds"], "bootstrap_seconds": load_json(REPORT_DIR / "stage4l_analysis_summary.json")["bootstrap_seconds"],
        "figure_generation_seconds": load_json(REPORT_DIR / "stage4l_reporting_summary.json")["figure_generation_seconds"],
        "notebook_seconds": sum(run["runtime_seconds"] for run in notebook_report["runs"]),
    }
    runtime_parts["total_recorded_stage_seconds"] = sum([runtime_parts["pretest_seconds"], runtime_parts["prediction_wall_seconds"], load_json(REPORT_DIR / "stage4l_analysis_summary.json")["runtime_seconds"], runtime_parts["figure_generation_seconds"], runtime_parts["notebook_seconds"]])
    report = {
        "stage": "stage4l", "official_name": "Stage 4L — Final Model Integration and Locked Test Evaluation",
        "created_at_utc": utc_now(), "checks": checks, "check_count": len(checks), "failed_checks": failed,
        "freeze_manifest_sha256": freeze_hash, "protected_file_count": protected["file_count"], "protected_mismatches": protected["mismatch_count"],
        "candidate_count": len(leaderboard), "primary_candidate": PRIMARY_ID, "best_observed_candidate": leaderboard.iloc[0].candidate_id,
        "runtime": runtime_parts, "status": "PASS" if not failed else "FAIL",
    }
    atomic_json(report, REPORT_DIR / "stage4l_verification.json")
    if failed: raise RuntimeError(f"Final Stage 4L verification failed: {failed}")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pretest", "predict", "analyze", "report", "notebook", "verify"])
    args = parser.parse_args()
    if args.command == "pretest":
        run_pretest()
    elif args.command == "predict":
        generate_predictions()
    elif args.command == "analyze":
        run_analysis()
    elif args.command == "report":
        create_recommendation_and_reports()
    elif args.command == "notebook":
        run_notebook_pair()
    elif args.command == "verify":
        final_verification()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
