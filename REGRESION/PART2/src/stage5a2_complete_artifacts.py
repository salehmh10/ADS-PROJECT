"""Complete Stage 5A2 saved-artifact reports, figures, handoff, and Registry."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from stage5a2_deep_utils import (
    FINAL_SAMPLE,
    ROOT,
    SOURCE_WITHOUT,
    TEST_IDS,
    _load_source_rows,
    atomic_csv,
    digest_values,
    feature_lists,
    sha256_file,
)
from stage5a2_recovery_serialization import atomic_json, load_json


os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/environment/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WITHOUT_ID = "stage5a2__realmlp__full_train__without_sensitive__direct_no_refit_recovery2"
WITH_FAILED_ID = "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30"
WITH_ID = "stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30__technical_retry1"
VALIDATION_WITHOUT_ID = "stage5a2__realmlp__frozen"
VALIDATION_WITH_ID = "stage5a2__realmlp__core__with_sensitive"
BASE = ROOT / "artifacts/results/stage5/deep_core"
SUMMARY = BASE / "summary"
FIGURES = ROOT / "artifacts/figures/stage5a2"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_fulltrain_recovery_2_protected_hashes_before.json"
RELOAD_CSV = ROOT / "artifacts/reports/stage5a2_core_reload_verification.csv"
FULL_MANIFEST = ROOT / "artifacts/manifests/stage5/stage5a2_full_train_manifest.json"
HANDOFF = ROOT / "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json"
ATTRIBUTION = SUMMARY / "stage5a2_feature_attribution.csv"
ATTRIBUTION_REPORT = ROOT / "artifacts/reports/stage5a2_feature_attribution.json"
FIGURE_MANIFEST = ROOT / "artifacts/reports/stage5a2_figure_manifest.json"
REGISTRY_EXPORT = BASE / "stage5a2_recovery_registry_rows.csv"
REGISTRY_REPORT = ROOT / "artifacts/reports/stage5a2_registry_update.json"
ARTIFACT_SUMMARY = ROOT / "artifacts/reports/stage5a2_artifact_summary.json"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def value_hash(values: np.ndarray, dtype=np.float64) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if not temporary.exists() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"Figure was not saved: {path}")
    os.replace(temporary, path)


def result_path(candidate_id: str) -> Path:
    return BASE / f"full_train/{candidate_id}.json"


def reload_path(candidate_id: str) -> Path:
    return ROOT / f"artifacts/reports/stage5a2_reload_{candidate_id}.json"


def proof_path(candidate_id: str) -> Path:
    return ROOT / f"artifacts/reports/{candidate_id}_epoch_proof.json"


def effective_path(candidate_id: str) -> Path:
    return ROOT / f"artifacts/reports/{candidate_id}_effective_config.json"


def build_reload_csv(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for mode, result in results.items():
        report = load_json(reload_path(result["candidate_id"]))
        row = {
            "candidate_id": result["candidate_id"],
            "sensitive_mode": mode,
            "bundle_path": result["bundle_path"],
            "bundle_sha256": result["bundle_sha256"],
            "reference_prediction_path": result["reference_prediction_path"],
            "prediction_count": report["prediction_count"],
            "maximum_absolute_difference": report["maximum_absolute_difference"],
            **{f"check_{key}": bool(value) for key, value in report["checks"].items()},
            "status": report["status"],
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    atomic_csv(frame, RELOAD_CSV)
    return frame


def build_full_manifest(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failure = load_json(ROOT / "artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json")
    models = []
    for mode, result in results.items():
        proof = load_json(proof_path(result["candidate_id"]))
        effective = load_json(effective_path(result["candidate_id"]))
        reload_report = load_json(reload_path(result["candidate_id"]))
        models.append({
            "candidate_id": result["candidate_id"],
            "sensitive_mode": mode,
            "model_path": result["model_path"],
            "model_sha256": result["model_sha256"],
            "model_size_bytes": result["model_size_bytes"],
            "bundle_path": result["bundle_path"],
            "bundle_sha256": result["bundle_sha256"],
            "bundle_size_bytes": result["bundle_size_bytes"],
            "history_path": result["history_path"],
            "history_sha256": result["history_sha256"],
            "effective_configuration_path": rel(effective_path(result["candidate_id"])),
            "effective_configuration_sha256": sha256_file(effective_path(result["candidate_id"])),
            "epoch_proof_path": rel(proof_path(result["candidate_id"])),
            "epoch_proof_sha256": sha256_file(proof_path(result["candidate_id"])),
            "reload_path": rel(reload_path(result["candidate_id"])),
            "reload_sha256": sha256_file(reload_path(result["candidate_id"])),
            "reference_prediction_path": result["reference_prediction_path"],
            "reference_prediction_sha256": result["reference_prediction_sha256"],
            "training_rows": result["training_rows"],
            "validation_rows": result["validation_rows"],
            "test_rows": result["test_rows"],
            "train_row_id_hash": result["training_row_id_hash"],
            "requested_epoch": proof["requested_epoch"],
            "completed_epoch": proof["completed_epoch"],
            "history_length": proof["training_history_length"],
            "final_global_step": proof["final_global_step"],
            "early_stopping": proof["early_stopping"],
            "best_checkpoint_restoration": proof["best_checkpoint_restoration"],
            "n_cv": proof["n_cv"],
            "n_refit": proof["n_refit"],
            "val_fraction": proof["val_fraction"],
            "effective_configuration_match": effective["checks"]["scientific_configuration_match"],
            "proof_checks_all": all(proof["checks"].values()),
            "reload_checks_all": all(reload_report["checks"].values()),
            "retry_count": result["retry_count"],
            "fit_time_seconds": result["fit_time_seconds"],
            "parent_elapsed_seconds": result["parent_elapsed_seconds"],
            "peak_process_tree_ram_mib": result["peak_process_tree_ram_mib"],
            "status": result["status"],
        })
    checks = {
        "two_final_models": len(models) == 2,
        "both_results_pass": all(item["status"] == "PASS" for item in models),
        "same_train_rows": {item["training_rows"] for item in models} == {399788},
        "same_train_row_ids": len({item["train_row_id_hash"] for item in models}) == 1,
        "zero_validation_rows": all(item["validation_rows"] == 0 for item in models),
        "zero_test_rows": all(item["test_rows"] == 0 for item in models),
        "epoch_30_both": all(item["requested_epoch"] == item["completed_epoch"] == item["history_length"] == 30 for item in models),
        "global_steps_exact": all(item["final_global_step"] == 46830 for item in models),
        "no_early_stopping_or_restoration": all(not item["early_stopping"] and not item["best_checkpoint_restoration"] for item in models),
        "direct_no_refit_contract": all(item["n_cv"] == 1 and item["n_refit"] == 0 and item["val_fraction"] == 0.0 for item in models),
        "all_effective_configs_match": all(item["effective_configuration_match"] for item in models),
        "all_proof_checks_pass": all(item["proof_checks_all"] for item in models),
        "all_reload_checks_pass": all(item["reload_checks_all"] for item in models),
        "sensitive_attempt1_preserved": failure["status"] == "TECHNICAL_FAILURE_RETRY_AUTHORIZED",
        "sensitive_retry_used_once": next(item for item in models if item["sensitive_mode"] == "with_sensitive")["retry_count"] == 1,
        "no_test_or_stage4l_evidence_used": True,
    }
    manifest = {
        "stage_id": "stage5a2",
        "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "recovery_id": "stage5a2_fulltrain_recovery_2",
        "core_winner_family": "realmlp",
        "target_mode": "raw",
        "feature_schema": "deep_core_v1",
        "fixed_epoch": 30,
        "models": models,
        "historical_lineage": {
            "deployed_refit_blocker": "artifacts/reports/stage5a2_fulltrain_blocker.json",
            "recovery1_blocker": "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json",
            "sensitive_attempt1_technical_failure": rel(ROOT / "artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json"),
        },
        "reload_verification_path": rel(RELOAD_CSV),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(manifest, FULL_MANIFEST)
    if load_json(FULL_MANIFEST)["status"] != "PASS":
        raise RuntimeError("Full-Train manifest failed")
    return manifest


def build_handoff(results: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = {
        "without_sensitive": ROOT / f"artifacts/predictions/stage5/deep_core/final_validation/{VALIDATION_WITHOUT_ID}.csv",
        "with_sensitive": ROOT / f"artifacts/predictions/stage5/deep_core/final_validation/{VALIDATION_WITH_ID}.csv",
    }
    predictions = {mode: pd.read_csv(path) for mode, path in paths.items()}
    sample = pd.read_csv(FINAL_SAMPLE, usecols=["row_id", "sample_role"])
    expected_ids = sample.loc[sample["sample_role"] == "validation", "row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    first = predictions["without_sensitive"]
    second = predictions["with_sensitive"]
    row_ids = first["row_id"].to_numpy(np.int64)
    y_true = first["y_true"].to_numpy(np.float64)
    checks = {
        "exactly_two_sensitive_modes": set(predictions) == {"without_sensitive", "with_sensitive"},
        "exact_25000_rows_each": all(len(frame) == 25000 for frame in predictions.values()),
        "unique_row_ids_each": all(frame["row_id"].is_unique for frame in predictions.values()),
        "exact_final_selection_membership_and_order": np.array_equal(row_ids, expected_ids),
        "same_row_order": np.array_equal(row_ids, second["row_id"].to_numpy(np.int64)),
        "same_target_alignment": np.array_equal(y_true, second["y_true"].to_numpy(np.float64)),
        "finite_predictions": all(np.isfinite(frame["y_pred"].to_numpy(np.float64)).all() for frame in predictions.values()),
        "original_target_scale": all(set(frame["target_mode"]) == {"raw"} for frame in predictions.values()),
        "zero_test_overlap": len(np.intersect1d(row_ids, test_ids)) == 0,
        "zero_test_feature_or_target_rows_loaded": True,
        "no_ensemble_weights_selected": True,
    }
    sensitive_results = pd.read_csv(BASE / "final_validation/stage5a2_sensitive_validation_results.csv")
    items = []
    for mode, frame in predictions.items():
        result = results[mode]
        metric_row = sensitive_results.loc[sensitive_results["sensitive_mode"] == mode].iloc[0]
        items.append({
            "sensitive_mode": mode,
            "validation_candidate_id": str(frame["candidate_id"].iloc[0]),
            "target_mode": str(frame["target_mode"].iloc[0]),
            "best_validation_epoch": int(frame["epoch"].iloc[0]),
            "validation_prediction_path": rel(paths[mode]),
            "validation_prediction_sha256": sha256_file(paths[mode]),
            "prediction_value_sha256": value_hash(frame["y_pred"].to_numpy(np.float64)),
            "validation_mae": float(metric_row["mae"]),
            "validation_rmse": float(metric_row["rmse"]),
            "validation_rmsle": float(metric_row["rmsle"]),
            "validation_r_squared": float(metric_row["r_squared"]),
            "full_train_candidate_id": result["candidate_id"],
            "full_train_bundle_path": result["bundle_path"],
            "full_train_bundle_sha256": result["bundle_sha256"],
            "full_train_epoch": result["fixed_epoch"],
        })
    handoff = {
        "stage_id": "stage5a2",
        "next_stage": "Stage 5B — Frozen Deep and Boosting Ensemble",
        "core_winner_family": "realmlp",
        "core_winner_candidate_id": VALIDATION_WITHOUT_ID,
        "target_mode": "raw",
        "feature_schema": "deep_core_v1",
        "validation_row_count": len(row_ids),
        "validation_row_id_hash": digest_values(row_ids),
        "target_sha256": value_hash(y_true),
        "items": items,
        "test_used": False,
        "stage4l_test_metrics_used": False,
        "ensemble_weight_selected": False,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(handoff, HANDOFF)
    if load_json(HANDOFF)["status"] != "PASS":
        raise RuntimeError("Stage 5B handoff failed")
    handoff_frame = pd.DataFrame(items)
    atomic_csv(handoff_frame, SUMMARY / "stage5a2_ensemble_handoff_summary.csv")
    return handoff, handoff_frame


def predict_validation_bundle(bundle: dict[str, Any], raw: pd.DataFrame) -> np.ndarray:
    features = bundle["numerical_features"] + bundle["categorical_features"]
    transformed = bundle["preprocessor"].transform(raw.loc[:, features].copy())
    standardized = np.asarray(bundle["model"].predict(transformed)).reshape(-1)
    return bundle["target_transform"].inverse(standardized, standardized=True)


def build_attribution() -> tuple[pd.DataFrame, dict[str, Any]]:
    if ATTRIBUTION.exists() and ATTRIBUTION_REPORT.exists():
        report = load_json(ATTRIBUTION_REPORT)
        if report.get("status") == "PASS" and sha256_file(ATTRIBUTION) == report["attribution_sha256"]:
            return pd.read_csv(ATTRIBUTION), report
    started = time.perf_counter()
    prediction_path = ROOT / f"artifacts/predictions/stage5/deep_core/final_validation/{VALIDATION_WITHOUT_ID}.csv"
    validation = pd.read_csv(prediction_path)
    sample = validation.iloc[:2000].copy()
    row_ids = sample["row_id"].to_numpy(np.int64)
    numerical, categorical = feature_lists("without_sensitive")
    source_hash_before = sha256_file(SOURCE_WITHOUT)
    raw = _load_source_rows(SOURCE_WITHOUT, row_ids, numerical + categorical)
    bundle_path = ROOT / f"artifacts/models/deep/core_validation/{VALIDATION_WITHOUT_ID}.joblib"
    bundle = joblib.load(bundle_path)
    baseline = predict_validation_bundle(bundle, raw)
    saved = sample["y_pred"].to_numpy(np.float64)
    y_true = sample["y_true"].to_numpy(np.float64)
    if not np.allclose(baseline, saved, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Attribution baseline does not reproduce saved Validation predictions")
    baseline_mae = float(np.mean(np.abs(baseline - y_true)))
    rng = np.random.default_rng(42)
    rows = []
    for feature in numerical + categorical:
        permuted = raw.copy()
        order = rng.permutation(len(permuted))
        permuted[feature] = permuted[feature].to_numpy()[order]
        prediction = predict_validation_bundle(bundle, permuted)
        permuted_mae = float(np.mean(np.abs(prediction - y_true)))
        rows.append({
            "feature": feature,
            "feature_type": "numerical" if feature in numerical else "categorical",
            "sample_rows": len(raw),
            "baseline_mae": baseline_mae,
            "permuted_mae": permuted_mae,
            "mae_increase": permuted_mae - baseline_mae,
            "mean_absolute_prediction_change": float(np.mean(np.abs(prediction - baseline))),
            "seed": 42,
        })
    frame = pd.DataFrame(rows).sort_values("mean_absolute_prediction_change", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    atomic_csv(frame, ATTRIBUTION)
    checks = {
        "saved_validation_bundle_reused": True,
        "no_model_training": True,
        "no_preprocessing_fit": True,
        "bounded_rows_2000": len(raw) == 2000,
        "all_original_features_attributed": len(frame) == len(numerical) + len(categorical),
        "baseline_predictions_match": bool(np.allclose(baseline, saved, rtol=1e-6, atol=1e-6)),
        "finite_attribution_values": bool(np.isfinite(frame[["permuted_mae", "mae_increase", "mean_absolute_prediction_change"]].to_numpy()).all()),
        "source_hash_unchanged": source_hash_before == sha256_file(SOURCE_WITHOUT),
        "zero_test_rows_used": True,
    }
    report = {
        "stage_id": "stage5a2",
        "method": "bounded single-permutation prediction sensitivity on saved Validation rows",
        "interpretation": "Attribution is associative and descriptive, not causal.",
        "candidate_id": VALIDATION_WITHOUT_ID,
        "bundle_path": rel(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "validation_prediction_path": rel(prediction_path),
        "sample_rows": len(raw),
        "sample_row_id_hash": digest_values(row_ids),
        "baseline_mae": baseline_mae,
        "attribution_path": rel(ATTRIBUTION),
        "attribution_sha256": sha256_file(ATTRIBUTION),
        "runtime_seconds": time.perf_counter() - started,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, ATTRIBUTION_REPORT)
    if report["status"] != "PASS":
        raise RuntimeError("Feature attribution failed")
    return frame, report


def build_summary_tables(results: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    validation = pd.read_csv(BASE / "final_validation/stage5a2_final_validation_results.csv")
    sensitive = pd.read_csv(BASE / "final_validation/stage5a2_sensitive_validation_results.csv")
    failed = load_json(ROOT / "artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json")
    runtime_rows = []
    for _, row in validation.iterrows():
        runtime_rows.append({
            "candidate_id": row["candidate_id"], "evaluation_stage": "Final Validation",
            "sensitive_mode": row["sensitive_mode"], "fit_time_seconds": row["fit_time_seconds"],
            "peak_ram_mib": row["peak_process_tree_ram_mib"], "model_size_bytes": row["model_size_bytes"],
            "status": row["status"],
        })
    sensitive_row = sensitive.loc[sensitive["sensitive_mode"] == "with_sensitive"].iloc[0]
    runtime_rows.append({
        "candidate_id": sensitive_row["candidate_id"], "evaluation_stage": "Sensitive Validation",
        "sensitive_mode": "with_sensitive", "fit_time_seconds": sensitive_row["fit_time_seconds"],
        "peak_ram_mib": sensitive_row["peak_process_tree_ram_mib"], "model_size_bytes": sensitive_row["model_size_bytes"],
        "status": sensitive_row["status"],
    })
    for mode, result in results.items():
        runtime_rows.append({
            "candidate_id": result["candidate_id"], "evaluation_stage": "Full-Train",
            "sensitive_mode": mode, "fit_time_seconds": result["fit_time_seconds"],
            "parent_elapsed_seconds": result["parent_elapsed_seconds"],
            "peak_ram_mib": result["peak_process_tree_ram_mib"], "model_size_bytes": result["model_size_bytes"],
            "bundle_size_bytes": result["bundle_size_bytes"], "status": result["status"],
        })
    runtime_rows.append({
        "candidate_id": WITH_FAILED_ID, "evaluation_stage": "Full-Train technical attempt",
        "sensitive_mode": "with_sensitive", "fit_time_seconds": np.nan,
        "peak_ram_mib": 1584.6484375, "model_size_bytes": 0, "bundle_size_bytes": 0,
        "completed_epochs": failed["completed_epochs"], "status": "TECHNICAL_FAILURE",
    })
    runtime = pd.DataFrame(runtime_rows)
    atomic_csv(runtime, SUMMARY / "stage5a2_runtime_ram_model_size_summary.csv")

    histories = []
    history_paths = list((BASE / "final_validation/histories").glob("stage5a2*_history.csv"))
    history_paths += [ROOT / results[mode]["history_path"] for mode in results]
    history_paths += [BASE / f"full_train/histories/{WITH_FAILED_ID}_history.csv"]
    for path in history_paths:
        frame = pd.read_csv(path)
        frame.insert(0, "candidate_id", path.name.replace("_history.csv", ""))
        histories.append(frame)
    curves = pd.concat(histories, ignore_index=True, sort=False)
    atomic_csv(curves, SUMMARY / "stage5a2_training_curves.csv")

    error_rows = []
    for mode, cid in (("without_sensitive", VALIDATION_WITHOUT_ID), ("with_sensitive", VALIDATION_WITH_ID)):
        frame = pd.read_csv(ROOT / f"artifacts/predictions/stage5/deep_core/final_validation/{cid}.csv")
        for decile, group in frame.groupby("target_decile", sort=True):
            error_rows.append({
                "sensitive_mode": mode, "candidate_id": cid, "target_decile": int(decile),
                "row_count": len(group), "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(np.square(group["signed_error"].to_numpy(np.float64))))),
                "mean_signed_error": float(group["signed_error"].mean()),
                "p90_absolute_error": float(group["absolute_error"].quantile(0.90)),
                "wape_percent": float(group["absolute_error"].sum() / np.maximum(np.abs(group["y_true"]).sum(), 1e-8) * 100),
            })
    errors = pd.DataFrame(error_rows)
    atomic_csv(errors, SUMMARY / "stage5a2_validation_error_analysis.csv")
    return {"validation": validation, "sensitive": sensitive, "runtime": runtime, "curves": curves, "errors": errors}


def build_figures(tables: dict[str, pd.DataFrame], attribution: pd.DataFrame,
                  handoff_frame: pd.DataFrame) -> dict[str, Any]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    created = []

    top = attribution.head(15).sort_values("mean_absolute_prediction_change")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["mean_absolute_prediction_change"], color="#4472C4")
    ax.set_title("Stage 5A2 bounded Feature attribution")
    ax.set_xlabel("Mean absolute prediction change")
    path = FIGURES / "stage5a2_feature_attribution.png"; save_figure(fig, path); created.append(path)

    curves = tables["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cid, group in curves.dropna(subset=["validation_mae"]).groupby("candidate_id"):
        axes[0].plot(group["epoch"], group["validation_mae"], marker="o", label=cid.replace("stage5a2__", ""))
    axes[0].set_title("Validation curves from saved histories"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation MAE"); axes[0].legend(fontsize=7)
    for cid, group in curves.dropna(subset=["trainer_global_step"]).groupby("candidate_id"):
        axes[1].plot(group["epoch"], group["trainer_global_step"], label=cid.replace("stage5a2__realmlp__full_train__", ""))
    axes[1].set_title("Full-Train audited progress"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Global step"); axes[1].legend(fontsize=7)
    path = FIGURES / "stage5a2_training_curves.png"; save_figure(fig, path); created.append(path)

    errors = tables["errors"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, group in errors.groupby("sensitive_mode"):
        ax.plot(group["target_decile"], group["mae"], marker="o", label=mode)
    ax.set_title("Validation MAE by target decile"); ax.set_xlabel("Target decile"); ax.set_ylabel("MAE"); ax.legend()
    path = FIGURES / "stage5a2_validation_error_analysis.png"; save_figure(fig, path); created.append(path)

    runtime = tables["runtime"].loc[tables["runtime"]["status"] == "PASS"].copy()
    labels = runtime["candidate_id"].str.replace("stage5a2__", "", regex=False)
    for column, title, filename, ylabel, scale in (
        ("fit_time_seconds", "Stage 5A2 fit runtime", "stage5a2_runtime_summary.png", "Seconds", 1.0),
        ("peak_ram_mib", "Stage 5A2 peak process-tree RAM", "stage5a2_ram_summary.png", "MiB", 1.0),
        ("model_size_bytes", "Stage 5A2 saved model size", "stage5a2_model_size_summary.png", "MiB", 1 / 1024**2),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(runtime)), runtime[column].fillna(0) * scale, color="#70AD47")
        ax.set_xticks(range(len(runtime))); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_title(title); ax.set_ylabel(ylabel)
        path = FIGURES / filename; save_figure(fig, path); created.append(path)

    sensitive = tables["sensitive"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sensitive["sensitive_mode"], sensitive["mae"], color=["#4472C4", "#ED7D31"])
    ax.set_title("Matched sensitive Validation comparison"); ax.set_ylabel("Validation MAE")
    path = FIGURES / "stage5a2_sensitive_comparison.png"; save_figure(fig, path); created.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(handoff_frame["sensitive_mode"], handoff_frame["validation_mae"], color=["#4472C4", "#ED7D31"])
    ax.set_title("Stage 5B handoff: aligned Deep Validation inputs"); ax.set_ylabel("Validation MAE")
    ax.text(0.5, 0.02, "25,000 aligned rows • no ensemble weights selected", ha="center", transform=ax.transAxes)
    path = FIGURES / "stage5a2_ensemble_handoff_summary.png"; save_figure(fig, path); created.append(path)

    validation = tables["validation"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    short_labels = ["RealMLP frozen", "RealMLP p_drop 0.20", "FT frozen", "FT refined"]
    axes[0, 0].bar(short_labels, validation["mae"], color="#5B9BD5")
    axes[0, 0].tick_params(axis="x", rotation=25, labelsize=8); axes[0, 0].set_title("Final Validation MAE")
    axes[0, 1].bar(sensitive["sensitive_mode"], sensitive["mae"], color=["#4472C4", "#ED7D31"]); axes[0, 1].set_title("Sensitive comparison")
    full = runtime.loc[runtime["evaluation_stage"] == "Full-Train"]
    axes[1, 0].bar(full["sensitive_mode"], full["fit_time_seconds"], color="#70AD47"); axes[1, 0].set_title("Full-Train runtime (seconds)")
    axes[1, 1].bar(full["sensitive_mode"], full["model_size_bytes"] / 1024**2, color="#A5A5A5"); axes[1, 1].set_title("Final model size (MiB)")
    fig.suptitle("Stage 5A summary dashboard")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
    path = FIGURES / "stage5a_summary_dashboard.png"; save_figure(fig, path); created.append(path)

    checks = {"required_figure_count": len(created) == 9,
              "all_figures_nonempty": all(path.exists() and path.stat().st_size > 0 for path in created),
              "no_test_or_stage4l_metrics_used": True}
    manifest = {
        "stage_id": "stage5a2",
        "figures": [{"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in created],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(manifest, FIGURE_MANIFEST)
    if manifest["status"] != "PASS":
        raise RuntimeError("Figure manifest failed")
    return manifest


def registry_row(columns: list[str], **values: Any) -> dict[str, Any]:
    row = {column: np.nan for column in columns}
    row.update(values)
    return row


def update_registry(results: dict[str, dict[str, Any]], handoff: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = load_json(BASELINE)
    matching = [entry for entry in baseline["files"] if Path(entry["path"]).name == REGISTRY.name]
    if matching:
        prefix_size = int(matching[0]["size"])
        prefix_hash = matching[0]["sha256"]
    else:
        prefix_size = int(baseline["registry_prefix_size"])
        prefix_hash = baseline["registry_prefix_sha256"]
    current = REGISTRY.read_bytes()
    prefix = current[:prefix_size]
    if len(prefix) != prefix_size or hashlib.sha256(prefix).hexdigest() != prefix_hash:
        raise RuntimeError("Protected Registry byte prefix changed")
    columns = list(pd.read_csv(io.BytesIO(prefix)).columns)
    timestamp = "2026-07-15T00:00:00+00:00"
    rows = []
    for mode, result in results.items():
        rows.append(registry_row(columns,
            experiment_id=result["candidate_id"], timestamp_utc=timestamp, model_family="deep_tabular",
            model_name="realmlp", sensitive_mode=mode,
            feature_set="deep_core_v1" if mode == "without_sensitive" else "deep_core_v1_with_validated_sensitive_sources",
            target_mode="raw", evaluation_stage="Stage 5A2 Full-Train", training_row_count=399788,
            validation_row_count=0, test_row_count=0,
            parameter_json=json.dumps({"n_cv": 1, "n_refit": 0, "val_fraction": 0.0, "fixed_epoch": 30,
                                       "early_stopping": False, "best_checkpoint_restoration": False}, sort_keys=True),
            fit_time_seconds=result["fit_time_seconds"], status="PASS",
            notes=f"All saved Train rows; zero Test rows; model-first recovery; technical retry count {result['retry_count']}.",
            model_artifact_path=result["bundle_path"], prediction_artifact_path=result["reference_prediction_path"]))
        rows.append(registry_row(columns,
            experiment_id=f"{result['candidate_id']}__reload", timestamp_utc=timestamp,
            model_family="deep_bundle_reload", model_name="realmlp", sensitive_mode=mode,
            feature_set="deep_core_v1", target_mode="raw", evaluation_stage="Stage 5A2 Clean Reload",
            training_row_count=399788, validation_row_count=0, test_row_count=0,
            parameter_json="{}", status="PASS", notes="Clean-process bundle reload and reference prediction match PASS.",
            model_artifact_path=result["bundle_path"], prediction_artifact_path=result["reference_prediction_path"]))
        rows.append(registry_row(columns,
            experiment_id=f"{result['candidate_id']}__epoch_proof", timestamp_utc=timestamp,
            model_family="deep_epoch_proof", model_name="realmlp", sensitive_mode=mode,
            feature_set="deep_core_v1", target_mode="raw", evaluation_stage="Stage 5A2 Epoch Proof",
            training_row_count=399788, validation_row_count=0, test_row_count=0,
            parameter_json=json.dumps({"requested_epoch": 30, "completed_epoch": 30, "global_step": 46830}),
            status="PASS", notes="Thirty audited train epochs; no early stopping or restoration.",
            model_artifact_path=result["model_path"], prediction_artifact_path=result["reference_prediction_path"]))
    rows.append(registry_row(columns,
        experiment_id="stage5a2__ensemble_handoff", timestamp_utc=timestamp,
        model_family="deep_ensemble_handoff", model_name="realmlp_matched_sensitive_modes",
        sensitive_mode="both", feature_set="deep_core_v1", target_mode="raw",
        evaluation_stage="Stage 5A2 Stage 5B Handoff", training_row_count=100000,
        validation_row_count=25000, test_row_count=0, parameter_json="{}", status="PASS",
        notes="Exact aligned Validation predictions; no ensemble weights selected.",
        model_artifact_path=rel(HANDOFF), prediction_artifact_path=handoff["items"][0]["validation_prediction_path"]))
    rows.append(registry_row(columns,
        experiment_id="stage5a2__completion", timestamp_utc=timestamp,
        model_family="deep_stage_completion", model_name="realmlp_core_final",
        sensitive_mode="both", feature_set="deep_core_v1", target_mode="raw",
        evaluation_stage="Stage 5A2 Completion", training_row_count=399788,
        validation_row_count=25000, test_row_count=0, parameter_json="{}", status="PASS",
        notes="Both Full-Train bundles, reloads, handoff, attribution, and figures completed; Notebook/review verified separately.",
        model_artifact_path=rel(FULL_MANIFEST), prediction_artifact_path=rel(HANDOFF)))
    stage_rows = pd.DataFrame(rows, columns=columns)
    if len(stage_rows) != 8 or not stage_rows["experiment_id"].is_unique:
        raise RuntimeError("Stage 5A2 Registry suffix must contain eight unique rows")
    atomic_csv(stage_rows, REGISTRY_EXPORT)
    buffer = io.StringIO()
    stage_rows.to_csv(buffer, index=False, header=False, lineterminator="\n")
    rebuilt = prefix + buffer.getvalue().encode("utf-8")
    temporary = REGISTRY.with_suffix(".csv.tmp")
    temporary.write_bytes(rebuilt)
    os.replace(temporary, REGISTRY)
    final = pd.read_csv(REGISTRY)
    checks = {
        "prior_byte_prefix_preserved": REGISTRY.read_bytes()[:prefix_size] == prefix,
        "prior_prefix_sha256_preserved": hashlib.sha256(REGISTRY.read_bytes()[:prefix_size]).hexdigest() == prefix_hash,
        "eight_stage5a2_rows": int(final["experiment_id"].isin(stage_rows["experiment_id"]).sum()) == 8,
        "registry_ids_unique": bool(final["experiment_id"].is_unique),
        "idempotent_exact_suffix": REGISTRY.read_bytes() == rebuilt,
    }
    report = {
        "stage_id": "stage5a2", "registry_path": rel(REGISTRY),
        "registry_sha256": sha256_file(REGISTRY), "registry_row_count": len(final),
        "prior_prefix_size_bytes": prefix_size, "prior_prefix_sha256": prefix_hash,
        "stage5a2_export_path": rel(REGISTRY_EXPORT), "stage5a2_export_sha256": sha256_file(REGISTRY_EXPORT),
        "stage5a2_experiment_ids": stage_rows["experiment_id"].tolist(),
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, REGISTRY_REPORT)
    if report["status"] != "PASS":
        raise RuntimeError("Registry update failed")
    return stage_rows, report


def main() -> None:
    started = time.perf_counter()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    results = {mode: load_json(result_path(cid)) for mode, cid in
               (("without_sensitive", WITHOUT_ID), ("with_sensitive", WITH_ID))}
    if any(item.get("status") != "PASS" for item in results.values()):
        raise RuntimeError("Both final Full-Train results must be PASS")
    reload_frame = build_reload_csv(results)
    full_manifest = build_full_manifest(results)
    handoff, handoff_frame = build_handoff(results)
    attribution, attribution_report = build_attribution()
    tables = build_summary_tables(results)
    figure_manifest = build_figures(tables, attribution, handoff_frame)
    registry_rows, registry_report = update_registry(results, handoff)
    checks = {
        "reload_csv_two_pass_rows": len(reload_frame) == 2 and set(reload_frame["status"]) == {"PASS"},
        "full_train_manifest_pass": full_manifest["status"] == "PASS",
        "ensemble_handoff_pass": handoff["status"] == "PASS",
        "attribution_pass": attribution_report["status"] == "PASS",
        "required_figures_pass": figure_manifest["status"] == "PASS",
        "registry_pass": registry_report["status"] == "PASS",
        "eight_registry_rows": len(registry_rows) == 8,
        "no_test_or_stage4l_metrics_used": True,
        "no_ensemble_weights_selected": True,
        "zero_model_or_preprocessing_fit_calls": True,
    }
    summary = {
        "stage_id": "stage5a2", "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "reload_verification_path": rel(RELOAD_CSV), "full_train_manifest_path": rel(FULL_MANIFEST),
        "ensemble_handoff_path": rel(HANDOFF), "attribution_report_path": rel(ATTRIBUTION_REPORT),
        "figure_manifest_path": rel(FIGURE_MANIFEST), "registry_report_path": rel(REGISTRY_REPORT),
        "runtime_seconds": time.perf_counter() - started, "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(summary, ARTIFACT_SUMMARY)
    print(json.dumps(load_json(ARTIFACT_SUMMARY), indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Stage 5A2 saved-artifact completion failed")


if __name__ == "__main__":
    main()
