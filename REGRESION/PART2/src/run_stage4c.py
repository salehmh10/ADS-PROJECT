"""Run recovery-safe Stage 4C CatBoost phases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4


ROOT = Path(__file__).resolve().parent


def _worker_config_path(candidate_id: str) -> Path:
    return c4.paths(ROOT)["checkpoints"] / f"{candidate_id}__config.json"


def _checkpoint_path(candidate_id: str) -> Path:
    return c4.paths(ROOT)["checkpoints"] / f"{candidate_id}.json"


def _valid_checkpoint(config: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = ROOT / config["checkpoint_path"]
    model = ROOT / config["model_path"]
    predictions = ROOT / config["prediction_path"]
    if not all(path.is_file() for path in (checkpoint, model, predictions)):
        return None
    try:
        metadata = c4.read_json(checkpoint)
        if metadata.get("status") != "PASS":
            return None
        if metadata.get("configuration_digest") != s4.configuration_digest(config, length=64):
            return None
        prediction_frame = pd.read_csv(predictions)
        expected_rows = 10_000 if config["kind"] == "screening" else 15_000
        if len(prediction_frame) != expected_rows or prediction_frame["row_id"].duplicated().any():
            return None
        if not np.isfinite(prediction_frame["y_pred"].to_numpy(dtype=float)).all():
            return None
        bundle = joblib.load(model)
        if bundle.get("experiment_id") != config["experiment_id"]:
            return None
        return metadata
    except Exception:
        return None


def _run_fit(config: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    cached = _valid_checkpoint(config)
    if cached is not None:
        return cached
    config_path = _worker_config_path(config["candidate_id"])
    s4.atomic_write_json(config_path, config)
    reports = c4.paths(ROOT)["reports"] / "stage4c_workers"
    reports.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        run = s4.run_worker_process(
            [sys.executable, "-B", str(ROOT / "stage4_catboost_worker.py"), "--root", str(ROOT), "--config", str(config_path)],
            timeout_seconds=timeout_seconds,
            cwd=ROOT,
        )
        run["attempt"] = attempt
        attempts.append(run)
        (reports / f"{config['candidate_id']}__attempt-{attempt}.log").write_text(
            f"STDOUT\n{run.get('stdout', '')}\nSTDERR\n{run.get('stderr', '')}\n",
            encoding="utf-8",
        )
        cached = _valid_checkpoint(config)
        if run["status"] == "success" and cached is not None:
            s4.atomic_write_json(reports / f"{config['candidate_id']}__parent.json", {"attempts": attempts, "status": "PASS"})
            return cached
        if attempt == 1:
            # The retry keeps the scientific configuration unchanged. It only starts a fresh process.
            continue
    s4.atomic_write_json(reports / f"{config['candidate_id']}__parent.json", {"attempts": attempts, "status": "FAIL"})
    raise RuntimeError(f"Fit {config['candidate_id']} failed after one retry: {attempts[-1]}")


def _screening_config(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    experiment_id = f"stage4c__screening__{candidate_id}__without_sensitive__cfg-{s4.configuration_digest(candidate)}"
    return {
        **candidate,
        "kind": "screening",
        "experiment_id": experiment_id,
        "sensitive_mode": "without_sensitive",
        "checkpoint_path": str(_checkpoint_path(candidate_id).relative_to(ROOT)),
        "prediction_path": str((c4.paths(ROOT)["predictions"] / f"{candidate_id}.csv").relative_to(ROOT)),
        "model_path": str((c4.paths(ROOT)["candidate_models"] / f"{candidate_id}.joblib").relative_to(ROOT)),
    }


def _controlled_config(frozen: dict[str, Any], sensitive_mode: str) -> dict[str, Any]:
    candidate_id = f"controlled_{sensitive_mode}"
    value = {
        "candidate_id": candidate_id,
        "kind": "controlled",
        "feature_pack": frozen["feature_pack"],
        "target_mode": frozen["target_mode"],
        "parameters": frozen["parameters"],
        "sensitive_mode": sensitive_mode,
    }
    value["experiment_id"] = f"stage4c__controlled__catboost__{sensitive_mode}__cfg-{s4.configuration_digest(value)}"
    value["checkpoint_path"] = str(_checkpoint_path(candidate_id).relative_to(ROOT))
    value["prediction_path"] = str((c4.paths(ROOT)["predictions"] / f"catboost_validation_predictions_{sensitive_mode}.csv").relative_to(ROOT))
    value["model_path"] = str((c4.paths(ROOT)["models"] / f"catboost_preliminary_{sensitive_mode}.joblib").relative_to(ROOT))
    return value


def phase_preflight() -> dict[str, Any]:
    c4.ensure_directories(ROOT)
    baseline = c4.capture_protected_baseline(ROOT)
    start = c4.validate_start(ROOT)
    subset = c4.create_screening_subset(ROOT) if not (c4.paths(ROOT)["splits"] / "stage4c_screening_subset.csv").is_file() else c4.validate_screening_subset(ROOT)
    result = {"baseline": baseline, "start": start, "screening_subset": subset, "status": "PASS"}
    s4.atomic_write_json(c4.paths(ROOT)["reports"] / "stage4c_preflight.json", result)
    return result


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    row = {
        "candidate_id": metadata["candidate_id"],
        "experiment_id": metadata["experiment_id"],
        "feature_pack": metadata["feature_pack"],
        "target_mode": metadata["target_mode"],
        "depth": metadata["parameters"]["depth"],
        "learning_rate": metadata["parameters"]["learning_rate"],
        "l2_leaf_reg": metadata["parameters"]["l2_leaf_reg"],
        "random_strength": metadata["parameters"]["random_strength"],
        "maximum_iterations": metadata["parameters"]["iterations"],
        "best_iteration": metadata["fixed_iteration_count"],
        "fit_time_seconds": metadata["fit_seconds"],
        "prediction_time_seconds": metadata["prediction_seconds"],
        "training_mae": metadata["training_metrics"]["mae"],
        "status": metadata["status"],
    }
    row.update(metadata["metrics"])
    return row


def _choose(rows: pd.DataFrame) -> pd.Series:
    best_mae = float(rows["mae"].min())
    close = rows.loc[rows["mae"] <= best_mae * 1.005].copy()
    close["native_preference"] = close["feature_pack"].ne("catboost_native_v1").astype(int)
    close["raw_preference"] = close["target_mode"].ne("raw").astype(int)
    close = close.sort_values(
        ["mae", "top_decile_mae", "top_five_percent_mae", "rmse", "rmsle_clipped_zero", "native_preference", "depth", "best_iteration", "fit_time_seconds", "raw_preference"],
        kind="mergesort",
    )
    return close.iloc[0]


def phase_screening() -> dict[str, Any]:
    phase_preflight()
    order = ["candidate_02_native_raw", "candidate_01_base_raw", "candidate_03_native_log1p"]
    metadata: list[dict[str, Any]] = []
    for candidate_id in order:
        metadata.append(_run_fit(_screening_config(c4.candidate_config(candidate_id)), timeout_seconds=480))
    initial = pd.DataFrame([_flatten_metadata(item) for item in metadata])
    provisional = _choose(initial)
    gap = (float(provisional["mae"]) - float(provisional["training_mae"])) / max(float(provisional["training_mae"]), 1e-12)
    refinement_reason = ""
    if gap > 0.05:
        base = c4.candidate_config(str(provisional["candidate_id"]))
        parameters = dict(base["parameters"])
        parameters.update({"depth": 6, "l2_leaf_reg": 20})
        refinement = {
            "candidate_id": "candidate_04_safer_refinement",
            "feature_pack": base["feature_pack"],
            "target_mode": base["target_mode"],
            "parameters": parameters,
            "required": False,
        }
        refinement_reason = f"The provisional best validation MAE was {gap:.1%} above its training MAE, so one safer depth and regularization direction was tested."
        metadata.append(_run_fit(_screening_config(refinement), timeout_seconds=480))
    rows = pd.DataFrame([_flatten_metadata(item) for item in metadata])
    selected = _choose(rows)
    rows["selected"] = rows["candidate_id"].eq(selected["candidate_id"])
    rows["selection_rank"] = rows["mae"].rank(method="first").astype(int)
    output = c4.paths(ROOT)["results"] / "catboost_initial_screening.csv"
    s4.atomic_write_csv(rows.sort_values("selection_rank"), output)
    selected_metadata = next(item for item in metadata if item["candidate_id"] == selected["candidate_id"])
    fixed_parameters = dict(selected_metadata["parameters"])
    fixed_parameters.pop("early_stopping_rounds", None)
    fixed_parameters["iterations"] = int(selected_metadata["fixed_iteration_count"])
    frozen = {
        "stage": c4.STAGE_ID,
        "selected_candidate_id": selected_metadata["candidate_id"],
        "selection_source": "without_sensitive Stage 4C screening only",
        "feature_pack": selected_metadata["feature_pack"],
        "target_mode": selected_metadata["target_mode"],
        "parameters": fixed_parameters,
        "fixed_iteration_count": int(selected_metadata["fixed_iteration_count"]),
        "random_seed": c4.SEED,
        "execution_mode": c4.EXECUTION_MODE,
        "thread_count": c4.THREAD_COUNT,
        "preprocessing_policy": "Stage 4B fixed features, stable selection, category sanitizing, and training-fit rare grouping inside each complete Pipeline",
        "feature_pack_digest": selected_metadata["feature_pack_digest"],
        "screening_subset_digest": selected_metadata["sample_digest"],
        "selection_reason": f"Lowest controlled screening MAE under the tie rules; MAE={selected_metadata['metrics']['mae']:.6f}.",
        "optional_refinement_run": len(metadata) == 4,
        "optional_refinement_reason": refinement_reason,
        "candidate_count": len(metadata),
        "created_at_utc": s4.utc_now(),
    }
    frozen["frozen_configuration_digest"] = s4.configuration_digest({key: frozen[key] for key in ("feature_pack", "target_mode", "parameters", "random_seed", "execution_mode", "thread_count", "preprocessing_policy")}, length=64)
    s4.atomic_write_json(c4.paths(ROOT)["results"] / "catboost_preliminary_configuration.json", frozen)
    return {"screening": rows.to_dict(orient="records"), "frozen": frozen, "status": "PASS"}


def _metric_row(metadata: dict[str, Any]) -> dict[str, Any]:
    row = {
        "experiment_id": metadata["experiment_id"],
        "sensitive_mode": metadata["sensitive_mode"],
        "feature_pack": metadata["feature_pack"],
        "target_mode": metadata["target_mode"],
        "fixed_iteration_count": metadata["fixed_iteration_count"],
        "feature_count": metadata["feature_count"],
        "fit_time_seconds": metadata["fit_seconds"],
        "prediction_time_seconds": metadata["prediction_seconds"],
        "status": metadata["status"],
    }
    row.update(metadata["metrics"])
    return row


def phase_controlled() -> dict[str, Any]:
    frozen_path = c4.paths(ROOT)["results"] / "catboost_preliminary_configuration.json"
    if not frozen_path.is_file():
        phase_screening()
    frozen = c4.read_json(frozen_path)
    metadata: list[dict[str, Any]] = []
    for mode in ("without_sensitive", "with_sensitive"):
        metadata.append(_run_fit(_controlled_config(frozen, mode), timeout_seconds=900))
    rows = pd.DataFrame([_metric_row(item) for item in metadata])
    s4.atomic_write_csv(rows, c4.paths(ROOT)["results"] / "catboost_controlled_validation_results.csv")
    without = rows.set_index("sensitive_mode").loc["without_sensitive"]
    with_sensitive = rows.set_index("sensitive_mode").loc["with_sensitive"]
    metric_names = ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "top_decile_mae", "top_five_percent_mae", "underestimation_rate", "overestimation_rate", "fit_time_seconds", "prediction_time_seconds", "feature_count"]
    comparison_rows = []
    for metric in metric_names:
        left = float(without[metric])
        right = float(with_sensitive[metric])
        comparison_rows.append({
            "metric": metric,
            "without_sensitive": left,
            "with_sensitive": right,
            "difference_with_minus_without": right - left,
            "relative_difference_percent": ((right - left) / abs(left) * 100) if left != 0 else np.nan,
        })
    comparison = pd.DataFrame(comparison_rows)
    s4.atomic_write_csv(comparison, c4.paths(ROOT)["results"] / "catboost_sensitive_comparison.csv")
    reload_rows = []
    for item in metadata:
        mode = item["sensitive_mode"]
        reload_result = c4.clean_reload_check(
            ROOT,
            ROOT / item["model_path"],
            ROOT / item["prediction_path"],
            c4.paths(ROOT)["reports"] / f"stage4c_reload_{mode}.json",
        )
        reload_result["sensitive_mode"] = mode
        reload_rows.append(reload_result)
    reload_frame = pd.DataFrame(reload_rows)
    s4.atomic_write_csv(reload_frame, c4.paths(ROOT)["reports"] / "stage4c_reload_verification.csv")
    if not reload_frame["status"].eq("PASS").all():
        raise AssertionError("A preliminary model did not pass clean-process reload.")
    manifest = {
        "stage": c4.STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "frozen_configuration_digest": frozen["frozen_configuration_digest"],
        "models": {
            item["sensitive_mode"]: {
                "model_path": item["model_path"],
                "prediction_path": item["prediction_path"],
                "model_sha256": c4.file_digest(ROOT / item["model_path"]),
                "prediction_sha256": c4.file_digest(ROOT / item["prediction_path"]),
                "source_hash": item["source_hash_digest"],
                "feature_pack_digest": item["feature_pack_digest"],
                "sample_digest": item["sample_digest"],
                "fixed_iteration_count": item["fixed_iteration_count"],
                "reload_status": "PASS",
            } for item in metadata
        },
        "status": "PASS",
    }
    s4.atomic_write_json(c4.paths(ROOT)["manifests"] / "catboost_preliminary_model_manifest.json", manifest)
    return {"results": rows.to_dict(orient="records"), "comparison": comparison.to_dict(orient="records"), "reload": reload_rows, "status": "PASS"}


def _bundle(mode: str) -> dict[str, Any]:
    s4.activate_local_packages(ROOT)
    return joblib.load(c4.paths(ROOT)["models"] / f"catboost_preliminary_{mode}.joblib")


def _feature_group(feature: str, bundle: dict[str, Any]) -> str:
    if feature in c4.SENSITIVE_COLUMNS:
        return "sensitive feature"
    if feature in {"respondent_id", "msamd_name", "county_name", "census_tract_number"}:
        return "high-cardinality category"
    fixed = set(s4.STAGE4B_FIXED_FEATURES)
    if feature in fixed:
        return "Stage 4B fixed feature"
    if feature in bundle["categorical_features"]:
        return "original categorical feature"
    return "original numeric feature"


def _importance(mode: str) -> pd.DataFrame:
    bundle = _bundle(mode)
    model = bundle["pipeline"].named_steps["model"]
    values = model.get_feature_importance(type="PredictionValuesChange")
    frame = pd.DataFrame({"feature": bundle["selected_features"], "importance": values})
    frame["rank"] = frame["importance"].rank(method="first", ascending=False).astype(int)
    frame["feature_group"] = frame["feature"].map(lambda name: _feature_group(name, bundle))
    frame["sensitive_mode"] = mode
    frame["importance_type"] = "PredictionValuesChange"
    frame = frame.sort_values("rank").reset_index(drop=True)
    s4.atomic_write_csv(frame, c4.paths(ROOT)["features"] / f"catboost_importance_{mode}.csv")
    s4.atomic_write_csv(frame.head(20), c4.paths(ROOT)["features"] / f"catboost_importance_top20_{mode}.csv")
    return frame


def _shap(mode: str, sample_ids: np.ndarray) -> pd.DataFrame:
    bundle = _bundle(mode)
    raw = bundle["raw_columns"]
    frame = s4.read_training_rows(c4.source_path(ROOT, mode), sample_ids, raw)
    X = frame.loc[sample_ids, raw].copy()
    preprocess = Pipeline(bundle["pipeline"].steps[:-1])
    ready = preprocess.transform(X)
    s4.activate_local_packages(ROOT)
    from catboost import Pool
    pool = Pool(ready, cat_features=bundle["categorical_features"])
    started = time.perf_counter()
    values = bundle["pipeline"].named_steps["model"].get_feature_importance(pool, type="ShapValues")
    elapsed = time.perf_counter() - started
    mean_absolute = np.abs(values[:, :-1]).mean(axis=0)
    result = pd.DataFrame({"feature": bundle["selected_features"], "mean_absolute_shap": mean_absolute})
    result["rank"] = result["mean_absolute_shap"].rank(method="first", ascending=False).astype(int)
    result["feature_group"] = result["feature"].map(lambda name: _feature_group(name, bundle))
    result["sensitive_mode"] = mode
    result = result.sort_values("rank").reset_index(drop=True)
    s4.atomic_write_csv(result, c4.paths(ROOT)["features"] / f"catboost_shap_importance_{mode}.csv")
    s4.atomic_write_csv(result.head(20), c4.paths(ROOT)["features"] / f"catboost_shap_top20_{mode}.csv")
    top = result.head(20).sort_values("mean_absolute_shap")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature"], top["mean_absolute_shap"], color="#3B82A0")
    axis.set_title(f"CatBoost mean absolute SHAP: {mode.replace('_', ' ')}")
    axis.set_xlabel("Mean absolute native CatBoost SHAP value")
    fig.tight_layout()
    fig.savefig(c4.paths(ROOT)["figures"] / f"catboost_shap_summary_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    s4.atomic_write_json(c4.paths(ROOT)["manifests"] / f"catboost_shap_metadata_{mode}.json", {
        "stage": c4.STAGE_ID,
        "sensitive_mode": mode,
        "sample_rows": len(sample_ids),
        "sample_ids_path": str((c4.paths(ROOT)["manifests"] / "catboost_shap_sample_row_ids.csv").relative_to(ROOT)),
        "native_catboost_shap": True,
        "wall_seconds": elapsed,
        "status": "PASS",
    })
    return result


def _errors(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(c4.paths(ROOT)["predictions"] / f"catboost_validation_predictions_{mode}.csv")
    predictions["target_decile"] = pd.qcut(predictions["y_true"], q=10, labels=False, duplicates="drop") + 1
    grouped = predictions.groupby("target_decile", observed=True).agg(
        rows=("row_id", "size"),
        target_min=("y_true", "min"),
        target_max=("y_true", "max"),
        mae=("absolute_error", "mean"),
        mean_signed_error=("residual", "mean"),
        underestimation_rate=("residual", lambda values: float(np.mean(values < 0))),
        overestimation_rate=("residual", lambda values: float(np.mean(values > 0))),
    ).reset_index()
    grouped["sensitive_mode"] = mode
    s4.atomic_write_csv(grouped, c4.paths(ROOT)["results"] / f"catboost_error_by_decile_{mode}.csv")
    worst = predictions.nlargest(20, "absolute_error").copy()
    worst["sensitive_mode"] = mode
    return grouped, worst


def _feature_proposals(importances: dict[str, pd.DataFrame], deciles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    without = importances["without_sensitive"].set_index("feature")
    ranks = {name: int(without.loc[name, "rank"]) if name in without.index else None for name in ("applicant_income_000s", "respondent_id", "loan_purpose_name", "lien_status_name", "hud_median_family_income", "tract_income_ratio")}
    tail = float(deciles["without_sensitive"].sort_values("target_decile").iloc[-1]["mae"])
    proposals = [
        {
            "feature_name": "applicant_to_estimated_tract_income_ratio",
            "formula": "applicant_income_000s / max((hud_median_family_income / 1000) * tract_income_ratio, epsilon)",
            "source_columns": "applicant_income_000s|hud_median_family_income|tract_income_ratio",
            "evidence": f"Income and area-income components had importance ranks {ranks['applicant_income_000s']}, {ranks['hud_median_family_income']}, and {ranks['tract_income_ratio']}; top target-decile MAE was {tail:.3f}.",
            "expected_benefit": "May give CatBoost a direct non-linear household-to-area capacity ratio for tail loans.",
            "fixed_or_learned": "fixed",
            "missing_handling": "Return missing when any source is missing.",
            "zero_denominator_handling": "Use a small positive epsilon and add a denominator-zero flag.",
            "target_derived": False,
            "sensitive_derived": False,
            "leakage_review": "PASS; uses only same-row application and area fields.",
            "duplicate_review": "Not an exact copy of the existing applicant-to-area or tract-income ratios because it combines both scales.",
        },
        {
            "feature_name": "respondent_purpose_group",
            "formula": "respondent_id + ' | ' + loan_purpose_name",
            "source_columns": "respondent_id|loan_purpose_name",
            "evidence": f"Respondent and purpose importance ranks were {ranks['respondent_id']} and {ranks['loan_purpose_name']} in the non-sensitive model.",
            "expected_benefit": "May represent stable lender specialization by loan purpose.",
            "fixed_or_learned": "learned rare grouping inside Pipeline",
            "missing_handling": "Map missing parts to <MISSING> before joining.",
            "zero_denominator_handling": "No denominator.",
            "target_derived": False,
            "sensitive_derived": False,
            "leakage_review": "PASS with training-fit rare grouping; no target statistic is used.",
            "duplicate_review": "No Stage 4B pack contains this lender-purpose combination.",
        },
        {
            "feature_name": "income_band_lien_status_group",
            "formula": "training-fit applicant_income_000s quantile band + ' | ' + lien_status_name",
            "source_columns": "applicant_income_000s|lien_status_name",
            "evidence": f"Applicant income and lien status importance ranks were {ranks['applicant_income_000s']} and {ranks['lien_status_name']}; tail MAE remained {tail:.3f}.",
            "expected_benefit": "May expose different lien effects across broad income capacity bands.",
            "fixed_or_learned": "learned band edges inside Pipeline",
            "missing_handling": "Use a separate missing band before joining.",
            "zero_denominator_handling": "No denominator.",
            "target_derived": False,
            "sensitive_derived": False,
            "leakage_review": "PASS only when band edges are fitted on each training partition.",
            "duplicate_review": "Stage 4B has a broad fixed income group but no income-band and lien interaction.",
        },
    ]
    return pd.DataFrame(proposals)


def _registry_rows() -> pd.DataFrame:
    registry = pd.read_csv(c4.paths(ROOT)["registry"])
    screening = pd.read_csv(c4.paths(ROOT)["results"] / "catboost_initial_screening.csv")
    controlled = pd.read_csv(c4.paths(ROOT)["results"] / "catboost_controlled_validation_results.csv")
    rows: list[dict[str, Any]] = []
    for _, item in screening.iterrows():
        metadata = c4.read_json(_checkpoint_path(str(item["candidate_id"])))
        rows.append(_registry_record(metadata, "screening", "candidate_fit"))
    for _, item in controlled.iterrows():
        metadata = c4.read_json(_checkpoint_path(f"controlled_{item['sensitive_mode']}"))
        rows.append(_registry_record(metadata, "controlled_discovery", "controlled_fit"))
        for analysis in ("model_save", "importance", "shap"):
            record = _registry_record(metadata, f"{analysis}_analysis", analysis)
            record["experiment_id"] = f"stage4c__{analysis}__catboost__{item['sensitive_mode']}__cfg-{s4.configuration_digest({'base': metadata['experiment_id'], 'analysis': analysis})}"
            if analysis != "model_save":
                record["model_artifact_path"] = metadata["model_path"]
                record["prediction_artifact_path"] = ""
            rows.append(record)
    new_rows = pd.DataFrame(rows)
    export = c4.paths(ROOT)["results"] / "stage4c_registry_rows.csv"
    s4.atomic_write_csv(new_rows[registry.columns], export)
    combined = s4.upsert_registry(registry, new_rows[registry.columns], allowed_stage_ids=(c4.STAGE_ID,))
    s4.atomic_write_csv(combined, c4.paths(ROOT)["registry"])
    if combined["experiment_id"].duplicated().any():
        raise AssertionError("Registry IDs are duplicated after Stage 4C upsert.")
    return new_rows[registry.columns]


def _registry_record(metadata: dict[str, Any], evaluation_stage: str, note: str) -> dict[str, Any]:
    metrics = metadata["metrics"]
    return {
        "experiment_id": metadata["experiment_id"], "timestamp_utc": metadata["completed_at_utc"],
        "model_family": "boosting_family", "model_name": "catboost", "sensitive_mode": metadata["sensitive_mode"],
        "feature_set": metadata["feature_pack"], "target_mode": metadata["target_mode"], "evaluation_stage": evaluation_stage,
        "fold_number": np.nan, "training_row_count": metadata["train_rows"], "validation_row_count": metadata["validation_rows"],
        "test_row_count": 0, "parameter_json": json.dumps(metadata["parameters"], sort_keys=True),
        "mae": metrics["mae"], "mse": metrics["mse"], "rmse": metrics["rmse"], "mape_percent": metrics["mape_percent"],
        "r_squared": metrics["r_squared"], "rmsle": metrics["rmsle"], "rmsle_clipped_zero": metrics["rmsle_clipped_zero"],
        "median_absolute_error": metrics["median_absolute_error"], "wape_percent": metrics["wape_percent"],
        "mean_signed_error": metrics["mean_signed_error"], "p90_absolute_error": metrics["p90_absolute_error"],
        "negative_prediction_rate": metrics["negative_prediction_rate"], "fit_time_seconds": metadata["fit_seconds"],
        "prediction_time_seconds": metadata["prediction_seconds"], "status": "success", "notes": note,
        "model_artifact_path": metadata["model_path"], "prediction_artifact_path": metadata["prediction_path"],
    }


def phase_analysis() -> dict[str, Any]:
    if not (c4.paths(ROOT)["results"] / "catboost_controlled_validation_results.csv").is_file():
        phase_controlled()
    importances = {mode: _importance(mode) for mode in ("without_sensitive", "with_sensitive")}
    _, validation_ids = c4.discovery_ids(ROOT)
    rng = np.random.default_rng(c4.SEED)
    sample_ids = np.sort(rng.choice(validation_ids, size=min(c4.MAX_SHAP_ROWS, len(validation_ids)), replace=False))
    s4.atomic_write_csv(pd.DataFrame({"row_id": sample_ids}), c4.paths(ROOT)["manifests"] / "catboost_shap_sample_row_ids.csv")
    shap = {mode: _shap(mode, sample_ids) for mode in ("without_sensitive", "with_sensitive")}
    deciles: dict[str, pd.DataFrame] = {}
    worst_frames = []
    for mode in ("without_sensitive", "with_sensitive"):
        deciles[mode], worst = _errors(mode)
        worst_frames.append(worst)
    s4.atomic_write_csv(pd.concat(worst_frames, ignore_index=True), c4.paths(ROOT)["results"] / "catboost_tail_error.csv")
    error_comparison = pd.concat(deciles.values(), ignore_index=True).pivot(index="target_decile", columns="sensitive_mode", values=["mae", "mean_signed_error", "underestimation_rate", "overestimation_rate"])
    error_comparison.columns = [f"{metric}__{mode}" for metric, mode in error_comparison.columns]
    s4.atomic_write_csv(error_comparison.reset_index(), c4.paths(ROOT)["results"] / "catboost_error_sensitive_comparison.csv")
    proposals = _feature_proposals(importances, deciles)
    s4.atomic_write_csv(proposals, c4.paths(ROOT)["features"] / "catboost_round2_feature_candidates.csv")
    registry_rows = _registry_rows()
    result = {
        "importance_rows": {mode: len(frame) for mode, frame in importances.items()},
        "shap_rows": {mode: len(frame) for mode, frame in shap.items()},
        "shap_sample_rows": len(sample_ids),
        "error_deciles": {mode: len(frame) for mode, frame in deciles.items()},
        "feature_proposals": len(proposals),
        "registry_rows": len(registry_rows),
        "status": "PASS",
    }
    s4.atomic_write_json(c4.paths(ROOT)["reports"] / "stage4c_analysis_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "screening", "controlled", "analysis", "heavy"), default="heavy")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.phase == "preflight":
        result = phase_preflight()
    elif args.phase == "screening":
        result = phase_screening()
    elif args.phase == "controlled":
        result = phase_controlled()
    elif args.phase == "analysis":
        result = phase_analysis()
    else:
        phase_preflight(); phase_screening(); phase_controlled(); result = phase_analysis()
    result["phase"] = args.phase
    result["wall_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
