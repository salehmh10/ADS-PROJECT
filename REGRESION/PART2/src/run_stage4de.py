"""Run recovery-safe Stage 4D–E CatBoost phases."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
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
import stage4de_catboost_utils as de


ROOT = Path(__file__).resolve().parent


def _checkpoint(fit_id: str) -> Path:
    return de.paths(ROOT)["checkpoints"] / f"{fit_id}.json"


def _config_path(fit_id: str) -> Path:
    return de.paths(ROOT)["checkpoints"] / f"{fit_id}__config.json"


def _valid(config: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = ROOT / config["checkpoint_path"]
    model = ROOT / config["model_path"]
    prediction = ROOT / config["prediction_path"] if config.get("prediction_path") else None
    native = ROOT / config["native_model_path"] if config.get("native_model_path") else None
    required = [checkpoint, model] + ([prediction] if prediction else []) + ([native] if native else [])
    if not all(path.is_file() for path in required):
        return None
    try:
        metadata = de.read_json(checkpoint)
        if metadata.get("status") != "PASS" or metadata.get("configuration_digest") != s4.configuration_digest(config, length=64):
            return None
        s4.activate_local_packages(ROOT)
        bundle = joblib.load(model)
        if bundle.metadata["experiment_id"] != config["experiment_id"]:
            return None
        if prediction:
            frame = pd.read_csv(prediction)
            expected = 500 if config["kind"] == "full_train" else (20_000 if config["sample_name"] == "feature_confirmation" else 25_000)
            if len(frame) != expected or frame["row_id"].duplicated().any() or not np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all():
                return None
        return metadata
    except Exception:
        return None


def _run_fit(config: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    cached = _valid(config)
    if cached is not None:
        return cached
    de.ensure_directories(ROOT)
    config_path = _config_path(config["fit_id"])
    s4.atomic_write_json(config_path, config)
    run = s4.run_worker_process(
        [sys.executable, "-B", str(ROOT / "stage4de_catboost_worker.py"), "--root", str(ROOT), "--config", str(config_path)],
        timeout_seconds=timeout_seconds,
        cwd=ROOT,
    )
    reports = de.paths(ROOT)["reports"] / "stage4de_workers"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{config['fit_id']}.log").write_text(f"STDOUT\n{run.get('stdout','')}\nSTDERR\n{run.get('stderr','')}\n", encoding="utf-8")
    s4.atomic_write_json(reports / f"{config['fit_id']}__parent.json", run)
    cached = _valid(config)
    if run["status"] != "success" or cached is None:
        raise RuntimeError(f"Stage 4D–E fit {config['fit_id']} failed: {run}")
    return cached


def _validation_config(
    fit_id: str,
    evaluation_stage: str,
    sample_name: str,
    spec: dict[str, Any],
    parameters: dict[str, Any],
    sensitive_mode: str = "without_sensitive",
    early_stopping_rounds: int | None = None,
) -> dict[str, Any]:
    base = {
        "fit_id": fit_id,
        "kind": "validation",
        "evaluation_stage": evaluation_stage,
        "sample_name": sample_name,
        "feature_pack": spec,
        "target_mode": "log1p",
        "sensitive_mode": sensitive_mode,
        "parameters": parameters,
        "early_stopping_rounds": early_stopping_rounds,
    }
    base["experiment_id"] = f"stage4de__{evaluation_stage}__{fit_id}__{sensitive_mode}__cfg-{s4.configuration_digest(base)}"
    base["checkpoint_path"] = str(_checkpoint(fit_id).relative_to(ROOT))
    base["prediction_path"] = str((de.paths(ROOT)["predictions"] / f"{fit_id}.csv").relative_to(ROOT))
    base["model_path"] = str((de.paths(ROOT)["candidate_models"] / f"{fit_id}.joblib").relative_to(ROOT))
    base["native_model_path"] = None
    return base


def _full_config(mode: str, final: dict[str, Any]) -> dict[str, Any]:
    fit_id = f"full_train_{mode}"
    base = {
        "fit_id": fit_id,
        "kind": "full_train",
        "evaluation_stage": "full_train",
        "feature_pack": final["feature_pack"],
        "target_mode": final["target_mode"],
        "sensitive_mode": mode,
        "parameters": final["parameters"],
    }
    base["experiment_id"] = f"stage4de__full_train__catboost__{mode}__cfg-{s4.configuration_digest(base)}"
    base["checkpoint_path"] = str(_checkpoint(fit_id).relative_to(ROOT))
    base["prediction_path"] = str((de.paths(ROOT)["predictions"] / f"catboost_full_train_reference_{mode}.csv").relative_to(ROOT))
    base["model_path"] = str((de.paths(ROOT)["models"] / f"catboost_final_{mode}.joblib").relative_to(ROOT))
    base["native_model_path"] = str((de.paths(ROOT)["models"] / f"catboost_final_{mode}.cbm").relative_to(ROOT))
    return base


def _smoke_v2() -> dict[str, Any]:
    p = de.paths(ROOT)
    train_ids, validation_ids = de.sample_ids(ROOT, "feature_confirmation")
    ids = np.concatenate([train_ids[:4000], validation_ids[:1000]])
    original = c4.load_feature_pack(ROOT, "catboost_native_v1")
    X = s4.read_training_rows(ROOT / "data/regression_without_sensitive_features.csv", ids, original["raw"]).loc[ids, original["raw"]].copy()
    source_digest = de.frame_digest(X)
    fit = X.iloc[:4000].copy()
    validation = X.iloc[4000:].copy()
    transformer = de.CatBoostFeatureEngineerV2()
    transformed_fit = transformer.fit_transform(fit)
    transformed_validation = transformer.transform(validation)
    edge_cases = validation.iloc[:2].copy()
    edge_cases.iloc[0, edge_cases.columns.get_loc("hud_median_family_income")] = 0.0
    edge_cases.iloc[1, edge_cases.columns.get_loc("applicant_income_000s")] = np.nan
    transformed_edge_cases = transformer.transform(edge_cases)
    artifact = p["manifests"] / "catboost_feature_engineer_v2_smoke.joblib"
    s4.atomic_write_joblib(transformer, artifact)
    clean_code = (
        "import joblib,pandas as pd,pathlib,sys; root=pathlib.Path.cwd(); sys.path.insert(0,str(root)); "
        "import stage4de_catboost_utils; t=joblib.load(root/'artifacts/manifests/stage4/catboost/catboost_feature_engineer_v2_smoke.joblib'); "
        "x=pd.read_csv(root/'artifacts/manifests/stage4/catboost/catboost_feature_engineer_v2_smoke_input.csv'); "
        "y=t.transform(x); assert len(y)==len(x); assert y.index.equals(x.index); print('PASS')"
    )
    input_path = p["manifests"] / "catboost_feature_engineer_v2_smoke_input.csv"
    s4.atomic_write_csv(validation, input_path)
    completed = subprocess.run([sys.executable, "-B", "-c", clean_code], cwd=ROOT, env=s4.worker_environment(ROOT), capture_output=True, text=True, timeout=60, check=False)
    checks = {
        "row_order_preserved": transformed_fit.index.equals(fit.index) and transformed_validation.index.equals(validation.index),
        "source_unchanged": de.frame_digest(X) == source_digest,
        "all_v1_features_preserved": set(s4.STAGE4B_FIXED_FEATURES).issubset(transformed_fit.columns),
        "all_approved_features_created": set((de.PROPOSAL_RATIO, de.RATIO_ZERO_FLAG, de.PROPOSAL_RESPONDENT_PURPOSE, de.PROPOSAL_INCOME_LIEN)).issubset(transformed_fit.columns),
        "ratio_finite_or_missing": np.isfinite(transformed_validation[de.PROPOSAL_RATIO].dropna()).all(),
        "ratio_zero_uses_epsilon_and_flag": bool(np.isfinite(transformed_edge_cases.iloc[0][de.PROPOSAL_RATIO]) and transformed_edge_cases.iloc[0][de.RATIO_ZERO_FLAG] == 1.0),
        "ratio_missing_source_stays_missing": bool(pd.isna(transformed_edge_cases.iloc[1][de.PROPOSAL_RATIO]) and transformed_edge_cases.iloc[1][de.RATIO_ZERO_FLAG] == 0.0),
        "learned_edges_training_fit": hasattr(transformer, "income_band_edges_"),
        "serialization_roundtrip": de.frame_digest(joblib.load(artifact).transform(validation)) == de.frame_digest(transformed_validation),
        "clean_process_import_transform": completed.returncode == 0 and "PASS" in completed.stdout,
        "target_absent": de.TARGET not in transformer.feature_names_in_,
        "sensitive_sources_absent": set(c4.SENSITIVE_COLUMNS).isdisjoint(transformer.feature_names_in_),
    }
    result = {
        "stage": de.STAGE_ID,
        "version": de.VERSION,
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "selected_proposals": list(de.APPROVED_PROPOSALS),
        "output_columns": list(transformed_validation.columns),
        "income_band_edges": transformer.income_band_edges_.tolist(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(p["manifests"] / "catboost_feature_engineer_v2_manifest.json", result)
    if result["status"] != "PASS":
        raise AssertionError(f"Feature Engineer v2 smoke failed: {result}")
    return result


def phase_preflight() -> dict[str, Any]:
    de.ensure_directories(ROOT)
    protected = de.recheck_protected(ROOT)
    stage4c = de.read_json(de.paths(ROOT)["reports"] / "stage4c_verification.json")
    samples = de.read_json(ROOT / "artifacts/splits/stage4/stage4_sample_verification.json")
    review = de.proposal_review(ROOT)
    smoke = _smoke_v2()
    checks = {
        "stage4c_pass": stage4c["status"] == "PASS",
        "protected_inputs_unchanged": protected["status"] == "PASS",
        "feature_confirmation_valid": samples["samples"]["feature_confirmation"]["valid"] is True,
        "final_selection_valid": samples["samples"]["final_selection"]["valid"] is True,
        "samples_disjoint_and_test_free": samples["all_sample_rows_unique"] is True and samples["test_overlap_rows"] == 0,
        "three_proposals_reviewed": len(review) == 3,
        "approved_proposals_safe": review["approved_for_combined_confirmation"].all(),
        "feature_engineer_v2_pass": smoke["status"] == "PASS",
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {"stage": de.STAGE_ID, "created_at_utc": s4.utc_now(), "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    s4.atomic_write_json(de.paths(ROOT)["reports"] / "stage4de_preflight.json", result)
    if result["status"] != "PASS":
        raise AssertionError(f"Stage 4D–E preflight failed: {checks}")
    return result


def _flat(metadata: dict[str, Any]) -> dict[str, Any]:
    row = {
        "fit_id": metadata["fit_id"], "experiment_id": metadata["experiment_id"],
        "feature_pack_id": metadata["feature_pack_id"], "sensitive_mode": metadata["sensitive_mode"],
        "target_mode": metadata["target_mode"], "depth": metadata["parameters"]["depth"],
        "learning_rate": metadata["parameters"]["learning_rate"], "l2_leaf_reg": metadata["parameters"]["l2_leaf_reg"],
        "random_strength": metadata["parameters"]["random_strength"], "best_iteration": metadata["fixed_iteration_count"],
        "feature_count": metadata["feature_count"], "fit_time_seconds": metadata["fit_seconds"],
        "prediction_time_seconds": metadata["prediction_seconds"], "status": metadata["status"],
    }
    row.update(metadata["metrics"])
    return row


def _confirmation_accept(base: pd.Series, candidate: pd.Series) -> tuple[bool, str, dict[str, Any]]:
    mae_change = (float(candidate["mae"]) - float(base["mae"])) / float(base["mae"]) * 100
    tail_change = (float(candidate["top_decile_mae"]) - float(base["top_decile_mae"])) / float(base["top_decile_mae"]) * 100
    p90_change = (float(candidate["p90_absolute_error"]) - float(base["p90_absolute_error"])) / float(base["p90_absolute_error"]) * 100
    main = mae_change <= -0.5
    tail = tail_change <= -2.0 and mae_change <= 0.2
    stability = abs(mae_change) <= 0.2 and abs(float(candidate["mean_signed_error"])) < abs(float(base["mean_signed_error"])) and tail_change < 0 and p90_change < 0 and float(candidate["fit_time_seconds"]) <= float(base["fit_time_seconds"]) * 1.5
    accepted = main or tail or stability
    rule = "main_mae" if main else "tail" if tail else "stability" if stability else "none"
    return accepted, rule, {"mae_change_percent": mae_change, "top_decile_mae_change_percent": tail_change, "p90_change_percent": p90_change, "main_rule": main, "tail_rule": tail, "stability_rule": stability}


def phase_confirmation() -> dict[str, Any]:
    phase_preflight()
    frozen = de.read_json(de.paths(ROOT)["results"].parent / "initial/catboost_preliminary_configuration.json")
    parameters = dict(frozen["parameters"])
    original = de.pack_spec(ROOT, "catboost_native_v1")
    combined = de.pack_spec(ROOT, "catboost_round2_combined_v2", de.APPROVED_PROPOSALS)
    configs = [
        _validation_config("confirmation_original", "feature_confirmation", "feature_confirmation", original, parameters),
        _validation_config("confirmation_combined_v2", "feature_confirmation", "feature_confirmation", combined, parameters),
    ]
    metadata = [_run_fit(config, 900) for config in configs]
    rows = pd.DataFrame([_flat(item) for item in metadata])
    base = rows.set_index("fit_id").loc["confirmation_original"]
    candidate = rows.set_index("fit_id").loc["confirmation_combined_v2"]
    accepted, rule, combined_impact = _confirmation_accept(base, candidate)
    combined_rule = rule
    rescue_ran = False
    rescue_accepted = False
    rescue_rule = None
    rescue_impact = None
    if not accepted:
        rescue_ran = True
        rescue_spec = de.pack_spec(ROOT, "catboost_round2_ratio_rescue_v2", (de.PROPOSAL_RATIO,))
        rescue_metadata = _run_fit(_validation_config("confirmation_ratio_rescue_v2", "feature_confirmation", "feature_confirmation", rescue_spec, parameters), 900)
        metadata.append(rescue_metadata)
        rows = pd.DataFrame([_flat(item) for item in metadata])
        rescue = rows.set_index("fit_id").loc["confirmation_ratio_rescue_v2"]
        rescue_accepted, rescue_rule, rescue_impact = _confirmation_accept(base, rescue)
    if accepted:
        final_spec = combined
        selected_proposals = list(de.APPROVED_PROPOSALS)
        selection = "combined_v2"
    elif rescue_accepted:
        final_spec = de.pack_spec(ROOT, "catboost_round2_ratio_rescue_v2", (de.PROPOSAL_RATIO,))
        selected_proposals = [de.PROPOSAL_RATIO]
        selection = "ratio_rescue_v2"
        rule = rescue_rule
    else:
        final_spec = original
        selected_proposals = []
        selection = "original_stage4c_pack"
        rule = "retain_original"
    rows["selected_final_pack"] = rows["feature_pack_id"].eq(final_spec["pack_id"])
    s4.atomic_write_csv(rows, de.paths(ROOT)["confirmation"] / "catboost_feature_confirmation_results.csv")
    s4.atomic_write_csv(rows, de.paths(ROOT)["results"] / "catboost_feature_confirmation_results.csv")
    all_predictions = []
    for item in metadata:
        frame = pd.read_csv(ROOT / item["prediction_path"])
        frame["fit_id"] = item["fit_id"]
        all_predictions.append(frame)
    s4.atomic_write_csv(pd.concat(all_predictions, ignore_index=True), de.paths(ROOT)["confirmation"] / "catboost_feature_confirmation_predictions.csv")
    final_pack = {
        "stage": de.STAGE_ID, "version": de.VERSION, "selection": selection,
        "feature_pack": final_spec, "selected_proposals": selected_proposals,
        "acceptance_rule": rule, "confirmation_sample_digest": metadata[0]["sample_digest"],
        "target_mode": "log1p", "parameters_frozen_from_stage4c": parameters,
        "created_at_utc": s4.utc_now(), "status": "PASS",
    }
    s4.atomic_write_json(de.paths(ROOT)["confirmation"] / "catboost_final_feature_pack.json", final_pack)
    s4.atomic_write_json(de.paths(ROOT)["results"] / "catboost_final_feature_pack.json", final_pack)
    review = pd.read_csv(de.paths(ROOT)["features"] / "catboost_round2_proposal_review.csv")
    selected = review.loc[review["feature_name"].isin(selected_proposals)].copy()
    rejected = review.loc[~review["feature_name"].isin(selected_proposals)].copy()
    rejected["final_rejection_reason"] = "The proposal did not pass the independent combined or rescue acceptance rule."
    s4.atomic_write_csv(selected, de.paths(ROOT)["features"] / "catboost_selected_round2_features.csv")
    s4.atomic_write_csv(rejected, de.paths(ROOT)["features"] / "catboost_rejected_round2_features.csv")
    impact_report = {
        "stage": de.STAGE_ID,
        "combined_accepted": accepted,
        "combined_acceptance_rule": combined_rule,
        "combined_impact": combined_impact,
        "rescue_ran": rescue_ran,
        "rescue_accepted": rescue_accepted,
        "rescue_acceptance_rule": rescue_rule,
        "rescue_impact": rescue_impact,
        "selection": selection,
        "final_acceptance_rule": rule,
        "status": "PASS",
    }
    s4.atomic_write_json(de.paths(ROOT)["confirmation"] / "catboost_feature_engineering_impact.json", impact_report)
    s4.atomic_write_json(de.paths(ROOT)["results"] / "catboost_feature_engineering_impact.json", impact_report)
    return {"rows": rows.to_dict(orient="records"), "final_pack": final_pack, "impact": impact_report, "status": "PASS"}


def _tuning_parameters() -> list[tuple[str, dict[str, Any]]]:
    shared = {"iterations": 2000, "random_seed": de.SEED, "loss_function": "MAE", "eval_metric": "MAE", "task_type": de.EXECUTION_MODE, "thread_count": de.THREAD_COUNT, "verbose": False, "allow_writing_files": False, "random_strength": 1}
    return [
        ("tuning_a_baseline", {**shared, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10}),
        ("tuning_b_safer", {**shared, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20}),
        ("tuning_c_flexible", {**shared, "depth": 9, "learning_rate": 0.04, "l2_leaf_reg": 10}),
    ]


def phase_tuning() -> dict[str, Any]:
    final_pack_path = de.paths(ROOT)["results"] / "catboost_final_feature_pack.json"
    if not final_pack_path.is_file():
        phase_confirmation()
    pack_record = de.read_json(final_pack_path)
    spec = pack_record["feature_pack"]
    metadata = []
    for fit_id, parameters in _tuning_parameters():
        metadata.append(_run_fit(_validation_config(fit_id, "final_tuning", "final_selection", spec, parameters, early_stopping_rounds=100), 900))
    rows = pd.DataFrame([_flat(item) for item in metadata])
    best_mae = float(rows["mae"].min())
    close = rows.loc[rows["mae"] <= best_mae * 1.0025].copy()
    close = close.sort_values(["depth", "best_iteration", "fit_time_seconds", "l2_leaf_reg"], ascending=[True, True, True, False], kind="mergesort")
    selected = close.iloc[0]
    rows["selected"] = rows["fit_id"].eq(selected["fit_id"])
    rows["mae_rank"] = rows["mae"].rank(method="first").astype(int)
    s4.atomic_write_csv(rows.sort_values("mae_rank"), de.paths(ROOT)["results"] / "catboost_final_tuning.csv")
    selected_metadata = next(item for item in metadata if item["fit_id"] == selected["fit_id"])
    parameters = dict(selected_metadata["parameters"])
    parameters["iterations"] = int(selected_metadata["fixed_iteration_count"])
    final = {
        "stage": de.STAGE_ID, "version": de.VERSION, "selected_fit_id": selected_metadata["fit_id"],
        "selection_source": "non-sensitive Final Selection sample only", "feature_pack": spec,
        "target_mode": "log1p", "parameters": parameters,
        "fixed_iteration_count": int(selected_metadata["fixed_iteration_count"]),
        "random_seed": de.SEED, "execution_mode": de.EXECUTION_MODE, "thread_count": de.THREAD_COUNT,
        "selection_reason": f"Selected under the 0.25 percent MAE tie rule; validation MAE={selected_metadata['metrics']['mae']:.6f}.",
        "created_at_utc": s4.utc_now(),
    }
    final["frozen_configuration_digest"] = s4.configuration_digest({key: final[key] for key in ("feature_pack", "target_mode", "parameters", "random_seed", "execution_mode", "thread_count")}, length=64)
    s4.atomic_write_json(de.paths(ROOT)["results"] / "catboost_final_configuration.json", final)
    return {"rows": rows.to_dict(orient="records"), "final": final, "status": "PASS"}


def phase_controlled() -> dict[str, Any]:
    final_path = de.paths(ROOT)["results"] / "catboost_final_configuration.json"
    if not final_path.is_file():
        phase_tuning()
    final = de.read_json(final_path)
    metadata = []
    for mode in ("without_sensitive", "with_sensitive"):
        fit_id = f"final_controlled_{mode}"
        metadata.append(_run_fit(_validation_config(fit_id, "final_controlled", "final_selection", final["feature_pack"], final["parameters"], sensitive_mode=mode, early_stopping_rounds=None), 1200))
    rows = pd.DataFrame([_flat(item) for item in metadata])
    s4.atomic_write_csv(rows, de.paths(ROOT)["results"] / "catboost_final_validation_results.csv")
    for item in metadata:
        mode = item["sensitive_mode"]
        source = ROOT / item["prediction_path"]
        destination = de.paths(ROOT)["results"] / f"catboost_final_validation_predictions_{mode}.csv"
        s4.atomic_write_csv(pd.read_csv(source), destination)
    indexed = rows.set_index("sensitive_mode")
    comparison = []
    for metric in ("mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate", "top_decile_mae", "top_five_percent_mae", "fit_time_seconds", "prediction_time_seconds", "feature_count"):
        left = float(indexed.loc["without_sensitive", metric]); right = float(indexed.loc["with_sensitive", metric])
        comparison.append({"metric": metric, "without_sensitive": left, "with_sensitive": right, "difference_with_minus_without": right-left, "relative_difference_percent": (right-left)/abs(left)*100 if left else np.nan})
    comparison_frame = pd.DataFrame(comparison)
    s4.atomic_write_csv(comparison_frame, de.paths(ROOT)["results"] / "catboost_final_sensitive_comparison.csv")
    return {"rows": rows.to_dict(orient="records"), "comparison": comparison, "status": "PASS"}


def _reload_final(mode: str, metadata: dict[str, Any]) -> dict[str, Any]:
    model_path = ROOT / metadata["model_path"]
    reference_path = ROOT / metadata["prediction_path"]
    code = (
        "import hashlib,joblib,numpy as np,pandas as pd,pathlib,sys; root=pathlib.Path.cwd(); sys.path.insert(0,str(root)); "
        "import stage4_boosting_utils as s4,stage4de_catboost_utils as de; "
        f"bundle=joblib.load(root/{str(model_path.relative_to(ROOT))!r}); ref=pd.read_csv(root/{str(reference_path.relative_to(ROOT))!r}); "
        f"mode={mode!r}; ids=ref['row_id'].to_numpy(dtype=np.int64); raw=bundle.metadata['raw_columns']; "
        "X=s4.read_training_rows(de.source_path(root,mode),ids,raw).loc[ids,raw].copy(); before=pd.util.hash_pandas_object(X,index=True).sum(); "
        "X2=X.loc[:,list(reversed(raw))].copy(); X2['unused_extra_column']=1; "
        "pred=bundle.predict(X2); direct=bundle.pipeline.predict(X2); manual=np.expm1(direct); after=pd.util.hash_pandas_object(X,index=True).sum(); "
        "prep=__import__('sklearn.pipeline',fromlist=['Pipeline']).Pipeline(bundle.pipeline.steps[:-1]); ready=prep.transform(X2); "
        "from catboost import CatBoostRegressor; native=CatBoostRegressor(); native.load_model(str(root/metadata_native)); native_direct=native.predict(ready); "
        "assert before==after; assert len(pred)==len(ref); assert np.isfinite(pred).all(); "
        "assert np.allclose(pred,manual,rtol=1e-12,atol=1e-12); assert np.allclose(direct,native_direct,rtol=1e-10,atol=1e-10); "
        "assert np.allclose(pred,ref['y_pred'].to_numpy(),rtol=1e-10,atol=1e-10); "
        "assert bundle.metadata['target_transform_contract']['prediction_inverse']=='numpy.expm1'; "
        "assert set(bundle.metadata['split_hashes'])=={'train_row_ids_sha256','test_row_ids_sha256','cv_fold_assignments_sha256','split_config_sha256'}; "
        "assert bundle.metadata['environment']['catboost']; print('PASS reordered_columns expm1 native_match provenance')"
    )
    code = code.replace("metadata_native", repr(metadata["native_model_path"]))
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=s4.worker_environment(ROOT), capture_output=True, text=True, timeout=300, check=False)
    return {"sensitive_mode": mode, "model_path": metadata["model_path"], "native_model_path": metadata["native_model_path"], "reference_path": metadata["prediction_path"], "model_bytes": model_path.stat().st_size, "native_model_bytes": (ROOT/metadata["native_model_path"]).stat().st_size, "wall_seconds": time.perf_counter()-started, "return_code": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(), "status": "PASS" if completed.returncode==0 and "PASS" in completed.stdout else "FAIL"}


def phase_full_train() -> dict[str, Any]:
    final_path = de.paths(ROOT)["results"] / "catboost_final_configuration.json"
    if not final_path.is_file():
        phase_tuning()
    final = de.read_json(final_path)
    metadata = []
    for mode in ("without_sensitive", "with_sensitive"):
        metadata.append(_run_fit(_full_config(mode, final), 2700))
    reloads = [_reload_final(item["sensitive_mode"], item) for item in metadata]
    reload_frame = pd.DataFrame(reloads)
    s4.atomic_write_csv(reload_frame, de.paths(ROOT)["reports"] / "stage4de_catboost_reload_verification.csv")
    if not reload_frame["status"].eq("PASS").all():
        raise AssertionError("A final full-Train model failed clean-process reload.")
    manifest = {
        "stage": de.STAGE_ID, "version": de.VERSION, "created_at_utc": s4.utc_now(),
        "frozen_configuration_digest": final["frozen_configuration_digest"],
        "environment": de.environment_metadata(ROOT),
        "models": {item["sensitive_mode"]: {**{key:item[key] for key in ("experiment_id","training_row_count","fit_seconds","fixed_iteration_count","model_path","native_model_path","prediction_path","source_hash_digest","sample_digest","feature_count")}, "model_sha256":s4.sha256_file(ROOT/item["model_path"]), "native_model_sha256":s4.sha256_file(ROOT/item["native_model_path"]), "reload_status":"PASS"} for item in metadata},
        "status": "PASS",
    }
    s4.atomic_write_json(de.paths(ROOT)["results"] / "catboost_full_train_manifest.json", manifest)
    return {"metadata": metadata, "reload": reloads, "manifest": manifest, "status": "PASS"}


def _feature_group(feature: str, sensitive_mode: str) -> str:
    if feature in set(c4.SENSITIVE_COLUMNS):
        return "sensitive"
    if feature in set(s4.STAGE4B_FIXED_FEATURES):
        return "stage4b_engineered"
    if feature in set(de.APPROVED_PROPOSALS) | {de.RATIO_ZERO_FLAG}:
        return "stage4d_proposal"
    if feature in {"respondent_id", "msamd_name", "county_name", "census_tract_number"}:
        return "high_cardinality_categorical"
    return "original_numeric_or_categorical"


def _interpret_mode(mode: str, validation_ids: np.ndarray) -> dict[str, Any]:
    """Create final importance and native SHAP summaries for one saved model."""
    s4.activate_local_packages(ROOT)
    from catboost import Pool

    p = de.paths(ROOT)
    model_path = p["models"] / f"catboost_final_{mode}.joblib"
    bundle = joblib.load(model_path)
    model = bundle.pipeline.named_steps["model"]
    names = list(model.feature_names_)
    importance_started = time.perf_counter()
    values = np.asarray(model.get_feature_importance(type="PredictionValuesChange"), dtype=float)
    importance_seconds = time.perf_counter() - importance_started
    if len(names) != len(values):
        raise AssertionError(f"Importance length mismatch for {mode}.")
    importance = pd.DataFrame({
        "feature": names,
        "importance": values,
        "feature_group": [_feature_group(name, mode) for name in names],
        "sensitive_mode": mode,
        "importance_type": "PredictionValuesChange",
    }).sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance_path = p["results"] / f"catboost_final_importance_{mode}.csv"
    s4.atomic_write_csv(importance, importance_path)
    s4.atomic_write_csv(importance.head(20), p["results"] / f"catboost_final_importance_{mode}_top20.csv")

    raw = list(bundle.metadata["raw_columns"])
    X = s4.read_training_rows(de.source_path(ROOT, mode), validation_ids, raw).loc[validation_ids, raw].copy()
    source_digest = de.frame_digest(X)
    preprocess = Pipeline(bundle.pipeline.steps[:-1])
    ready = preprocess.transform(X)
    categorical = list(bundle.metadata["categorical_features"])
    pool = Pool(ready, cat_features=categorical)
    shap_started = time.perf_counter()
    shap_values = np.asarray(model.get_feature_importance(pool, type="ShapValues"), dtype=float)
    shap_seconds = time.perf_counter() - shap_started
    if shap_values.shape != (len(validation_ids), len(names) + 1):
        raise AssertionError(f"Unexpected SHAP shape for {mode}: {shap_values.shape}")
    if de.frame_digest(X) != source_digest or not np.isfinite(shap_values).all():
        raise AssertionError(f"SHAP source mutation or non-finite values for {mode}.")
    contributions = shap_values[:, :-1]
    shap = pd.DataFrame({
        "feature": names,
        "mean_absolute_shap": np.mean(np.abs(contributions), axis=0),
        "mean_signed_shap": np.mean(contributions, axis=0),
        "feature_group": [_feature_group(name, mode) for name in names],
        "sensitive_mode": mode,
        "shap_space": "log1p_target",
        "sample_rows": len(validation_ids),
    }).sort_values(["mean_absolute_shap", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    shap["rank"] = np.arange(1, len(shap) + 1)
    shap_path = p["results"] / f"catboost_final_shap_{mode}.csv"
    s4.atomic_write_csv(shap, shap_path)
    s4.atomic_write_csv(shap.head(20), p["results"] / f"catboost_final_shap_{mode}_top20.csv")

    for frame, value_column, title, filename in (
        (importance.head(20).iloc[::-1], "importance", f"Final CatBoost importance: {mode}", f"catboost_final_importance_{mode}.png"),
        (shap.head(20).iloc[::-1], "mean_absolute_shap", f"Final CatBoost mean absolute SHAP: {mode}", f"catboost_final_shap_{mode}.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(frame["feature"], frame[value_column], color="#31688e")
        ax.set_title(title)
        ax.set_xlabel(value_column.replace("_", " "))
        fig.tight_layout()
        figure_path = p["figures"] / filename
        temp_path = figure_path.with_name(f"{figure_path.stem}.tmp{figure_path.suffix}")
        fig.savefig(temp_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        os.replace(temp_path, figure_path)

    result = {
        "sensitive_mode": mode,
        "model_path": str(model_path.relative_to(ROOT)),
        "importance_path": str(importance_path.relative_to(ROOT)),
        "shap_path": str(shap_path.relative_to(ROOT)),
        "importance_seconds": importance_seconds,
        "shap_seconds": shap_seconds,
        "shap_rows": len(validation_ids),
        "feature_count": len(names),
        "expected_value_mean": float(np.mean(shap_values[:, -1])),
        "status": "PASS",
    }
    del pool, ready, X, shap_values, contributions, model, bundle
    gc.collect()
    return result


def _previous_stage_table() -> pd.DataFrame:
    stage2 = pd.read_csv(ROOT / "artifacts/results/prompt2/cv_oof_summary.csv")
    stage3 = pd.read_csv(ROOT / "artifacts/results/stage3/cv_oof_summary.csv")
    stage4c = pd.read_csv(ROOT / "artifacts/results/stage4/catboost/initial/catboost_controlled_validation_results.csv")
    stage4de = pd.read_csv(de.paths(ROOT)["results"] / "catboost_final_validation_results.csv")
    s2 = stage2.loc[stage2["sensitive_mode"].eq("without_sensitive")].sort_values("mae", kind="mergesort").iloc[0]
    s3 = stage3.loc[stage3["sensitive_mode"].eq("without_sensitive") & stage3["status"].eq("success")].sort_values("mae", kind="mergesort").iloc[0]
    s4c = stage4c.loc[stage4c["sensitive_mode"].eq("without_sensitive")].iloc[0]
    s4de = stage4de.loc[stage4de["sensitive_mode"].eq("without_sensitive")].iloc[0]
    rows = [
        {"stage": "Stage 2", "model_name": s2["model_name"], "sensitive_mode": s2["sensitive_mode"], "evaluation_scheme": "saved-fold OOF", "evaluation_rows": 399788, "mae": s2["mae"], "rmse": s2["rmse"], "rmsle": s2["rmsle"], "r_squared": s2["r_squared"], "runtime_seconds": s2["total_fit_time_seconds"], "prediction_runtime_seconds": s2["total_prediction_time_seconds"]},
        {"stage": "Stage 3", "model_name": s3["model_name"], "sensitive_mode": s3["sensitive_mode"], "evaluation_scheme": "saved-fold OOF", "evaluation_rows": int(s3["oof_rows"]), "mae": s3["mae"], "rmse": s3["rmse"], "rmsle": s3["rmsle"], "r_squared": s3["r_squared"], "runtime_seconds": s3["total_fit_time_seconds"], "prediction_runtime_seconds": s3["total_prediction_time_seconds"]},
        {"stage": "Stage 4C", "model_name": "catboost", "sensitive_mode": s4c["sensitive_mode"], "evaluation_scheme": "Discovery validation", "evaluation_rows": 15000, "mae": s4c["mae"], "rmse": s4c["rmse"], "rmsle": s4c["rmsle"], "r_squared": s4c["r_squared"], "runtime_seconds": s4c["fit_time_seconds"], "prediction_runtime_seconds": s4c["prediction_time_seconds"]},
        {"stage": "Stage 4D-E", "model_name": "catboost", "sensitive_mode": s4de["sensitive_mode"], "evaluation_scheme": "Final Selection validation", "evaluation_rows": 25000, "mae": s4de["mae"], "rmse": s4de["rmse"], "rmsle": s4de["rmsle"], "r_squared": s4de["r_squared"], "runtime_seconds": s4de["fit_time_seconds"], "prediction_runtime_seconds": s4de["prediction_time_seconds"]},
    ]
    frame = pd.DataFrame(rows)
    frame["directly_comparable_across_all_rows"] = False
    frame["comparison_note"] = "OOF, Discovery validation, and Final Selection validation use different evaluation rows."
    frame["locked_test_note"] = "The locked Test Set will later provide the common final comparison."
    return frame


def _registry_row(metadata: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    metrics = dict(metadata.get("metrics", {}))
    row = {
        "experiment_id": metadata["experiment_id"],
        "timestamp_utc": metadata.get("completed_at_utc", metadata.get("started_at_utc")),
        "model_family": "boosting_family",
        "model_name": "catboost",
        "sensitive_mode": metadata["sensitive_mode"],
        "feature_set": metadata["feature_pack_id"],
        "target_mode": metadata["target_mode"],
        "evaluation_stage": metadata["evaluation_stage"],
        "fold_number": np.nan,
        "training_row_count": metadata["training_row_count"],
        "validation_row_count": metadata["validation_row_count"],
        "test_row_count": 0,
        "parameter_json": json.dumps(metadata["parameters"], sort_keys=True),
        "fit_time_seconds": metadata.get("fit_seconds", 0.0),
        "prediction_time_seconds": metadata.get("prediction_seconds", 0.0),
        "status": "success",
        "notes": metadata["fit_id"],
        "model_artifact_path": metadata.get("model_path"),
        "prediction_artifact_path": metadata.get("prediction_path"),
    }
    for name in ("mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"):
        row[name] = metrics.get(name, np.nan)
    row.update(overrides)
    return row


def _write_registry(interpretation: list[dict[str, Any]]) -> pd.DataFrame:
    checkpoint_names = [
        "confirmation_original", "confirmation_combined_v2", "confirmation_ratio_rescue_v2",
        "tuning_a_baseline", "tuning_b_safer", "tuning_c_flexible",
        "final_controlled_without_sensitive", "final_controlled_with_sensitive",
        "full_train_without_sensitive", "full_train_with_sensitive",
    ]
    metadata = {name: de.read_json(_checkpoint(name)) for name in checkpoint_names}
    rows = [_registry_row(metadata[name]) for name in checkpoint_names]
    for mode in ("without_sensitive", "with_sensitive"):
        full = metadata[f"full_train_{mode}"]
        digest_source = {"full_experiment_id": full["experiment_id"], "mode": mode}
        for kind, validation_rows, notes in (("reload", 500, "clean_process_reload"), ("importance", 0, "PredictionValuesChange"), ("shap", 300, "native_SHAP_log1p_target")):
            experiment_id = f"stage4de__{kind}__catboost__{mode}__cfg-{s4.configuration_digest({**digest_source, 'kind': kind})}"
            rows.append(_registry_row(
                full,
                experiment_id=experiment_id,
                evaluation_stage=f"{kind}_analysis",
                validation_row_count=validation_rows,
                fit_time_seconds=0.0,
                prediction_time_seconds=0.0,
                notes=notes,
            ))
    columns = list(pd.read_csv(de.paths(ROOT)["registry"], nrows=0).columns)
    stage_rows = pd.DataFrame(rows)
    for column in columns:
        if column not in stage_rows.columns:
            stage_rows[column] = np.nan
    stage_rows = stage_rows[columns]
    s4.atomic_write_csv(stage_rows, de.paths(ROOT)["results"] / "stage4de_registry_rows.csv")
    if len(stage_rows) != 16 or stage_rows["experiment_id"].duplicated().any():
        raise AssertionError("Stage 4D-E Registry must contain 16 unique rows.")
    prior_paths = [
        ROOT / "artifacts/results/prompt2/prompt2_registry_rows.csv",
        ROOT / "artifacts/results/stage3/stage3_registry_rows.csv",
        ROOT / "artifacts/results/stage4/catboost/initial/stage4c_registry_rows.csv",
    ]
    prior = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in prior_paths], ignore_index=True)
    current = pd.read_csv(de.paths(ROOT)["results"] / "stage4de_registry_rows.csv", dtype=str, keep_default_na=False)
    aligned = pd.concat([prior, current], ignore_index=True)
    if list(aligned.columns) != columns or len(prior) != 227 or len(aligned) != 243 or aligned["experiment_id"].duplicated().any():
        raise AssertionError("Canonical prior Registry reconstruction failed.")
    s4.atomic_write_csv(aligned, de.paths(ROOT)["registry"])
    return stage_rows


def _enrich_final_bundles() -> dict[str, Any]:
    """Add required provenance to existing bundles without fitting a model."""
    s4.activate_local_packages(ROOT)
    manifest_path = de.paths(ROOT)["results"] / "catboost_full_train_manifest.json"
    manifest = de.read_json(manifest_path)
    for mode in ("without_sensitive", "with_sensitive"):
        item = manifest["models"][mode]
        model_path = ROOT / item["model_path"]
        bundle = joblib.load(model_path)
        bundle.metadata.update(de.bundle_provenance(ROOT, mode, bundle.target_mode, bundle.metadata["raw_columns"]))
        s4.atomic_write_joblib(bundle, model_path)
        item["model_sha256"] = s4.sha256_file(model_path)
        item["bundle_metadata_contract"] = {
            "environment_in_bundle": True,
            "split_hashes_in_bundle": True,
            "target_inverse_in_bundle": True,
            "name_based_input_contract_in_bundle": True,
        }
    manifest["bundle_metadata_enriched_at_utc"] = s4.utc_now()
    s4.atomic_write_json(manifest_path, manifest)
    return manifest


def phase_analysis() -> dict[str, Any]:
    if not (de.paths(ROOT)["results"] / "catboost_full_train_manifest.json").is_file():
        phase_full_train()
    _enrich_final_bundles()
    validation_ids = np.sort(de.sample_ids(ROOT, "final_selection")[1])[:300]
    sample = pd.DataFrame({"row_id": validation_ids, "sample_role": "shap", "sample_order": np.arange(len(validation_ids))})
    s4.atomic_write_csv(sample, de.paths(ROOT)["results"] / "catboost_final_shap_sample_ids.csv")
    interpretation = [_interpret_mode(mode, validation_ids) for mode in ("without_sensitive", "with_sensitive")]
    s4.atomic_write_json(de.paths(ROOT)["results"] / "catboost_final_interpretation_manifest.json", {
        "stage": de.STAGE_ID,
        "sample_source": "first 300 sorted Final Selection validation row IDs",
        "same_ids_both_modes": True,
        "test_rows": 0,
        "modes": interpretation,
        "status": "PASS",
    })
    previous = _previous_stage_table()
    s4.atomic_write_csv(previous, de.paths(ROOT)["results"] / "catboost_previous_stage_reference.csv")
    registry = _write_registry(interpretation)
    fit_names = [
        "confirmation_original", "confirmation_combined_v2", "confirmation_ratio_rescue_v2",
        "tuning_a_baseline", "tuning_b_safer", "tuning_c_flexible",
        "final_controlled_without_sensitive", "final_controlled_with_sensitive",
        "full_train_without_sensitive", "full_train_with_sensitive",
    ]
    fits = [de.read_json(_checkpoint(name)) for name in fit_names]
    confirmation_seconds = float(sum(item["fit_seconds"] for item in fits if item["evaluation_stage"] == "feature_confirmation"))
    tuning_seconds = float(sum(item["fit_seconds"] for item in fits if item["evaluation_stage"] == "final_tuning"))
    controlled_seconds = float(sum(item["fit_seconds"] for item in fits if item["evaluation_stage"] == "final_controlled"))
    full_seconds = float(sum(item["fit_seconds"] for item in fits if item["evaluation_stage"] == "full_train"))
    fit_seconds = confirmation_seconds + tuning_seconds + controlled_seconds + full_seconds
    analysis_seconds = float(sum(item["importance_seconds"] + item["shap_seconds"] for item in interpretation))
    recovery = de.read_json(de.paths(ROOT)["reports"] / "stage4c_notebook_recovery.json")
    stage4c_recovery_seconds = float(sum(item.get("wall_seconds", 0.0) for item in recovery.get("attempts", []) if item.get("status") == "success"))
    notebook_report = de.paths(ROOT)["reports"] / "stage4de_notebook_executions.json"
    notebook_seconds = float(sum(item.get("wall_seconds", 0.0) for item in de.read_json(notebook_report).get("attempts", []) if item.get("status") == "success")) if notebook_report.is_file() else 0.0
    total_accounted = fit_seconds + analysis_seconds + notebook_seconds
    runtime = {
        "stage": de.STAGE_ID,
        "fit_count": len(fits),
        "stage4c_recovery_seconds_reference": stage4c_recovery_seconds,
        "feature_confirmation_fit_seconds": confirmation_seconds,
        "final_tuning_fit_seconds": tuning_seconds,
        "controlled_comparison_fit_seconds": controlled_seconds,
        "full_train_fit_seconds": full_seconds,
        "fit_seconds_sum": fit_seconds,
        "interpretation_seconds_sum": analysis_seconds,
        "notebook_execution_seconds": notebook_seconds,
        "accounted_seconds": total_accounted,
        "budget_seconds": 210 * 60,
        "measurement_scope": "Sum of Stage 4D-E fit, interpretation, and notebook execution times. Stage 4C recovery is reported separately.",
        "within_budget": total_accounted <= 210 * 60,
        "status": "PASS" if total_accounted <= 210 * 60 else "FAIL",
    }
    s4.atomic_write_json(de.paths(ROOT)["reports"] / "stage4de_runtime_report.json", runtime)
    if runtime["status"] != "PASS":
        raise AssertionError("Stage 4D-E exceeded its runtime budget.")
    return {"interpretation": interpretation, "previous_stage_rows": previous.to_dict(orient="records"), "registry_rows": len(registry), "runtime": runtime, "status": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "confirmation", "tuning", "controlled", "full_train", "analysis"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    functions = {"preflight": phase_preflight, "confirmation": phase_confirmation, "tuning": phase_tuning, "controlled": phase_controlled, "full_train": phase_full_train, "analysis": phase_analysis}
    result = functions[args.phase]()
    result["phase"] = args.phase
    result["wall_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
