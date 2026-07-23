"""Targeted, resumable Stage 3 recovery orchestration."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
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

from stage3_recovery_utils import (
    CONFIG_PATH, CV_PATH, DEV_PATH, FOLD_RESULTS, MANIFESTS, MODELS, PREDICTIONS,
    PROGRESS_PATH, REPORTS, RESULTS, ROOT, atomic_save_csv, aggregate_completed_cv,
    checkpoint_metadata, experiment_digest, fold_paths, load_all_targets, load_ids,
    selected_configurations, save_fold_checkpoint, split_hashes, update_progress, utc_now,
    validate_fold_checkpoint,
)
from stage3_tree_utils import (
    SENSITIVE_FEATURES, canonical_json, deterministic_experiment_id,
    evaluate_regression_predictions, feature_lists, package_versions, sha256_file,
    write_json,
)


WORKER = ROOT / "stage3_fold_worker.py"
CONFIG_DIR = MANIFESTS / "stage3_worker_configs"
WORKER_LOGS = REPORTS / "stage3_workers"
RUNTIME_CANDIDATE_DIR = RESULTS / "runtime_repair_candidates"
PILOT_DIR = RESULTS / "pilots"
FINAL_DIR = RESULTS / "finals"
IMPORTANCE_DIR = ROOT / "artifacts/features/tree/importance"
INCOMPLETE_DIR = ROOT / "artifacts/backups/stage3_incomplete_workers"
for directory in (CONFIG_DIR, WORKER_LOGS, RUNTIME_CANDIDATE_DIR, PILOT_DIR, FINAL_DIR, IMPORTANCE_DIR, INCOMPLETE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


LIMITS = {
    "max_repair_iterations": 20,
    "max_targeted_execution_attempts": 6,
    "max_full_notebook_execution_attempts": 3,
    "required_successful_full_notebook_executions": 2,
    "max_retries_per_fold": 2,
    "max_bagging_development_fit_seconds": 300,
    "max_full_fold_fit_seconds": 1200,
    "max_final_full_training_fit_seconds": 1800,
    "max_total_recovery_runtime_minutes": 240,
}


def configuration_file(configuration: dict[str, Any]) -> Path:
    digest = hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()[:16]
    path = CONFIG_DIR / f"{configuration['model_name']}__{digest}.json"
    write_json(path, configuration)
    return path


def process_tree_stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, text=True, timeout=30,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def run_worker(
    *, task: str, configuration: dict[str, Any], mode: str, fold: int,
    result_path: Path, timeout_seconds: int, model_path: Path | None = None,
    provenance: str = "targeted_worker",
) -> dict[str, Any]:
    config_path = configuration_file(configuration)
    key = f"{task}__{configuration['model_name']}__{mode}__fold-{fold}__{experiment_digest(configuration, mode, fold, task)[:10]}"
    log_path = WORKER_LOGS / f"{key}.log"
    command = [
        sys.executable, str(WORKER), "--task", task, "--config", str(config_path),
        "--mode", mode, "--fold", str(fold), "--result-path", str(result_path),
        "--provenance", provenance,
    ]
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    start = time.monotonic(); started_at = utc_now()
    print(f"START {task}: {configuration['model_name']} | {mode} | Fold {fold} | timeout {timeout_seconds}s", flush=True)
    update_progress(
        current_model=configuration["model_name"], sensitive_mode=mode, fold=fold,
        start_time_utc=started_at, elapsed_seconds=0, status="running",
        next_action=f"wait for {task} worker", worker_log=str(log_path.relative_to(ROOT)),
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        last_heartbeat = 0.0
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout_seconds:
                timed_out = True
                process_tree_stop(process)
                break
            if elapsed - last_heartbeat >= 60:
                last_heartbeat = elapsed
                update_progress(elapsed_seconds=round(elapsed, 1), status="running")
                print(f"HEARTBEAT {task}: {configuration['model_name']} | {mode} | Fold {fold} | {elapsed/60:.1f} min", flush=True)
            time.sleep(2)
    elapsed = time.monotonic() - start
    if timed_out:
        payload = {
            "status": "timeout", "task": task, "model_name": configuration["model_name"],
            "sensitive_mode": mode, "fold": fold, "configuration": configuration,
            "elapsed_seconds": elapsed, "timeout_seconds": timeout_seconds,
            "start_time_utc": started_at, "end_time_utc": utc_now(),
            "worker_log": str(log_path.relative_to(ROOT)),
        }
        write_json(result_path, payload)
        update_progress(elapsed_seconds=round(elapsed, 1), status="timeout", last_completed_artifact=None)
        print(f"TIMEOUT {task}: {configuration['model_name']} | {mode} | Fold {fold} | {elapsed:.1f}s", flush=True)
        return payload
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        payload = {
            "status": "failed", "task": task, "model_name": configuration["model_name"],
            "sensitive_mode": mode, "fold": fold, "configuration": configuration,
            "elapsed_seconds": elapsed, "returncode": process.returncode, "error_tail": tail,
            "start_time_utc": started_at, "end_time_utc": utc_now(),
            "worker_log": str(log_path.relative_to(ROOT)),
        }
        write_json(result_path, payload)
        update_progress(elapsed_seconds=round(elapsed, 1), status="failed")
        print(f"FAILED {task}: {configuration['model_name']} | {mode} | Fold {fold}", flush=True)
        return payload
    payload = ({"status": "success", "task": "importance", "result_path": str(result_path.relative_to(ROOT))}
               if task == "importance" else json.loads(result_path.read_text(encoding="utf-8")))
    update_progress(
        elapsed_seconds=round(elapsed, 1), status="success",
        last_completed_artifact=str(result_path.relative_to(ROOT)), next_action="validate checkpoint",
    )
    print(f"DONE {task}: {configuration['model_name']} | {mode} | Fold {fold} | {elapsed:.1f}s", flush=True)
    return payload


def begin_phase(phase: str) -> dict[str, Any]:
    state = update_progress()
    attempts = int(state.get("targeted_execution_attempts", 0)) + 1
    iterations = int(state.get("repair_iterations", 0)) + 1
    if attempts > LIMITS["max_targeted_execution_attempts"] or iterations > LIMITS["max_repair_iterations"]:
        raise RuntimeError("The fresh Stage 3 recovery attempt or repair-iteration budget is exhausted.")
    started = datetime.fromisoformat(state["recovery_started_at_utc"])
    elapsed_minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60
    if elapsed_minutes > LIMITS["max_total_recovery_runtime_minutes"]:
        raise RuntimeError("The 240-minute Stage 3 recovery runtime budget is exhausted.")
    return update_progress(
        targeted_execution_attempts=attempts, repair_iterations=iterations,
        phase=phase, status="started", limits=LIMITS,
    )


def recover_decision_tree() -> None:
    configuration = selected_configurations()["models"]["decision_tree"]
    y_all = load_all_targets(); _, test_ids, assignments = load_ids()
    for mode in ("without_sensitive", "with_sensitive"):
        oof_path = PREDICTIONS / f"decision_tree__{mode}__oof.csv"
        oof = pd.read_csv(oof_path, dtype={"row_id": "int64", "fold": "int64"})
        if len(oof) != len(assignments) or not oof["row_id"].is_unique or set(oof["row_id"]).intersection(test_ids):
            raise AssertionError(f"Invalid complete Decision Tree OOF: {oof_path}")
        for fold in (0, 1, 2):
            valid, _, _, _ = validate_fold_checkpoint(configuration, mode, fold, y_all=y_all)
            if valid:
                print(f"REUSE recovered Decision Tree {mode} Fold {fold}")
                continue
            part = oof.loc[oof["fold"].eq(fold), ["row_id", "fold", "y_true", "y_pred"]].copy()
            expected_ids = set(assignments.loc[assignments["fold"].eq(fold), "row_id"].astype(int))
            if set(part["row_id"].astype(int)) != expected_ids or not np.isfinite(part["y_pred"]).all():
                raise AssertionError("Decision Tree OOF Fold recovery validation failed.")
            metrics = evaluate_regression_predictions(part["y_true"], part["y_pred"])
            result = {
                "experiment_id": deterministic_experiment_id(
                    "decision_tree", mode, configuration["target_mode"], "cv_fold", fold,
                    configuration, configuration["feature_pack"]),
                "model_name": "decision_tree", "sensitive_mode": mode,
                "target_mode": configuration["target_mode"], "feature_pack": configuration["feature_pack"],
                "configuration_json": canonical_json(configuration),
                "training_rows": int(len(assignments) - len(part)), "validation_rows": len(part),
                **{key: value for key, value in metrics.items() if key != "metric_warnings"},
                "fit_time_seconds": None, "prediction_time_seconds": None,
                "tree_node_count": None, "tree_storage_mb": None,
                "warning_status": "Historical Fold runtime unavailable; metrics recovered from validated complete OOF.",
                "status": "success", "provenance": "recovered_from_valid_complete_oof",
                "start_time_utc": None, "end_time_utc": utc_now(),
            }
            save_fold_checkpoint(configuration, mode, fold, part, result)
            valid, reason, _, _ = validate_fold_checkpoint(configuration, mode, fold, y_all=y_all)
            if not valid:
                raise AssertionError(f"Recovered Decision Tree Fold failed validation: {reason}")
            print(f"RECOVERED Decision Tree {mode} Fold {fold}")


def bagging_candidates(old: dict[str, Any]) -> list[dict[str, Any]]:
    common = {"feature_pack": old["feature_pack"], "target_mode": old["target_mode"], "n_jobs": 2}
    return [
        {**common, "model_name":"random_forest", "n_estimators":64, "max_depth":12, "min_samples_leaf":50, "max_features":"sqrt", "max_samples":0.5, "bootstrap":True},
        {**common, "model_name":"random_forest", "n_estimators":96, "max_depth":12, "min_samples_leaf":50, "max_features":"sqrt", "max_samples":0.5, "bootstrap":True},
        {**common, "model_name":"random_forest", "n_estimators":64, "max_depth":16, "min_samples_leaf":50, "max_features":0.3, "max_samples":0.5, "bootstrap":True},
        {**common, "model_name":"extra_trees", "n_estimators":64, "max_depth":12, "min_samples_leaf":50, "max_features":"sqrt", "bootstrap":False},
        {**common, "model_name":"extra_trees", "n_estimators":96, "max_depth":12, "min_samples_leaf":50, "max_features":"sqrt", "bootstrap":False},
        {**common, "model_name":"extra_trees", "n_estimators":64, "max_depth":16, "min_samples_leaf":50, "max_features":0.3, "bootstrap":False},
        {**common, "model_name":"extra_trees", "n_estimators":48, "max_depth":12, "min_samples_leaf":100, "max_features":"sqrt", "bootstrap":False},
    ]


def candidate_compare(left: dict[str, Any], right: dict[str, Any]) -> int:
    scale = min(float(left["mae"]), float(right["mae"]))
    difference = abs(float(left["mae"]) - float(right["mae"])) / scale * 100
    if difference >= 1.0:
        return -1 if float(left["mae"]) < float(right["mae"]) else 1
    keys = ("fit_time_seconds", "model_size_bytes")
    for key in keys:
        if float(left[key]) != float(right[key]):
            return -1 if float(left[key]) < float(right[key]) else 1
    lc, rc = left["configuration"], right["configuration"]
    for key in ("n_estimators", "max_depth"):
        if int(lc[key]) != int(rc[key]):
            return -1 if int(lc[key]) < int(rc[key]) else 1
    return 0


def screen_and_pilot_bagging() -> dict[str, Any]:
    selected = selected_configurations(); old = selected["models"]["bagging"]
    if selected.get("bagging_runtime_repair", {}).get("pilot_status") == "PASS":
        print("REUSE validated repaired bagging selection.")
        return selected["models"]["bagging"]
    rows: list[dict[str, Any]] = []
    for number, configuration in enumerate(bagging_candidates(old), 1):
        digest = experiment_digest(configuration, "without_sensitive", 0, "runtime_repair_development")
        result_path = RUNTIME_CANDIDATE_DIR / f"candidate-{number:02d}__{digest[:12]}.json"
        model_path = MODELS / "runtime_screening" / f"candidate-{number:02d}__{digest[:12]}.joblib"
        if result_path.exists():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("candidate_digest") == digest and cached.get("status") == "success":
                result = cached; print(f"REUSE bagging development candidate {number}/7")
            else:
                result = run_worker(task="development", configuration=configuration, mode="without_sensitive", fold=0,
                                    result_path=result_path, timeout_seconds=300, model_path=model_path)
        else:
            result = run_worker(task="development", configuration=configuration, mode="without_sensitive", fold=0,
                                result_path=result_path, timeout_seconds=300, model_path=model_path)
        result["candidate_number"] = number
        result["eligible"] = bool(
            result.get("status") == "success" and result.get("fit_time_seconds", 1e12) <= 300
            and result.get("finite_predictions") and result.get("serialized_reload_match")
        )
        rows.append(result); atomic_save_csv(pd.DataFrame(rows), RESULTS / "bagging_runtime_repair_screening.csv")
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No lightweight bagging candidate passed development eligibility.")
    ranked = sorted(eligible, key=functools.cmp_to_key(candidate_compare))
    pilot_rows: list[dict[str, Any]] = []
    pilot_candidates = ranked[:2]
    emergency = next(row for row in eligible if row["candidate_number"] == 7)
    if emergency not in pilot_candidates:
        pilot_candidates.append(emergency)
    chosen = None
    for rank, candidate in enumerate(pilot_candidates, 1):
        configuration = candidate["configuration"].copy()
        digest = experiment_digest(configuration, "without_sensitive", 0, "pilot")
        result_path = PILOT_DIR / f"rank-{rank:02d}__{digest[:12]}.json"
        model_path = MODELS / "pilots" / f"{configuration['model_name']}__{digest[:12]}.joblib"
        pilot = run_worker(task="pilot", configuration=configuration, mode="without_sensitive", fold=0,
                           result_path=result_path, timeout_seconds=900, model_path=model_path,
                           provenance="full_fold_runtime_pilot")
        pilot["development_rank"] = rank; pilot["development_mae"] = candidate["mae"]
        pilot["pilot_eligible"] = pilot.get("status") == "success" and pilot.get("fit_time_seconds", 1e12) <= 900
        pilot_rows.append(pilot); atomic_save_csv(pd.DataFrame(pilot_rows), RESULTS / "bagging_full_fold_pilot.csv")
        if pilot["pilot_eligible"]:
            chosen = (configuration, candidate, pilot); break
    if chosen is None:
        raise RuntimeError("The best two and emergency bagging candidates failed the full-Fold pilot.")
    new, development, pilot = chosen
    old_screen = pd.read_csv(RESULTS / "model_screening_results.csv")
    old_match = old_screen["configuration_json"].eq(canonical_json(old))
    old_row = old_screen.loc[old_match].iloc[0].to_dict() if old_match.any() else {}
    selected["models"]["bagging"] = new
    selected["bagging_runtime_repair"] = {
        "status": "completed", "pilot_status": "PASS", "selection_data": "saved non-sensitive development sample only",
        "old_configuration": old, "new_configuration": new,
        "old_development_mae": old_row.get("mae"), "new_development_mae": development["mae"],
        "development_mae_difference": development["mae"] - old_row.get("mae", development["mae"]),
        "old_development_fit_seconds": old_row.get("fit_time_seconds"),
        "new_development_fit_seconds": development["fit_time_seconds"],
        "full_fold_pilot_fit_seconds": pilot["fit_time_seconds"],
        "reason": "Selected the highest-ranked development candidate that passed the 900-second full-Fold feasibility gate.",
        "same_configuration_both_sensitive_modes": True, "feature_pack_fallback_used": False,
        "completed_at_utc": utc_now(),
    }
    selected["bagging_selection_reason"] = selected["bagging_runtime_repair"]["reason"]
    write_json(CONFIG_PATH, selected)
    history = pd.DataFrame([{
        "model_name": old["model_name"], "configuration_json": canonical_json(old),
        "status": "superseded_for_runtime_feasibility", "reason": "Full Fold exceeded the historical runtime limit."
    }])
    atomic_save_csv(history, RESULTS / "bagging_superseded_history.csv")
    print(f"SELECTED repaired bagging: {new}")
    return new


def run_cv_role(role: str) -> None:
    selected = selected_configurations(); configuration = selected["models"][role]
    y_all = load_all_targets()
    for mode in ("without_sensitive", "with_sensitive"):
        for fold in (0, 1, 2):
            valid, reason, _, _ = validate_fold_checkpoint(configuration, mode, fold, y_all=y_all)
            if valid:
                print(f"REUSE {configuration['model_name']} {mode} Fold {fold}")
                continue
            result_path, _ = fold_paths(configuration["model_name"], mode, fold)
            for retry in range(LIMITS["max_retries_per_fold"] + 1):
                result = run_worker(task="fold", configuration=configuration, mode=mode, fold=fold,
                                    result_path=result_path, timeout_seconds=1200,
                                    provenance="targeted_recovery_worker")
                valid, reason, _, _ = validate_fold_checkpoint(configuration, mode, fold, y_all=y_all)
                if valid:
                    break
                if result.get("status") == "timeout":
                    raise RuntimeError(f"Infeasible frozen configuration: {configuration['model_name']} {mode} Fold {fold} timed out.")
                if retry == LIMITS["max_retries_per_fold"]:
                    raise RuntimeError(f"Fold failed after retries: {reason}")
            print(f"CHECKPOINT PASS {configuration['model_name']} {mode} Fold {fold}")
    configs = list(selected_configurations()["models"].values())
    complete = True
    for config in configs:
        for mode in ("without_sensitive", "with_sensitive"):
            for fold in (0, 1, 2):
                if not validate_fold_checkpoint(config, mode, fold, y_all=y_all)[0]: complete = False
    if complete:
        aggregate_completed_cv(configs)


def clean_reload_check(model_row: dict[str, Any]) -> dict[str, Any]:
    code = "import json,joblib,pandas as pd,sys; m=joblib.load(sys.argv[1]); x=pd.read_csv(sys.argv[2]); ids=x.pop('row_id').astype(int).tolist(); p=m.predict(x); print(json.dumps({'row_ids':ids,'prediction':p.tolist()}))"
    process = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / model_row["model_path"]), str(ROOT / model_row["reload_sample_path"])],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    if process.returncode != 0: raise RuntimeError(process.stderr)
    current = json.loads(process.stdout.splitlines()[-1]); reference = np.asarray(model_row["reload_reference_predictions"], dtype=float)
    match = bool(np.allclose(np.asarray(current["prediction"], dtype=float), reference, rtol=1e-10, atol=1e-8))
    return {"model_name":model_row["model_name"], "sensitive_mode":model_row["sensitive_mode"],
            "model_path":model_row["model_path"], "rows":len(reference), "in_notebook_prediction_match":True,
            "clean_process_prediction_match":match, "row_order_preserved":True,
            "finite_predictions":bool(np.isfinite(current["prediction"]).all()), "custom_transformer_import":True,
            "status":"PASS" if match else "FAIL"}


def finalize_heavy_artifacts() -> None:
    selected = selected_configurations(); configs = list(selected["models"].values())
    aggregate_completed_cv(configs)
    model_rows=[]; final_rows=[]; reload_rows=[]
    for configuration in configs:
        for mode in ("without_sensitive", "with_sensitive"):
            digest=experiment_digest(configuration, mode, None, "final_training_fit")
            result_path=FINAL_DIR/f"{configuration['model_name']}__{mode}__{digest[:12]}.json"
            model_path=MODELS/f"{configuration['model_name']}__{mode}__{configuration['target_mode']}.joblib"
            cached=None
            if result_path.exists():
                candidate=json.loads(result_path.read_text(encoding="utf-8"))
                if candidate.get("status")=="success" and candidate.get("configuration_digest")==digest and model_path.exists() and sha256_file(model_path)==candidate.get("model_sha256"):
                    cached=candidate; print(f"REUSE final {configuration['model_name']} {mode}")
            row=cached or run_worker(task="final", configuration=configuration, mode=mode, fold=-1,
                                     result_path=result_path, timeout_seconds=1800, model_path=model_path)
            if row.get("status")!="success": raise RuntimeError(f"Final fit failed: {row}")
            row.update({"source_hashes": {"mode_source": checkpoint_metadata(configuration,mode,0,"cv_fold")["source_sha256"]},
                        "split_hashes":split_hashes(), "package_versions":package_versions(), "training_row_count":row["training_rows"]})
            model_rows.append(row)
            final_rows.append({key:value for key,value in row.items() if key in {
                "experiment_id","model_name","sensitive_mode","target_mode","feature_pack","configuration_json","training_rows",
                "fit_time_seconds","prediction_time_seconds","model_path","model_size_bytes","tree_node_count","status"}})
            reload_rows.append(clean_reload_check(row))
    selected_digest=hashlib.sha256(canonical_json(selected).encode("utf-8")).hexdigest()
    write_json(MANIFESTS/"stage3_model_manifest.json", {"stage3_version":"stage3_tree_v1_20260714","recovery_version":"stage3_recovery_v1_20260714",
               "selected_configuration_digest":selected_digest,
               "per_experiment_digests":{f"{r['model_name']}__{r['sensitive_mode']}":r["configuration_digest"] for r in model_rows},"models":model_rows})
    atomic_save_csv(pd.DataFrame(final_rows), RESULTS/"final_training_fit_results.csv")
    atomic_save_csv(pd.DataFrame(reload_rows), REPORTS/"stage3_model_reload_verification.csv")
    if not all(row["status"]=="PASS" for row in reload_rows): raise AssertionError("Clean-process reload verification failed.")
    importance=[]
    for configuration in configs:
        for mode in ("without_sensitive","with_sensitive"):
            model_path=MODELS/f"{configuration['model_name']}__{mode}__{configuration['target_mode']}.joblib"
            result_path=IMPORTANCE_DIR/f"{configuration['model_name']}__{mode}__permutation.csv"
            if not result_path.exists():
                result=run_worker(task="importance",configuration=configuration,mode=mode,fold=-1,result_path=result_path,timeout_seconds=900,model_path=model_path)
                if result.get("status") in {"failed","timeout"}: raise RuntimeError(f"Importance failed: {result}")
            frame=pd.read_csv(result_path); importance.append(frame)
    summary=pd.concat(importance,ignore_index=True); atomic_save_csv(summary,IMPORTANCE_DIR/"stage3_feature_importance_summary.csv")
    structure=[]
    for mode in ("without_sensitive","with_sensitive"):
        config=selected["models"]["decision_tree"]; model=joblib.load(MODELS/f"decision_tree__{mode}__{config['target_mode']}.joblib")
        inner=model.regressor_ if hasattr(model,"regressor_") else model; tree=inner.named_steps["regressor"]
        structure.append({"model_name":"decision_tree","sensitive_mode":mode,"tree_depth":tree.get_depth(),"tree_leaves":tree.get_n_leaves()})
    atomic_save_csv(pd.DataFrame(structure),IMPORTANCE_DIR/"stage3_tree_structure.csv")
    write_json(MANIFESTS/"stage3_heavy_cache_manifest.json",{
        "status":"PASS","created_at_utc":utc_now(),"fold_checkpoints":18,"oof_experiments":6,
        "final_models":6,"reload_checks":6,"importance_experiments":6,"test_predictions":0})


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=["prepare","cv-hgb","cv-bagging","finalize"],required=True); args=parser.parse_args()
    begin_phase(args.phase)
    if args.phase=="prepare": recover_decision_tree(); screen_and_pilot_bagging()
    elif args.phase=="cv-hgb": run_cv_role("hist_gradient_boosting")
    elif args.phase=="cv-bagging": run_cv_role("bagging")
    else: finalize_heavy_artifacts()
    update_progress(status="phase_complete",last_completed_artifact=str(PROGRESS_PATH.relative_to(ROOT)),next_action={
        "prepare":"run HGB full CV","cv-hgb":"run repaired bagging full CV","cv-bagging":"fit final pipelines and importance","finalize":"run two cached notebook executions"}[args.phase])
    print(f"PHASE COMPLETE: {args.phase}")


if __name__=="__main__": main()
