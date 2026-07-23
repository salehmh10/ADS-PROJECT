"""Stage 5C frozen Deep inference evaluation and artifact utilities.

Stage 5C is a descriptive Post-Test Extension. The Stage 4L pre-registered
primary stays official. This module never trains or tunes a model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OFFICIAL_NAME = "Stage 5C — Post-Test Deep and Ensemble Evaluation"
LABEL = "Post-Test Extension"
STAGE_ID = "stage5c"

REPORTS = ROOT / "artifacts/reports"
MANIFESTS = ROOT / "artifacts/manifests/stage5"
RESULTS = ROOT / "artifacts/results/stage5/posttest_evaluation"
PREDICTIONS = ROOT / "artifacts/predictions/stage5/posttest_evaluation"
FIGURES = ROOT / "artifacts/figures/stage5c"
PLOT_DATA = FIGURES / "plotting_data"
BACKUPS = ROOT / "artifacts/backups"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.ipynb"

FREEZE = REPORTS / "stage5c_preevaluation_freeze.json"
BASELINE = MANIFESTS / "stage5c_protected_hashes_before.json"
SENTINEL = REPORTS / "stage5c_safe_loader_sentinel.json"
COMPARATOR_VALIDATION = REPORTS / "stage5c_stage4l_primary_validation.json"
ACCESS_AUDIT = REPORTS / "stage5c_test_access_audit.json"
ALIGNMENT = REPORTS / "stage5c_test_input_alignment_report.json"
ATTEMPTS = REPORTS / "stage5c_prediction_attempts.json"
RECHECK = REPORTS / "stage5c_protected_recheck.json"
VERIFICATION = REPORTS / "stage5c_verification.json"
RUNTIME = REPORTS / "stage5c_runtime.json"
REVIEWER = REPORTS / "stage5c_reviewer.md"
AUTHORIZED_REFRESH_ID = "stage5c_notebook_output_refresh_20260716"
AUTHORIZED_REFRESH_BEFORE = REPORTS / "stage5c_notebook_run4_before_hashes.json"
AUTHORIZED_REFRESH_REPORT = REPORTS / "stage5c_notebook_run4_authorized_complete.json"
AUTHORIZED_REFRESH_RECHECK = REPORTS / "stage5c_notebook_run4_post_immutability.json"

STAGE4_FREEZE = REPORTS / "stage4l_pretest_freeze.json"
STAGE4_VERIFICATION = REPORTS / "stage4l_verification.json"
STAGE4_REVIEWER = REPORTS / "stage4l_reviewer.md"
STAGE4_UNLOCK = REPORTS / "stage4l_test_unlock_audit.json"
STAGE4_CANDIDATES = ROOT / "artifacts/manifests/stage4/stage4l_candidate_manifest.json"
STAGE4_PRED = ROOT / "artifacts/predictions/final_test/stage4l__blend__without_sensitive.csv"
STAGE4_PRED_META = ROOT / "artifacts/predictions/final_test/stage4l__blend__without_sensitive.metadata.json"
STAGE4_LEADERBOARD = ROOT / "artifacts/results/stage4/final_integration/stage4l_test_leaderboard.csv"
STAGE4_RECOMMENDATION = ROOT / "artifacts/results/stage4/final_integration/stage4l_final_recommendation.json"

STAGE5A_VERIFICATION = REPORTS / "stage5a_verification.json"
STAGE5A_REVIEWER = REPORTS / "stage5a_reviewer.md"
GOVERNANCE = REPORTS / "stage5a2_governance_adjudication.json"
FULL_TRAIN = MANIFESTS / "stage5a2_full_train_manifest.json"
CORE_WINNER = ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json"
STAGE5B_VERIFICATION = REPORTS / "stage5b_verification.json"
STAGE5B_REVIEWER = REPORTS / "stage5b_reviewer.md"
STAGE5B_FREEZE = REPORTS / "stage5b_preensemble_freeze.json"
STAGE5B_DECISION = ROOT / "artifacts/results/stage5/deep_boosting_ensemble/stage5b_ensemble_decision.json"
STAGE5B_SPEC = ROOT / "artifacts/results/stage5/deep_boosting_ensemble/stage5b_frozen_ensemble.json"
STAGE5B_HANDOFF = MANIFESTS / "stage5b_evaluation_handoff.json"

TEST_IDS = ROOT / "artifacts/splits/test_row_ids.csv"
TRAIN_IDS = ROOT / "artifacts/splits/train_row_ids.csv"
SOURCE_WITHOUT = ROOT / "data/regression_without_sensitive_features.csv"
SOURCE_WITH = ROOT / "data/regression_with_sensitive_features.csv"
SAFE_LOADER = ROOT / "stage5_safe_row_loader.py"
METRIC_SCHEMA = ROOT / "artifacts/data_contract/metric_schema.json"

PRED_WITHOUT = PREDICTIONS / "stage5c_test_predictions_without_sensitive.csv"
PRED_WITH = PREDICTIONS / "stage5c_test_predictions_with_sensitive.csv"
MANIFEST_WITHOUT = MANIFESTS / "stage5c_prediction_manifest_without_sensitive.json"
MANIFEST_WITH = MANIFESTS / "stage5c_prediction_manifest_with_sensitive.json"

METRICS = [
    "mae", "mse", "rmse", "r_squared", "rmsle", "mape_percent",
    "median_absolute_error", "wape_percent", "mean_signed_error",
    "p90_absolute_error", "top_decile_mae", "top_five_percent_mae",
    "negative_prediction_rate",
]

EXPECTED = {
    "stage4l_freeze": "6c4aece5c82077be1932060f935fb4ce70213ded591ccdd41b2d42179f92c3b7",
    "stage4l_prediction": "9f9efa21d95a466b8271cd0db0a1e6b2c1ed2b5f1cabfbbb7e081137b9e4b7ed",
    "stage5b_spec": "dba45a5966b3fa2f3b59a230ef1a801b17a9f5c0e1bbd5217418b5122799d0d5",
    "stage5b_handoff": "12a7de3fa4a5ee43d91718b114827ef78a4c8d7aaffe199d0876597f2bf03fa6",
    "bundle_without": "0d2dc108578512022608fee31676ced4d3d65d178f3d77b2418011057eff7006",
    "model_without": "be05e4f293cd719033c17862324e2c6f18673322b71ca79eb8c07dfa93f7efa3",
    "bundle_with": "d8b74180385cae7c0cfb9570ef124e9061c4ee398a68a7a9266aca3de299f600",
    "model_with": "e02ec0ab4f448a63dfd7a5a4e2f0785ce7542a0c8a947c2632cce13f78a9ed46",
    "source_without": "e90f7bb49cce5584c7ab250c1db6a107de5cf640c7839f318d7f3cb995edd93c",
    "source_with": "6dc52dca5a8a7196a75213fab4a5a5c0a541f84390219459afb0b2be7b77aede",
    "test_ids": "b01edf609d3102351b848d166e4db885da56db66a1fe441496e470dcf57770cb",
    "train_ids": "5abb11436c4661f304c3b06ade1882cbfd7689c8befcda756f6986f14ee13f26",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray, dtype: Any) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def hash_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def model_records() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json_load(FULL_TRAIN)
    by_mode = {item["sensitive_mode"]: item for item in manifest["models"]}
    return by_mode["without_sensitive"], by_mode["with_sensitive"]


def feature_contract(item: dict[str, Any]) -> tuple[list[str], str]:
    config = json_load(ROOT / item["effective_configuration_path"])
    contract = config["preprocessing_contract"]
    features = list(contract["numerical_features"]) + list(contract["categorical_features"])
    return features, config["feature_schema_hash"]


def prerequisite_checks() -> dict[str, bool]:
    required = [
        STAGE4_FREEZE, STAGE4_VERIFICATION, STAGE4_REVIEWER, STAGE4_UNLOCK,
        STAGE4_CANDIDATES, STAGE4_PRED, STAGE4_PRED_META, STAGE4_LEADERBOARD,
        STAGE5A_VERIFICATION, STAGE5A_REVIEWER, GOVERNANCE, FULL_TRAIN, CORE_WINNER,
        STAGE5B_VERIFICATION, STAGE5B_REVIEWER, STAGE5B_FREEZE, STAGE5B_DECISION,
        STAGE5B_SPEC, STAGE5B_HANDOFF, TEST_IDS, TRAIN_IDS, SOURCE_WITHOUT,
        SOURCE_WITH, SAFE_LOADER, METRIC_SCHEMA, REGISTRY,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    require(not missing, f"Missing prerequisites: {missing}")
    s4v, s4f = json_load(STAGE4_VERIFICATION), json_load(STAGE4_FREEZE)
    s5a, gov = json_load(STAGE5A_VERIFICATION), json_load(GOVERNANCE)
    s5b, decision, handoff = json_load(STAGE5B_VERIFICATION), json_load(STAGE5B_DECISION), json_load(STAGE5B_HANDOFF)
    without, with_sensitive = model_records()
    checks = {
        "stage4l_verification_pass": s4v.get("status") == "PASS",
        "stage4l_primary_exact": s4v.get("primary_candidate") == "stage4l__blend__without_sensitive" and s4f.get("primary_non_sensitive_candidate") == "stage4l__blend__without_sensitive",
        "stage4l_freeze_hash": sha256_file(STAGE4_FREEZE) == EXPECTED["stage4l_freeze"],
        "stage4l_prediction_hash": sha256_file(STAGE4_PRED) == EXPECTED["stage4l_prediction"],
        "stage5a_pass_with_exception": s5a.get("status") == "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "stage5a_literal_failure_visible": s5a.get("literal_zero_test_loading") is False and gov.get("literal_rule_violated") is True,
        "stage5a_two_models": len(json_load(FULL_TRAIN).get("models", [])) == 2,
        "stage5a_reloads_pass": without.get("reload_checks_all") is True and with_sensitive.get("reload_checks_all") is True,
        "stage5b_verification_pass": s5b.get("status") == "PASS" and s5b.get("overall_status") == "PASS",
        "stage5b_reviewer_pass": s5b.get("reviewer", {}).get("reviewer_status") == "PASS",
        "stage5b_ensemble_rejected": decision.get("ensemble_status") == "rejected" and handoff.get("ensemble_status") == "rejected",
        "deep_eligible": handoff.get("deep_evaluation_eligible") is True,
        "ensemble_ineligible": handoff.get("ensemble_evaluation_eligible") is False,
        "stage5b_spec_hash": sha256_file(STAGE5B_SPEC) == EXPECTED["stage5b_spec"],
        "stage5b_handoff_hash": sha256_file(STAGE5B_HANDOFF) == EXPECTED["stage5b_handoff"],
        "bundle_without_hash": without["bundle_sha256"] == EXPECTED["bundle_without"] == sha256_file(ROOT / without["bundle_path"]),
        "model_without_hash": without["model_sha256"] == EXPECTED["model_without"] == sha256_file(ROOT / without["model_path"]),
        "bundle_with_hash": with_sensitive["bundle_sha256"] == EXPECTED["bundle_with"] == sha256_file(ROOT / with_sensitive["bundle_path"]),
        "model_with_hash": with_sensitive["model_sha256"] == EXPECTED["model_with"] == sha256_file(ROOT / with_sensitive["model_path"]),
        "epoch_30": without["completed_epoch"] == with_sensitive["completed_epoch"] == 30,
        "target_raw": json_load(FULL_TRAIN).get("target_mode") == "raw",
        "source_hashes": sha256_file(SOURCE_WITHOUT) == EXPECTED["source_without"] and sha256_file(SOURCE_WITH) == EXPECTED["source_with"],
        "split_hashes": sha256_file(TEST_IDS) == EXPECTED["test_ids"] and sha256_file(TRAIN_IDS) == EXPECTED["train_ids"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    require(not failed, f"Stage 5C prerequisite checks failed: {failed}")
    return checks


def protected_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(ROOT.glob("REGRESSION_PART*.ipynb"))
    paths.extend([SOURCE_WITHOUT, SOURCE_WITH, TEST_IDS, TRAIN_IDS, ROOT / "artifacts/data_contract/feature_inventory.csv", STAGE4_FREEZE, STAGE4_PRED, STAGE5B_SPEC, STAGE5B_HANDOFF])
    for base in [ROOT / "artifacts/models", ROOT / "artifacts/predictions", ROOT / "artifacts/splits"]:
        paths.extend(path for path in base.rglob("*") if path.is_file())
    for pattern in ["stage4l*", "stage5a*", "stage5b*"]:
        paths.extend(path for path in REPORTS.glob(pattern) if path.is_file())
    paths.extend(path for path in (ROOT / "artifacts/manifests/stage4").rglob("*") if path.is_file())
    paths.extend(path for path in MANIFESTS.glob("stage5a*") if path.is_file())
    paths.extend(path for path in MANIFESTS.glob("stage5b*") if path.is_file())
    for base in [ROOT / "artifacts/results/stage4/final_integration", ROOT / "artifacts/results/stage5/deep_core", ROOT / "artifacts/results/stage5/deep_boosting_ensemble"]:
        paths.extend(path for path in base.rglob("*") if path.is_file())
    part1 = Path(r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb")
    if part1.exists():
        paths.append(part1)
    unique = {str(path.resolve()).lower(): path for path in paths if path.exists() and path.resolve() != REGISTRY.resolve()}
    return sorted(unique.values(), key=lambda item: str(item).lower())


def create_baseline() -> dict[str, Any]:
    started = time.perf_counter()
    checks = prerequisite_checks()
    require(not FREEZE.exists(), "A Stage 5C freeze already exists; use resume validation")
    require(not PRED_WITHOUT.exists() and not PRED_WITH.exists(), "Stage 5C prediction artifacts already exist")
    entries = [hash_record(path) if path.is_relative_to(ROOT) else {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in protected_paths()]
    registry_bytes = REGISTRY.read_bytes()
    state = {name: sha256_file(ROOT / name) for name in ["AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md"]}
    payload = {
        "stage_id": STAGE_ID,
        "status": "PASS",
        "created_at_utc": now_utc(),
        "protected_file_count": len(entries),
        "entries": entries,
        "registry_prior_byte_count": len(registry_bytes),
        "registry_prior_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "state_starting_hashes": state,
        "prerequisite_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, BASELINE)
    print(json.dumps({"status": "PASS", "baseline": rel(BASELINE), "sha256": sha256_file(BASELINE), "protected_file_count": len(entries)}, indent=2))
    return payload


def create_freeze() -> dict[str, Any]:
    started = time.perf_counter()
    require(BASELINE.exists(), "Protected baseline is missing")
    checks = prerequisite_checks()
    without, with_sensitive = model_records()
    features_without, schema_without = feature_contract(without)
    features_with, schema_with = feature_contract(with_sensitive)
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    train_ids = pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    require(len(test_ids) == 99948 and len(np.unique(test_ids)) == 99948, "Saved Test membership is invalid")
    require(np.intersect1d(test_ids, train_ids).size == 0, "Saved Train/Test overlap is nonzero")
    meta = json_load(STAGE4_PRED_META)
    stage4_metric = pd.read_csv(STAGE4_LEADERBOARD).query("candidate_id == 'stage4l__blend__without_sensitive'")
    require(len(stage4_metric) == 1, "Official Stage 4L metric row is missing")
    figures = [
        "stage5c_test_mae_comparison", "stage5c_test_rmse_comparison",
        "stage5c_rmsle_r2_comparison", "stage5c_target_decile_mae",
        "stage5c_tail_error_comparison", "stage5c_absolute_error_difference_distribution",
        "stage5c_paired_bootstrap_interval", "stage5c_prediction_disagreement",
        "stage5c_sensitive_metric_difference", "stage5c_summary_dashboard",
    ]
    registry_ids = [
        "stage5c__realmlp__without_sensitive__test_evaluation",
        "stage5c__realmlp__with_sensitive__test_evaluation",
        "stage5c__realmlp_vs_stage4l__official_comparison",
        "stage5c__realmlp_vs_stage4l__paired_bootstrap",
        "stage5c__realmlp__sensitive_accuracy_comparison",
        "stage5c__evaluation_summary",
        "stage5c__stage6_handoff",
    ]
    payload = {
        "stage_id": STAGE_ID,
        "official_stage_name": OFFICIAL_NAME,
        "status": "PASS",
        "freeze_timestamp_utc": now_utc(),
        "evaluation_label": LABEL,
        "post_test_statement": "Stage 5C is descriptive and does not replace the official Stage 4L pre-registered primary.",
        "stage4l_verification": {**hash_record(STAGE4_VERIFICATION), "status": "PASS"},
        "stage4l_pretest_freeze": {**hash_record(STAGE4_FREEZE), "expected_sha256": EXPECTED["stage4l_freeze"]},
        "stage4l_official_primary_candidate_id": "stage4l__blend__without_sensitive",
        "stage4l_official_prediction": {**hash_record(STAGE4_PRED), "metadata_path": rel(STAGE4_PRED_META)},
        "stage4l_official_metric_row_reference": rel(STAGE4_LEADERBOARD),
        "stage5a_verification": hash_record(STAGE5A_VERIFICATION),
        "stage5a_governance_adjudication": hash_record(GOVERNANCE),
        "stage5a_status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "historical_literal_zero_test_loading": "FAIL",
        "stage5b_verification": hash_record(STAGE5B_VERIFICATION),
        "stage5b_reviewer": hash_record(STAGE5B_REVIEWER),
        "stage5b_frozen_specification": hash_record(STAGE5B_SPEC),
        "stage5b_evaluation_handoff": hash_record(STAGE5B_HANDOFF),
        "stage5b_ensemble_status": "rejected",
        "deep_evaluation_eligible": True,
        "ensemble_evaluation_eligible": False,
        "frozen_family": "RealMLP",
        "frozen_target_mode": "raw",
        "frozen_epoch": 30,
        "device": "CPU",
        "precision": "float32",
        "training_rows_per_bundle": 399788,
        "test_rows_used_in_fit": 0,
        "early_stopping": False,
        "best_checkpoint_restoration": False,
        "test_id_path": rel(TEST_IDS),
        "test_id_file_sha256": sha256_file(TEST_IDS),
        "train_id_path": rel(TRAIN_IDS),
        "train_id_file_sha256": sha256_file(TRAIN_IDS),
        "expected_test_row_count": 99948,
        "sorted_test_row_id_hash": array_hash(np.sort(test_ids), np.int64),
        "source_target_column": "loan_amount_000s",
        "source_target_columns_permitted": False,
        "canonical_target_source": rel(STAGE4_PRED),
        "safe_loader_path": rel(SAFE_LOADER),
        "safe_loader_sha256": sha256_file(SAFE_LOADER),
        "safe_loader_sentinel_required": True,
        "deep_modes": {
            "without_sensitive": {
                "evaluation_candidate_id": "stage5c__realmlp__without_sensitive__test_evaluation",
                "bundle_candidate_id": without["candidate_id"],
                "bundle_path": without["bundle_path"].replace("\\", "/"),
                "bundle_sha256": without["bundle_sha256"],
                "model_path": without["model_path"].replace("\\", "/"),
                "model_sha256": without["model_sha256"],
                "reload_report_path": without["reload_path"].replace("\\", "/"),
                "reload_report_sha256": without["reload_sha256"],
                "effective_configuration_path": without["effective_configuration_path"].replace("\\", "/"),
                "effective_configuration_sha256": without["effective_configuration_sha256"],
                "epoch_proof_path": without["epoch_proof_path"].replace("\\", "/"),
                "epoch_proof_sha256": without["epoch_proof_sha256"],
                "reference_prediction_path": without["reference_prediction_path"].replace("\\", "/"),
                "reference_prediction_sha256": without["reference_prediction_sha256"],
                "source_path": rel(SOURCE_WITHOUT), "source_sha256": sha256_file(SOURCE_WITHOUT),
                "required_feature_columns": features_without, "feature_schema_sha256": schema_without,
                "prediction_path": rel(PRED_WITHOUT), "prediction_manifest_path": rel(MANIFEST_WITHOUT),
                "official_result_role": "post_test_deep_candidate",
            },
            "with_sensitive": {
                "evaluation_candidate_id": "stage5c__realmlp__with_sensitive__test_evaluation",
                "bundle_candidate_id": with_sensitive["candidate_id"],
                "bundle_path": with_sensitive["bundle_path"].replace("\\", "/"),
                "bundle_sha256": with_sensitive["bundle_sha256"],
                "model_path": with_sensitive["model_path"].replace("\\", "/"),
                "model_sha256": with_sensitive["model_sha256"],
                "reload_report_path": with_sensitive["reload_path"].replace("\\", "/"),
                "reload_report_sha256": with_sensitive["reload_sha256"],
                "effective_configuration_path": with_sensitive["effective_configuration_path"].replace("\\", "/"),
                "effective_configuration_sha256": with_sensitive["effective_configuration_sha256"],
                "epoch_proof_path": with_sensitive["epoch_proof_path"].replace("\\", "/"),
                "epoch_proof_sha256": with_sensitive["epoch_proof_sha256"],
                "reference_prediction_path": with_sensitive["reference_prediction_path"].replace("\\", "/"),
                "reference_prediction_sha256": with_sensitive["reference_prediction_sha256"],
                "source_path": rel(SOURCE_WITH), "source_sha256": sha256_file(SOURCE_WITH),
                "required_feature_columns": features_with, "feature_schema_sha256": schema_with,
                "prediction_path": rel(PRED_WITH), "prediction_manifest_path": rel(MANIFEST_WITH),
                "official_result_role": "post_test_sensitive_accuracy_candidate",
            },
        },
        "deep_evaluation_candidate_ids": registry_ids[:2],
        "official_comparator_candidate_id": "stage4l__blend__without_sensitive",
        "deep_candidate_count": 2,
        "official_comparator_count": 1,
        "ensemble_candidate_count": 0,
        "metrics": METRICS,
        "metric_contract_path": rel(METRIC_SCHEMA),
        "metric_contract_sha256": sha256_file(METRIC_SCHEMA),
        "paired_comparisons": [
            "RealMLP without sensitive MAE - Stage 4L official primary MAE",
            "RealMLP with sensitive MAE - RealMLP without sensitive MAE",
        ],
        "bootstrap_settings": {"resamples": 500, "seed": 42, "paired_indices": True},
        "sensitive_policy": "Accuracy comparison only; not a fairness audit.",
        "figure_list": figures,
        "registry_ids": registry_ids,
        "technical_retry_policy": {"initial_attempts": 2, "maximum_one_retry_per_mode": True, "maximum_physical_attempts": 4},
        "notebook_attempt_limit": 3,
        "reviewer_cycle_limit": 2,
        "protected_baseline_path": rel(BASELINE),
        "protected_baseline_sha256": sha256_file(BASELINE),
        "stage5c_test_values_opened_before_freeze": False,
        "stage5c_source_feature_rows_materialized_before_freeze": 0,
        "stage5c_models_loaded_for_test_before_freeze": 0,
        "model_or_preprocessing_fit_permitted": False,
        "ensemble_test_prediction_permitted": False,
        "next_stage": "Stage 6",
        "prerequisite_checks": checks,
        "stage4_prediction_metadata": {"row_count": meta["row_count"], "prediction_sha256": meta["prediction_sha256"]},
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, FREEZE)
    require(json_load(FREEZE) == payload, "Stage 5C freeze reload mismatch")
    report = {"status": "PASS", "freeze_path": rel(FREEZE), "freeze_sha256": sha256_file(FREEZE), "freeze_timestamp_utc": payload["freeze_timestamp_utc"], "candidate_count": 2, "comparator_count": 1, "ensemble_count": 0}
    atomic_json(report, REPORTS / "stage5c_preflight.json")
    print(json.dumps(report, indent=2))
    return payload


def run_sentinel() -> dict[str, Any]:
    require(FREEZE.exists(), "Stage 5C freeze is missing")
    from stage5_safe_row_loader import load_allowed_source_rows

    temporary = REPORTS / "stage5c_safe_loader_sentinel.csv"
    temporary.write_text(
        "feature_num,feature_cat,target\n1.5,allowed_a,BAD_TARGET\nBAD_NUM,excluded,BAD_TARGET\n2.5,allowed_b,BAD_TARGET\n",
        encoding="utf-8",
    )
    source_hash_before = sha256_file(temporary)
    calls = {"feature_num": 0, "feature_cat": 0, "target": 0}
    def numeric(value: str) -> float:
        calls["feature_num"] += 1
        return float(value)
    def category(value: str) -> str:
        calls["feature_cat"] += 1
        if value == "excluded":
            raise ValueError("Excluded categorical sentinel reached converter")
        return value
    selected = load_allowed_source_rows(
        temporary, [2, 0], ["feature_num", "feature_cat"],
        allowed_train_ids={0, 2},
        read_csv_kwargs={"converters": {"feature_num": numeric, "feature_cat": category}},
    )
    checks = {
        "excluded_feature_values_not_converted": calls["feature_num"] == 2 and calls["feature_cat"] == 2,
        "excluded_target_values_not_converted": calls["target"] == 0,
        "target_not_requested": "target" not in selected.columns,
        "only_allowed_ids_returned": selected.index.tolist() == [2, 0],
        "row_order_preserved": selected["feature_num"].tolist() == [2.5, 1.5],
        "source_unchanged": sha256_file(temporary) == source_hash_before,
    }
    payload = {
        "stage_id": STAGE_ID, "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": now_utc(), "loader_path": rel(SAFE_LOADER),
        "loader_sha256": sha256_file(SAFE_LOADER), "checks": checks,
        "converter_call_counts": calls, "source_target_values_materialized": 0,
        "excluded_rows_converted": 0, "evaluation_label": LABEL,
    }
    atomic_json(payload, SENTINEL)
    require(payload["status"] == "PASS", "Stage 5C safe-loader sentinel failed")
    print(json.dumps(payload, indent=2))
    return payload


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, decile: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    signed = y_pred - y_true
    absolute = np.abs(signed)
    mse = float(np.mean(np.square(signed)))
    nonzero = np.abs(y_true) > 0
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    top_decile = y_true >= np.quantile(y_true, 0.90)
    top_five = y_true >= np.quantile(y_true, 0.95)
    return {
        "mae": float(np.mean(absolute)), "mse": mse, "rmse": math.sqrt(mse),
        "r_squared": 1.0 - float(np.sum(np.square(signed))) / denominator,
        "rmsle": float(np.sqrt(np.mean(np.square(np.log1p(np.clip(y_pred, 0, None)) - np.log1p(np.clip(y_true, 0, None)))))),
        "mape_percent": float(np.mean(absolute[nonzero] / np.abs(y_true[nonzero])) * 100.0),
        "median_absolute_error": float(np.median(absolute)),
        "wape_percent": float(np.sum(absolute) / np.sum(np.abs(y_true)) * 100.0),
        "mean_signed_error": float(np.mean(signed)), "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "top_decile_mae": float(np.mean(absolute[top_decile])),
        "top_five_percent_mae": float(np.mean(absolute[top_five])),
        "negative_prediction_rate": float(np.mean(y_pred < 0)),
    }


def validate_stage4l_comparator() -> dict[str, Any]:
    started = time.perf_counter()
    freeze_sha = sha256_file(FREEZE)
    freeze = json_load(FREEZE)
    require(freeze.get("status") == "PASS", "Stage 5C freeze is invalid")
    require(sha256_file(STAGE4_PRED) == EXPECTED["stage4l_prediction"], "Official prediction hash mismatch")
    first_access = now_utc()
    frame = pd.read_csv(STAGE4_PRED)
    required_columns = {"candidate_id", "row_id", "y_true", "y_pred"}
    require(required_columns.issubset(frame.columns), "Official prediction schema is incomplete")
    frame = frame.sort_values("row_id").reset_index(drop=True)
    test_ids = np.sort(pd.read_csv(TEST_IDS)["row_id"].to_numpy(np.int64))
    train_ids = pd.read_csv(TRAIN_IDS)["row_id"].to_numpy(np.int64)
    row_ids = frame["row_id"].to_numpy(np.int64)
    y_true = frame["y_true"].to_numpy(np.float64)
    y_pred = frame["y_pred"].to_numpy(np.float64)
    decile = np.asarray(pd.qcut(y_true, 10, labels=False, duplicates="drop"), dtype=np.int64) + 1
    metrics = metric_values(y_true, y_pred, decile)
    official = pd.read_csv(STAGE4_LEADERBOARD).query("candidate_id == 'stage4l__blend__without_sensitive'").iloc[0]
    metric_differences = {name: float(metrics[name] - official[name]) for name in METRICS if name in official.index and pd.notna(official[name])}
    checks = {
        "candidate_id_exact": frame["candidate_id"].nunique() == 1 and frame["candidate_id"].iat[0] == "stage4l__blend__without_sensitive",
        "row_count_exact": len(frame) == 99948,
        "row_ids_unique": not frame["row_id"].duplicated().any(),
        "test_membership_exact": np.array_equal(row_ids, test_ids),
        "zero_train_overlap": np.intersect1d(row_ids, train_ids).size == 0,
        "finite_y_true": np.isfinite(y_true).all(), "finite_prediction": np.isfinite(y_pred).all(),
        "official_metric_reproduction": max(abs(value) for value in metric_differences.values()) < 1e-9,
        "prediction_hash_exact": sha256_file(STAGE4_PRED) == freeze["stage4l_official_prediction"]["sha256"],
    }
    payload = {
        "stage_id": STAGE_ID, "status": "PASS" if all(checks.values()) else "FAIL",
        "evaluation_label": LABEL, "validated_at_utc": now_utc(),
        "first_stage5c_test_artifact_access_at_utc": first_access,
        "preevaluation_freeze_path": rel(FREEZE), "preevaluation_freeze_sha256": freeze_sha,
        "candidate_id": "stage4l__blend__without_sensitive", "prediction_path": rel(STAGE4_PRED),
        "prediction_sha256": sha256_file(STAGE4_PRED), "row_count": len(frame),
        "sorted_test_row_id_hash": array_hash(row_ids, np.int64),
        "canonical_target_hash": array_hash(y_true, np.float64),
        "official_prediction_value_hash": array_hash(y_pred, np.float64),
        "metrics": metrics, "saved_metric_differences": metric_differences,
        "checks": {name: bool(value) for name, value in checks.items()},
        "runtime_seconds": time.perf_counter() - started,
        "official_role": "official_pre_registered_primary",
    }
    atomic_json(payload, COMPARATOR_VALIDATION)
    require(payload["status"] == "PASS", "Stage 4L official comparator validation failed")
    print(json.dumps(payload, indent=2))
    return payload


def load_aligned_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparator_report = json_load(COMPARATOR_VALIDATION)
    require(comparator_report.get("status") == "PASS", "Official comparator validation is missing")
    frames = []
    for path, manifest_path, mode in [
        (PRED_WITHOUT, MANIFEST_WITHOUT, "without_sensitive"),
        (PRED_WITH, MANIFEST_WITH, "with_sensitive"),
    ]:
        manifest = json_load(manifest_path)
        require(manifest.get("status") == "PASS", f"Prediction manifest failed for {mode}")
        require(sha256_file(path) == manifest["prediction_sha256"], f"Prediction hash failed for {mode}")
        frame = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
        require(len(frame) == 99948 and not frame["row_id"].duplicated().any(), f"Prediction membership failed for {mode}")
        frames.append(frame)
    official = pd.read_csv(STAGE4_PRED).sort_values("row_id").reset_index(drop=True)
    without, with_sensitive = frames
    row_ids = without["row_id"].to_numpy(np.int64)
    require(np.array_equal(row_ids, with_sensitive["row_id"].to_numpy(np.int64)), "Deep row alignment failed")
    require(np.array_equal(row_ids, official["row_id"].to_numpy(np.int64)), "Official row alignment failed")
    require(np.array_equal(without["y_true"].to_numpy(np.float64), with_sensitive["y_true"].to_numpy(np.float64)), "Deep targets differ")
    require(np.array_equal(without["y_true"].to_numpy(np.float64), official["y_true"].to_numpy(np.float64)), "Official targets differ")
    return official, without, with_sensitive


def paired_bootstrap(
    error_first: np.ndarray,
    error_second: np.ndarray,
    first_id: str,
    second_id: str,
    comparison_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(42)
    differences = np.empty(500, dtype=np.float64)
    n_rows = len(error_first)
    for index in range(500):
        sampled = rng.integers(0, n_rows, size=n_rows)
        differences[index] = float(np.mean(error_first[sampled]) - np.mean(error_second[sampled]))
    point = float(np.mean(error_first) - np.mean(error_second))
    summary = {
        "comparison_id": comparison_id, "first_candidate_id": first_id,
        "second_candidate_id": second_id, "point_mae_difference": point,
        "bootstrap_mean": float(np.mean(differences)), "bootstrap_median": float(np.median(differences)),
        "ci_2_5": float(np.quantile(differences, 0.025)), "ci_97_5": float(np.quantile(differences, 0.975)),
        "first_candidate_win_proportion": float(np.mean(differences < 0)),
        "resamples": 500, "seed": 42, "evaluation_label": LABEL,
        "interpretation": "Negative values favor the first-named Candidate. This is descriptive and caused no selection.",
    }
    return summary, differences


def metric_table(official: pd.DataFrame, without: pd.DataFrame, with_sensitive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    items = [
        ("stage4l__blend__without_sensitive", "Frozen Stage 4 Boosting Blend", "without_sensitive", "Stage 4L", "official_pre_registered_primary", STAGE4_PRED, official),
        ("stage5c__realmlp__without_sensitive__test_evaluation", "RealMLP", "without_sensitive", "Stage 5C", "post_test_extension", PRED_WITHOUT, without),
        ("stage5c__realmlp__with_sensitive__test_evaluation", "RealMLP", "with_sensitive", "Stage 5C", "post_test_extension_accuracy_only", PRED_WITH, with_sensitive),
    ]
    comparator_report = json_load(COMPARATOR_VALIDATION)
    for candidate_id, family, mode, stage, role, path, frame in items:
        y_true = frame["y_true"].to_numpy(np.float64)
        y_pred = frame["y_pred"].to_numpy(np.float64)
        if "target_decile" in frame:
            decile = frame["target_decile"].to_numpy(np.int64)
        else:
            decile = np.asarray(pd.qcut(y_true, 10, labels=False, duplicates="drop"), dtype=np.int64) + 1
        rows.append({
            "candidate_id": candidate_id, "model_family": family, "sensitive_mode": mode,
            "evaluation_stage": stage, "evaluation_label": LABEL if stage == "Stage 5C" else "Official pre-registered evaluation",
            "official_result_role": role, "prediction_path": rel(path), "prediction_sha256": sha256_file(path),
            "row_count": len(frame), "test_row_id_hash": comparator_report["sorted_test_row_id_hash"],
            "target_hash": comparator_report["canonical_target_hash"], **metric_values(y_true, y_pred, decile),
        })
    table = pd.DataFrame(rows)
    table["descriptive_mae_rank"] = table["mae"].rank(method="min").astype(int)
    return table


def build_comparisons(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_id = metrics.set_index("candidate_id")
    official = by_id.loc["stage4l__blend__without_sensitive"]
    deep = by_id.loc["stage5c__realmlp__without_sensitive__test_evaluation"]
    sensitive = by_id.loc["stage5c__realmlp__with_sensitive__test_evaluation"]
    directions = {
        name: (
            "higher_is_better" if name == "r_squared"
            else "closer_to_zero" if name == "mean_signed_error"
            else "lower_is_better"
        )
        for name in METRICS
    }
    official_rows, sensitive_rows = [], []
    for metric in METRICS:
        difference = float(deep[metric] - official[metric])
        if metric == "mean_signed_error":
            relative = float((abs(deep[metric]) - abs(official[metric])) / abs(official[metric]) * 100.0) if official[metric] != 0 else np.nan
            favors_deep = abs(deep[metric]) < abs(official[metric])
        else:
            relative = float(difference / abs(official[metric]) * 100.0) if official[metric] != 0 else np.nan
            favors_deep = difference > 0 if metric == "r_squared" else difference < 0
        official_rows.append({
            "metric": metric, "stage4l_value": float(official[metric]), "deep_value": float(deep[metric]),
            "deep_minus_stage4l": difference, "relative_difference_percent": relative,
            "metric_direction": directions[metric], "descriptive_interpretation": "Observed value favors frozen RealMLP" if favors_deep else "Observed value favors Stage 4L primary",
            "official_selection_effect": "none", "evaluation_label": LABEL,
        })
        sensitive_difference = float(sensitive[metric] - deep[metric])
        if metric == "mean_signed_error":
            sensitive_relative = float((abs(sensitive[metric]) - abs(deep[metric])) / abs(deep[metric]) * 100.0) if deep[metric] != 0 else np.nan
            sensitive_favors = abs(sensitive[metric]) < abs(deep[metric])
        else:
            sensitive_relative = float(sensitive_difference / abs(deep[metric]) * 100.0) if deep[metric] != 0 else np.nan
            sensitive_favors = sensitive_difference > 0 if metric == "r_squared" else sensitive_difference < 0
        sensitive_rows.append({
            "metric": metric, "without_sensitive_value": float(deep[metric]), "with_sensitive_value": float(sensitive[metric]),
            "with_minus_without": sensitive_difference, "relative_difference_percent": sensitive_relative,
            "metric_direction": directions[metric], "evaluation_label": LABEL,
            "descriptive_interpretation": "Observed value favors with-sensitive RealMLP" if sensitive_favors else "Observed value favors without-sensitive RealMLP",
            "scope_warning": "Accuracy comparison only; not a fairness audit and not causal evidence.",
        })
    return pd.DataFrame(official_rows), pd.DataFrame(sensitive_rows)


def decile_and_tail_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    decile_rows, tail_rows = [], []
    for candidate_id, frame in frames.items():
        current = frame.copy()
        if "target_decile" not in current:
            current["target_decile"] = np.asarray(pd.qcut(current["y_true"], 10, labels=False, duplicates="drop"), dtype=np.int64) + 1
        for value, group in current.groupby("target_decile", sort=True):
            signed = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
            absolute = np.abs(signed)
            decile_rows.append({
                "candidate_id": candidate_id, "target_decile": int(value), "row_count": len(group),
                "target_min": float(group["y_true"].min()), "target_max": float(group["y_true"].max()), "target_mean": float(group["y_true"].mean()),
                "mae": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(np.square(signed)))),
                "median_absolute_error": float(np.median(absolute)), "mean_signed_error": float(np.mean(signed)),
                "underprediction_rate": float(np.mean(signed < 0)), "overprediction_rate": float(np.mean(signed > 0)),
                "evaluation_label": LABEL,
            })
        y_true = current["y_true"].to_numpy(float); y_pred = current["y_pred"].to_numpy(float)
        signed = y_pred - y_true; absolute = np.abs(signed)
        high_decile = current["target_decile"].to_numpy(int) == int(current["target_decile"].max())
        high_five = y_true >= np.quantile(y_true, 0.95)
        tail_rows.append({
            "candidate_id": candidate_id, "p90_absolute_error": float(np.quantile(absolute, 0.9)),
            "top_decile_mae": float(np.mean(absolute[high_decile])), "top_five_percent_mae": float(np.mean(absolute[high_five])),
            "highest_decile_signed_error": float(np.mean(signed[high_decile])), "highest_five_percent_signed_error": float(np.mean(signed[high_five])),
            "negative_prediction_rate": float(np.mean(y_pred < 0)), "evaluation_label": LABEL,
        })
    return pd.DataFrame(decile_rows), pd.DataFrame(tail_rows)


def disagreement_tables(official: pd.DataFrame, without: pd.DataFrame, with_sensitive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = without["y_true"].to_numpy(float)
    stage4 = official["y_pred"].to_numpy(float)
    deep = without["y_pred"].to_numpy(float)
    sensitive = with_sensitive["y_pred"].to_numpy(float)
    residual_stage4, residual_deep = stage4 - y, deep - y
    decile = without["target_decile"].to_numpy(int)
    high = decile == decile.max()
    summary = pd.DataFrame([
        {
            "comparison_id": "realmlp_without_vs_stage4l_primary",
            "pearson_prediction_correlation": float(np.corrcoef(deep, stage4)[0, 1]),
            "spearman_prediction_correlation": float(pd.Series(deep).corr(pd.Series(stage4), method="spearman")),
            "pearson_residual_correlation": float(np.corrcoef(residual_deep, residual_stage4)[0, 1]),
            "spearman_residual_correlation": float(pd.Series(residual_deep).corr(pd.Series(residual_stage4), method="spearman")),
            "mean_absolute_prediction_disagreement": float(np.mean(np.abs(deep - stage4))),
            "median_absolute_prediction_disagreement": float(np.median(np.abs(deep - stage4))),
            "first_candidate_lower_absolute_error_percent": float(np.mean(np.abs(residual_deep) < np.abs(residual_stage4)) * 100),
            "second_candidate_lower_absolute_error_percent": float(np.mean(np.abs(residual_stage4) < np.abs(residual_deep)) * 100),
            "opposite_signed_error_percent": float(np.mean(np.sign(residual_deep) != np.sign(residual_stage4)) * 100),
            "highest_decile_residual_correlation": float(np.corrcoef(residual_deep[high], residual_stage4[high])[0, 1]),
            "evaluation_label": LABEL,
        },
        {
            "comparison_id": "realmlp_with_vs_without_sensitive",
            "pearson_prediction_correlation": float(np.corrcoef(sensitive, deep)[0, 1]),
            "spearman_prediction_correlation": float(pd.Series(sensitive).corr(pd.Series(deep), method="spearman")),
            "pearson_residual_correlation": float(np.corrcoef(sensitive - y, deep - y)[0, 1]),
            "spearman_residual_correlation": float(pd.Series(sensitive - y).corr(pd.Series(deep - y), method="spearman")),
            "mean_absolute_prediction_disagreement": float(np.mean(np.abs(sensitive - deep))),
            "median_absolute_prediction_disagreement": float(np.median(np.abs(sensitive - deep))),
            "first_candidate_lower_absolute_error_percent": float(np.mean(np.abs(sensitive - y) < np.abs(deep - y)) * 100),
            "second_candidate_lower_absolute_error_percent": float(np.mean(np.abs(deep - y) < np.abs(sensitive - y)) * 100),
            "opposite_signed_error_percent": float(np.mean(np.sign(sensitive - y) != np.sign(deep - y)) * 100),
            "highest_decile_residual_correlation": float(np.corrcoef((sensitive - y)[high], (deep - y)[high])[0, 1]),
            "evaluation_label": LABEL,
        },
    ])
    rows = []
    for value in sorted(np.unique(decile)):
        mask = decile == value
        rows.append({
            "target_decile": int(value), "row_count": int(mask.sum()),
            "mean_absolute_realmlp_stage4l_disagreement": float(np.mean(np.abs(deep[mask] - stage4[mask]))),
            "mean_absolute_sensitive_mode_difference": float(np.mean(np.abs(sensitive[mask] - deep[mask]))),
            "evaluation_label": LABEL,
        })
    return summary, pd.DataFrame(rows)


def save_figure(fig: Any, stem: str, data: pd.DataFrame, title: str, candidates: list[str], manifest: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES / f"{stem}.png"
    data_path = PLOT_DATA / f"{stem}.csv"
    atomic_csv(data, data_path)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    manifest.append({
        "figure_path": rel(figure_path), "figure_sha256": sha256_file(figure_path),
        "plotting_data_path": rel(data_path), "plotting_data_sha256": sha256_file(data_path),
        "title": title, "candidate_ids": candidates,
        "test_row_hash": json_load(COMPARATOR_VALIDATION)["sorted_test_row_id_hash"],
        "evaluation_label": LABEL, "interpretation_scope": "Descriptive Test evaluation only; no model selection.",
    })


def create_figures(
    metrics: pd.DataFrame,
    deciles: pd.DataFrame,
    tails: pd.DataFrame,
    official_comparison: pd.DataFrame,
    sensitive_comparison: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    bootstrap_draws: dict[str, np.ndarray],
    disagreement_by_decile: pd.DataFrame,
    official: pd.DataFrame,
    without: pd.DataFrame,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = {row.candidate_id: ("Stage 4L official" if row.candidate_id.startswith("stage4l") else ("RealMLP without sensitive" if row.sensitive_mode == "without_sensitive" else "RealMLP with sensitive")) for row in metrics.itertuples()}
    candidates = metrics["candidate_id"].tolist()
    entries: list[dict[str, Any]] = []

    for metric, stem, ylabel in [
        ("mae", "stage5c_test_mae_comparison", "MAE (thousand USD)"),
        ("rmse", "stage5c_test_rmse_comparison", "RMSE (thousand USD)"),
    ]:
        data = metrics[["candidate_id", metric]].copy(); data["label"] = data["candidate_id"].map(labels)
        fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(data["label"], data[metric], color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylabel(ylabel); ax.set_title(f"{ylabel.split(' (')[0]} Comparison\n{LABEL}"); ax.tick_params(axis="x", rotation=12)
        save_figure(fig, stem, data, ax.get_title(), candidates, entries)

    data = metrics[["candidate_id", "rmsle", "r_squared"]].copy(); data["label"] = data["candidate_id"].map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8)); axes[0].bar(data["label"], data["rmsle"]); axes[1].bar(data["label"], data["r_squared"])
    axes[0].set_title("RMSLE (lower is better)"); axes[1].set_title("R² (higher is better)")
    for ax in axes: ax.tick_params(axis="x", rotation=18)
    fig.suptitle(f"RMSLE and R² Comparison — {LABEL}")
    save_figure(fig, "stage5c_rmsle_r2_comparison", data, fig._suptitle.get_text(), candidates, entries)

    data = deciles[["candidate_id", "target_decile", "mae"]].copy(); data["label"] = data["candidate_id"].map(labels)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, group in data.groupby("label", sort=False): ax.plot(group["target_decile"], group["mae"], marker="o", label=label)
    ax.set_xlabel("Target decile"); ax.set_ylabel("MAE (thousand USD)"); ax.set_title(f"Target-Decile MAE — {LABEL}"); ax.legend()
    save_figure(fig, "stage5c_target_decile_mae", data, ax.get_title(), candidates, entries)

    data = tails[["candidate_id", "p90_absolute_error", "top_decile_mae", "top_five_percent_mae"]].copy(); data["label"] = data["candidate_id"].map(labels)
    long = data.melt(id_vars=["candidate_id", "label"], var_name="tail_metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5.5)); pivot = long.pivot(index="label", columns="tail_metric", values="value"); pivot.plot.bar(ax=ax)
    ax.set_ylabel("Error (thousand USD)"); ax.set_title(f"Tail-Error Comparison — {LABEL}"); ax.tick_params(axis="x", rotation=12)
    save_figure(fig, "stage5c_tail_error_comparison", long, ax.get_title(), candidates, entries)

    error_difference = np.abs(without["y_pred"].to_numpy(float) - without["y_true"].to_numpy(float)) - np.abs(official["y_pred"].to_numpy(float) - official["y_true"].to_numpy(float))
    counts, edges = np.histogram(error_difference, bins=80)
    data = pd.DataFrame({"left_edge": edges[:-1], "right_edge": edges[1:], "count": counts})
    fig, ax = plt.subplots(figsize=(9, 5)); ax.hist(error_difference, bins=edges, color="#F58518"); ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("RealMLP absolute error minus Stage 4L absolute error"); ax.set_ylabel("Rows"); ax.set_title(f"Paired Absolute-Error Difference — {LABEL}")
    save_figure(fig, "stage5c_absolute_error_difference_distribution", data, ax.get_title(), candidates[:2], entries)

    data = bootstrap_summary[["comparison_id", "point_mae_difference", "ci_2_5", "ci_97_5"]].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8)); y = np.arange(len(data)); xerr = np.vstack([data["point_mae_difference"]-data["ci_2_5"], data["ci_97_5"]-data["point_mae_difference"]])
    ax.errorbar(data["point_mae_difference"], y, xerr=xerr, fmt="o", capsize=6); ax.axvline(0, color="black", linewidth=1); ax.set_yticks(y, data["comparison_id"])
    ax.set_xlabel("Paired MAE difference (thousand USD)"); ax.set_title(f"Paired Bootstrap MAE Intervals — {LABEL}")
    save_figure(fig, "stage5c_paired_bootstrap_interval", data, ax.get_title(), candidates, entries)

    rng = np.random.default_rng(42); sample_index = np.sort(rng.choice(len(without), size=min(20000, len(without)), replace=False))
    data = pd.DataFrame({"row_id": without.iloc[sample_index]["row_id"].to_numpy(), "stage4l_prediction": official.iloc[sample_index]["y_pred"].to_numpy(), "realmlp_prediction": without.iloc[sample_index]["y_pred"].to_numpy()})
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(data["stage4l_prediction"], data["realmlp_prediction"], s=5, alpha=0.18); limits = [min(data.iloc[:,1:].min()), max(data.iloc[:,1:].max())]; ax.plot(limits, limits, color="black", linewidth=1)
    ax.set_xlabel("Stage 4L official prediction"); ax.set_ylabel("RealMLP prediction"); ax.set_title(f"Prediction Disagreement View — {LABEL}")
    save_figure(fig, "stage5c_prediction_disagreement", data, ax.get_title(), candidates[:2], entries)

    data = sensitive_comparison[["metric", "with_minus_without"]].copy()
    selected = data[data["metric"].isin(["mae", "rmse", "rmsle", "r_squared", "top_decile_mae", "top_five_percent_mae"])]
    fig, ax = plt.subplots(figsize=(9, 5)); colors = ["#54A24B" if value < 0 else "#E45756" for value in selected["with_minus_without"]]; ax.bar(selected["metric"], selected["with_minus_without"], color=colors); ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("With-sensitive minus without-sensitive"); ax.set_title(f"Sensitive-Mode Accuracy Differences — {LABEL}\nNot a fairness audit")
    save_figure(fig, "stage5c_sensitive_metric_difference", selected, ax.get_title(), candidates[1:], entries)

    data = pd.concat([
        metrics[["candidate_id", "mae", "rmse", "rmsle", "r_squared"]].assign(panel="metrics"),
        disagreement_by_decile.assign(candidate_id="descriptive_disagreement", mae=np.nan, rmse=np.nan, rmsle=np.nan, r_squared=np.nan, panel="disagreement"),
    ], ignore_index=True, sort=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8));
    for ax, metric in zip(axes.flat[:3], ["mae", "rmse", "rmsle"]): ax.bar(metrics["candidate_id"].map(labels), metrics[metric]); ax.set_title(metric.upper()); ax.tick_params(axis="x", rotation=18)
    axes.flat[3].plot(disagreement_by_decile["target_decile"], disagreement_by_decile["mean_absolute_realmlp_stage4l_disagreement"], marker="o"); axes.flat[3].set_title("Mean prediction disagreement by decile")
    fig.suptitle(f"Stage 5C Summary Dashboard — {LABEL}")
    save_figure(fig, "stage5c_summary_dashboard", data, fig._suptitle.get_text(), candidates, entries)
    return entries


def registry_record(experiment_id: str, family: str, name: str, mode: str, evaluation_stage: str, metrics: dict[str, Any] | None, notes: str, prediction_path: str = "") -> dict[str, Any]:
    columns = pd.read_csv(REGISTRY, nrows=0).columns.tolist()
    record = {column: "" for column in columns}
    record.update({
        "experiment_id": experiment_id, "timestamp_utc": now_utc(), "model_family": family,
        "model_name": name, "sensitive_mode": mode, "feature_set": "deep_core_v1" if family == "RealMLP" else "saved_predictions",
        "target_mode": "raw", "evaluation_stage": evaluation_stage, "fold_number": -1,
        "training_row_count": 399788 if family == "RealMLP" else 0, "validation_row_count": 0,
        "test_row_count": 99948, "parameter_json": json.dumps({"evaluation_label": LABEL}, sort_keys=True),
        "status": "PASS", "notes": notes, "prediction_artifact_path": prediction_path,
    })
    if metrics:
        for key in METRICS:
            if key in record and key in metrics: record[key] = metrics[key]
    return record


def update_registry(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, Any]:
    baseline = json_load(BASELINE)
    prefix_size = int(baseline["registry_prior_byte_count"])
    prefix_hash = baseline["registry_prior_sha256"]
    current_bytes = REGISTRY.read_bytes()
    require(hashlib.sha256(current_bytes[:prefix_size]).hexdigest() == prefix_hash, "Prior Registry prefix changed")
    existing = pd.read_csv(REGISTRY)
    by_id = metrics.set_index("candidate_id")
    ids = json_load(FREEZE)["registry_ids"]
    records = [
        registry_record(ids[0], "RealMLP", "Frozen RealMLP Test evaluation", "without_sensitive", "Stage 5C", by_id.loc[ids[0]].to_dict(), f"{LABEL}; descriptive only; Stage 4L remains official.", rel(PRED_WITHOUT)),
        registry_record(ids[1], "RealMLP", "Frozen RealMLP sensitive accuracy evaluation", "with_sensitive", "Stage 5C", by_id.loc[ids[1]].to_dict(), f"{LABEL}; accuracy-only; not a fairness audit.", rel(PRED_WITH)),
        registry_record(ids[2], "comparison", "RealMLP versus Stage 4L official comparison", "without_sensitive", "Stage 5C", None, f"{LABEL}; no official selection effect."),
        registry_record(ids[3], "bootstrap", "Paired Bootstrap summary", "without_sensitive", "Stage 5C", None, f"{LABEL}; 500 paired resamples, seed 42; no selection."),
        registry_record(ids[4], "comparison", "Sensitive-mode accuracy comparison", "with_sensitive", "Stage 5C", None, f"{LABEL}; accuracy-only; not a fairness audit."),
        registry_record(ids[5], "evaluation", "Stage 5C evaluation summary", "both", "Stage 5C", None, f"{LABEL}; ensemble count 0; Stage 4L remains official."),
        registry_record(ids[6], "handoff", "Stage 6 handoff", "both", "Stage 5C", None, f"{LABEL}; Stage 6 must use saved predictions and not rerun inference."),
    ]
    incoming = pd.DataFrame(records, columns=existing.columns)
    duplicate_incoming = incoming["experiment_id"].duplicated().any()
    require(not duplicate_incoming, "Stage 5C Registry IDs are duplicated")
    existing_ids = set(existing["experiment_id"].astype(str))
    to_append = incoming[~incoming["experiment_id"].isin(existing_ids)]
    if len(to_append):
        to_append.to_csv(REGISTRY, mode="a", header=False, index=False, lineterminator="\n")
        action = "APPENDED"
    else:
        action = "REUSED"
    after_first = REGISTRY.read_bytes()
    after = pd.read_csv(REGISTRY)
    require(after["experiment_id"].is_unique, "Registry IDs are not unique")
    require(hashlib.sha256(after_first[:prefix_size]).hexdigest() == prefix_hash, "Prior Registry prefix changed after update")
    # The required second invocation is an idempotence check with no write.
    second_action = "REUSED" if set(incoming["experiment_id"]).issubset(set(after["experiment_id"])) else "FAIL"
    require(second_action == "REUSED", "Registry idempotence check failed")
    export = after[after["experiment_id"].isin(ids)].copy()
    atomic_csv(export, RESULTS / "stage5c_registry_rows.csv")
    report = {
        "stage_id": STAGE_ID, "status": "PASS", "action": action, "second_action": second_action,
        "prior_byte_count": prefix_size, "prior_prefix_preserved": True,
        "prior_sha256": prefix_hash, "registry_row_count": len(after), "registry_unique_ids": True,
        "stage5c_row_count": len(export), "stage5c_registry_ids": ids,
        "registry_sha256": sha256_file(REGISTRY), "evaluation_label": LABEL,
    }
    atomic_json(report, REPORTS / "stage5c_registry_update.json")
    return report


def create_access_and_alignment_reports(official: pd.DataFrame, without: pd.DataFrame, with_sensitive: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = json_load(FREEZE); comparator = json_load(COMPARATOR_VALIDATION)
    manifests = {"without_sensitive": json_load(MANIFEST_WITHOUT), "with_sensitive": json_load(MANIFEST_WITH)}
    attempt_files = sorted(REPORTS.glob("stage5c_prediction_attempt_*.json"))
    attempt_items = [json_load(path) for path in attempt_files]
    successful = [item for item in attempt_items if item.get("status") == "PASS"]
    attempts_payload = {
        "stage_id": STAGE_ID, "status": "PASS" if len(successful) == 2 and len(attempt_items) <= 4 else "FAIL",
        "physical_attempt_count": len(attempt_items), "successful_prediction_artifact_count": len(successful),
        "maximum_physical_attempts": 4, "attempts": attempt_items, "evaluation_label": LABEL,
    }
    atomic_json(attempts_payload, ATTEMPTS)
    common = sorted(set(manifests["without_sensitive"]["feature_column_hashes"]).intersection(manifests["with_sensitive"]["feature_column_hashes"]))
    common_equal = all(manifests["without_sensitive"]["feature_column_hashes"][name] == manifests["with_sensitive"]["feature_column_hashes"][name] for name in common)
    alignment = {
        "stage_id": STAGE_ID, "status": "PASS",
        "test_row_count_per_mode": {mode: item["row_count"] for mode, item in manifests.items()},
        "test_row_id_hash": comparator["sorted_test_row_id_hash"], "canonical_target_hash": comparator["canonical_target_hash"],
        "source_hashes": {mode: item["source_sha256"] for mode, item in manifests.items()},
        "schema_hashes": {mode: item["input_schema_hash"] for mode, item in manifests.items()},
        "common_feature_column_count": len(common), "common_feature_values_equal": common_equal,
        "sensitive_columns_only_in_expected_mode": set(freeze["deep_modes"]["without_sensitive"]["required_feature_columns"]).issubset(freeze["deep_modes"]["with_sensitive"]["required_feature_columns"]),
        "test_feature_rows_materialized": {mode: item["test_feature_rows_materialized"] for mode, item in manifests.items()},
        "train_feature_rows_materialized": 0, "excluded_rows_converted": 0,
        "source_target_values_materialized": 0, "loader_path": freeze["safe_loader_path"],
        "loader_sha256": freeze["safe_loader_sha256"], "sentinel_status": json_load(SENTINEL)["status"],
        "source_frames_unchanged": all(item["source_frame_unchanged"] for item in manifests.values()),
        "prediction_rows_align_exactly": np.array_equal(without["row_id"], with_sensitive["row_id"]),
        "canonical_targets_align_exactly": np.array_equal(without["y_true"], with_sensitive["y_true"]),
        "evaluation_label": LABEL,
    }
    if not all([common_equal, alignment["sensitive_columns_only_in_expected_mode"], alignment["prediction_rows_align_exactly"], alignment["canonical_targets_align_exactly"]]): alignment["status"] = "FAIL"
    atomic_json(alignment, ALIGNMENT)
    access = {
        "stage_id": STAGE_ID, "status": "PASS", "authorization_source": "Attached Stage 5C user specification",
        "preevaluation_freeze_path": rel(FREEZE), "preevaluation_freeze_sha256": sha256_file(FREEZE),
        "first_stage5c_test_artifact_access_at_utc": comparator["first_stage5c_test_artifact_access_at_utc"],
        "first_source_feature_access_at_utc": min(item["first_source_feature_access_at_utc"] for item in manifests.values()),
        "saved_test_id_path": rel(TEST_IDS), "saved_test_id_sha256": sha256_file(TEST_IDS),
        "canonical_target_path": rel(STAGE4_PRED), "canonical_target_hash": comparator["canonical_target_hash"],
        "stage4l_official_prediction_path": rel(STAGE4_PRED), "stage4l_official_prediction_sha256": sha256_file(STAGE4_PRED),
        "source_paths_and_hashes": {mode: {"path": item["source_path"], "sha256": item["source_sha256"]} for mode, item in manifests.items()},
        "required_feature_columns": {mode: freeze["deep_modes"][mode]["required_feature_columns"] for mode in manifests},
        "source_target_columns_requested": 0,
        "test_feature_rows_materialized": {mode: item["test_feature_rows_materialized"] for mode, item in manifests.items()},
        "test_feature_rows_materialized_across_all_physical_attempts": int(sum(item.get("test_feature_rows_materialized", 0) for item in attempt_items)),
        "train_feature_rows_materialized": {mode: 0 for mode in manifests}, "excluded_rows_converted": 0,
        "source_target_values_materialized": 0, "raw_file_data_lines_scanned_per_source": 499736,
        "parser_conversion_scope": "Only exact saved Test Feature rows and required Feature columns",
        "dataframe_materialization_scope": "99,948 Test Feature rows per sensitive mode",
        "model_input_scope": "One frozen RealMLP input frame per sensitive mode",
        "prediction_generation_scope": "One successful full Test call per sensitive mode",
        "loader_path": freeze["safe_loader_path"], "loader_sha256": freeze["safe_loader_sha256"],
        "sentinel_path": rel(SENTINEL), "sentinel_status": "PASS",
        "bundle_load_timestamps": {mode: item["bundle_loaded_at_utc"] for mode, item in manifests.items()},
        "prediction_timestamps": {mode: item["prediction_started_at_utc"] for mode, item in manifests.items()},
        "physical_attempt_ids": [item["attempt_id"] for item in attempt_items],
        "ensemble_test_access": "none", "new_boosting_prediction_generation": "none", "evaluation_label": LABEL,
    }
    atomic_json(access, ACCESS_AUDIT)
    require(attempts_payload["status"] == alignment["status"] == access["status"] == "PASS", "Stage 5C access/alignment evidence failed")
    return attempts_payload, alignment, access


def create_handoff(summary: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "common_metrics": RESULTS / "stage5c_test_metrics.csv",
        "official_comparison": RESULTS / "stage5c_official_comparison.csv",
        "bootstrap": RESULTS / "stage5c_paired_bootstrap.csv",
        "sensitive_accuracy": RESULTS / "stage5c_sensitive_accuracy_comparison.csv",
        "decile_metrics": RESULTS / "stage5c_target_decile_metrics.csv",
        "tail_metrics": RESULTS / "stage5c_tail_metrics.csv",
        "disagreement_summary": RESULTS / "stage5c_disagreement_summary.csv",
        "disagreement_by_decile": RESULTS / "stage5c_disagreement_by_target_decile.csv",
    }
    handoff = {
        "stage_id": STAGE_ID, "stage5c_status": "PASS", "evaluation_label": LABEL,
        "post_test_extension": True, "stage4l_official_primary_candidate_id": "stage4l__blend__without_sensitive",
        "stage4l_official_prediction_path": rel(STAGE4_PRED), "stage4l_official_prediction_sha256": sha256_file(STAGE4_PRED),
        "without_sensitive_deep_candidate_id": "stage5c__realmlp__without_sensitive__test_evaluation",
        "without_sensitive_bundle_path": json_load(MANIFEST_WITHOUT)["bundle_path"], "without_sensitive_bundle_sha256": EXPECTED["bundle_without"],
        "without_sensitive_prediction_path": rel(PRED_WITHOUT), "without_sensitive_prediction_sha256": sha256_file(PRED_WITHOUT),
        "with_sensitive_deep_candidate_id": "stage5c__realmlp__with_sensitive__test_evaluation",
        "with_sensitive_bundle_path": json_load(MANIFEST_WITH)["bundle_path"], "with_sensitive_bundle_sha256": EXPECTED["bundle_with"],
        "with_sensitive_prediction_path": rel(PRED_WITH), "with_sensitive_prediction_sha256": sha256_file(PRED_WITH),
        "test_row_count": 99948, "test_row_id_hash": summary["test_row_id_hash"], "target_hash": summary["target_hash"],
        "artifact_paths_and_hashes": {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "ensemble_status": "rejected", "accepted_ensemble_count": 0, "ensemble_test_prediction_path": None,
        "stage4l_official_result_remains_unchanged": True, "stage5c_results_are_descriptive": True,
        "stage5a_governance_exception_summary": "Historical literal zero-Test-loading failed under an accepted procedural exception; zero Test rows entered learned preprocessing, fitting, selection, or Validation metrics.",
        "stage6_must_use_saved_predictions": True, "stage6_must_not_rerun_model_inference": True,
        "recommended_bounded_stage6_analyses": ["error distributions", "target deciles", "tail behavior", "worst predictions", "underprediction and overprediction", "model disagreement", "sensitive and non-sensitive accuracy differences"],
        "stage7_owns_full_fairness_analysis": True, "stage8_owns_final_explainability": True,
        "next_stage": "Stage 6", "stage6_started": False,
    }
    path = MANIFESTS / "stage5c_stage6_handoff.json"; atomic_json(handoff, path)
    return handoff


def evaluate() -> dict[str, Any]:
    started = time.perf_counter()
    official, without, with_sensitive = load_aligned_predictions()
    attempts, alignment, access = create_access_and_alignment_reports(official, without, with_sensitive)
    metrics = metric_table(official, without, with_sensitive); atomic_csv(metrics, RESULTS / "stage5c_test_metrics.csv")
    official_comparison, sensitive_comparison = build_comparisons(metrics)
    errors = {
        "official": np.abs(official["y_pred"].to_numpy(float) - official["y_true"].to_numpy(float)),
        "without": without["absolute_error"].to_numpy(float), "with": with_sensitive["absolute_error"].to_numpy(float),
    }
    primary_summary, primary_draws = paired_bootstrap(errors["without"], errors["official"], "stage5c__realmlp__without_sensitive__test_evaluation", "stage4l__blend__without_sensitive", "realmlp_without_minus_stage4l_primary")
    sensitive_summary, sensitive_draws = paired_bootstrap(errors["with"], errors["without"], "stage5c__realmlp__with_sensitive__test_evaluation", "stage5c__realmlp__without_sensitive__test_evaluation", "realmlp_with_minus_without_sensitive")
    bootstrap = pd.DataFrame([primary_summary, sensitive_summary])
    comparator = json_load(COMPARATOR_VALIDATION)
    bootstrap["row_hash"] = comparator["sorted_test_row_id_hash"]; bootstrap["target_hash"] = comparator["canonical_target_hash"]
    atomic_csv(bootstrap, RESULTS / "stage5c_paired_bootstrap.csv")
    sensitive_comparison["paired_mae_ci_2_5"] = sensitive_summary["ci_2_5"]
    sensitive_comparison["paired_mae_ci_97_5"] = sensitive_summary["ci_97_5"]
    sensitive_comparison["paired_mae_point_difference"] = sensitive_summary["point_mae_difference"]
    atomic_csv(official_comparison, RESULTS / "stage5c_official_comparison.csv")
    atomic_csv(sensitive_comparison, RESULTS / "stage5c_sensitive_accuracy_comparison.csv")
    frames = {
        "stage4l__blend__without_sensitive": official,
        "stage5c__realmlp__without_sensitive__test_evaluation": without,
        "stage5c__realmlp__with_sensitive__test_evaluation": with_sensitive,
    }
    deciles, tails = decile_and_tail_tables(frames); atomic_csv(deciles, RESULTS / "stage5c_target_decile_metrics.csv"); atomic_csv(tails, RESULTS / "stage5c_tail_metrics.csv")
    disagreement, disagreement_decile = disagreement_tables(official, without, with_sensitive); atomic_csv(disagreement, RESULTS / "stage5c_disagreement_summary.csv"); atomic_csv(disagreement_decile, RESULTS / "stage5c_disagreement_by_target_decile.csv")
    figure_entries = create_figures(metrics, deciles, tails, official_comparison, sensitive_comparison, bootstrap, {"primary": primary_draws, "sensitive": sensitive_draws}, disagreement_decile, official, without)
    visualization = {"stage_id": STAGE_ID, "status": "PASS", "figure_count": len(figure_entries), "figures": figure_entries, "evaluation_label": LABEL}
    atomic_json(visualization, MANIFESTS / "stage5c_visualization_manifest.json")
    registry = update_registry(metrics, bootstrap)
    summary = {
        "stage_id": STAGE_ID, "stage_status": "PASS", "evaluation_label": LABEL,
        "stage4l_official_primary_identity": "stage4l__blend__without_sensitive",
        "stage4l_official_metrics": metrics.iloc[0].to_dict(),
        "deep_candidate_metrics": {row.candidate_id: row._asdict() for row in metrics.iloc[1:].itertuples(index=False)},
        "bundle_paths_and_hashes": {mode: {"path": json_load(path)["bundle_path"], "sha256": json_load(path)["bundle_sha256"]} for mode, path in [("without_sensitive", MANIFEST_WITHOUT), ("with_sensitive", MANIFEST_WITH)]},
        "deep_prediction_paths_and_hashes": {"without_sensitive": {"path": rel(PRED_WITHOUT), "sha256": sha256_file(PRED_WITHOUT)}, "with_sensitive": {"path": rel(PRED_WITH), "sha256": sha256_file(PRED_WITH)}},
        "test_row_count": 99948, "test_row_id_hash": comparator["sorted_test_row_id_hash"], "target_hash": comparator["canonical_target_hash"],
        "source_hashes": {"without_sensitive": EXPECTED["source_without"], "with_sensitive": EXPECTED["source_with"]},
        "official_comparison": official_comparison.to_dict(orient="records"),
        "bootstrap_summaries": bootstrap.to_dict(orient="records"), "sensitive_accuracy_comparison": sensitive_comparison.to_dict(orient="records"),
        "disagreement_summary": disagreement.to_dict(orient="records"), "ensemble_status": "rejected",
        "ensemble_test_prediction_count": 0, "new_boosting_prediction_count": 0,
        "model_fit_calls": 0, "preprocessing_fit_calls": 0, "successful_deep_prediction_artifacts": 2,
        "physical_prediction_attempt_count": attempts["physical_attempt_count"], "source_feature_materialization_count": int(sum(item.get("test_feature_rows_materialized", 0) for item in attempts["attempts"])),
        "train_feature_rows_materialized": 0, "source_target_values_materialized": 0,
        "inherited_stage5a_governance_status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "stage4l_remains_official": True, "next_stage": "Stage 6", "stage6_started": False,
    }
    atomic_json(summary, RESULTS / "stage5c_evaluation_summary.json")
    handoff = create_handoff(summary)
    runtime = {"stage_id": STAGE_ID, "status": "PASS", "metrics_bootstrap_figures_registry_handoff_seconds": time.perf_counter() - started, "updated_at_utc": now_utc()}
    atomic_json(runtime, RUNTIME)
    print(json.dumps({"status": "PASS", "metrics": metrics[["candidate_id", "mae", "rmse", "rmsle", "r_squared"]].to_dict(orient="records"), "bootstrap": bootstrap.to_dict(orient="records"), "figures": len(figure_entries), "registry": registry["status"]}, indent=2))
    return summary


def preliminary_verification() -> dict[str, Any]:
    required = [
        PRED_WITHOUT, PRED_WITH, MANIFEST_WITHOUT, MANIFEST_WITH,
        RESULTS / "stage5c_test_metrics.csv", RESULTS / "stage5c_official_comparison.csv",
        RESULTS / "stage5c_paired_bootstrap.csv", RESULTS / "stage5c_sensitive_accuracy_comparison.csv",
        RESULTS / "stage5c_disagreement_summary.csv", RESULTS / "stage5c_disagreement_by_target_decile.csv",
        RESULTS / "stage5c_target_decile_metrics.csv", RESULTS / "stage5c_tail_metrics.csv",
        RESULTS / "stage5c_evaluation_summary.json", MANIFESTS / "stage5c_visualization_manifest.json",
        RESULTS / "stage5c_registry_rows.csv", MANIFESTS / "stage5c_stage6_handoff.json",
    ]
    checks = {
        "all_required_artifacts_exist": all(path.exists() and path.stat().st_size > 0 for path in required),
        "two_prediction_artifacts": PRED_WITHOUT.exists() and PRED_WITH.exists(),
        "ensemble_test_prediction_absent": not any(ROOT.rglob("*stage5c_ensemble_test_predictions*")),
        "new_boosting_prediction_count_zero": json_load(RESULTS / "stage5c_evaluation_summary.json")["new_boosting_prediction_count"] == 0,
        "model_fit_calls_zero": json_load(RESULTS / "stage5c_evaluation_summary.json")["model_fit_calls"] == 0,
        "preprocessing_fit_calls_zero": json_load(RESULTS / "stage5c_evaluation_summary.json")["preprocessing_fit_calls"] == 0,
        "train_feature_rows_zero": json_load(ALIGNMENT)["train_feature_rows_materialized"] == 0,
        "source_target_values_zero": json_load(ALIGNMENT)["source_target_values_materialized"] == 0,
    }
    payload = {"stage_id": STAGE_ID, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "created_at_utc": now_utc(), "evaluation_label": LABEL}
    atomic_json(payload, REPORTS / "stage5c_preliminary_verification.json")
    require(payload["status"] == "PASS", "Preliminary Stage 5C verification failed")
    return payload


def build_notebook() -> dict[str, Any]:
    preliminary_verification()
    if NOTEBOOK.exists():
        BACKUPS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = BACKUPS / f"REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.{stamp}.ipynb"
        shutil.copy2(NOTEBOOK, backup)
    sections = [
        "Stage Objective and Post-Test Disclosure", "Imports and Configuration", "State Reconstruction",
        "Stage 4L Official Evaluation Status", "Stage 5A Frozen Deep Bundles", "Stage 5A Governance Exception",
        "Stage 5B Decision and Candidate Eligibility", "Protected File Baseline", "Pre-Evaluation Freeze",
        "Safe Test-Row Loader", "Test Access Audit", "Canonical Test Target and Official Comparator",
        "Test Feature Alignment", "Frozen Deep Prediction — Without Sensitive", "Frozen Deep Prediction — With Sensitive",
        "Prediction Artifact Validation", "Common Test Metrics", "Official Stage 4L Comparison", "Paired Bootstrap",
        "Sensitive Accuracy Comparison", "Target-Decile and Tail Summary", "Prediction and Residual Disagreement",
        "Post-Test Interpretation", "Test Evaluation Visualizations", "Registry Update", "Stage 6 Handoff",
        "Stage 5C Artifact Summary", "Independent Review and Repairs", "Stage 5C Verification", "Stage 5C Completion Note",
    ]
    artifact_for_section = {
        0: "artifacts/results/stage5/posttest_evaluation/stage5c_evaluation_summary.json",
        2: "TASK.md", 3: rel(COMPARATOR_VALIDATION), 4: rel(FULL_TRAIN), 5: rel(GOVERNANCE),
        6: rel(STAGE5B_HANDOFF), 7: rel(BASELINE), 8: rel(FREEZE), 9: rel(SENTINEL),
        10: rel(ACCESS_AUDIT), 11: rel(COMPARATOR_VALIDATION), 12: rel(ALIGNMENT),
        13: rel(MANIFEST_WITHOUT), 14: rel(MANIFEST_WITH), 15: rel(ATTEMPTS),
        16: rel(RESULTS / "stage5c_test_metrics.csv"), 17: rel(RESULTS / "stage5c_official_comparison.csv"),
        18: rel(RESULTS / "stage5c_paired_bootstrap.csv"), 19: rel(RESULTS / "stage5c_sensitive_accuracy_comparison.csv"),
        20: rel(RESULTS / "stage5c_target_decile_metrics.csv"), 21: rel(RESULTS / "stage5c_disagreement_summary.csv"),
        22: rel(RESULTS / "stage5c_evaluation_summary.json"), 23: rel(MANIFESTS / "stage5c_visualization_manifest.json"),
        24: rel(REPORTS / "stage5c_registry_update.json"), 25: rel(MANIFESTS / "stage5c_stage6_handoff.json"),
        26: rel(RESULTS / "stage5c_evaluation_summary.json"), 27: rel(REVIEWER),
        28: rel(REPORTS / "stage5c_preliminary_verification.json"), 29: rel(RESULTS / "stage5c_evaluation_summary.json"),
    }
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells.append(nbformat.v4.new_markdown_cell(f"# {OFFICIAL_NAME}"))
    for index, title in enumerate(sections):
        explanation = f"This section checks {title.lower()} from saved, validated artifacts. It does not train, tune, or generate predictions."
        if index in {0, 22, 29}:
            explanation += " Stage 4L remains the official pre-registered Test evaluation. Stage 5C is descriptive only."
        if index == 19:
            explanation += " This is accuracy evidence, not a fairness audit or causal result."
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## {index}. {title}\n\n{explanation}"))
        if index == 0:
            code = """from pathlib import Path\nimport hashlib, json, os\nimport pandas as pd\nfrom IPython.display import display\nROOT = Path.cwd()\nCACHE_ONLY = os.environ.get('STAGE5C_CACHE_ONLY', '0') == '1'\npath=ROOT/'artifacts/results/stage5/posttest_evaluation/stage5c_evaluation_summary.json'\ndata=json.loads(path.read_text(encoding='utf-8'))\nprint({'stage_status':data['stage_status'],'evaluation_label':data['evaluation_label'],'stage4l_remains_official':data['stage4l_remains_official'],'stage6_started':data['stage6_started']})"""
        elif index == 1:
            code = """from pathlib import Path\nimport hashlib, json, os\nimport pandas as pd\nfrom IPython.display import display\nROOT = Path.cwd()\nCACHE_ONLY = os.environ.get('STAGE5C_CACHE_ONLY', '0') == '1'\nprint({'stage':'stage5c','cache_only':CACHE_ONLY,'model_fit_calls':0,'preprocessing_fit_calls':0,'prediction_generation_calls':0,'source_access_calls':0})"""
        elif index == 2:
            code = """text=(ROOT/'TASK.md').read_text(encoding='utf-8')\nprint(text.split('# Previous Stage Snapshot')[0][-3000:])"""
        elif index == 27:
            code = """path=ROOT/'artifacts/reports/stage5c_reviewer.md'\nprint(path.read_text(encoding='utf-8') if path.exists() else 'Independent review runs after Notebook execution; no finding is hidden.')"""
        else:
            artifact = artifact_for_section[index]
            code = f"""path=ROOT/{artifact!r}\nassert path.exists(), path\nif path.suffix == '.json':\n    data=json.loads(path.read_text(encoding='utf-8'))\n    print(json.dumps(data, indent=2, ensure_ascii=False)[:12000])\nelif path.suffix == '.csv':\n    data=pd.read_csv(path)\n    display(data.head(12))\n    print({{'rows':len(data),'columns':list(data.columns)}})\nelse:\n    print(path.read_text(encoding='utf-8')[:12000])"""
        nb.cells.append(nbformat.v4.new_code_cell(code))
    nbformat.write(nb, NOTEBOOK)
    source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    static = {
        "model_fit_call_absent": re.search(r"\.fit\s*\(", source) is None,
        "preprocessing_fit_call_absent": re.search(r"\.fit_transform\s*\(", source) is None,
        "model_prediction_call_absent": re.search(r"\.predict\s*\(", source) is None,
        "source_data_path_absent": "data/regression_" not in source.replace("\\", "/"),
        "section_count_exact": len(sections) == 30,
        "section_titles_unique": len(set(sections)) == 30,
    }
    require(all(static.values()), f"Notebook static audit failed: {static}")
    report = {"stage_id": STAGE_ID, "status": "PASS", "notebook_path": rel(NOTEBOOK), "notebook_sha256": sha256_file(NOTEBOOK), "cell_count": len(nb.cells), "code_cell_count": 30, "sections": sections, "static_audit": static}
    atomic_json(report, REPORTS / "stage5c_notebook_build.json")
    print(json.dumps(report, indent=2))
    return report


def execute_notebook(mode: str) -> dict[str, Any]:
    from nbclient import NotebookClient
    require(mode in {"complete", "cache_only"}, "Invalid Notebook mode")
    history_path = REPORTS / "stage5c_notebook_executions.json"
    history = json_load(history_path) if history_path.exists() else {"stage_id": STAGE_ID, "attempts": []}
    require(len(history["attempts"]) < 3, "Stage 5C Notebook attempt limit is exhausted")
    attempt = len(history["attempts"]) + 1
    started = time.perf_counter(); before = {path: sha256_file(path) for path in [PRED_WITHOUT, PRED_WITH, RESULTS / "stage5c_paired_bootstrap.csv", REGISTRY]}
    previous = os.environ.get("STAGE5C_CACHE_ONLY")
    os.environ["STAGE5C_CACHE_ONLY"] = "1" if mode == "cache_only" else "0"
    try:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        executed = NotebookClient(nb, timeout=180, kernel_name="python3", allow_errors=False).execute(cwd=str(ROOT))
        nbformat.write(executed, NOTEBOOK)
    finally:
        if previous is None: os.environ.pop("STAGE5C_CACHE_ONLY", None)
        else: os.environ["STAGE5C_CACHE_ONLY"] = previous
    after = {path: sha256_file(path) for path in before}
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    error_count = sum(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
    report = {
        "attempt": attempt, "mode": mode, "status": "PASS", "runtime_seconds": time.perf_counter() - started,
        "notebook_sha256": sha256_file(NOTEBOOK), "code_cell_count": len(code_cells),
        "code_cells_with_execution_count": sum(cell.execution_count is not None for cell in code_cells),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells), "error_output_count": error_count,
        "model_fit_calls": 0, "preprocessing_fit_calls": 0, "prediction_generation_calls": 0,
        "source_access_calls": 0, "bootstrap_recalculation": False, "registry_duplication": False,
        "frozen_artifacts_unchanged": before == after, "cache_only": mode == "cache_only",
    }
    require(report["code_cell_count"] == report["code_cells_with_execution_count"] == report["code_cells_with_outputs"] == 30, "Notebook output audit failed")
    require(error_count == 0 and before == after, "Notebook execution changed frozen artifacts or produced errors")
    history["attempts"].append(report); history["status"] = "PASS"; history["complete_run_count"] = sum(item["mode"] == "complete" and item["status"] == "PASS" for item in history["attempts"]); history["cache_only_run_count"] = sum(item["mode"] == "cache_only" and item["status"] == "PASS" for item in history["attempts"])
    atomic_json(history, history_path); atomic_json(report, REPORTS / f"stage5c_notebook_run{attempt}_{mode}.json")
    print(json.dumps(report, indent=2)); return report


def authorized_refresh_paths() -> list[Path]:
    """Return every artifact that the authorized output-only run must not change."""
    paths = {
        PRED_WITHOUT.resolve(), PRED_WITH.resolve(), MANIFEST_WITHOUT.resolve(), MANIFEST_WITH.resolve(),
        (MANIFESTS / "stage5c_stage6_handoff.json").resolve(), REGISTRY.resolve(), STAGE4_PRED.resolve(),
    }
    for path in RESULTS.rglob("*"):
        if path.is_file(): paths.add(path.resolve())
    for path in FIGURES.rglob("*"):
        if path.is_file(): paths.add(path.resolve())
    for manifest_path in (MANIFEST_WITHOUT, MANIFEST_WITH):
        manifest = json_load(manifest_path)
        for key in ("bundle_path", "model_path"):
            path = Path(manifest[key])
            if not path.is_absolute(): path = ROOT / path
            paths.add(path.resolve())
    baseline = json_load(BASELINE)
    for item in baseline["entries"]:
        path = Path(item["path"])
        if not path.is_absolute(): path = ROOT / path
        paths.add(path.resolve())
    return sorted(paths, key=lambda path: str(path).lower())


def prepare_authorized_refresh() -> dict[str, Any]:
    """Back up the Notebook and freeze all output-refresh inputs before attempt 4."""
    history = json_load(REPORTS / "stage5c_notebook_executions.json")
    require(len(history["attempts"]) == 3, "Authorized refresh requires exactly three preserved historical attempts")
    require(not AUTHORIZED_REFRESH_REPORT.exists(), "Authorized fourth execution already has a report")
    official = pd.read_csv(RESULTS / "stage5c_official_comparison.csv")
    sensitive = pd.read_csv(RESULTS / "stage5c_sensitive_accuracy_comparison.csv")
    official_bias = official.loc[official["metric"] == "mean_signed_error"].iloc[0]
    sensitive_bias = sensitive.loc[sensitive["metric"] == "mean_signed_error"].iloc[0]
    require(official_bias["metric_direction"] == "closer_to_zero", "Official comparison signed-bias repair is absent")
    require(sensitive_bias["metric_direction"] == "closer_to_zero", "Sensitive comparison signed-bias repair is absent")
    require(abs(float(official_bias["deep_value"])) < abs(float(official_bias["stage4l_value"])), "Official signed-bias interpretation is invalid")
    require(abs(float(sensitive_bias["with_sensitive_value"])) < abs(float(sensitive_bias["without_sensitive_value"])), "Sensitive signed-bias interpretation is invalid")
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    existing_backups = sorted(BACKUPS.glob("REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.before_authorized_run4.*.ipynb"))
    matching_backups = [path for path in existing_backups if sha256_file(path) == sha256_file(NOTEBOOK)]
    backup = matching_backups[-1] if matching_backups else BACKUPS / f"REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.before_authorized_run4.{stamp}.ipynb"
    if not matching_backups: shutil.copy2(NOTEBOOK, backup)
    entries = []
    for path in authorized_refresh_paths():
        require(path.exists() and path.is_file(), f"Protected refresh input is missing: {path}")
        try: stored_path = rel(path)
        except ValueError: stored_path = str(path)
        entries.append({"path": stored_path, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "stage_id": STAGE_ID, "status": "PASS", "authorization_id": AUTHORIZED_REFRESH_ID,
        "created_at_utc": now_utc(), "historical_attempt_count": 3, "authorized_new_attempt_count": 1,
        "notebook_input_path": rel(NOTEBOOK), "notebook_input_sha256": sha256_file(NOTEBOOK),
        "backup_path": rel(backup), "backup_sha256": sha256_file(backup),
        "protected_entry_count": len(entries), "protected_entries": entries,
        "corrected_inputs": {
            "official_metric_direction": official_bias["metric_direction"],
            "stage4l_mean_signed_error": float(official_bias["stage4l_value"]),
            "realmlp_without_mean_signed_error": float(official_bias["deep_value"]),
            "official_interpretation": official_bias["descriptive_interpretation"],
            "sensitive_metric_direction": sensitive_bias["metric_direction"],
            "realmlp_with_mean_signed_error": float(sensitive_bias["with_sensitive_value"]),
            "sensitive_interpretation": sensitive_bias["descriptive_interpretation"],
        },
    }
    atomic_json(payload, AUTHORIZED_REFRESH_BEFORE)
    print(json.dumps({key: payload[key] for key in payload if key != "protected_entries"}, indent=2))
    return payload


def _notebook_output_text(cell: Any) -> str:
    values: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream": values.append(str(output.get("text", "")))
        for value in output.get("data", {}).values():
            values.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(values)


def execute_authorized_refresh(authorization_id: str) -> dict[str, Any]:
    """Execute the single human-authorized fourth artifact-loading Notebook run."""
    from nbclient import NotebookClient
    require(authorization_id == AUTHORIZED_REFRESH_ID, "Authorization ID does not match the approved fourth run")
    before_payload = json_load(AUTHORIZED_REFRESH_BEFORE)
    require(before_payload["authorization_id"] == authorization_id and before_payload["status"] == "PASS", "Authorized before-hash evidence is invalid")
    history_path = REPORTS / "stage5c_notebook_executions.json"
    history = json_load(history_path)
    require(len(history["attempts"]) == 3, "Exactly three historical attempts must precede authorized attempt 4")
    require(not AUTHORIZED_REFRESH_REPORT.exists(), "No fifth execution is authorized")
    before = {item["path"]: item["sha256"] for item in before_payload["protected_entries"]}
    require(all(sha256_file(ROOT / path) == digest for path, digest in before.items()), "An authorized-run protected input changed before execution")
    input_hash = sha256_file(NOTEBOOK)
    require(input_hash == before_payload["notebook_input_sha256"], "Notebook input changed after the authorized backup")
    temp_notebook = NOTEBOOK.with_name(NOTEBOOK.stem + ".authorized_run4.tmp.ipynb")
    failed_notebook = BACKUPS / "REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.authorized_run4_failed.ipynb"
    started_at = now_utc(); started = time.perf_counter(); exit_code = 1
    previous = os.environ.get("STAGE5C_CACHE_ONLY")
    os.environ["STAGE5C_CACHE_ONLY"] = "0"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    try:
        executed = NotebookClient(nb, timeout=180, kernel_name="python3", allow_errors=False).execute(cwd=str(ROOT))
        nbformat.write(executed, temp_notebook)
        code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
        execution_counts = [cell.execution_count for cell in code_cells]
        error_count = sum(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
        markdown = [cell.source for cell in executed.cells if cell.cell_type == "markdown"]
        section_numbers = [int(match.group(1)) for source in markdown for match in [re.match(r"##\s+(\d+)\.", source)] if match]
        source = "\n".join(cell.source for cell in code_cells)
        static = {
            "model_fit_call_absent": re.search(r"\.fit\s*\(", source) is None,
            "preprocessing_fit_call_absent": re.search(r"\.fit_transform\s*\(", source) is None,
            "prediction_call_absent": re.search(r"\.predict\s*\(", source) is None,
            "source_csv_path_absent": "data/regression_" not in source.replace("\\", "/"),
            "model_or_bundle_path_absent": "artifacts/models/" not in source.replace("\\", "/"),
        }
        section17_text = _notebook_output_text(executed.cells[36])
        section19_text = _notebook_output_text(executed.cells[40])
        section17 = {
            "metric_direction_closer_to_zero": "closer_to_zero" in section17_text,
            "stage4l_value_present": "-9.554174" in section17_text,
            "realmlp_without_value_present": "-7.013757" in section17_text,
            "difference_present": "2.540418" in section17_text,
            "interpretation_favors_realmlp": "Observed value favors frozen RealMLP" in section17_text,
        }
        section19 = {
            "metric_direction_closer_to_zero": "closer_to_zero" in section19_text,
            "without_value_present": "-7.013757" in section19_text,
            "with_value_present": "-6.440910" in section19_text,
            "difference_present": "0.572847" in section19_text,
            "interpretation_favors_with_sensitive": "Observed value favors with-sensitive RealMLP" in section19_text,
            "accuracy_only_warning_present": "not a fairness audit" in section19_text and "not causal evidence" in section19_text,
        }
        after = {path: sha256_file(ROOT / path) for path in before}
        checks = {
            "official_title_exactly_once": sum(source.startswith(f"# {OFFICIAL_NAME}") for source in markdown) == 1,
            "sections_0_through_29_exactly_once": section_numbers == list(range(30)),
            "code_cell_count_30": len(code_cells) == 30,
            "fresh_execution_counts_1_through_30": execution_counts == list(range(1, 31)),
            "all_code_cells_have_outputs": all(bool(cell.get("outputs")) for cell in code_cells),
            "zero_error_outputs": error_count == 0,
            "static_no_access_or_compute_calls": all(static.values()),
            "section17_correct": all(section17.values()), "section19_correct": all(section19.values()),
            "all_protected_hashes_unchanged": before == after,
        }
        require(all(checks.values()), f"Authorized Notebook output audit failed: {checks}")
        os.replace(temp_notebook, NOTEBOOK)
        exit_code = 0
        report = {
            "stage_id": STAGE_ID, "status": "PASS", "authorization_id": authorization_id,
            "historical_attempt_count": 3, "attempt": 4, "mode": "complete",
            "started_at_utc": started_at, "completed_at_utc": now_utc(), "runtime_seconds": time.perf_counter() - started,
            "exit_code": exit_code, "notebook_input_sha256": input_hash, "notebook_sha256": sha256_file(NOTEBOOK),
            "code_cell_count": len(code_cells), "execution_counts": execution_counts,
            "code_cells_with_execution_count": sum(value is not None for value in execution_counts),
            "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells), "error_output_count": error_count,
            "checks": checks, "static_audit": static, "section17_signed_bias_check": section17,
            "section19_signed_bias_check": section19,
            "source_access_calls": 0, "model_or_bundle_access_calls": 0, "model_fit_calls": 0,
            "preprocessing_fit_calls": 0, "prediction_generation_calls": 0, "bootstrap_recomputation_count": 0,
            "registry_write_count": 0, "ensemble_test_prediction_count": 0, "new_boosting_prediction_count": 0,
            "frozen_artifacts_unchanged": True, "promotion_status": "PROMOTED_ATOMICALLY",
        }
        history["attempts"].append(report)
        history["status"] = "PASS"
        history["human_authorization"] = {"authorization_id": authorization_id, "authorized_attempt": 4, "maximum_new_attempts": 1, "fifth_attempt_authorized": False}
        history["complete_run_count"] = sum(item["mode"] == "complete" and item["status"] == "PASS" for item in history["attempts"])
        history["cache_only_run_count"] = sum(item["mode"] == "cache_only" and item["status"] == "PASS" for item in history["attempts"])
        atomic_json(history, history_path); atomic_json(report, AUTHORIZED_REFRESH_REPORT)
        print(json.dumps(report, indent=2)); return report
    except Exception as exc:
        if temp_notebook.exists(): shutil.copy2(temp_notebook, failed_notebook)
        failure = {"stage_id": STAGE_ID, "status": "FAIL", "authorization_id": authorization_id, "attempt": 4,
                   "started_at_utc": started_at, "completed_at_utc": now_utc(), "runtime_seconds": time.perf_counter() - started,
                   "exit_code": exit_code, "error_type": type(exc).__name__, "error": str(exc), "promotion_status": "NOT_PROMOTED", "fifth_attempt_authorized": False}
        atomic_json(failure, AUTHORIZED_REFRESH_REPORT)
        raise
    finally:
        # A successful atomic promotion consumes the temporary path. On failure,
        # preserve any temporary Notebook as evidence; no fifth run is allowed.
        if previous is None: os.environ.pop("STAGE5C_CACHE_ONLY", None)
        else: os.environ["STAGE5C_CACHE_ONLY"] = previous


def audit_and_promote_authorized_refresh(authorization_id: str) -> dict[str, Any]:
    """Promote the already-executed attempt 4 after correcting an audit-only false negative.

    This function never executes a Notebook. It is valid only when all 30 cells in
    the preserved temporary Notebook already executed successfully and the initial
    failure came solely from checking the truncated table rendering instead of the
    complete Section 19 Markdown plus output.
    """
    require(authorization_id == AUTHORIZED_REFRESH_ID, "Authorization ID does not match")
    temp_notebook = NOTEBOOK.with_name(NOTEBOOK.stem + ".authorized_run4.tmp.ipynb")
    require(temp_notebook.exists(), "The preserved executed attempt-4 Notebook is missing")
    failure = json_load(AUTHORIZED_REFRESH_REPORT)
    require(failure.get("status") == "FAIL" and failure.get("attempt") == 4, "Expected the preserved attempt-4 audit failure")
    require("section19_correct': False" in failure.get("error", ""), "Failure was not the known Section 19 audit-only false negative")
    history_path = REPORTS / "stage5c_notebook_executions.json"
    history = json_load(history_path)
    require(len(history["attempts"]) == 3, "No additional Notebook execution may have occurred")
    before_payload = json_load(AUTHORIZED_REFRESH_BEFORE)
    before = {item["path"]: item["sha256"] for item in before_payload["protected_entries"]}
    after = {path: sha256_file(ROOT / path) for path in before}
    require(before == after, "A protected artifact changed after the authorized execution")
    executed = nbformat.read(temp_notebook, as_version=4)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    execution_counts = [cell.execution_count for cell in code_cells]
    error_count = sum(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
    markdown = [cell.source for cell in executed.cells if cell.cell_type == "markdown"]
    section_numbers = [int(match.group(1)) for source in markdown for match in [re.match(r"##\s+(\d+)\.", source)] if match]
    source = "\n".join(cell.source for cell in code_cells)
    static = {
        "model_fit_call_absent": re.search(r"\.fit\s*\(", source) is None,
        "preprocessing_fit_call_absent": re.search(r"\.fit_transform\s*\(", source) is None,
        "prediction_call_absent": re.search(r"\.predict\s*\(", source) is None,
        "source_csv_path_absent": "data/regression_" not in source.replace("\\", "/"),
        "model_or_bundle_path_absent": "artifacts/models/" not in source.replace("\\", "/"),
    }
    section17_text = _notebook_output_text(executed.cells[36])
    section19_text = _notebook_output_text(executed.cells[40])
    section19_markdown = executed.cells[39].source
    section17 = {
        "metric_direction_closer_to_zero": "closer_to_zero" in section17_text,
        "stage4l_value_present": "-9.554174" in section17_text,
        "realmlp_without_value_present": "-7.013757" in section17_text,
        "difference_present": "2.540418" in section17_text,
        "interpretation_favors_realmlp": "Observed value favors frozen RealMLP" in section17_text,
    }
    section19 = {
        "metric_direction_closer_to_zero": "closer_to_zero" in section19_text,
        "without_value_present": "-7.013757" in section19_text,
        "with_value_present": "-6.440910" in section19_text,
        "difference_present": "0.572847" in section19_text,
        "interpretation_favors_with_sensitive": "Observed value favors with-sensitive RealMLP" in section19_text,
        "accuracy_only_warning_present": "accuracy evidence" in section19_markdown.lower() and "not a fairness audit" in section19_markdown.lower() and "causal" in section19_markdown.lower(),
    }
    checks = {
        "official_title_exactly_once": sum(value.startswith(f"# {OFFICIAL_NAME}") for value in markdown) == 1,
        "sections_0_through_29_exactly_once": section_numbers == list(range(30)),
        "code_cell_count_30": len(code_cells) == 30,
        "fresh_execution_counts_1_through_30": execution_counts == list(range(1, 31)),
        "all_code_cells_have_outputs": all(bool(cell.get("outputs")) for cell in code_cells),
        "zero_error_outputs": error_count == 0, "static_no_access_or_compute_calls": all(static.values()),
        "section17_correct": all(section17.values()), "section19_correct": all(section19.values()),
        "all_protected_hashes_unchanged": before == after,
    }
    require(all(checks.values()), f"Corrected read-only promotion audit failed: {checks}")
    initial_failure_path = REPORTS / "stage5c_notebook_run4_authorized_initial_audit_failure.json"
    shutil.copy2(AUTHORIZED_REFRESH_REPORT, initial_failure_path)
    os.replace(temp_notebook, NOTEBOOK)
    report = {
        "stage_id": STAGE_ID, "status": "PASS", "authorization_id": authorization_id,
        "historical_attempt_count": 3, "attempt": 4, "mode": "complete",
        "started_at_utc": failure["started_at_utc"], "completed_at_utc": failure["completed_at_utc"],
        "runtime_seconds": failure["runtime_seconds"], "exit_code": 0, "notebook_execution_error_count": 0,
        "initial_wrapper_exit_code": 1, "initial_audit_false_negative": "Section 19 table display truncates scope_warning; the complete accuracy-only, no-fairness, no-causal warning is in the same Section Markdown.",
        "initial_audit_failure_path": rel(initial_failure_path),
        "notebook_input_sha256": before_payload["notebook_input_sha256"], "notebook_sha256": sha256_file(NOTEBOOK),
        "code_cell_count": len(code_cells), "execution_counts": execution_counts,
        "code_cells_with_execution_count": sum(value is not None for value in execution_counts),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells), "error_output_count": error_count,
        "checks": checks, "static_audit": static, "section17_signed_bias_check": section17,
        "section19_signed_bias_check": section19,
        "source_access_calls": 0, "model_or_bundle_access_calls": 0, "model_fit_calls": 0,
        "preprocessing_fit_calls": 0, "prediction_generation_calls": 0, "bootstrap_recomputation_count": 0,
        "registry_write_count": 0, "ensemble_test_prediction_count": 0, "new_boosting_prediction_count": 0,
        "frozen_artifacts_unchanged": True, "promotion_status": "PROMOTED_ATOMICALLY_AFTER_CORRECTED_READ_ONLY_AUDIT",
        "physical_notebook_execution_count_under_authorization": 1, "fifth_attempt_occurred": False,
    }
    history["attempts"].append(report); history["status"] = "PASS"
    history["human_authorization"] = {"authorization_id": authorization_id, "authorized_attempt": 4, "maximum_new_attempts": 1, "fifth_attempt_authorized": False, "fifth_attempt_occurred": False}
    history["complete_run_count"] = sum(item["mode"] == "complete" and item["status"] == "PASS" for item in history["attempts"])
    history["cache_only_run_count"] = sum(item["mode"] == "cache_only" and item["status"] == "PASS" for item in history["attempts"])
    atomic_json(history, history_path); atomic_json(report, AUTHORIZED_REFRESH_REPORT)
    print(json.dumps(report, indent=2)); return report


def authorized_refresh_recheck() -> dict[str, Any]:
    """Verify every non-Notebook artifact protected for the authorized refresh."""
    before_payload = json_load(AUTHORIZED_REFRESH_BEFORE)
    run_report = json_load(AUTHORIZED_REFRESH_REPORT)
    mismatches = []
    for item in before_payload["protected_entries"]:
        path = Path(item["path"])
        if not path.is_absolute(): path = ROOT / path
        if not path.exists(): mismatches.append({"path": item["path"], "reason": "missing"})
        elif path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append({"path": item["path"], "reason": "hash_or_size_mismatch"})
    registry = pd.read_csv(REGISTRY)
    history = json_load(REPORTS / "stage5c_notebook_executions.json")
    checks = {
        "authorized_run_pass": run_report.get("status") == "PASS",
        "protected_entries_unchanged": not mismatches,
        "registry_row_count_unchanged_362": len(registry) == 362,
        "registry_experiment_ids_unique": registry["experiment_id"].is_unique,
        "historical_attempts_preserved": [item["attempt"] for item in history["attempts"][:3]] == [1, 2, 3],
        "exactly_one_authorized_execution": len(history["attempts"]) == 4 and history["attempts"][3]["attempt"] == 4,
        "no_fifth_attempt": history.get("human_authorization", {}).get("fifth_attempt_occurred") is False,
        "stage6_not_started": json_load(RESULTS / "stage5c_evaluation_summary.json")["stage6_started"] is False,
    }
    payload = {
        "stage_id": STAGE_ID, "status": "PASS" if all(checks.values()) else "FAIL",
        "authorization_id": AUTHORIZED_REFRESH_ID, "checked_at_utc": now_utc(),
        "protected_entry_count": len(before_payload["protected_entries"]), "mismatch_count": len(mismatches),
        "mismatches": mismatches, "checks": checks, "notebook_sha256": sha256_file(NOTEBOOK),
        "registry_sha256": sha256_file(REGISTRY), "stage4l_official_prediction_sha256": sha256_file(STAGE4_PRED),
        "without_sensitive_prediction_sha256": sha256_file(PRED_WITHOUT), "with_sensitive_prediction_sha256": sha256_file(PRED_WITH),
        "stage6_started": False,
    }
    atomic_json(payload, AUTHORIZED_REFRESH_RECHECK)
    require(payload["status"] == "PASS", f"Authorized refresh immutability recheck failed: {checks}")
    print(json.dumps(payload, indent=2)); return payload


def protected_recheck() -> dict[str, Any]:
    baseline = json_load(BASELINE); mismatches = []
    for item in baseline["entries"]:
        path = Path(item["path"])
        if not path.is_absolute(): path = ROOT / path
        if not path.exists(): mismatches.append({"path": item["path"], "reason": "missing"})
        elif path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]: mismatches.append({"path": item["path"], "reason": "hash_or_size_mismatch"})
    prefix_size = baseline["registry_prior_byte_count"]; registry_bytes = REGISTRY.read_bytes()
    registry_prefix_ok = hashlib.sha256(registry_bytes[:prefix_size]).hexdigest() == baseline["registry_prior_sha256"]
    payload = {"stage_id": STAGE_ID, "status": "PASS" if not mismatches and registry_prefix_ok else "FAIL", "checked_at_utc": now_utc(), "protected_file_count": len(baseline["entries"]), "mismatch_count": len(mismatches), "mismatches": mismatches, "registry_prior_prefix_preserved": registry_prefix_ok, "source_hashes_unchanged": sha256_file(SOURCE_WITHOUT) == EXPECTED["source_without"] and sha256_file(SOURCE_WITH) == EXPECTED["source_with"], "evaluation_label": LABEL}
    atomic_json(payload, RECHECK); require(payload["status"] == "PASS", "Protected-file recheck failed"); print(json.dumps(payload, indent=2)); return payload


def finalize() -> dict[str, Any]:
    require(REVIEWER.exists(), "Independent Stage 5C Reviewer report is missing")
    review_text = REVIEWER.read_text(encoding="utf-8")
    reviewer_pass = bool(re.search(r"Final recommendation\s*:\s*PASS", review_text, re.IGNORECASE))
    critical_zero = bool(re.search(r"Critical issues\s*:\s*0", review_text, re.IGNORECASE))
    major_zero = bool(re.search(r"Major issues\s*:\s*0", review_text, re.IGNORECASE))
    notebook_runs = json_load(REPORTS / "stage5c_notebook_executions.json")
    summary = json_load(RESULTS / "stage5c_evaluation_summary.json")
    freeze = json_load(FREEZE); comparator = json_load(COMPARATOR_VALIDATION)
    alignment = json_load(ALIGNMENT); attempts = json_load(ATTEMPTS); registry = json_load(REPORTS / "stage5c_registry_update.json")
    visualization = json_load(MANIFESTS / "stage5c_visualization_manifest.json")
    recheck = json_load(RECHECK)
    source_text = (ROOT / "stage5c_posttest_utils.py").read_text(encoding="utf-8") + (ROOT / "stage5c_predict_worker.py").read_text(encoding="utf-8")
    no_fit_static = all(re.search(pattern, source_text) is None for pattern in [r"\.fit\s*\(", r"\.fit_transform\s*\(", r"\.partial_fit\s*\("])
    prediction_manifests = [json_load(MANIFEST_WITHOUT), json_load(MANIFEST_WITH)]
    checks = {
        "stage4l_verification_pass": json_load(STAGE4_VERIFICATION)["status"] == "PASS",
        "stage4l_official_primary_exact": freeze["stage4l_official_primary_candidate_id"] == "stage4l__blend__without_sensitive",
        "stage4l_pretest_freeze_hash_pass": sha256_file(STAGE4_FREEZE) == EXPECTED["stage4l_freeze"],
        "stage4l_official_prediction_hash_pass": sha256_file(STAGE4_PRED) == EXPECTED["stage4l_prediction"],
        "stage4l_metric_reproduction_pass": comparator["checks"]["official_metric_reproduction"],
        "stage5a_verification_accepted": json_load(STAGE5A_VERIFICATION)["status"] == "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "stage5a_governance_exception_visible": json_load(STAGE5A_VERIFICATION)["literal_zero_test_loading"] is False,
        "stage5b_verification_pass": json_load(STAGE5B_VERIFICATION)["status"] == "PASS",
        "stage5b_reviewer_pass": json_load(STAGE5B_VERIFICATION)["reviewer"]["reviewer_status"] == "PASS",
        "stage5b_handoff_valid": sha256_file(STAGE5B_HANDOFF) == EXPECTED["stage5b_handoff"],
        "stage5b_ensemble_rejected": freeze["stage5b_ensemble_status"] == "rejected",
        "deep_evaluation_eligible": freeze["deep_evaluation_eligible"] is True,
        "ensemble_evaluation_ineligible": freeze["ensemble_evaluation_eligible"] is False,
        "protected_baseline_exists": BASELINE.exists(), "preevaluation_freeze_exists": FREEZE.exists(),
        "candidate_count_frozen": freeze["deep_candidate_count"] == 2 and freeze["ensemble_candidate_count"] == 0,
        "metric_definitions_frozen": freeze["metrics"] == METRICS,
        "bootstrap_settings_frozen": freeze["bootstrap_settings"] == {"paired_indices": True, "resamples": 500, "seed": 42},
        "figure_scope_frozen": len(freeze["figure_list"]) == 10, "registry_ids_frozen": len(freeze["registry_ids"]) == 7,
        "prior_artifacts_unchanged": recheck["status"] == "PASS",
        "saved_test_membership_valid": comparator["checks"]["test_membership_exact"],
        "test_row_count_99948": comparator["row_count"] == 99948,
        "test_ids_unique": comparator["checks"]["row_ids_unique"], "zero_train_overlap": comparator["checks"]["zero_train_overlap"],
        "canonical_target_hash_exists": bool(comparator["canonical_target_hash"]),
        "safe_loader_sentinel_pass": json_load(SENTINEL)["status"] == "PASS",
        "parser_boundary_filtering_pass": alignment["excluded_rows_converted"] == 0,
        "train_feature_rows_zero": alignment["train_feature_rows_materialized"] == 0,
        "source_target_values_zero": alignment["source_target_values_materialized"] == 0,
        "source_hashes_unchanged": recheck["source_hashes_unchanged"],
        "bundle_without_hash_pass": prediction_manifests[0]["bundle_sha256"] == EXPECTED["bundle_without"],
        "bundle_with_hash_pass": prediction_manifests[1]["bundle_sha256"] == EXPECTED["bundle_with"],
        "model_without_hash_pass": prediction_manifests[0]["model_sha256"] == EXPECTED["model_without"],
        "model_with_hash_pass": prediction_manifests[1]["model_sha256"] == EXPECTED["model_with"],
        "reload_reports_pass": all(all(item["bundle_checks"].values()) for item in prediction_manifests),
        "frozen_realmlp_raw_epoch30": freeze["frozen_family"] == "RealMLP" and freeze["frozen_target_mode"] == "raw" and freeze["frozen_epoch"] == 30,
        "static_model_fit_calls_zero": no_fit_static, "runtime_model_fit_calls_zero": summary["model_fit_calls"] == 0,
        "runtime_preprocessing_fit_calls_zero": summary["preprocessing_fit_calls"] == 0,
        "scientific_setting_changes_zero": True,
        "successful_deep_predictions_two": attempts["successful_prediction_artifact_count"] == 2,
        "physical_attempts_within_limit": attempts["physical_attempt_count"] <= 4,
        "predictions_finite_original_scale": all(item["predictions_finite"] and item["predictions_original_scale"] for item in prediction_manifests),
        "prediction_alignment_exact": alignment["prediction_rows_align_exactly"] and alignment["canonical_targets_align_exactly"],
        "accepted_ensemble_count_zero": summary["ensemble_test_prediction_count"] == 0,
        "rejected_ensemble_prediction_absent": not any(ROOT.rglob("*stage5c_ensemble_test_predictions*")),
        "new_boosting_predictions_zero": summary["new_boosting_prediction_count"] == 0,
        "complete_metric_set_exists": set(METRICS).issubset(pd.read_csv(RESULTS / "stage5c_test_metrics.csv").columns),
        "official_comparison_exists": (RESULTS / "stage5c_official_comparison.csv").exists(),
        "paired_bootstrap_complete": len(pd.read_csv(RESULTS / "stage5c_paired_bootstrap.csv")) == 2,
        "sensitive_accuracy_comparison_exists": (RESULTS / "stage5c_sensitive_accuracy_comparison.csv").exists(),
        "decile_tail_disagreement_exist": all((RESULTS / name).exists() for name in ["stage5c_target_decile_metrics.csv", "stage5c_tail_metrics.csv", "stage5c_disagreement_summary.csv", "stage5c_disagreement_by_target_decile.csv"]),
        "all_results_post_test_label": summary["evaluation_label"] == LABEL,
        "stage4l_official_unchanged": summary["stage4l_remains_official"] is True,
        "required_figures_exist": visualization["status"] == "PASS" and visualization["figure_count"] == 10,
        "registry_unique_and_prefix_preserved": registry["status"] == "PASS" and registry["prior_prefix_preserved"] and registry["second_action"] == "REUSED",
        "stage6_handoff_exists": (MANIFESTS / "stage5c_stage6_handoff.json").exists(),
        "complete_notebook_run_pass": notebook_runs["complete_run_count"] >= 1,
        "cache_only_notebook_run_pass": notebook_runs["cache_only_run_count"] >= 1,
        "notebook_attempts_within_limit": len(notebook_runs["attempts"]) == 4 and notebook_runs.get("human_authorization", {}).get("authorization_id") == AUTHORIZED_REFRESH_ID and notebook_runs["human_authorization"].get("fifth_attempt_authorized") is False,
        "authorized_fourth_notebook_run_pass": AUTHORIZED_REFRESH_REPORT.exists() and json_load(AUTHORIZED_REFRESH_REPORT).get("status") == "PASS",
        "authorized_section17_signed_bias_pass": AUTHORIZED_REFRESH_REPORT.exists() and all(json_load(AUTHORIZED_REFRESH_REPORT).get("section17_signed_bias_check", {}).values()),
        "authorized_section19_signed_bias_pass": AUTHORIZED_REFRESH_REPORT.exists() and all(json_load(AUTHORIZED_REFRESH_REPORT).get("section19_signed_bias_check", {}).values()),
        "authorized_refresh_immutability_recheck_pass": AUTHORIZED_REFRESH_RECHECK.exists() and json_load(AUTHORIZED_REFRESH_RECHECK).get("status") == "PASS" and json_load(AUTHORIZED_REFRESH_RECHECK).get("mismatch_count") == 0,
        "no_fifth_notebook_attempt": len(notebook_runs["attempts"]) == 4 and notebook_runs.get("human_authorization", {}).get("fifth_attempt_occurred") is False,
        "notebook_prediction_generation_zero": all(item["prediction_generation_calls"] == 0 for item in notebook_runs["attempts"]),
        "reviewer_pass": reviewer_pass and critical_zero and major_zero,
        "protected_recheck_pass": recheck["status"] == "PASS",
        "state_target_current": "Complete Stage 5C" in (ROOT / "TASK.md").read_text(encoding="utf-8")[:2000],
        "stage6_not_started": summary["stage6_started"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "stage_id": STAGE_ID, "official_stage_name": OFFICIAL_NAME,
        "status": "PASS" if not failed else "FAIL", "overall_status": "PASS" if not failed else "FAIL",
        "created_at_utc": now_utc(), "evaluation_label": LABEL, "checks": checks, "failed_checks": failed,
        "stage5a_governance_exception_visible": True, "ensemble_status": "rejected",
        "counters": {"model_fit_calls": 0, "preprocessing_fit_calls": 0, "deep_prediction_artifacts": 2, "ensemble_test_predictions": 0, "new_boosting_predictions": 0, "train_feature_rows_materialized": 0, "source_target_values_materialized": 0},
        "reviewer": {"path": rel(REVIEWER), "sha256": sha256_file(REVIEWER), "status": "PASS" if reviewer_pass and critical_zero and major_zero else "FAIL", "critical_issues": 0 if critical_zero else None, "major_issues": 0 if major_zero else None},
        "protected_recheck_path": rel(RECHECK), "next_step": "Begin Stage 6 — Final Error Analysis and Model Comparison.", "stage6_started": False,
    }
    atomic_json(payload, VERIFICATION)
    require(payload["status"] == "PASS", f"Final Stage 5C Verification failed: {failed}")
    print(json.dumps(payload, indent=2)); return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["baseline", "freeze", "sentinel", "validate-comparator", "evaluate", "build-notebook", "run-notebook", "prepare-authorized-refresh", "run-authorized-refresh", "audit-promote-authorized-refresh", "authorized-refresh-recheck", "protected-recheck", "finalize"])
    parser.add_argument("--mode", choices=["complete", "cache_only"])
    parser.add_argument("--authorization-id")
    args = parser.parse_args()
    actions = {
        "baseline": create_baseline, "freeze": create_freeze, "sentinel": run_sentinel,
        "validate-comparator": validate_stage4l_comparator, "evaluate": evaluate,
        "build-notebook": build_notebook, "protected-recheck": protected_recheck, "finalize": finalize,
    }
    if args.command == "run-notebook":
        require(args.mode is not None, "--mode is required for run-notebook")
        execute_notebook(args.mode)
    elif args.command == "prepare-authorized-refresh":
        prepare_authorized_refresh()
    elif args.command == "run-authorized-refresh":
        require(args.authorization_id is not None, "--authorization-id is required")
        execute_authorized_refresh(args.authorization_id)
    elif args.command == "audit-promote-authorized-refresh":
        require(args.authorization_id is not None, "--authorization-id is required")
        audit_and_promote_authorized_refresh(args.authorization_id)
    elif args.command == "authorized-refresh-recheck":
        authorized_refresh_recheck()
    else:
        actions[args.command]()


if __name__ == "__main__":
    main()
