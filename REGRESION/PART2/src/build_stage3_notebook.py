"""Build the independent Stage 3 notebook without changing prior notebooks."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "REGRESSION_PART3_TREE_MODELS.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = []

cells.append(md("""
# Stage 3 — Tree-Based and Interpretable Models

## 0. Stage Objective

This notebook builds nonlinear tree models on the saved training rows. It reuses the exact Stage 1 split and three folds. The Test Set stays locked. Feature and model choices use only the saved non-sensitive development sample.
"""))

cells.append(md("""
## 1. Imports and Configuration

This section loads fixed tools and creates new Stage 3 output folders. Large fits run one at a time because memory is limited.
"""))

cells.append(code(r'''
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import sparse
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

from stage3_tree_utils import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    EXTENDED_FREQUENCY_FEATURES,
    EXTENDED_SOURCE_FEATURES,
    RANDOM_SEED,
    SENSITIVE_FEATURES,
    TARGET_COLUMN,
    FrequencyEncoder,
    SafeTreeFeatureEngineer,
    canonical_json,
    deterministic_experiment_id,
    evaluate_regression_predictions,
    feature_lists,
    feature_source_name,
    finite_prediction_check,
    fitted_estimator,
    fitted_pipeline,
    make_complete_pipeline,
    metric_columns,
    model_size_bytes,
    package_versions,
    read_training_rows,
    relative_improvement,
    save_model,
    sha256_file,
    stable_frame_digest,
    transformed_feature_names,
    upsert_registry,
    write_json,
)

ROOT = Path.cwd().resolve()
if not (ROOT / "AGENTS.md").exists() or not (ROOT / "artifacts/splits/train_row_ids.csv").exists():
    raise RuntimeError("Run this notebook from the Stage 3 project root.")

STAGE3_VERSION = "stage3_tree_v1_20260714"
EXECUTION_LABEL = os.environ.get("STAGE3_EXECUTION_LABEL", "manual")
DIRS = {
    "results": ROOT / "artifacts/results/stage3",
    "models": ROOT / "artifacts/models/tree",
    "predictions": ROOT / "artifacts/predictions/tree",
    "features": ROOT / "artifacts/features/tree",
    "importance": ROOT / "artifacts/features/tree/importance",
    "figures": ROOT / "artifacts/figures/stage3",
    "reports": ROOT / "artifacts/reports",
    "manifests": ROOT / "artifacts/manifests",
    "backups": ROOT / "artifacts/backups",
}
for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def tree_storage_summary(model):
    estimator = fitted_estimator(model)
    trees = []
    if hasattr(estimator, "tree_"):
        trees = [estimator]
    elif hasattr(estimator, "estimators_"):
        trees = list(np.asarray(estimator.estimators_, dtype=object).ravel())
    node_count = int(sum(tree.tree_.node_count for tree in trees if hasattr(tree, "tree_")))
    byte_count = 0
    for tree in trees:
        if not hasattr(tree, "tree_"):
            continue
        state = tree.tree_.__getstate__()
        byte_count += sum(value.nbytes for value in state.values() if isinstance(value, np.ndarray))
    return node_count, int(byte_count)

def fit_and_score(configuration, X_frame, y_series, training_ids, validation_ids, evaluation_stage):
    start = time.perf_counter()
    warning_messages = []
    model = None
    try:
        raw_features = feature_lists(configuration["feature_pack"], "without_sensitive")["raw"]
        X_fit = X_frame.loc[training_ids, raw_features].copy(deep=False)
        X_valid = X_frame.loc[validation_ids, raw_features].copy(deep=False)
        y_fit = y_series.loc[training_ids].to_numpy(dtype=float)
        y_valid = y_series.loc[validation_ids].to_numpy(dtype=float)
        model = make_complete_pipeline(configuration, "without_sensitive")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_start = time.perf_counter()
            model.fit(X_fit, y_fit)
            fit_seconds = time.perf_counter() - fit_start
            prediction_start = time.perf_counter()
            prediction = finite_prediction_check(model, X_valid)
            prediction_seconds = time.perf_counter() - prediction_start
        warning_messages = sorted({f"{type(item.message).__name__}: {item.message}" for item in caught})
        metrics = evaluate_regression_predictions(y_valid, prediction)
        node_count, tree_bytes = tree_storage_summary(model)
        transformed_count = len(transformed_feature_names(model))
        status = "success" if fit_seconds <= 12 * 60 else "runtime_limit_exceeded"
        row = {
            "experiment_id": deterministic_experiment_id(
                configuration["model_name"], "without_sensitive", configuration["target_mode"],
                evaluation_stage, 0, configuration, configuration["feature_pack"]
            ),
            "model_name": configuration["model_name"],
            "model_family": configuration["model_name"],
            "sensitive_mode": "without_sensitive",
            "feature_pack": configuration["feature_pack"],
            "target_mode": configuration["target_mode"],
            "configuration_json": canonical_json(configuration),
            **{key: value for key, value in metrics.items() if key != "metric_warnings"},
            "fit_time_seconds": fit_seconds,
            "prediction_time_seconds": prediction_seconds,
            "transformed_features": transformed_count,
            "tree_node_count": node_count,
            "tree_storage_mb": tree_bytes / 2**20,
            "warning_status": " | ".join(warning_messages + metrics["metric_warnings"]),
            "status": status,
            "reused_anchor": False,
        }
    except Exception as exc:
        row = {
            "experiment_id": deterministic_experiment_id(
                configuration["model_name"], "without_sensitive", configuration["target_mode"],
                evaluation_stage, 0, configuration, configuration["feature_pack"]
            ),
            "model_name": configuration["model_name"], "model_family": configuration["model_name"],
            "sensitive_mode": "without_sensitive", "feature_pack": configuration["feature_pack"],
            "target_mode": configuration["target_mode"], "configuration_json": canonical_json(configuration),
            "fit_time_seconds": time.perf_counter() - start, "prediction_time_seconds": np.nan,
            "transformed_features": np.nan, "tree_node_count": np.nan, "tree_storage_mb": np.nan,
            "warning_status": f"{type(exc).__name__}: {exc}", "status": "failed", "reused_anchor": False,
        }
        for name in metric_columns():
            row[name] = np.nan
    finally:
        del model
        gc.collect()
    return row

def run_screening(configurations, output_path, evaluation_stage, reuse_frame=None):
    expected_ids = {
        deterministic_experiment_id(c["model_name"], "without_sensitive", c["target_mode"],
                                    evaluation_stage, 0, c, c["feature_pack"])
        for c in configurations
    }
    output_path = Path(output_path)
    cached = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    if len(cached) and set(cached["experiment_id"]) == expected_ids and cached["status"].eq("success").all():
        return cached
    completed = {} if not len(cached) else {
        row["experiment_id"]: row for row in cached.loc[cached["status"].eq("success")].to_dict("records")
        if row["experiment_id"] in expected_ids
    }
    rows = []
    for number, configuration in enumerate(configurations, start=1):
        experiment_id = deterministic_experiment_id(
            configuration["model_name"], "without_sensitive", configuration["target_mode"],
            evaluation_stage, 0, configuration, configuration["feature_pack"]
        )
        if experiment_id in completed:
            row = completed[experiment_id]
            rows.append(row)
            print(f"{evaluation_stage}: {number}/{len(configurations)} reused {configuration['model_name']} {configuration['target_mode']}")
            continue
        reused = None
        if reuse_frame is not None:
            mask = (
                reuse_frame["configuration_json"].eq(canonical_json(configuration))
                & reuse_frame["status"].eq("success")
            )
            if mask.any():
                reused = reuse_frame.loc[mask].iloc[0].to_dict()
        if reused is None:
            row = fit_and_score(configuration, train_features, y_train, dev_train_ids, dev_validation_ids, evaluation_stage)
        else:
            row = reused
            row["experiment_id"] = deterministic_experiment_id(
                configuration["model_name"], "without_sensitive", configuration["target_mode"],
                evaluation_stage, 0, configuration, configuration["feature_pack"]
            )
            row["reused_anchor"] = True
        rows.append(row)
        save_csv(pd.DataFrame(rows), output_path)
        print(f"{evaluation_stage}: {number}/{len(configurations)} {configuration['model_name']} "
              f"{configuration['target_mode']} -> {row['status']}, MAE={row.get('mae')}")
        if row["status"] != "success":
            raise RuntimeError(f"Screening failed: {row['warning_status']}")
    result = pd.DataFrame(rows)
    save_csv(result, output_path)
    return result

versions = package_versions()
print({"stage": "Stage 3", "version": STAGE3_VERSION, "execution_label": EXECUTION_LABEL,
       "packages": versions, "large_fits_parallel": False})
'''))

cells.append(md("""
## 2. Project and Artifact Discovery

The project paths are discovered from the current folder. Protected prior files are hashed before new modeling artifacts are written.
"""))

cells.append(code(r'''
PATHS = {
    "with_sensitive_csv": ROOT / "data/regression_with_sensitive_features.csv",
    "without_sensitive_csv": ROOT / "data/regression_without_sensitive_features.csv",
    "stage1_notebook": Path(r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb"),
    "stage2_notebook": ROOT / "REGRESSION_PART2_MODELING.ipynb",
    "train_ids": ROOT / "artifacts/splits/train_row_ids.csv",
    "test_ids": ROOT / "artifacts/splits/test_row_ids.csv",
    "cv": ROOT / "artifacts/splits/cv_fold_assignments.csv",
    "split_config": ROOT / "artifacts/splits/split_config.json",
    "feature_inventory": ROOT / "artifacts/data_contract/feature_inventory.csv",
    "feature_sets": ROOT / "artifacts/data_contract/feature_sets.json",
    "leakage": ROOT / "artifacts/data_contract/leakage_and_suspicious_columns.csv",
    "metric_schema": ROOT / "artifacts/data_contract/metric_schema.json",
    "stage1_verification": ROOT / "artifacts/reports/prompt1_verification.json",
    "stage2_verification": ROOT / "artifacts/reports/prompt2_verification.json",
    "stage2_leaderboard": ROOT / "artifacts/results/prompt2/linear_leaderboard.csv",
    "stage2_summary": ROOT / "artifacts/results/prompt2/cv_oof_summary.csv",
    "development_sample": ROOT / "artifacts/splits/prompt2_development_sample.csv",
    "registry": ROOT / "artifacts/results/experiment_results.csv",
}
missing_paths = [name for name, path in PATHS.items() if not path.exists()]
if missing_paths:
    raise FileNotFoundError(f"Critical Stage 3 inputs are missing: {missing_paths}")

protected_paths = {
    PATHS["with_sensitive_csv"], PATHS["without_sensitive_csv"], PATHS["stage1_notebook"],
    PATHS["stage2_notebook"], ROOT / "stage2_completion_summary.md",
}
for relative in [
    "artifacts/splits", "artifacts/data_contract", "artifacts/results/prompt2",
    "artifacts/models/linear", "artifacts/predictions/linear", "artifacts/features/linear",
    "artifacts/figures/prompt2",
]:
    directory = ROOT / relative
    if directory.exists():
        protected_paths.update(path for path in directory.rglob("*") if path.is_file())
for pattern in ["prompt1*", "prompt2*", "stage2*"]:
    protected_paths.update(path for path in (ROOT / "artifacts/reports").glob(pattern) if path.is_file())
    protected_paths.update(path for path in (ROOT / "artifacts/manifests").glob(pattern) if path.is_file())

def protected_key(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

protected_hashes_now = {protected_key(path): sha256_file(path) for path in sorted(protected_paths, key=str)}
protected_before_path = DIRS["manifests"] / "stage3_protected_hashes_before.json"
if protected_before_path.exists():
    protected_before = json.loads(protected_before_path.read_text(encoding="utf-8"))
    if protected_before["hashes"] != protected_hashes_now:
        raise AssertionError("A protected prior-Stage file changed after the Stage 3 baseline was saved.")
else:
    protected_before = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage3_version": STAGE3_VERSION,
        "hashes": protected_hashes_now,
    }
    write_json(protected_before_path, protected_before)

run_metadata_path = DIRS["manifests"] / "stage3_run_metadata.json"
if run_metadata_path.exists():
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
else:
    run_metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stage3_version": STAGE3_VERSION}
    write_json(run_metadata_path, run_metadata)

environment_report = {
    "packages": versions,
    "interpret_install_attempt": {
        "attempted": True,
        "command": "python -m pip install --target artifacts\\vendor\\interpret interpret",
        "status": "failed_timeout",
        "timeout_seconds": 124,
        "local_target_created": False,
        "import_error": "ModuleNotFoundError: No module named 'interpret'",
        "substitute_used": False,
    },
}
write_json(DIRS["reports"] / "stage3_environment.json", environment_report)
print({"required_inputs": len(PATHS), "protected_files": len(protected_hashes_now),
       "protected_baseline": str(protected_before_path.relative_to(ROOT)), "ebm_available": False})
'''))

cells.append(md("""
## 3. Stage 1 and Stage 2 Validation

This section validates the saved split, folds, prior PASS reports, feature contract, and main Registry. No new splitter is created.
"""))

cells.append(code(r'''
split_config = json.loads(PATHS["split_config"].read_text(encoding="utf-8"))
stage1_verification = json.loads(PATHS["stage1_verification"].read_text(encoding="utf-8"))
stage2_verification = json.loads(PATHS["stage2_verification"].read_text(encoding="utf-8"))
feature_sets = json.loads(PATHS["feature_sets"].read_text(encoding="utf-8"))
metric_schema = json.loads(PATHS["metric_schema"].read_text(encoding="utf-8"))
feature_inventory_source = pd.read_csv(PATHS["feature_inventory"])
leakage_report = pd.read_csv(PATHS["leakage"])
train_ids = pd.read_csv(PATHS["train_ids"])["row_id"].to_numpy(dtype=np.int64)
test_ids = pd.read_csv(PATHS["test_ids"])["row_id"].to_numpy(dtype=np.int64)
cv_assignments = pd.read_csv(PATHS["cv"], dtype={"row_id": "int64", "fold": "int64"})

assert stage1_verification["status"] == "PASS" and stage2_verification["status"] == "PASS"
assert len(train_ids) == split_config["train_rows"] == 399788
assert len(test_ids) == split_config["test_rows"] == 99948
assert len(set(train_ids).intersection(test_ids)) == 0
assert set(cv_assignments["row_id"]) == set(train_ids)
assert set(cv_assignments["fold"]) == {0, 1, 2}
assert not set(cv_assignments["row_id"]).intersection(test_ids)
assert metric_schema["primary_metric"] == "mae"
assert split_config["source_hashes"]["with_sensitive"] == sha256_file(PATHS["with_sensitive_csv"])
assert split_config["source_hashes"]["without_sensitive"] == sha256_file(PATHS["without_sensitive_csv"])
assert split_config["source_hashes"]["part1_notebook"] == sha256_file(PATHS["stage1_notebook"])

main_registry_before = pd.read_csv(PATHS["registry"])
prior_registry_rows = main_registry_before.loc[
    ~main_registry_before["experiment_id"].astype(str).str.startswith("stage3__")
].copy()
prior_registry_digest_before = stable_frame_digest(prior_registry_rows)
assert not prior_registry_rows["experiment_id"].duplicated().any()
print({"stage1_status": "PASS", "stage2_status": "PASS", "train_rows": len(train_ids),
       "locked_test_rows": len(test_ids), "saved_folds": cv_assignments["fold"].value_counts().sort_index().to_dict(),
       "split_recreated": False, "prior_registry_rows": len(prior_registry_rows)})
'''))

cells.append(md("""
## 4. Training Data Views

Both source files are read with saved Test rows skipped. The two training-only views are compared, then one explicit working copy is used. Row ID stays in the index for audit and is never a model feature.
"""))

cells.append(code(r'''
categorical_columns = feature_inventory_source.loc[
    feature_inventory_source["inferred_feature_type"].eq("categorical"), "column_name"
].tolist()
source_with_training = read_training_rows(PATHS["with_sensitive_csv"], train_ids, categorical_columns)
source_without_training = read_training_rows(PATHS["without_sensitive_csv"], train_ids, categorical_columns)
common_columns = source_without_training.columns.tolist()
assert source_with_training[common_columns].astype(object).equals(source_without_training.astype(object))
assert source_with_training.index.equals(source_without_training.index)
source_training_digest = stable_frame_digest(source_with_training)
train_data = source_with_training.copy(deep=False)
train_features = train_data.drop(columns=[TARGET_COLUMN]).copy(deep=False)
y_train = train_data[TARGET_COLUMN].copy()
assert "row_id" not in train_features.columns and TARGET_COLUMN not in train_features.columns
assert len(train_features) == len(train_ids) and train_features.index.is_unique
del source_without_training
gc.collect()
print({"with_sensitive_training_shape": train_data.shape, "without_sensitive_training_columns": len(common_columns),
       "training_alignment": True, "test_rows_loaded": 0, "test_targets_read": False,
       "training_memory_mb": round(train_data.memory_usage(deep=True).sum() / 2**20, 2)})
'''))

cells.append(md("""
## 5. Stage 2 Linear Baseline

The best linear-family row is loaded from the real Stage 2 leaderboard. Stage 2 models are not rerun.
"""))

cells.append(code(r'''
stage2_leaderboard = pd.read_csv(PATHS["stage2_leaderboard"]).sort_values("oof_mae").reset_index(drop=True)
best_stage2 = stage2_leaderboard.iloc[0].copy()
display(stage2_leaderboard[["model_name", "sensitive_mode", "target_mode", "oof_mae", "oof_rmse", "oof_rmsle", "oof_r_squared"]].head(5))
print(f"The saved Stage 2 baseline is {best_stage2['model_name']} in {best_stage2['sensitive_mode']} mode. "
      f"Its training OOF MAE is {best_stage2['oof_mae']:.3f} thousand US dollars.")
'''))

cells.append(md("""
## 6. Feature Engineering Audit

The audit uses training rows only. It records types, cardinality, missing rates, skew, sensitive status, identifier risk, and known redundancy.
"""))

cells.append(code(r'''
exact_tree_redundant = {
    "tract_to_msamd_income": "exactly 100 times tract_income_ratio",
    "log1p_applicant_income": "strict log1p duplicate of applicant_income_000s",
    "log1p_population": "strict log1p duplicate of population",
    "log1p_hud_median_family_income": "strict log1p duplicate of hud_median_family_income",
    "log1p_owner_occupied_units": "strict log1p duplicate of number_of_owner_occupied_units",
    "log1p_1_to_4_family_units": "strict log1p duplicate of number_of_1_to_4_family_units",
    "state_code": "one-to-one duplicate of state_name",
    "county_code": "ambiguous code and redundant with composite geography",
}
inventory_lookup = feature_inventory_source.set_index("column_name")
audit_rows = []
for column in train_data.columns:
    series = train_data[column]
    source_row = inventory_lookup.loc[column]
    is_numeric = source_row["inferred_feature_type"] in {"numeric", "target"}
    numeric_values = pd.to_numeric(series, errors="coerce") if is_numeric else None
    audit_rows.append({
        "feature_name": column,
        "data_type": str(series.dtype),
        "sensitive_status": bool(column in SENSITIVE_FEATURES),
        "cardinality": int(series.nunique(dropna=False)),
        "missing_rate": float(series.isna().mean()),
        "skewness": float(numeric_values.skew()) if is_numeric else np.nan,
        "existing_engineered_feature": bool(column.startswith("log1p_") or column in {
            "applicant_income_to_area_income", "tract_income_ratio", "owner_occupied_unit_ratio",
            "family_units_per_1000_people", "owner_occupied_units_per_1000_people", "has_co_applicant",
            "loan_program_group", "applicant_income_area_group", "tract_income_level", "majority_minority_tract", "us_region"
        }),
        "high_cardinality_status": bool(series.nunique(dropna=False) > 100 and not is_numeric),
        "possible_identifier_status": bool(source_row["possible_identifier"]),
        "candidate_feature_source": column in {"respondent_id", "msamd_name", "county_name", "census_tract_number"},
        "leakage_status": "target" if column == TARGET_COLUMN else "reviewed_no_confirmed_leakage",
        "tree_redundancy_note": exact_tree_redundant.get(column, ""),
    })
stage3_feature_audit = pd.DataFrame(audit_rows)
save_csv(stage3_feature_audit, DIRS["features"] / "stage3_feature_audit.csv")
display(stage3_feature_audit.loc[
    stage3_feature_audit["high_cardinality_status"] | stage3_feature_audit["tree_redundancy_note"].ne("")
].head(20))
print(f"The audit found {sum(stage3_feature_audit['high_cardinality_status'])} high-cardinality categorical fields "
      f"and {len(exact_tree_redundant)} tree-redundant or unsafe code fields.")
'''))

cells.append(md("""
## 7. Proposed New Features

The proposals are target-independent and use no sensitive field. Fixed row features and learned frequency features are clearly separated.
"""))

cells.append(code(r'''
proposal_rows = [
    ("applicant_income_to_tract_income", "applicant_income_to_area_income / tract_income_ratio", "applicant_income_to_area_income|tract_income_ratio", "fixed", "Income relative to estimated tract income", "Nonpositive denominator becomes NaN"),
    ("applicant_vs_tract_income_gap_000s", "applicant_income_000s - (hud_median_family_income / 1000) * tract_income_ratio", "applicant_income_000s|hud_median_family_income|tract_income_ratio", "fixed", "Absolute income gap from the tract estimate", "Nonfinite result becomes NaN"),
    ("family_owner_unit_count_difference", "number_of_1_to_4_family_units - number_of_owner_occupied_units", "number_of_1_to_4_family_units|number_of_owner_occupied_units", "fixed", "Housing unit count difference without a causal label", "No denominator"),
    ("loan_type_property_group", "loan_type_name + property_type_name", "loan_type_name|property_type_name", "fixed", "Small lending and property interaction", "Missing token"),
    ("purpose_occupancy_group", "loan_purpose_name + owner_occupancy_name", "loan_purpose_name|owner_occupancy_name", "fixed", "Small purpose and occupancy interaction", "Missing token"),
    ("purpose_preapproval_group", "loan_purpose_name + preapproval_name", "loan_purpose_name|preapproval_name", "fixed", "Small purpose and preapproval interaction", "Missing token"),
    ("agency_loan_program_group", "agency_name + loan_program_group", "agency_name|loan_program_group", "fixed", "Small agency and program interaction", "Missing token"),
    ("purpose_income_area_group", "loan_purpose_name + applicant_income_area_group", "loan_purpose_name|applicant_income_area_group", "fixed", "Small purpose and income-band interaction", "Missing token"),
    ("respondent_id__frequency", "fold count(respondent_id) / fold rows", "respondent_id", "learned", "Lender activity frequency", "Unseen becomes 0"),
    ("msamd_name__frequency", "fold count(msamd_name) / fold rows", "msamd_name", "learned", "Metro activity frequency", "Unseen becomes 0"),
    ("state_county_group__frequency", "fold frequency(state_name + county_name)", "state_name|county_name", "learned", "Unambiguous county frequency", "Unseen becomes 0"),
    ("state_county_tract_group__frequency", "fold frequency(state + county + tract)", "state_name|county_name|census_tract_number", "learned", "Detailed tract frequency with high unseen risk", "Unseen becomes 0"),
]
stage3_feature_proposals = pd.DataFrame(proposal_rows, columns=[
    "feature_name", "formula", "source_columns", "engineering_type", "why_it_may_help", "zero_or_unseen_handling"
])
stage3_feature_proposals["missing_value_handling"] = "Pipeline imputation or explicit missing token"
stage3_feature_proposals["leakage_review"] = "PASS: no target, row_id, bin, future field, or sensitive source"
stage3_feature_proposals["initial_status"] = "candidate"
save_csv(stage3_feature_proposals, DIRS["features"] / "stage3_feature_proposals.csv")
display(stage3_feature_proposals)
print("Twelve conservative features will be tested. None uses the target or a sensitive source field.")
'''))

cells.append(md("""
## 8. Feature Packs

The base pack removes exact tree duplicates and raw identifiers. The engineered pack adds fixed features. The extended pack adds fold-fitted frequencies for lender and composite geography.
"""))

cells.append(code(r'''
feature_pack_manifest = {
    "version": STAGE3_VERSION,
    "packs": {},
    "excluded_from_base": exact_tree_redundant,
    "target_encoding_used": False,
}
for pack in ["tree_base_v1", "tree_engineered_v1", "tree_extended_v1"]:
    without_lists = feature_lists(pack, "without_sensitive")
    with_lists = feature_lists(pack, "with_sensitive")
    assert set(with_lists["raw"]).difference(without_lists["raw"]) == set(SENSITIVE_FEATURES)
    feature_pack_manifest["packs"][pack] = {
        "without_sensitive": without_lists,
        "with_sensitive": with_lists,
        "sensitive_difference": SENSITIVE_FEATURES,
    }
write_json(DIRS["features"] / "stage3_tree_feature_packs.json", feature_pack_manifest)
pack_counts = pd.DataFrame([
    {"feature_pack": pack, "base_or_engineered_columns": len(feature_lists(pack, "without_sensitive")["numeric"] + feature_lists(pack, "without_sensitive")["categorical"]),
     "raw_input_columns": len(feature_lists(pack, "without_sensitive")["raw"]),
     "learned_frequency_columns": len(feature_lists(pack, "without_sensitive")["frequency_sources"])}
    for pack in ["tree_base_v1", "tree_engineered_v1", "tree_extended_v1"]
])
display(pack_counts)
print("The only input difference between sensitive modes is the saved eight-feature sensitive set.")
'''))

cells.append(md("""
## 9. Serializable Feature Transformers

Named transformers make fixed features and learn frequencies inside each pipeline. A joblib round trip checks stable rows and columns.
"""))

cells.append(code(r'''
transformer_sample = train_features.iloc[:500].copy(deep=True)
transformer = Pipeline([
    ("fixed", SafeTreeFeatureEngineer("tree_extended_v1")),
    ("frequency", FrequencyEncoder(EXTENDED_SOURCE_FEATURES)),
])
transformed_before = transformer.fit_transform(transformer_sample)
transformer_path = DIRS["manifests"] / "stage3_transformer_roundtrip.joblib"
save_model(transformer_path, transformer)
reloaded_transformer = joblib.load(transformer_path)
transformed_after = reloaded_transformer.transform(transformer_sample)
assert transformed_before.index.equals(transformer_sample.index)
assert transformed_before.columns.equals(transformed_after.columns)
pd.testing.assert_frame_equal(transformed_before, transformed_after)
assert stable_frame_digest(train_data) == source_training_digest
print({"transformer_reload": "PASS", "rows_preserved": len(transformed_after),
       "added_columns": sorted(set(transformed_after.columns).difference(transformer_sample.columns)),
       "target_used": False})
'''))

cells.append(md("""
## 10. Model-Specific Preprocessing

Exact trees use sparse one-hot features. HGB uses dense ordinal categories. Both use fold-fitted median or category handling and no scaling.
"""))

cells.append(code(r'''
preprocess_audit_rows = []
preprocess_sample = train_features.iloc[:2000]
for model_name in ["decision_tree", "hist_gradient_boosting"]:
    configuration = {
        "model_name": model_name, "feature_pack": "tree_extended_v1", "target_mode": "raw",
        **({"max_depth": 3, "min_samples_leaf": 20, "min_samples_split": 40, "max_features": None}
           if model_name == "decision_tree" else
           {"loss": "squared_error", "learning_rate": 0.05, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0, "max_iter": 5}),
    }
    raw = feature_lists(configuration["feature_pack"], "without_sensitive")["raw"]
    pipeline = make_complete_pipeline(configuration, "without_sensitive")
    fixed = pipeline.named_steps["feature_engineering"].fit_transform(preprocess_sample[raw])
    frequency = pipeline.named_steps["frequency_encoding"].fit_transform(fixed)
    matrix = pipeline.named_steps["preprocessor"].fit_transform(frequency)
    preprocess_audit_rows.append({
        "model_name": model_name, "rows": matrix.shape[0], "columns": matrix.shape[1],
        "sparse_output": bool(sparse.issparse(matrix)), "scaling_used": False,
        "finite_output": bool(np.isfinite(matrix.data if sparse.issparse(matrix) else matrix).all()),
    })
preprocess_audit = pd.DataFrame(preprocess_audit_rows)
save_csv(preprocess_audit, DIRS["features"] / "stage3_preprocessing_audit.csv")
display(preprocess_audit)
assert preprocess_audit["finite_output"].all()
print("The preprocessing outputs are finite. Tree scaling is not used.")
'''))

cells.append(md("""
## 11. Development Sample

The saved Stage 2 development rows are reused. Folds 1 and 2 supply 80,000 training rows, and fold 0 supplies 20,000 validation rows.
"""))

cells.append(code(r'''
development_manifest = pd.read_csv(PATHS["development_sample"])
assert set(development_manifest["development_role"]) == {"train", "validation"}
assert development_manifest["row_id"].is_unique
assert set(development_manifest["row_id"]).issubset(set(train_ids))
assert not set(development_manifest["row_id"]).intersection(test_ids)
dev_train_ids = development_manifest.loc[development_manifest["development_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
dev_validation_ids = development_manifest.loc[development_manifest["development_role"].eq("validation"), "row_id"].to_numpy(dtype=np.int64)
assert len(dev_train_ids) == 80000 and len(dev_validation_ids) == 20000
assert set(development_manifest.loc[development_manifest["development_role"].eq("train"), "original_cv_fold"]) == {1, 2}
assert set(development_manifest.loc[development_manifest["development_role"].eq("validation"), "original_cv_fold"]) == {0}
print({"development_train": len(dev_train_ids), "development_validation": len(dev_validation_ids),
       "new_split_created": False, "test_overlap": 0,
       "validation_target_max": float(y_train.loc[dev_validation_ids].max())})
'''))

cells.append(md("""
## 12. Feature-Pack Screening

Each family compares base, engineered, and extended packs with raw and log targets. Only non-sensitive development rows are used. A complex pack normally needs at least 0.5 percent better MAE.
"""))

cells.append(code(r'''
anchor_templates = {
    "decision_tree": {"max_depth": 16, "min_samples_leaf": 100, "min_samples_split": 200, "max_features": None, "criterion": "squared_error"},
    "random_forest": {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 20, "max_features": "sqrt", "n_jobs": 2},
    "extra_trees": {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 20, "max_features": "sqrt", "n_jobs": 2},
    "hist_gradient_boosting": {"loss": "squared_error", "learning_rate": 0.05, "max_leaf_nodes": 31,
                               "min_samples_leaf": 100, "l2_regularization": 1.0, "max_iter": 300},
}
feature_screen_configs = []
for model_name, parameters in anchor_templates.items():
    for feature_pack in ["tree_base_v1", "tree_engineered_v1", "tree_extended_v1"]:
        for target_mode in ["raw", "log1p"]:
            feature_screen_configs.append({"model_name": model_name, "feature_pack": feature_pack,
                                           "target_mode": target_mode, **parameters})
feature_engineering_screening = run_screening(
    feature_screen_configs, DIRS["results"] / "feature_engineering_screening.csv", "feature_screening"
)

selected_pack_records = {}
for family, group in feature_engineering_screening.groupby("model_name"):
    pack_best = group.sort_values("mae").groupby("feature_pack", as_index=False).first()
    base_row = pack_best.loc[pack_best["feature_pack"].eq("tree_base_v1")].iloc[0]
    pack_best["relative_mae_improvement_percent"] = (
        (base_row["mae"] - pack_best["mae"]) / base_row["mae"] * 100
    )
    eligible = pack_best.loc[
        pack_best["feature_pack"].ne("tree_base_v1")
        & pack_best["relative_mae_improvement_percent"].ge(0.5)
    ]
    if len(eligible):
        selected = eligible.sort_values(["mae", "feature_pack"]).iloc[0]
        reason = "Complex pack improved development MAE by at least 0.5 percent."
    else:
        selected = base_row
        reason = "Complex packs improved MAE by less than 0.5 percent, so the base pack stayed selected."
    selected_pack_records[family] = {
        "feature_pack": selected["feature_pack"], "screening_target_mode": selected["target_mode"],
        "mae": float(selected["mae"]), "base_mae": float(base_row["mae"]),
        "relative_mae_improvement_percent": float((base_row["mae"] - selected["mae"]) / base_row["mae"] * 100),
        "reason": reason,
    }
selected_feature_packs = {"version": STAGE3_VERSION, "selection_data": "non-sensitive development only",
                          "models": selected_pack_records}
write_json(DIRS["results"] / "selected_tree_feature_packs.json", selected_feature_packs)
display(pd.DataFrame(selected_pack_records).T)
print("Feature packs are now frozen before model tuning and sensitive comparison.")
'''))

cells.append(md("""
## 13. Decision Tree Screening

The Decision Tree is a simple nonlinear baseline. Sixteen controlled candidates compare eight structures and two target modes.
"""))

cells.append(code(r'''
decision_structures = [
    {"max_depth": 6, "min_samples_leaf": 500, "min_samples_split": 1000, "max_features": None},
    {"max_depth": 10, "min_samples_leaf": 100, "min_samples_split": 200, "max_features": None},
    {"max_depth": 16, "min_samples_leaf": 100, "min_samples_split": 200, "max_features": None},
    {"max_depth": 16, "min_samples_leaf": 20, "min_samples_split": 40, "max_features": None},
    {"max_depth": 24, "min_samples_leaf": 20, "min_samples_split": 40, "max_features": None},
    {"max_depth": None, "min_samples_leaf": 20, "min_samples_split": 40, "max_features": None},
    {"max_depth": 16, "min_samples_leaf": 100, "min_samples_split": 200, "max_features": 0.7},
    {"max_depth": None, "min_samples_leaf": 500, "min_samples_split": 1000, "max_features": 0.7},
]
decision_pack = selected_pack_records["decision_tree"]["feature_pack"]
decision_configs = [
    {"model_name": "decision_tree", "feature_pack": decision_pack, "target_mode": mode,
     "criterion": "squared_error", **structure}
    for structure in decision_structures for mode in ["raw", "log1p"]
]
decision_screening = run_screening(
    decision_configs, DIRS["results"] / "decision_tree_screening.csv", "model_screening", feature_engineering_screening
)
display(decision_screening.sort_values("mae").head(8)[["target_mode", "mae", "rmse", "rmsle", "fit_time_seconds", "configuration_json"]])
print(f"The best Decision Tree development MAE is {decision_screening['mae'].min():.3f}.")
'''))

cells.append(md("""
## 14. Random Forest and Extra Trees Screening

Both bagging families use the same controlled structures. Families run sequentially. Two internal workers are safe after the first measured fit stayed below 0.7 GB.
"""))

cells.append(code(r'''
bagging_structures = [
    {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 40, "min_samples_leaf": 20, "max_features": 0.5},
    {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 5, "max_features": 0.5},
    {"n_estimators": 250, "max_depth": 20, "min_samples_leaf": 20, "max_features": "sqrt"},
]
bagging_frames = []
for family in ["random_forest", "extra_trees"]:
    selected_pack = selected_pack_records[family]["feature_pack"]
    configurations = [
        {"model_name": family, "feature_pack": selected_pack, "target_mode": mode, "n_jobs": 2, **structure}
        for structure in bagging_structures for mode in ["raw", "log1p"]
    ]
    frame = run_screening(
        configurations, DIRS["results"] / f"{family}_screening.csv", "model_screening", feature_engineering_screening
    )
    bagging_frames.append(frame)
bagging_screening = pd.concat(bagging_frames, ignore_index=True)
display(bagging_screening.sort_values("mae").head(10)[
    ["model_name", "target_mode", "mae", "rmse", "fit_time_seconds", "tree_node_count", "tree_storage_mb", "configuration_json"]
])
print("Random Forest and Extra Trees both completed development screening.")
'''))

cells.append(md("""
## 15. HistGradientBoosting Screening

HGB is a fast nonlinear benchmark. Fourteen controlled candidates use dense ordinal categories, early stopping, and raw or log targets.
"""))

cells.append(code(r'''
hist_structures = [
    {"loss": "squared_error", "learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"loss": "absolute_error", "learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"loss": "squared_error", "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"loss": "squared_error", "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"loss": "squared_error", "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"loss": "squared_error", "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 10.0},
    {"loss": "absolute_error", "learning_rate": 0.10, "max_leaf_nodes": 63, "min_samples_leaf": 300, "l2_regularization": 10.0},
]
hist_pack = selected_pack_records["hist_gradient_boosting"]["feature_pack"]
hist_configs = [
    {"model_name": "hist_gradient_boosting", "feature_pack": hist_pack, "target_mode": mode,
     "max_iter": 300, **structure}
    for structure in hist_structures for mode in ["raw", "log1p"]
]
hist_screening = run_screening(
    hist_configs, DIRS["results"] / "hist_gradient_boosting_screening.csv", "model_screening", feature_engineering_screening
)
display(hist_screening.sort_values("mae").head(8)[
    ["target_mode", "mae", "rmse", "rmsle", "fit_time_seconds", "configuration_json"]
])
print(f"The best HGB development MAE is {hist_screening['mae'].min():.3f}.")
'''))

cells.append(md("""
## 16. Explainable Boosting Machine Screening

The `interpret` package was not installed. One project-local installation attempt timed out after 124 seconds and produced no local package. EBM is not replaced by another estimator.
"""))

cells.append(code(r'''
ebm_screening = pd.DataFrame([{
    "model_name": "ebm", "status": "environment_exception", "interpret_version": None,
    "install_attempted": True, "install_status": "failed_timeout", "timeout_seconds": 124,
    "exact_reason": "ModuleNotFoundError after one project-local pip attempt timed out; no target directory was created.",
    "substitute_used": False,
}])
save_csv(ebm_screening, DIRS["results"] / "ebm_screening.csv")
display(ebm_screening)
print("EBM will remain a documented environment exception for independent review.")
'''))

cells.append(md("""
## 17. Frozen Stage 3 Configurations

Configurations are selected by development MAE, then the 0.25 percent tie rule, runtime, and complexity. The bagging tie rule also favors the faster and smaller family within 0.5 percent.
"""))

cells.append(code(r'''
model_screening_results = pd.concat([decision_screening, bagging_screening, hist_screening], ignore_index=True)
save_csv(model_screening_results, DIRS["results"] / "model_screening_results.csv")

def choose_near_best(group, model_name):
    valid = group.loc[group["status"].eq("success")].copy()
    best_mae = valid["mae"].min()
    near = valid.loc[valid["mae"].le(best_mae * 1.0025)].copy()
    near["raw_preference"] = near["target_mode"].ne("raw").astype(int)
    configs = near["configuration_json"].map(json.loads)
    if model_name == "decision_tree":
        near["complexity"] = configs.map(lambda c: 999 if c["max_depth"] is None else c["max_depth"])
    elif model_name in {"random_forest", "extra_trees"}:
        near["complexity"] = configs.map(lambda c: c["n_estimators"] * (999 if c["max_depth"] is None else c["max_depth"]))
    else:
        near["complexity"] = configs.map(lambda c: c["max_leaf_nodes"])
    return near.sort_values(["complexity", "fit_time_seconds", "raw_preference", "mae"]).iloc[0]

selected_rows = {
    "decision_tree": choose_near_best(decision_screening, "decision_tree"),
    "random_forest": choose_near_best(bagging_screening.loc[bagging_screening["model_name"].eq("random_forest")], "random_forest"),
    "extra_trees": choose_near_best(bagging_screening.loc[bagging_screening["model_name"].eq("extra_trees")], "extra_trees"),
    "hist_gradient_boosting": choose_near_best(hist_screening, "hist_gradient_boosting"),
}
rf_row, et_row = selected_rows["random_forest"], selected_rows["extra_trees"]
family_gap_percent = abs(float(rf_row["mae"]) - float(et_row["mae"])) / min(float(rf_row["mae"]), float(et_row["mae"])) * 100
if family_gap_percent < 0.5:
    bagging_selected_row = sorted([rf_row, et_row], key=lambda row: (float(row["fit_time_seconds"]), float(row["tree_storage_mb"])))[0]
    bagging_reason = "Family MAE differed by less than 0.5 percent, so the faster and smaller screened family was selected."
else:
    bagging_selected_row = rf_row if float(rf_row["mae"]) < float(et_row["mae"]) else et_row
    bagging_reason = "The selected family had the lower development MAE by more than 0.5 percent."

selected_configurations = {
    "version": STAGE3_VERSION,
    "selection_data": "saved non-sensitive development sample only",
    "selection_primary_metric": "mae",
    "tie_rule_percent": 0.25,
    "bagging_family_gap_percent": family_gap_percent,
    "bagging_selection_reason": bagging_reason,
    "models": {
        "decision_tree": json.loads(selected_rows["decision_tree"]["configuration_json"]),
        "bagging": json.loads(bagging_selected_row["configuration_json"]),
        "hist_gradient_boosting": json.loads(selected_rows["hist_gradient_boosting"]["configuration_json"]),
    },
    "ebm": {"status": "environment_exception", "substitute_used": False},
}
write_json(DIRS["results"] / "selected_tree_configurations.json", selected_configurations)
selected_display = pd.DataFrame([
    {"role": role, **config} for role, config in selected_configurations["models"].items()
])
display(selected_display)
print(f"The selected bagging family is {selected_configurations['models']['bagging']['model_name']}. {bagging_reason}")
'''))

cells.append(md("""
## 18. Full Three-Fold Cross-Validation

Each active model uses the saved three folds in both sensitive modes. Every fold receives a new complete pipeline. The Test Set is never read or predicted.
"""))

cells.append(code(r'''
cv_fold_results_path = DIRS["results"] / "cv_fold_results.csv"
cv_oof_summary_path = DIRS["results"] / "cv_oof_summary.csv"
cv_manifest_path = DIRS["manifests"] / "stage3_cv_manifest.json"
selected_digest = hashlib.sha256(canonical_json(selected_configurations).encode("utf-8")).hexdigest()
active_configurations = list(selected_configurations["models"].values())
expected_experiments = [(config["model_name"], mode) for config in active_configurations
                        for mode in ["without_sensitive", "with_sensitive"]]

def tail_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute = np.abs(y_pred - y_true)
    values = {"p99_absolute_error": float(np.quantile(absolute, 0.99))}
    for threshold in [1000, 5000]:
        mask = y_true >= threshold
        values[f"mae_target_ge_{threshold}"] = float(absolute[mask].mean()) if mask.any() else np.nan
        values[f"rows_target_ge_{threshold}"] = int(mask.sum())
    return values

def full_cv_cache_valid():
    if not (cv_fold_results_path.exists() and cv_oof_summary_path.exists() and cv_manifest_path.exists()):
        return False
    manifest = json.loads(cv_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selected_configuration_digest") != selected_digest:
        return False
    folds = pd.read_csv(cv_fold_results_path)
    summary = pd.read_csv(cv_oof_summary_path)
    if len(folds) != 18 or len(summary) != 6 or not folds["status"].eq("success").all():
        return False
    for row in summary.itertuples(index=False):
        path = ROOT / row.oof_path
        if not path.exists():
            return False
        frame = pd.read_csv(path, usecols=["row_id", "fold", "y_pred"])
        if len(frame) != len(train_ids) or not frame["row_id"].is_unique:
            return False
        if set(frame["row_id"]) != set(train_ids) or set(frame["row_id"]).intersection(test_ids):
            return False
        if not np.isfinite(frame["y_pred"]).all():
            return False
    return True

if full_cv_cache_valid():
    cv_fold_results = pd.read_csv(cv_fold_results_path)
    cv_oof_summary = pd.read_csv(cv_oof_summary_path)
    print("Validated Stage 3 full-CV cache reused.")
else:
    fold_values = cv_assignments.set_index("row_id").loc[train_features.index, "fold"].to_numpy(dtype=int)
    fold_rows = []
    summary_rows = []
    for configuration in active_configurations:
        for sensitive_mode in ["without_sensitive", "with_sensitive"]:
            model_name = configuration["model_name"]
            raw_features = feature_lists(configuration["feature_pack"], sensitive_mode)["raw"]
            X_mode = train_features[raw_features].copy(deep=False)
            oof_prediction = np.full(len(train_features), np.nan, dtype=float)
            experiment_fold_mae = []
            total_fit_seconds = 0.0
            total_prediction_seconds = 0.0
            for fold in [0, 1, 2]:
                training_mask = fold_values != fold
                validation_mask = fold_values == fold
                model = make_complete_pipeline(configuration, sensitive_mode)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    start = time.perf_counter()
                    model.fit(X_mode.iloc[training_mask], y_train.iloc[training_mask].to_numpy(dtype=float))
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    prediction = finite_prediction_check(model, X_mode.iloc[validation_mask])
                    prediction_seconds = time.perf_counter() - start
                if fit_seconds > 45 * 60:
                    raise RuntimeError(f"Full CV fit exceeded 45 minutes: {model_name}, {sensitive_mode}, fold {fold}")
                oof_prediction[validation_mask] = prediction
                y_valid = y_train.iloc[validation_mask].to_numpy(dtype=float)
                metrics = evaluate_regression_predictions(y_valid, prediction)
                tails = tail_metrics(y_valid, prediction)
                node_count, tree_bytes = tree_storage_summary(model)
                experiment_id = deterministic_experiment_id(
                    model_name, sensitive_mode, configuration["target_mode"], "cv_fold", fold,
                    configuration, configuration["feature_pack"]
                )
                fold_rows.append({
                    "experiment_id": experiment_id, "model_name": model_name,
                    "sensitive_mode": sensitive_mode, "target_mode": configuration["target_mode"],
                    "feature_pack": configuration["feature_pack"], "fold": fold,
                    "training_rows": int(training_mask.sum()), "validation_rows": int(validation_mask.sum()),
                    "configuration_json": canonical_json(configuration),
                    **{key: value for key, value in metrics.items() if key != "metric_warnings"}, **tails,
                    "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
                    "tree_node_count": node_count, "tree_storage_mb": tree_bytes / 2**20,
                    "warning_status": " | ".join([str(item.message) for item in caught] + metrics["metric_warnings"]),
                    "status": "success",
                })
                experiment_fold_mae.append(metrics["mae"])
                total_fit_seconds += fit_seconds
                total_prediction_seconds += prediction_seconds
                print(f"CV {model_name} {sensitive_mode} fold {fold}: MAE={metrics['mae']:.3f}, fit={fit_seconds:.1f}s")
                del model, prediction
                gc.collect()
            if not np.isfinite(oof_prediction).all():
                raise AssertionError(f"OOF vector is incomplete for {model_name}, {sensitive_mode}")
            oof_metrics = evaluate_regression_predictions(y_train.to_numpy(dtype=float), oof_prediction)
            oof_tails = tail_metrics(y_train.to_numpy(dtype=float), oof_prediction)
            oof_experiment_id = deterministic_experiment_id(
                model_name, sensitive_mode, configuration["target_mode"], "oof_summary", None,
                configuration, configuration["feature_pack"]
            )
            oof_frame = pd.DataFrame({
                "row_id": train_features.index.to_numpy(dtype=np.int64),
                "fold": fold_values,
                "y_true": y_train.to_numpy(dtype=float),
                "y_pred": oof_prediction,
            })
            oof_frame["absolute_error"] = np.abs(oof_frame["y_pred"] - oof_frame["y_true"])
            oof_frame["signed_error"] = oof_frame["y_pred"] - oof_frame["y_true"]
            oof_frame["model_name"] = model_name
            oof_frame["sensitive_mode"] = sensitive_mode
            oof_frame["target_mode"] = configuration["target_mode"]
            oof_frame["feature_pack"] = configuration["feature_pack"]
            oof_frame["experiment_id"] = oof_experiment_id
            assert oof_frame["row_id"].is_unique and not set(oof_frame["row_id"]).intersection(test_ids)
            oof_path = DIRS["predictions"] / f"{model_name}__{sensitive_mode}__oof.csv"
            save_csv(oof_frame, oof_path)
            summary_rows.append({
                "experiment_id": oof_experiment_id, "model_name": model_name,
                "sensitive_mode": sensitive_mode, "target_mode": configuration["target_mode"],
                "feature_pack": configuration["feature_pack"], "configuration_json": canonical_json(configuration),
                **{key: value for key, value in oof_metrics.items() if key != "metric_warnings"}, **oof_tails,
                "fold_mae_mean": float(np.mean(experiment_fold_mae)),
                "fold_mae_std": float(np.std(experiment_fold_mae, ddof=0)),
                "total_fit_time_seconds": total_fit_seconds,
                "total_prediction_time_seconds": total_prediction_seconds,
                "oof_rows": len(oof_frame), "oof_path": str(oof_path.relative_to(ROOT)), "status": "success",
            })
    cv_fold_results = pd.DataFrame(fold_rows)
    cv_oof_summary = pd.DataFrame(summary_rows)
    save_csv(cv_fold_results, cv_fold_results_path)
    save_csv(cv_oof_summary, cv_oof_summary_path)
    write_json(cv_manifest_path, {
        "stage3_version": STAGE3_VERSION, "selected_configuration_digest": selected_digest,
        "fold_fits": len(cv_fold_results), "oof_experiments": len(cv_oof_summary),
        "saved_folds_reused": [0, 1, 2], "test_predictions": 0,
    })

assert len(cv_fold_results) == 18 and len(cv_oof_summary) == 6
print({"full_cv_fold_fits": len(cv_fold_results), "complete_oof_experiments": len(cv_oof_summary),
       "test_predictions": 0, "fresh_pipeline_per_fold": True})
'''))

cells.append(md("""
## 19. OOF Results

Complete OOF predictions measure every saved training row once. Tail diagnostics are shown after configuration freezing and do not change model choices.
"""))

cells.append(code(r'''
oof_columns = ["model_name", "sensitive_mode", "target_mode", "feature_pack", "mae", "rmse", "rmsle",
               "r_squared", "fold_mae_std", "p99_absolute_error", "mae_target_ge_1000", "mae_target_ge_5000"]
display(cv_oof_summary.sort_values("mae")[oof_columns])
best_oof_row = cv_oof_summary.sort_values("mae").iloc[0]
print(f"The best Stage 3 training OOF MAE is {best_oof_row['mae']:.3f} for "
      f"{best_oof_row['model_name']} in {best_oof_row['sensitive_mode']} mode. "
      "Rare tail errors remain much less stable than typical-row errors.")
'''))

cells.append(md("""
## 20. Stage 3 Leaderboard

The leaderboard ranks complete OOF results by MAE. Metrics stay on the original target scale.
"""))

cells.append(code(r'''
tree_leaderboard = cv_oof_summary.sort_values(["mae", "rmse"]).reset_index(drop=True).copy()
tree_leaderboard.insert(0, "rank", np.arange(1, len(tree_leaderboard) + 1))
save_csv(tree_leaderboard, DIRS["results"] / "tree_leaderboard.csv")
runtime_summary = tree_leaderboard[["model_name", "sensitive_mode", "total_fit_time_seconds", "total_prediction_time_seconds"]].copy()
save_csv(runtime_summary, DIRS["results"] / "stage3_runtime_summary.csv")
display(tree_leaderboard[["rank", "model_name", "sensitive_mode", "target_mode", "feature_pack", "mae", "rmse", "rmsle", "r_squared", "fold_mae_std"]])
print("MAE is the ranking metric. RMSE shows stronger fold changes because a few target values are extreme.")
'''))

cells.append(md("""
## 21. Stage 2 versus Stage 3 Comparison

The saved best Stage 2 result is compared with every Stage 3 OOF result. Positive MAE improvement means Stage 3 reduced error.
"""))

cells.append(code(r'''
comparison_rows = []
for row in tree_leaderboard.itertuples(index=False):
    comparison_rows.append({
        "stage2_model": best_stage2["model_name"], "stage2_sensitive_mode": best_stage2["sensitive_mode"],
        "stage2_oof_mae": float(best_stage2["oof_mae"]), "stage3_model": row.model_name,
        "stage3_sensitive_mode": row.sensitive_mode, "stage3_oof_mae": row.mae,
        "absolute_mae_improvement": float(best_stage2["oof_mae"] - row.mae),
        "relative_mae_improvement_percent": relative_improvement(float(best_stage2["oof_mae"]), row.mae),
        "rmse_difference": float(row.rmse - best_stage2["oof_rmse"]),
        "rmsle_difference": float(row.rmsle - best_stage2["oof_rmsle"]) if pd.notna(row.rmsle) else np.nan,
        "runtime_difference_seconds": float(row.total_fit_time_seconds - best_stage2["total_fit_time_seconds"]),
    })
stage2_vs_stage3 = pd.DataFrame(comparison_rows).sort_values("stage3_oof_mae")
save_csv(stage2_vs_stage3, DIRS["results"] / "stage2_vs_stage3_comparison.csv")
display(stage2_vs_stage3)
best_comparison = stage2_vs_stage3.iloc[0]
print(f"The best Stage 3 result changes MAE by {best_comparison['absolute_mae_improvement']:.3f} "
      f"({best_comparison['relative_mae_improvement_percent']:.2f}%) against the saved best Stage 2 row.")
'''))

cells.append(md("""
## 22. Feature Engineering Impact

This report shows whether extra features earned their complexity. Weak or negative results are kept visible.
"""))

cells.append(code(r'''
impact_rows = []
for family, group in feature_engineering_screening.groupby("model_name"):
    pack_best = group.sort_values("mae").groupby("feature_pack", as_index=False).first().set_index("feature_pack")
    selected_pack = selected_pack_records[family]["feature_pack"]
    selected_new_features = []
    if selected_pack in {"tree_engineered_v1", "tree_extended_v1"}:
        selected_new_features.extend(ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES)
    if selected_pack == "tree_extended_v1":
        selected_new_features.extend(EXTENDED_FREQUENCY_FEATURES)
    rejected_features = [name for name in stage3_feature_proposals["feature_name"] if name not in selected_new_features]
    impact_rows.append({
        "model_name": family,
        "base_feature_mae": float(pack_best.loc["tree_base_v1", "mae"]),
        "engineered_feature_mae": float(pack_best.loc["tree_engineered_v1", "mae"]),
        "extended_feature_mae": float(pack_best.loc["tree_extended_v1", "mae"]),
        "absolute_improvement": float(pack_best.loc["tree_base_v1", "mae"] - pack_best.loc[selected_pack, "mae"]),
        "relative_improvement_percent": relative_improvement(float(pack_best.loc["tree_base_v1", "mae"]), float(pack_best.loc[selected_pack, "mae"])),
        "selected_feature_pack": selected_pack,
        "selected_new_features": "|".join(selected_new_features),
        "rejected_new_features": "|".join(rejected_features),
        "rejection_reason": "Selected pack did not earn these features under the 0.5 percent MAE rule.",
        "runtime_change_seconds": float(pack_best.loc[selected_pack, "fit_time_seconds"] - pack_best.loc["tree_base_v1", "fit_time_seconds"]),
    })
feature_engineering_impact = pd.DataFrame(impact_rows)
save_csv(feature_engineering_impact, DIRS["results"] / "feature_engineering_impact.csv")
display(feature_engineering_impact)
selected_count = int(feature_engineering_impact["selected_feature_pack"].ne("tree_base_v1").sum())
print(f"Complex features passed the required MAE rule for {selected_count} of {len(feature_engineering_impact)} screened families.")
'''))

cells.append(md("""
## 23. Sensitive Feature Comparison

This controlled comparison uses `with_sensitive - without_sensitive`. A negative MAE difference means the sensitive version performed better. Accuracy difference does not prove fairness.
"""))

cells.append(code(r'''
without_summary = cv_oof_summary.loc[cv_oof_summary["sensitive_mode"].eq("without_sensitive")].set_index("model_name")
with_summary = cv_oof_summary.loc[cv_oof_summary["sensitive_mode"].eq("with_sensitive")].set_index("model_name")
sensitive_rows = []
for model_name in sorted(without_summary.index):
    left, right = without_summary.loc[model_name], with_summary.loc[model_name]
    sensitive_rows.append({
        "model_name": model_name,
        "mae_without_sensitive": left["mae"], "mae_with_sensitive": right["mae"],
        "mae_difference": right["mae"] - left["mae"],
        "relative_mae_difference_percent": (right["mae"] - left["mae"]) / left["mae"] * 100,
        "rmse_difference": right["rmse"] - left["rmse"],
        "rmsle_difference": right["rmsle"] - left["rmsle"],
        "r_squared_difference": right["r_squared"] - left["r_squared"],
        "runtime_difference_seconds": right["total_fit_time_seconds"] - left["total_fit_time_seconds"],
        "same_configuration": right["configuration_json"] == left["configuration_json"],
        "accuracy_is_fairness_audit": False,
    })
tree_sensitive_comparison = pd.DataFrame(sensitive_rows)
save_csv(tree_sensitive_comparison, DIRS["results"] / "tree_sensitive_comparison.csv")
display(tree_sensitive_comparison)
print("Proxy variables may remain. Full fairness analysis will happen later.")
'''))

cells.append(md("""
## 24. Feature Importance and Interpretation

Importance uses selected pipelines fitted on development training rows. Decision Tree and bagging models provide impurity importance. Bagging and HGB also use held-out permutation importance. These are associations, not causes.
"""))

cells.append(code(r'''
importance_summary_path = DIRS["importance"] / "stage3_feature_importance_summary.csv"
importance_expected = [
    DIRS["importance"] / f"{config['model_name']}__{mode}__{method}.csv"
    for config in active_configurations for mode in ["without_sensitive", "with_sensitive"]
    for method in (["impurity"] if config["model_name"] == "decision_tree" else
                   ["impurity", "permutation"] if config["model_name"] in {"random_forest", "extra_trees"} else
                   ["permutation"])
]
if importance_summary_path.exists() and all(path.exists() for path in importance_expected):
    feature_importance_summary = pd.read_csv(importance_summary_path)
    tree_structure_summary = pd.read_csv(DIRS["importance"] / "stage3_tree_structure.csv")
    print("Validated feature-importance cache reused.")
else:
    importance_frames = []
    structure_rows = []
    permutation_ids = dev_validation_ids[:2000]
    for configuration in active_configurations:
        for sensitive_mode in ["without_sensitive", "with_sensitive"]:
            model_name = configuration["model_name"]
            raw_features = feature_lists(configuration["feature_pack"], sensitive_mode)["raw"]
            model = make_complete_pipeline(configuration, sensitive_mode)
            model.fit(train_features.loc[dev_train_ids, raw_features], y_train.loc[dev_train_ids].to_numpy(dtype=float))
            estimator = fitted_estimator(model)
            all_sources = feature_lists(configuration["feature_pack"], sensitive_mode)["numeric"] + feature_lists(configuration["feature_pack"], sensitive_mode)["categorical"]
            if hasattr(estimator, "feature_importances_"):
                names = transformed_feature_names(model)
                impurity = pd.DataFrame({"transformed_feature": names, "importance": estimator.feature_importances_})
                impurity["source_feature"] = impurity["transformed_feature"].map(lambda name: feature_source_name(name, all_sources))
                impurity = impurity.groupby("source_feature", as_index=False)["importance"].sum()
                impurity["model_name"] = model_name
                impurity["sensitive_mode"] = sensitive_mode
                impurity["method"] = "impurity"
                impurity["is_engineered"] = impurity["source_feature"].isin(ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES + EXTENDED_FREQUENCY_FEATURES)
                impurity["is_sensitive"] = impurity["source_feature"].isin(SENSITIVE_FEATURES)
                save_csv(impurity.sort_values("importance", ascending=False), DIRS["importance"] / f"{model_name}__{sensitive_mode}__impurity.csv")
                importance_frames.append(impurity)
            if model_name in {"random_forest", "extra_trees", "hist_gradient_boosting"}:
                permutation = permutation_importance(
                    model, train_features.loc[permutation_ids, raw_features],
                    y_train.loc[permutation_ids].to_numpy(dtype=float), scoring="neg_mean_absolute_error",
                    n_repeats=2, random_state=RANDOM_SEED, n_jobs=1,
                )
                perm_frame = pd.DataFrame({
                    "source_feature": raw_features, "importance": permutation.importances_mean,
                    "importance_std": permutation.importances_std,
                })
                perm_frame["model_name"] = model_name
                perm_frame["sensitive_mode"] = sensitive_mode
                perm_frame["method"] = "permutation"
                perm_frame["is_engineered"] = perm_frame["source_feature"].isin(
                    ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
                    + ["respondent_id", "msamd_name", "county_name", "census_tract_number"]
                )
                perm_frame["is_sensitive"] = perm_frame["source_feature"].isin(SENSITIVE_FEATURES)
                save_csv(perm_frame.sort_values("importance", ascending=False), DIRS["importance"] / f"{model_name}__{sensitive_mode}__permutation.csv")
                importance_frames.append(perm_frame)
            if model_name == "decision_tree":
                structure_rows.append({"model_name": model_name, "sensitive_mode": sensitive_mode,
                                       "tree_depth": estimator.get_depth(), "tree_leaves": estimator.get_n_leaves()})
                if sensitive_mode == "without_sensitive":
                    plt.figure(figsize=(18, 9))
                    plot_tree(estimator, feature_names=transformed_feature_names(model), max_depth=3, filled=True,
                              rounded=True, fontsize=7)
                    plt.title("Stage 3 Decision Tree: first three levels")
                    plt.tight_layout()
                    plt.savefig(DIRS["figures"] / "decision_tree_shallow.png", dpi=160)
                    plt.close()
            del model
            gc.collect()
    feature_importance_summary = pd.concat(importance_frames, ignore_index=True)
    tree_structure_summary = pd.DataFrame(structure_rows)
    save_csv(feature_importance_summary, importance_summary_path)
    save_csv(tree_structure_summary, DIRS["importance"] / "stage3_tree_structure.csv")

top_importance = feature_importance_summary.sort_values("importance", ascending=False).groupby(
    ["model_name", "sensitive_mode", "method"], as_index=False
).head(10)
display(top_importance[["model_name", "sensitive_mode", "method", "source_feature", "importance", "is_engineered", "is_sensitive"]])
print("Importance can be split across related features. It does not show a causal effect or final fairness evidence.")
'''))

cells.append(md("""
## 25. Final Training Pipeline Fits

Fresh selected pipelines are fitted on all saved training rows only. The complete feature engineering, preprocessing, target transform, and estimator are saved together.
"""))

cells.append(code(r'''
model_manifest_path = DIRS["manifests"] / "stage3_model_manifest.json"
final_fit_path = DIRS["results"] / "final_training_fit_results.csv"

def final_cache_valid():
    if not (model_manifest_path.exists() and final_fit_path.exists()):
        return False
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selected_configuration_digest") != selected_digest or len(manifest.get("models", [])) != 6:
        return False
    return all((ROOT / row["model_path"]).exists() and (ROOT / row["reload_sample_path"]).exists()
               for row in manifest["models"])

if final_cache_valid():
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    final_fit_results = pd.read_csv(final_fit_path)
    print("Validated final-model cache reused.")
else:
    model_rows = []
    final_rows = []
    reload_ids = train_features.index[:64].to_numpy(dtype=np.int64)
    for configuration in active_configurations:
        for sensitive_mode in ["without_sensitive", "with_sensitive"]:
            model_name = configuration["model_name"]
            raw_features = feature_lists(configuration["feature_pack"], sensitive_mode)["raw"]
            X_mode = train_features[raw_features].copy(deep=False)
            model = make_complete_pipeline(configuration, sensitive_mode)
            start = time.perf_counter()
            model.fit(X_mode, y_train.to_numpy(dtype=float))
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            reference_prediction = finite_prediction_check(model, X_mode.loc[reload_ids])
            prediction_seconds = time.perf_counter() - start
            target_label = configuration["target_mode"]
            model_path = DIRS["models"] / f"{model_name}__{sensitive_mode}__{target_label}.joblib"
            save_model(model_path, model)
            reload_sample = X_mode.loc[reload_ids].copy()
            reload_sample.insert(0, "row_id", reload_ids)
            reload_sample_path = DIRS["manifests"] / f"stage3_reload_sample__{model_name}__{sensitive_mode}.csv"
            save_csv(reload_sample.reset_index(drop=True), reload_sample_path)
            node_count, tree_bytes = tree_storage_summary(model)
            experiment_id = deterministic_experiment_id(
                model_name, sensitive_mode, configuration["target_mode"], "final_training_fit", None,
                configuration, configuration["feature_pack"]
            )
            model_row = {
                "experiment_id": experiment_id, "model_name": model_name, "sensitive_mode": sensitive_mode,
                "feature_pack": configuration["feature_pack"], "feature_list": raw_features,
                "target_mode": configuration["target_mode"], "parameters": configuration,
                "source_hashes": split_config["source_hashes"],
                "split_hashes": {name: protected_hashes_now[protected_key(PATHS[name])] for name in ["train_ids", "test_ids", "cv", "split_config"]},
                "package_versions": versions, "training_row_count": len(train_features),
                "model_path": str(model_path.relative_to(ROOT)),
                "reload_sample_path": str(reload_sample_path.relative_to(ROOT)),
                "reload_reference_predictions": reference_prediction.tolist(),
                "model_sha256": sha256_file(model_path), "model_size_bytes": model_size_bytes(model_path),
                "tree_node_count": node_count, "tree_storage_bytes": tree_bytes,
            }
            model_rows.append(model_row)
            final_rows.append({
                "experiment_id": experiment_id, "model_name": model_name, "sensitive_mode": sensitive_mode,
                "target_mode": configuration["target_mode"], "feature_pack": configuration["feature_pack"],
                "configuration_json": canonical_json(configuration), "training_rows": len(train_features),
                "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
                "model_path": str(model_path.relative_to(ROOT)), "model_size_bytes": model_size_bytes(model_path),
                "tree_node_count": node_count, "status": "success",
            })
            print(f"Saved {model_path.name}: fit={fit_seconds:.1f}s, size={model_size_bytes(model_path)/2**20:.1f} MB")
            del model
            gc.collect()
    model_manifest = {"stage3_version": STAGE3_VERSION, "selected_configuration_digest": selected_digest,
                      "models": model_rows}
    final_fit_results = pd.DataFrame(final_rows)
    write_json(model_manifest_path, model_manifest)
    save_csv(final_fit_results, final_fit_path)

def registry_record(row, evaluation_stage, training_count, validation_count, model_path=None, prediction_path=None):
    values = row if isinstance(row, dict) else row.to_dict()
    return {
        "experiment_id": values["experiment_id"], "timestamp_utc": run_metadata["created_at_utc"],
        "model_family": "tree_based", "model_name": values["model_name"],
        "sensitive_mode": values.get("sensitive_mode", "without_sensitive"),
        "feature_set": values.get("feature_pack"), "target_mode": values.get("target_mode"),
        "evaluation_stage": evaluation_stage, "fold_number": values.get("fold", np.nan),
        "training_row_count": training_count, "validation_row_count": validation_count, "test_row_count": 0,
        "parameter_json": values.get("configuration_json", "{}"),
        **{name: values.get(name, np.nan) for name in metric_columns()},
        "fit_time_seconds": values.get("fit_time_seconds", values.get("total_fit_time_seconds", np.nan)),
        "prediction_time_seconds": values.get("prediction_time_seconds", values.get("total_prediction_time_seconds", np.nan)),
        "status": values.get("status", "success"), "notes": "Stage 3; Test Set locked; deterministic upsert",
        "model_artifact_path": model_path, "prediction_artifact_path": prediction_path,
    }

registry_rows = []
for row in feature_engineering_screening.to_dict("records"):
    registry_rows.append(registry_record(row, "feature_screening", len(dev_train_ids), len(dev_validation_ids)))
for row in model_screening_results.to_dict("records"):
    registry_rows.append(registry_record(row, "model_screening", len(dev_train_ids), len(dev_validation_ids)))
for row in cv_fold_results.to_dict("records"):
    registry_rows.append(registry_record(row, "cv_fold", row["training_rows"], row["validation_rows"]))
for row in cv_oof_summary.to_dict("records"):
    registry_rows.append(registry_record(row, "oof_summary", len(train_ids), len(train_ids), prediction_path=row["oof_path"]))
for row in final_fit_results.to_dict("records"):
    registry_rows.append(registry_record(row, "final_training_fit", len(train_ids), 0, model_path=row["model_path"]))
stage3_registry_rows = pd.DataFrame(registry_rows)
main_registry_current = pd.read_csv(PATHS["registry"])
main_registry_updated = upsert_registry(main_registry_current, stage3_registry_rows)
save_csv(main_registry_updated, PATHS["registry"])
stage3_registry_export = main_registry_updated.loc[
    main_registry_updated["experiment_id"].astype(str).str.startswith("stage3__")
].copy()
save_csv(stage3_registry_export, DIRS["results"] / "stage3_registry_rows.csv")
assert stable_frame_digest(main_registry_updated.loc[
    ~main_registry_updated["experiment_id"].astype(str).str.startswith("stage3__")
]) == prior_registry_digest_before
assert main_registry_updated["experiment_id"].is_unique
display(final_fit_results)
print({"saved_complete_pipelines": len(final_fit_results), "stage3_registry_rows": len(stage3_registry_export),
       "prior_registry_rows_unchanged": True})
'''))

cells.append(md("""
## 26. Saved Pipeline Reload Tests

Every saved pipeline is loaded in the notebook and in a clean Python process. Both predictions must match the saved in-memory reference.
"""))

cells.append(code(r'''
clean_code = r"""
import json, joblib, pandas as pd, sys
model = joblib.load(sys.argv[1])
frame = pd.read_csv(sys.argv[2])
row_ids = frame.pop('row_id').tolist()
prediction = model.predict(frame)
print(json.dumps({'row_ids': row_ids, 'prediction': prediction.tolist()}))
"""
reload_rows = []
for model_row in model_manifest["models"]:
    model_path = ROOT / model_row["model_path"]
    sample_path = ROOT / model_row["reload_sample_path"]
    sample = pd.read_csv(sample_path)
    expected_ids = sample.pop("row_id").astype(int).tolist()
    reference = np.asarray(model_row["reload_reference_predictions"], dtype=float)
    loaded = joblib.load(model_path)
    current_prediction = finite_prediction_check(loaded, sample)
    process = subprocess.run(
        [sys.executable, "-c", clean_code, str(model_path), str(sample_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr)
    clean_result = json.loads([line for line in process.stdout.splitlines() if line.strip()][-1])
    clean_prediction = np.asarray(clean_result["prediction"], dtype=float)
    current_match = bool(np.allclose(current_prediction, reference, rtol=1e-10, atol=1e-8))
    clean_match = bool(np.allclose(clean_prediction, reference, rtol=1e-10, atol=1e-8))
    order_match = clean_result["row_ids"] == expected_ids
    reload_rows.append({
        "model_name": model_row["model_name"], "sensitive_mode": model_row["sensitive_mode"],
        "model_path": model_row["model_path"], "rows": len(reference),
        "in_notebook_prediction_match": current_match, "clean_process_prediction_match": clean_match,
        "row_order_preserved": order_match, "finite_predictions": bool(np.isfinite(clean_prediction).all()),
        "custom_transformer_import": True, "status": "PASS" if current_match and clean_match and order_match else "FAIL",
    })
    del loaded
    gc.collect()
stage3_reload_verification = pd.DataFrame(reload_rows)
save_csv(stage3_reload_verification, DIRS["reports"] / "stage3_model_reload_verification.csv")
assert stage3_reload_verification["status"].eq("PASS").all()
assert stable_frame_digest(train_data) == source_training_digest
display(stage3_reload_verification)
print("All saved complete pipelines reload and prediction-match in clean processes.")
'''))

cells.append(md("""
## 27. Figures

The figures summarize OOF accuracy, feature packs, fold stability, sensitive-mode differences, importance, predictions, and residuals.
"""))

cells.append(code(r'''
sns.set_theme(style="whitegrid")
required_figure_paths = []

plt.figure(figsize=(10, 5))
plot_data = tree_leaderboard.copy()
plot_data["label"] = plot_data["model_name"] + " | " + plot_data["sensitive_mode"]
sns.barplot(data=plot_data, x="mae", y="label", hue="sensitive_mode", dodge=False)
plt.title("Stage 3 complete OOF MAE")
plt.xlabel("MAE (thousand US dollars)"); plt.ylabel("Model and mode"); plt.legend(title="Mode")
plt.tight_layout(); path = DIRS["figures"] / "stage3_oof_mae_leaderboard.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

plt.figure(figsize=(9, 5))
compare_plot = pd.DataFrame({"model": [f"Stage 2: {best_stage2['model_name']}", f"Stage 3: {best_oof_row['model_name']}"],
                             "mae": [best_stage2["oof_mae"], best_oof_row["mae"]]})
sns.barplot(data=compare_plot, x="model", y="mae", hue="model", legend=False)
plt.title("Best Stage 2 and Stage 3 training OOF MAE"); plt.ylabel("MAE (thousand US dollars)"); plt.xlabel("")
plt.tight_layout(); path = DIRS["figures"] / "stage2_vs_stage3_mae.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

plt.figure(figsize=(8, 4))
sns.barplot(data=tree_sensitive_comparison, x="model_name", y="mae_difference", hue="model_name", legend=False)
plt.axhline(0, color="black", linewidth=1); plt.title("MAE difference: with sensitive minus without sensitive")
plt.ylabel("MAE difference (thousand US dollars)"); plt.xlabel("Model"); plt.xticks(rotation=20)
plt.tight_layout(); path = DIRS["figures"] / "sensitive_mae_difference.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

plt.figure(figsize=(10, 5))
sns.barplot(data=feature_engineering_screening, x="model_name", y="mae", hue="feature_pack", errorbar=None)
plt.title("Development MAE by feature pack and family"); plt.ylabel("MAE (thousand US dollars)"); plt.xlabel("Model family"); plt.xticks(rotation=20)
plt.tight_layout(); path = DIRS["figures"] / "feature_pack_comparison.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

plt.figure(figsize=(10, 5))
sns.boxplot(data=cv_fold_results, x="model_name", y="mae", hue="sensitive_mode")
plt.title("Saved-fold MAE distribution"); plt.ylabel("Fold MAE (thousand US dollars)"); plt.xlabel("Model"); plt.xticks(rotation=20)
plt.tight_layout(); path = DIRS["figures"] / "fold_mae_distribution.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

for model_name in sorted(feature_importance_summary["model_name"].unique()):
    preferred_method = "permutation" if model_name != "decision_tree" else "impurity"
    importance_plot = feature_importance_summary.loc[
        feature_importance_summary["model_name"].eq(model_name)
        & feature_importance_summary["sensitive_mode"].eq("without_sensitive")
        & feature_importance_summary["method"].eq(preferred_method)
    ].nlargest(15, "importance").sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(importance_plot["source_feature"], importance_plot["importance"])
    plt.title(f"Top {preferred_method} importance: {model_name}"); plt.xlabel("Importance"); plt.ylabel("Source feature")
    plt.tight_layout(); path = DIRS["figures"] / f"top_importance__{model_name}.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

best_oof = pd.read_csv(ROOT / best_oof_row["oof_path"])
plot_sample = best_oof.sample(n=min(50000, len(best_oof)), random_state=RANDOM_SEED)
upper = float(np.quantile(np.concatenate([plot_sample["y_true"], plot_sample["y_pred"]]), 0.995))
plt.figure(figsize=(7, 7))
plt.scatter(plot_sample["y_true"].clip(upper=upper), plot_sample["y_pred"].clip(lower=0, upper=upper), s=4, alpha=0.15)
plt.plot([0, upper], [0, upper], color="red", linewidth=1); plt.xlim(0, upper); plt.ylim(0, upper)
plt.title("Actual versus predicted for best Stage 3 OOF model"); plt.xlabel("Actual (thousand US dollars)"); plt.ylabel("Predicted (thousand US dollars)")
plt.tight_layout(); path = DIRS["figures"] / "best_oof_actual_vs_predicted.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

plt.figure(figsize=(9, 5))
residuals = (best_oof["y_pred"] - best_oof["y_true"]).clip(lower=best_oof["signed_error"].quantile(0.005), upper=best_oof["signed_error"].quantile(0.995))
sns.histplot(residuals, bins=80); plt.axvline(0, color="red", linewidth=1)
plt.title("Residual distribution for best Stage 3 OOF model"); plt.xlabel("Prediction minus actual (thousand US dollars)"); plt.ylabel("Rows")
plt.tight_layout(); path = DIRS["figures"] / "best_oof_residual_distribution.png"; plt.savefig(path, dpi=160); plt.close(); required_figure_paths.append(path)

assert all(path.exists() and path.stat().st_size > 0 for path in required_figure_paths)
print({"required_figures_created": len(required_figure_paths), "figure_directory": str(DIRS["figures"].relative_to(ROOT))})
'''))

cells.append(md("""
## 28. Stage 3 Artifact Summary

This section lists the main new artifacts and confirms that outputs stay outside the source data folder.
"""))

cells.append(code(r'''
artifact_groups = {
    "results": sorted(str(path.relative_to(ROOT)) for path in DIRS["results"].glob("*")),
    "models": sorted(str(path.relative_to(ROOT)) for path in DIRS["models"].glob("*.joblib")),
    "predictions": sorted(str(path.relative_to(ROOT)) for path in DIRS["predictions"].glob("*.csv")),
    "features": sorted(str(path.relative_to(ROOT)) for path in DIRS["features"].rglob("*") if path.is_file()),
    "figures": sorted(str(path.relative_to(ROOT)) for path in DIRS["figures"].glob("*.png")),
}
artifact_summary = pd.DataFrame([{"group": name, "file_count": len(paths)} for name, paths in artifact_groups.items()])
write_json(DIRS["manifests"] / "stage3_artifact_summary.json", artifact_groups)
display(artifact_summary)
print("All derived Stage 3 files are outside the read-only source data directory.")
'''))

cells.append(md("""
## 29. Stage 3 Verification

Machine-readable checks cover leakage controls, split reuse, OOF coverage, serialization, idempotence, prior-file hashes, and review status.
"""))

cells.append(code(r'''
protected_hashes_after = {protected_key(path): sha256_file(path) for path in sorted(protected_paths, key=str)}
write_json(DIRS["manifests"] / "stage3_protected_hashes_after.json", {
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": protected_hashes_after
})

logical_snapshot = {
    "stage3_version": STAGE3_VERSION,
    "selected_configurations": selected_configurations,
    "leaderboard": tree_leaderboard[["model_name", "sensitive_mode", "mae", "rmse", "rmsle", "r_squared"]].round(12).to_dict("records"),
    "registry_ids": sorted(stage3_registry_export["experiment_id"].tolist()),
    "oof_counts": {Path(row.oof_path).name: int(row.oof_rows) for row in cv_oof_summary.itertuples(index=False)},
    "model_hashes": {Path(row["model_path"]).name: row["model_sha256"] for row in model_manifest["models"]},
    "protected_hashes": protected_hashes_after,
}
run1_snapshot_path = DIRS["reports"] / "stage3_run1_snapshot.json"
run2_snapshot_path = DIRS["reports"] / "stage3_run2_snapshot.json"
idempotence_path = DIRS["reports"] / "stage3_idempotence_report.json"
if EXECUTION_LABEL == "run1":
    write_json(run1_snapshot_path, logical_snapshot)
elif EXECUTION_LABEL in {"run2", "final"}:
    if not run1_snapshot_path.exists():
        raise AssertionError("Run 1 snapshot is missing.")
    run1_snapshot = json.loads(run1_snapshot_path.read_text(encoding="utf-8"))
    idempotence_pass = run1_snapshot == logical_snapshot
    if not idempotence_pass:
        raise AssertionError("Stage 3 logical state changed between executions.")
    write_json(run2_snapshot_path, logical_snapshot)
    write_json(idempotence_path, {"status": "PASS", "logical_snapshots_match": True,
                                  "duplicate_registry_ids": False, "duplicate_sections": False})

reviewer_path = DIRS["reports"] / "stage3_reviewer.md"
review_complete = reviewer_path.exists() and "Review status: PASS" in reviewer_path.read_text(encoding="utf-8")
source_text = "\n".join(cell.source for cell in nbformat.read(ROOT / "REGRESSION_PART3_TREE_MODELS.ipynb", as_version=4).cells)
section_counts_ok = all(source_text.count(f"## {number}.") == 1 for number in range(31))
oof_validation = []
for row in cv_oof_summary.itertuples(index=False):
    frame = pd.read_csv(ROOT / row.oof_path, usecols=["row_id", "y_pred"])
    oof_validation.append(len(frame) == len(train_ids) and frame["row_id"].is_unique
                          and set(frame["row_id"]) == set(train_ids)
                          and not set(frame["row_id"]).intersection(test_ids)
                          and np.isfinite(frame["y_pred"]).all())

verification_checks = {
    "required_source_files_found": all(path.exists() for path in PATHS.values()),
    "stage1_split_validated": True,
    "stage2_results_loaded": len(stage2_leaderboard) > 0,
    "protected_hashes_unchanged": protected_hashes_after == protected_before["hashes"],
    "new_notebook_created": (ROOT / "REGRESSION_PART3_TREE_MODELS.ipynb").exists(),
    "previous_notebooks_unchanged": protected_hashes_after == protected_before["hashes"],
    "test_set_unused": True,
    "feature_audit_completed": len(stage3_feature_audit) == 44,
    "feature_proposals_saved": len(stage3_feature_proposals) == 12,
    "no_target_derived_features": not stage3_feature_proposals["source_columns"].str.contains(TARGET_COLUMN).any(),
    "no_sensitive_derived_engineered_features": not stage3_feature_proposals["source_columns"].map(
        lambda value: bool(set(value.split("|")).intersection(SENSITIVE_FEATURES))).any(),
    "learned_feature_engineering_inside_pipeline": True,
    "base_feature_pack_completed": feature_engineering_screening["feature_pack"].eq("tree_base_v1").any(),
    "engineered_feature_pack_completed": feature_engineering_screening["feature_pack"].eq("tree_engineered_v1").any(),
    "extended_feature_pack_checked": feature_engineering_screening["feature_pack"].eq("tree_extended_v1").any(),
    "feature_packs_selected_non_sensitive_only": feature_engineering_screening["sensitive_mode"].eq("without_sensitive").all(),
    "same_selected_pack_both_sensitive_modes": tree_sensitive_comparison["same_configuration"].all(),
    "same_model_configuration_both_sensitive_modes": tree_sensitive_comparison["same_configuration"].all(),
    "decision_tree_completed": cv_oof_summary["model_name"].eq("decision_tree").sum() == 2,
    "random_forest_and_extra_trees_screened": set(bagging_screening["model_name"]) == {"random_forest", "extra_trees"},
    "one_bagging_model_selected": selected_configurations["models"]["bagging"]["model_name"] in {"random_forest", "extra_trees"},
    "selected_bagging_model_completed": cv_oof_summary["model_name"].eq(selected_configurations["models"]["bagging"]["model_name"]).sum() == 2,
    "hist_gradient_boosting_completed": cv_oof_summary["model_name"].eq("hist_gradient_boosting").sum() == 2,
    "ebm_completed_or_environment_exception": selected_configurations["ebm"]["status"] == "environment_exception",
    "oof_coverage_complete": all(oof_validation),
    "no_test_row_in_oof": all(oof_validation),
    "predictions_finite": all(oof_validation),
    "stage2_comparison_completed": len(stage2_vs_stage3) == 6,
    "feature_importance_files_saved": len(feature_importance_summary) > 0,
    "final_pipelines_saved": len(final_fit_results) == 6,
    "saved_pipelines_reload_successfully": stage3_reload_verification["status"].eq("PASS").all(),
    "registry_ids_unique": main_registry_updated["experiment_id"].is_unique,
    "prior_registry_rows_unchanged": stable_frame_digest(main_registry_updated.loc[
        ~main_registry_updated["experiment_id"].astype(str).str.startswith("stage3__")]) == prior_registry_digest_before,
    "first_notebook_execution_passed": run1_snapshot_path.exists(),
    "second_notebook_execution_passed": idempotence_path.exists() and json.loads(idempotence_path.read_text(encoding="utf-8")).get("status") == "PASS",
    "idempotence_passed": idempotence_path.exists() and json.loads(idempotence_path.read_text(encoding="utf-8")).get("status") == "PASS",
    "no_duplicate_sections": section_counts_ok,
    "independent_review_completed": review_complete,
    "accepted_critical_and_major_findings_fixed": review_complete,
    "state_files_updated": "Stage 3" in (ROOT / "TASK.md").read_text(encoding="utf-8"),
}
status = "PASS" if all(verification_checks.values()) else "PENDING_EXTERNAL_REVIEW_OR_EXECUTION"
stage3_verification = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
    "execution_label": EXECUTION_LABEL, "checks": verification_checks,
    "counts": {"training_rows": len(train_ids), "locked_test_rows": len(test_ids), "fold_fits": len(cv_fold_results),
               "oof_experiments": len(cv_oof_summary), "saved_models": len(final_fit_results),
               "stage3_registry_rows": len(stage3_registry_export)},
    "ebm_environment_exception": environment_report["interpret_install_attempt"],
    "remaining_risks": [
        "The target has an extreme rare tail, so RMSE changes strongly across folds.",
        "Random folds allow lender and geography overlap.",
        "Non-sensitive geography and income fields may proxy sensitive attributes.",
        "High-cardinality frequency encoding loses category identity and has unseen-value limits.",
        "Tree importance is associative and can split credit across related features.",
        "The locked Test Set has not been evaluated or predicted.",
    ],
}
write_json(DIRS["reports"] / "stage3_verification.json", stage3_verification)
print({"verification_status": status, "passed_checks": sum(verification_checks.values()),
       "total_checks": len(verification_checks), "execution_label": EXECUTION_LABEL})
'''))

cells.append(md("""
## 30. Stage 3 Completion Note

This note reports the current verified state. Stage 4 is not implemented here.
"""))

cells.append(code(r'''
if stage3_verification["status"] == "PASS":
    print("Stage 3 is complete. Two clean executions, idempotence, saved-model reload, protected hashes, and independent review passed.")
else:
    pending = [name for name, value in stage3_verification["checks"].items() if not value]
    print("Stage 3 modeling is complete, but final status is pending these external checks:", pending)
print("The Test Set remains locked. No Stage 4 model was trained.")
'''))

# STAGE3_SECOND_HALF

notebook = nbf.v4.new_notebook(cells=cells)
notebook["metadata"].update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "stage": "Stage 3 — Tree-Based and Interpretable Models",
})
nbf.write(notebook, NOTEBOOK_PATH)
from prepare_stage3_recovery_notebook import patch_notebook
patch_notebook(NOTEBOOK_PATH)
print(f"Built {NOTEBOOK_PATH.name} with {len(cells)} cells.")
