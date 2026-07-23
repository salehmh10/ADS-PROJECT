"""Create the Prompt 1 regression foundation notebook.

This script only creates the notebook. The notebook performs and records all
data checks when it is executed.
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "REGRESSION_PART2_MODELING.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md("""
    # Regression Modeling

    ## 0. Project Objective

    This notebook builds a safe base for later regression models. It checks two processed loan datasets, creates one shared split, creates shared cross-validation folds, defines metrics, and saves audit artifacts. It does not train a real model.
    """),
    md("""
    ## 1. Imports and Configuration

    One configuration keeps paths and fixed settings in one place. Fixed seeds make the split and folds repeatable.
    """),
    code("""
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json
    import math
    import random
    import warnings

    import numpy as np
    import pandas as pd
    from IPython.display import display
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split

    PROJECT_ROOT = Path.cwd().resolve()
    required_root_markers = [PROJECT_ROOT / "data" / "regression_with_sensitive_features.csv", PROJECT_ROOT / "AGENTS.md"]
    if not all(path.exists() for path in required_root_markers):
        raise RuntimeError("Run this notebook from the regresionpart2 project root. Required root markers were not found.")
    CONFIG = {
        "project_root": PROJECT_ROOT,
        "with_sensitive_path": PROJECT_ROOT / "data" / "regression_with_sensitive_features.csv",
        "without_sensitive_path": PROJECT_ROOT / "data" / "regression_without_sensitive_features.csv",
        "source_notebook_path": (PROJECT_ROOT / ".." / "main" / "REGRESION_PART1.ipynb").resolve(),
        "artifact_root": PROJECT_ROOT / "artifacts",
        "target_column": "loan_amount_000s",
        "target_unit": "thousands of US dollars",
        "random_state": 42,
        "test_size": 0.20,
        "n_cv_folds": 3,
        "n_target_bins": 10,
        "encoding": "utf-8-sig",
        "display_rows": 8,
    }
    ARTIFACT_DIRS = {
        "backups": CONFIG["artifact_root"] / "backups",
        "data_contract": CONFIG["artifact_root"] / "data_contract",
        "splits": CONFIG["artifact_root"] / "splits",
        "results": CONFIG["artifact_root"] / "results",
        "reports": CONFIG["artifact_root"] / "reports",
    }
    for directory in ARTIFACT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    pd.set_option("display.max_rows", CONFIG["display_rows"])
    pd.set_option("display.max_columns", 12)
    print({k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()})
    """),
    md("""
    The project uses seed 42, a 20 percent test set, three CV folds, and ten requested target bins. The target unit was confirmed in the Part 1 notebook.
    """),
    md("""
    ## 2. Project and File Discovery

    File discovery prevents the notebook from using a guessed input. The exact CSV names are required. The source notebook is the unique close match in the project.
    """),
    code("""
    required_paths = {
        "with_sensitive_csv": CONFIG["with_sensitive_path"],
        "without_sensitive_csv": CONFIG["without_sensitive_path"],
        "source_notebook": CONFIG["source_notebook_path"],
    }
    discovery = []
    for name, path in required_paths.items():
        discovery.append({
            "name": name,
            "relative_or_resolved_path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        })
    discovery_df = pd.DataFrame(discovery)
    display(discovery_df)
    if not discovery_df["exists"].all():
        raise FileNotFoundError("A required source file was not found.")
    """),
    md("""
    Both processed CSV files and the Part 1 notebook were found. Derived files will be saved only under `artifacts/`.
    """),
    md("""
    ## 3. Source Data Protection

    Hashes, sizes, and modification times provide a source safety record. Hashing reads each file in small blocks and does not load the full file into memory.
    """),
    code("""
    def sha256_stream(path, block_size=1024 * 1024):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(block_size), b""):
                digest.update(block)
        return digest.hexdigest()

    def file_fingerprint(path):
        path = Path(path).resolve()
        stat = path.stat()
        return {
            "resolved_path": str(path),
            "size_bytes": stat.st_size,
            "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256_stream(path),
        }

    source_paths = {
        "with_sensitive": CONFIG["with_sensitive_path"],
        "without_sensitive": CONFIG["without_sensitive_path"],
    }
    protected_source_paths = {**source_paths, "part1_notebook": CONFIG["source_notebook_path"]}
    source_hashes_before = {name: file_fingerprint(path) for name, path in protected_source_paths.items()}
    before_path = ARTIFACT_DIRS["data_contract"] / "source_hashes_before.json"
    before_path.write_text(json.dumps(source_hashes_before, indent=2), encoding="utf-8")
    display(pd.DataFrame(source_hashes_before).T[["size_bytes", "sha256"]])
    """),
    md("""
    Source fingerprints were saved before the full data load. The final section will calculate them again and require exact hash matches.
    """),
    md("""
    ## 4. Data Loading

    The source files are loaded into canonical DataFrames. Text columns use the category dtype to reduce memory use. No source DataFrame is changed in place.
    """),
    code("""
    def load_source_csv(path):
        sample = pd.read_csv(path, nrows=5000, encoding=CONFIG["encoding"])
        dtype_map = {col: "category" for col in sample.select_dtypes(include="object").columns}
        return pd.read_csv(path, dtype=dtype_map, encoding=CONFIG["encoding"], low_memory=False)

    df_with_sensitive_raw = load_source_csv(CONFIG["with_sensitive_path"])
    df_without_sensitive_raw = load_source_csv(CONFIG["without_sensitive_path"])

    def compact_frame_summary(name, frame):
        return {
            "dataset": name,
            "rows": len(frame),
            "columns": frame.shape[1],
            "numeric_columns": frame.select_dtypes(include=np.number).shape[1],
            "categorical_columns": frame.select_dtypes(exclude=np.number).shape[1],
            "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 2),
        }

    loading_summary = pd.DataFrame([
        compact_frame_summary("with_sensitive", df_with_sensitive_raw),
        compact_frame_summary("without_sensitive", df_without_sensitive_raw),
    ])
    display(loading_summary)
    display(df_with_sensitive_raw.head(3))
    """),
    md("""
    The sensitive dataset has 44 columns and the non-sensitive dataset has 36 columns. Both contain the same 499,736 observations.
    """),
    md("""
    ## 5. Data Contract Validation

    These checks confirm file, row, common-feature, and target agreement. Comparisons are made one column at a time to limit memory use.
    """),
    code("""
    TARGET = CONFIG["target_column"]

    def duplicate_column_names(frame):
        return frame.columns[frame.columns.duplicated()].tolist()

    def semantically_equal(left, right):
        if not left.isna().equals(right.isna()):
            return False
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            return np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True)
        left_text = left.astype("string").fillna("<MISSING>")
        right_text = right.astype("string").fillna("<MISSING>")
        return left_text.equals(right_text)

    if duplicate_column_names(df_with_sensitive_raw) or duplicate_column_names(df_without_sensitive_raw):
        raise ValueError("Duplicate column names were found.")
    if len(df_with_sensitive_raw) != len(df_without_sensitive_raw):
        raise ValueError("The datasets have different row counts.")
    if TARGET not in df_with_sensitive_raw or TARGET not in df_without_sensitive_raw:
        raise KeyError(f"The target {TARGET} is missing.")

    common_columns = [c for c in df_without_sensitive_raw.columns if c in df_with_sensitive_raw.columns]
    missing_positions_equal = {}
    common_values_equal = {}
    dtype_differences = []
    for column in common_columns:
        left = df_with_sensitive_raw[column]
        right = df_without_sensitive_raw[column]
        missing_positions_equal[column] = left.isna().equals(right.isna())
        common_values_equal[column] = semantically_equal(left, right)
        if str(left.dtype) != str(right.dtype):
            dtype_differences.append({"column": column, "with_dtype": str(left.dtype), "without_dtype": str(right.dtype)})
    if not all(missing_positions_equal.values()):
        raise ValueError("Missing-value positions differ in common columns.")
    if not all(common_values_equal.values()):
        failed = [c for c, passed in common_values_equal.items() if not passed]
        raise ValueError(f"Common feature values differ: {failed[:10]}")

    y = pd.to_numeric(df_with_sensitive_raw[TARGET], errors="raise")
    target_checks = {
        "numeric": pd.api.types.is_numeric_dtype(y),
        "no_missing": not y.isna().any(),
        "finite": bool(np.isfinite(y.to_numpy(dtype=float)).all()),
        "positive": bool((y > 0).all()),
        "equal_row_by_row": common_values_equal[TARGET],
    }
    if not all(target_checks.values()):
        raise ValueError(f"Target validation failed: {target_checks}")

    quantiles = y.quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    target_summary = {
        "count": int(y.count()), "mean": float(y.mean()), "std": float(y.std()),
        "min": float(y.min()), "p01": float(quantiles.loc[0.01]),
        "p10": float(quantiles.loc[0.10]), "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]), "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]), "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]), "max": float(y.max()), "skewness": float(y.skew()),
    }
    display(pd.DataFrame([target_summary]).T.rename(columns={0: "value"}))
    print("Target checks:", target_checks)
    print("Common columns checked:", len(common_columns), "| dtype differences:", len(dtype_differences))
    """),
    md("""
    All common values and missing positions match by row. The target is positive, finite, and complete. Its skew is very large, so robust metrics and target-bin stratification are useful. No outlier is removed here.
    """),
    md("""
    ## 6. Target and Feature Review

    The feature inventory records types, missing values, cardinality, identifier risks, and leakage warnings without changing the source data.
    """),
    code("""
    with_columns = list(df_with_sensitive_raw.columns)
    without_columns = list(df_without_sensitive_raw.columns)
    sensitive_features = [c for c in with_columns if c not in without_columns]
    if TARGET in sensitive_features or not sensitive_features:
        raise ValueError("The sensitive feature difference is invalid.")
    if any(c not in with_columns for c in without_columns):
        raise ValueError("A non-sensitive feature is missing from the sensitive dataset.")

    def infer_feature_type(column, series):
        lower = column.lower()
        code_like = any(token in lower for token in ["_id", "_code", "tract_number"])
        if code_like or not pd.api.types.is_numeric_dtype(series):
            return "categorical"
        return "numeric"

    def cardinality_class(unique_count):
        if unique_count <= 2: return "Binary"
        if unique_count <= 20: return "Low cardinality"
        if unique_count <= 100: return "Medium cardinality"
        return "High cardinality"

    inventory_rows = []
    for column in with_columns:
        series = df_with_sensitive_raw[column]
        unique_count = int(series.nunique(dropna=True))
        lower = column.lower()
        possible_identifier = any(token in lower for token in ["_id", "_code", "tract_number", "msamd_name", "county_name"])
        possible_leakage = column != TARGET and (TARGET.lower() in lower or "loan_amount" in lower)
        inventory_rows.append({
            "column_name": column,
            "dataset_membership": "both" if column in without_columns else "with_sensitive_only",
            "is_sensitive": column in sensitive_features,
            "pandas_dtype": str(series.dtype),
            "inferred_feature_type": "target" if column == TARGET else infer_feature_type(column, series),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().mean() * 100),
            "unique_values": unique_count,
            "cardinality_class": cardinality_class(unique_count),
            "is_constant": unique_count <= 1,
            "possible_identifier": possible_identifier,
            "possible_leakage": possible_leakage,
            "is_target": column == TARGET,
        })
    feature_inventory = pd.DataFrame(inventory_rows)
    feature_inventory.to_csv(ARTIFACT_DIRS["data_contract"] / "feature_inventory.csv", index=False)

    def features_of_type(frame, columns, kind):
        return [c for c in columns if c != TARGET and infer_feature_type(c, frame[c]) == kind]

    features_with_sensitive = [c for c in with_columns if c != TARGET]
    features_without_sensitive = [c for c in without_columns if c != TARGET]
    feature_sets = {
        "target_column": TARGET,
        "features_with_sensitive": features_with_sensitive,
        "features_without_sensitive": features_without_sensitive,
        "sensitive_features": sensitive_features,
        "common_features": [c for c in common_columns if c != TARGET],
        "numeric_features_with_sensitive": features_of_type(df_with_sensitive_raw, features_with_sensitive, "numeric"),
        "categorical_features_with_sensitive": features_of_type(df_with_sensitive_raw, features_with_sensitive, "categorical"),
        "numeric_features_without_sensitive": features_of_type(df_without_sensitive_raw, features_without_sensitive, "numeric"),
        "categorical_features_without_sensitive": features_of_type(df_without_sensitive_raw, features_without_sensitive, "categorical"),
        "high_cardinality_features": feature_inventory.loc[(feature_inventory.cardinality_class == "High cardinality") & ~feature_inventory.is_target, "column_name"].tolist(),
        "high_cardinality_categorical_features": feature_inventory.loc[(feature_inventory.cardinality_class == "High cardinality") & (feature_inventory.inferred_feature_type == "categorical"), "column_name"].tolist(),
        "possible_identifier_features": feature_inventory.loc[feature_inventory.possible_identifier, "column_name"].tolist(),
    }
    (ARTIFACT_DIRS["data_contract"] / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    display(feature_inventory[["column_name", "dataset_membership", "inferred_feature_type", "unique_values", "possible_identifier"]].head(10))
    """),
    md("""
    The sensitive difference contains eight fields. Code and identifier-like fields are marked as categorical or suspicious for later pipeline decisions. They are not removed automatically.
    """),
    md("""
    ## 7. Sensitive Feature Comparison

    The non-sensitive dataset must be an aligned subset of the sensitive dataset. This is needed for fair model comparison later.
    """),
    code("""
    sensitive_comparison = {
        "with_sensitive_feature_count": len(features_with_sensitive),
        "without_sensitive_feature_count": len(features_without_sensitive),
        "sensitive_feature_count": len(sensitive_features),
        "sensitive_features": sensitive_features,
        "non_sensitive_is_ordered_subset": without_columns == [c for c in with_columns if c not in sensitive_features],
    }
    if not sensitive_comparison["non_sensitive_is_ordered_subset"]:
        raise ValueError("The non-sensitive schema is not the expected ordered subset.")
    print(json.dumps(sensitive_comparison, indent=2))
    """),
    md("""
    The non-sensitive schema is the exact ordered subset after eight sensitive fields are removed. The same `row_id` values can therefore be used for both modes.
    """),
    md("""
    ## 8. Leakage and Suspicious-Column Checks

    This scan checks exact target copies, target-like names, duplicate features, constants, accidental indexes, identifiers, and post-outcome names. Correlation alone is not treated as proof of leakage.
    """),
    code("""
    accidental_index_names = {"index", "level_0", "unnamed: 0", "row_id"}
    exact_target_duplicates = []
    numeric_target_diagnostics = []
    for column in features_with_sensitive:
        series = df_with_sensitive_raw[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            target_values = y.to_numpy(dtype=float)
            finite = np.isfinite(values) & np.isfinite(target_values)
            exact_copy = bool(finite.all() and np.array_equal(values, target_values, equal_nan=True))
            if exact_copy:
                exact_target_duplicates.append(column)
            correlation = float(np.corrcoef(values[finite], target_values[finite])[0, 1]) if finite.sum() > 1 and np.std(values[finite]) > 0 else None
            affine_r_squared = None
            scaled_residual_ratio = None
            if finite.sum() > 1 and np.std(values[finite]) > 0:
                slope, intercept = np.polyfit(values[finite], target_values[finite], 1)
                fitted = slope * values[finite] + intercept
                residual = target_values[finite] - fitted
                denominator = np.sum((target_values[finite] - target_values[finite].mean()) ** 2)
                affine_r_squared = float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None
                scaled_residual_ratio = float(np.max(np.abs(residual)) / max(np.ptp(target_values[finite]), 1.0))
            numeric_target_diagnostics.append({
                "column": column, "correlation_with_target": correlation,
                "affine_r_squared": affine_r_squared, "max_scaled_affine_residual": scaled_residual_ratio,
                "exact_target_copy": exact_copy,
                "near_affine_warning": bool(affine_r_squared is not None and affine_r_squared > 0.999999 and scaled_residual_ratio is not None and scaled_residual_ratio < 1e-6),
            })

    fingerprints = {}
    for column in features_with_sensitive:
        fingerprints[column] = int(pd.util.hash_pandas_object(df_with_sensitive_raw[column], index=False).sum())
    duplicate_pairs = []
    for i, left in enumerate(features_with_sensitive):
        for right in features_with_sensitive[i + 1:]:
            if fingerprints[left] == fingerprints[right] and semantically_equal(df_with_sensitive_raw[left], df_with_sensitive_raw[right]):
                duplicate_pairs.append((left, right))

    leakage_rows = []
    for column in features_with_sensitive:
        lower = column.lower()
        reasons = []
        status = "Safe"
        if column in exact_target_duplicates:
            status, reasons = "Confirmed leakage", ["exact target duplicate"]
        else:
            if lower in accidental_index_names: reasons.append("clear accidental or internal index")
            if TARGET.lower() in lower or "loan_amount" in lower: reasons.append("target-like column name")
            if feature_inventory.set_index("column_name").loc[column, "is_constant"]: reasons.append("constant column")
            if feature_inventory.set_index("column_name").loc[column, "possible_identifier"]: reasons.append("identifier or geographic code risk")
            if any(column in pair for pair in duplicate_pairs): reasons.append("duplicate feature content")
            if any(token in lower for token in ["loan_approved", "action_taken", "final_outcome"]): reasons.append("possible post-outcome name")
            if reasons: status = "Suspicious"
        leakage_rows.append({"column": column, "classification": status, "reasons": "; ".join(reasons) or "no rule-based warning"})

    leakage_report = pd.DataFrame(leakage_rows)
    target_diagnostic_frame = pd.DataFrame(numeric_target_diagnostics)
    if not target_diagnostic_frame.empty:
        diagnostic_rows = target_diagnostic_frame.assign(
            classification=lambda x: np.where(x.near_affine_warning, "Strong leakage candidate", "Numeric target diagnostic"),
            reasons=lambda x: np.where(x.near_affine_warning, "almost exact affine target transform", "warning-only numeric comparison; correlation is not proof of leakage")
        )[["column", "classification", "reasons", "correlation_with_target", "affine_r_squared", "max_scaled_affine_residual"]]
        leakage_report = pd.concat([leakage_report, diagnostic_rows], ignore_index=True, sort=False)
    leakage_report.to_csv(ARTIFACT_DIRS["data_contract"] / "leakage_and_suspicious_columns.csv", index=False)
    auto_exclusions = sorted(set(exact_target_duplicates + [c for c in features_with_sensitive if c.lower() in accidental_index_names]))
    print("Exact target duplicates:", exact_target_duplicates)
    print("Duplicate feature pairs:", duplicate_pairs[:10])
    print("Near-affine target warnings:", target_diagnostic_frame.loc[target_diagnostic_frame.near_affine_warning, "column"].tolist())
    display(leakage_report[leakage_report.classification != "Safe"].head(12))
    """),
    md("""
    No feature is an exact target copy. Identifier and geography fields remain warning items for later deployment-focused tests. Engineered input transforms are redundant with raw inputs but are not target leakage.
    """),
    md("""
    ## 9. Shared Train-Test Split

    A zero-based `row_id` keeps the original row position without changing either source file. Target quantile bins are used only for stratification and are never model features.
    """),
    code("""
    row_ids = np.arange(len(y), dtype=np.int64)

    def make_target_bins(values, requested_bins, minimum_count):
        values = pd.Series(np.asarray(values), index=np.arange(len(values)))
        for requested in range(requested_bins, 1, -1):
            try:
                labels, edges = pd.qcut(values, q=requested, labels=False, retbins=True, duplicates="drop")
                actual = int(pd.Series(labels).nunique())
                counts = pd.Series(labels).value_counts()
                if labels.isna().any() or actual < 2 or counts.min() < minimum_count:
                    continue
                return labels.to_numpy(dtype=int), edges.tolist(), requested, actual
            except ValueError:
                continue
        raise ValueError("Could not create safe target stratification bins.")

    full_bins, full_bin_edges, attempted_bins, actual_full_bins = make_target_bins(y, CONFIG["n_target_bins"], 2)
    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=full_bins,
    )
    train_ids = np.sort(train_ids)
    test_ids = np.sort(test_ids)
    pd.DataFrame({"row_id": train_ids}).to_csv(ARTIFACT_DIRS["splits"] / "train_row_ids.csv", index=False)
    pd.DataFrame({"row_id": test_ids}).to_csv(ARTIFACT_DIRS["splits"] / "test_row_ids.csv", index=False)

    split_config = {
        "random_seed": CONFIG["random_state"], "test_size": CONFIG["test_size"],
        "bin_method": "pandas.qcut on raw target with duplicate edges dropped",
        "requested_bin_count": CONFIG["n_target_bins"], "attempted_bin_count": attempted_bins,
        "actual_bin_count": actual_full_bins, "bin_edges": full_bin_edges,
        "train_rows": len(train_ids), "test_rows": len(test_ids),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "row_identity_rule": "zero-based original row position; never a model feature",
        "source_hashes": {k: v["sha256"] for k, v in source_hashes_before.items()},
        "test_set_policy": "locked; not used for model selection, preprocessing, tuning, or outlier decisions",
    }
    (ARTIFACT_DIRS["splits"] / "split_config.json").write_text(json.dumps(split_config, indent=2), encoding="utf-8")
    print({"train_rows": len(train_ids), "test_rows": len(test_ids), "actual_bins": actual_full_bins})
    """),
    md("""
    The shared split contains 399,788 training rows and 99,948 locked test rows. All ten requested target bins were valid.
    """),
    md("""
    ## 10. Shared Cross-Validation Folds

    Fold bins are created again from training targets only. One validation-fold number is saved for each training row.
    """),
    code("""
    y_array = y.to_numpy()
    train_target = y_array[train_ids]
    train_bins, train_bin_edges, attempted_cv_bins, actual_cv_bins = make_target_bins(
        train_target, CONFIG["n_target_bins"], CONFIG["n_cv_folds"]
    )
    splitter = StratifiedKFold(n_splits=CONFIG["n_cv_folds"], shuffle=True, random_state=CONFIG["random_state"])
    fold_assignment = np.full(len(train_ids), -1, dtype=int)
    for fold, (_, validation_position) in enumerate(splitter.split(train_ids, train_bins)):
        fold_assignment[validation_position] = fold
    cv_assignments = pd.DataFrame({"row_id": train_ids, "fold": fold_assignment}).sort_values("row_id")
    cv_assignments.to_csv(ARTIFACT_DIRS["splits"] / "cv_fold_assignments.csv", index=False)
    if (fold_assignment < 0).any() or cv_assignments.row_id.duplicated().any():
        raise ValueError("CV assignment coverage failed.")
    display(cv_assignments.groupby("fold").size().rename("rows").to_frame())
    """),
    md("""
    Every training row has one validation-fold assignment. Fold sizes differ by at most one row, and no test row is included.
    """),
    code("""
    def distribution_summary(name, values):
        values = pd.Series(values)
        return {"set": name, "rows": len(values), "mean": float(values.mean()), "std": float(values.std()),
                "median": float(values.median()), "p90": float(values.quantile(.90)),
                "p95": float(values.quantile(.95)), "p99": float(values.quantile(.99)),
                "max": float(values.max()), "skew": float(values.skew())}

    distribution_rows = [distribution_summary("full", y_array), distribution_summary("train", y_array[train_ids]), distribution_summary("test", y_array[test_ids])]
    for fold in range(CONFIG["n_cv_folds"]):
        fold_ids = cv_assignments.loc[cv_assignments.fold == fold, "row_id"].to_numpy()
        distribution_rows.append(distribution_summary(f"cv_fold_{fold}", y_array[fold_ids]))
    distribution_table = pd.DataFrame(distribution_rows)

    full_bin_frame = pd.DataFrame({"row_id": row_ids, "bin": full_bins})
    bin_proportions = {}
    for name, ids in {"full": row_ids, "train": train_ids, "test": test_ids}.items():
        proportions = full_bin_frame.loc[ids, "bin"].value_counts(normalize=True).sort_index()
        bin_proportions[name] = {str(int(k)): float(v) for k, v in proportions.items()}

    train_bin_reference = pd.Series(train_bins).value_counts(normalize=True).sort_index()
    cv_bin_proportions = {}
    cv_bin_max_absolute_deviation = {}
    for fold in range(CONFIG["n_cv_folds"]):
        fold_positions = np.flatnonzero(fold_assignment == fold)
        proportions = pd.Series(train_bins[fold_positions]).value_counts(normalize=True).reindex(train_bin_reference.index, fill_value=0).sort_index()
        cv_bin_proportions[str(fold)] = {str(int(k)): float(v) for k, v in proportions.items()}
        cv_bin_max_absolute_deviation[str(fold)] = float((proportions - train_bin_reference).abs().max())
    cv_bin_tolerance = 0.01

    recreated_train, recreated_test = train_test_split(
        row_ids, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=full_bins
    )
    recreated_train, recreated_test = np.sort(recreated_train), np.sort(recreated_test)
    recreated_fold_assignment = np.full(len(train_ids), -1, dtype=int)
    for fold, (_, validation_position) in enumerate(splitter.split(train_ids, train_bins)):
        recreated_fold_assignment[validation_position] = fold

    split_checks = {
        "train_test_overlap_zero": len(np.intersect1d(train_ids, test_ids)) == 0,
        "train_test_coverage_complete": np.array_equal(np.sort(np.concatenate([train_ids, test_ids])), row_ids),
        "cv_coverage_complete": np.array_equal(np.sort(cv_assignments.row_id.to_numpy()), train_ids),
        "no_test_row_in_cv": len(np.intersect1d(cv_assignments.row_id.to_numpy(), test_ids)) == 0,
        "fold_sizes_reasonable": int(cv_assignments.groupby("fold").size().max() - cv_assignments.groupby("fold").size().min()) <= 1,
        "cv_target_bin_distributions_similar": max(cv_bin_max_absolute_deviation.values()) <= cv_bin_tolerance,
        "split_reproducible": np.array_equal(train_ids, recreated_train) and np.array_equal(test_ids, recreated_test),
        "folds_reproducible": np.array_equal(fold_assignment, recreated_fold_assignment),
        "same_rows_for_both_sensitive_modes": len(df_with_sensitive_raw) == len(df_without_sensitive_raw) == len(row_ids),
        "target_order_correct": common_values_equal[TARGET],
    }
    if not all(split_checks.values()):
        raise AssertionError(f"Split verification failed: {split_checks}")
    split_verification = {"checks": split_checks, "distribution_summary": distribution_rows, "full_target_bin_proportions": bin_proportions,
                          "cv_target_bin_proportions": cv_bin_proportions, "cv_bin_max_absolute_deviation": cv_bin_max_absolute_deviation,
                          "cv_bin_similarity_tolerance": cv_bin_tolerance,
                          "cv_requested_bins": CONFIG["n_target_bins"], "cv_actual_bins": actual_cv_bins, "cv_bin_edges": train_bin_edges}
    (ARTIFACT_DIRS["splits"] / "split_verification.json").write_text(json.dumps(split_verification, indent=2), encoding="utf-8")
    display(distribution_table)
    """),
    md("""
    Target quantiles are close across train, test, and folds. The test maximum is lower because the full data has a very rare extreme tail. This is reported as a risk; the test set is not used to change the split.
    """),
    md("""
    ## 11. Regression Evaluation Metrics

    Metrics are calculated on the original target scale. Negative predictions are kept for ordinary metrics. Only the clearly named clipped RMSLE uses zero-clipped predictions.
    """),
    code("""
    def _validated_numeric_vector(values, name):
        array = np.asarray(values)
        if array.ndim != 1: raise ValueError(f"{name} must be one-dimensional.")
        if array.size == 0: raise ValueError(f"{name} must not be empty.")
        try: array = array.astype(float)
        except (TypeError, ValueError) as exc: raise TypeError(f"{name} must contain numeric values.") from exc
        if not np.isfinite(array).all(): raise ValueError(f"{name} contains NaN or infinite values.")
        return array

    def evaluate_regression_predictions(y_true, y_pred, unit_confirmed=True):
        true = _validated_numeric_vector(y_true, "y_true")
        pred = _validated_numeric_vector(y_pred, "y_pred")
        if len(true) != len(pred): raise ValueError("y_true and y_pred must have the same length.")
        errors = pred - true
        absolute_errors = np.abs(errors)
        mse = float(np.mean(errors ** 2))
        negative_rate = float(np.mean(pred < 0))
        metric_warnings = []
        if np.any(true == 0):
            mape = None
            metric_warnings.append("MAPE is unavailable because y_true contains zero.")
        else:
            mape = float(np.mean(np.abs(errors / true)) * 100)
        denominator = float(np.sum(np.abs(true)))
        wape = None if denominator == 0 else float(np.sum(absolute_errors) / denominator * 100)
        if denominator == 0: metric_warnings.append("WAPE is unavailable because sum(abs(y_true)) is zero.")
        r_squared = None if len(true) < 2 or np.all(true == true[0]) else float(r2_score(true, pred))
        if r_squared is None: metric_warnings.append("R-squared is unavailable for fewer than two values or a constant target.")
        rmsle = None
        rmsle_clipped_zero = None
        if np.any(true < 0):
            metric_warnings.append("RMSLE is unavailable because y_true contains negative values.")
        elif negative_rate > 0:
            rmsle_clipped_zero = float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(true)) ** 2)))
            metric_warnings.append("Negative predictions were clipped only for rmsle_clipped_zero.")
        else:
            rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
        mae = float(np.mean(absolute_errors))
        rmse = float(np.sqrt(mse))
        return {
            "mae": mae, "mse": mse, "rmse": rmse, "mape_percent": mape,
            "r_squared": r_squared, "rmsle": rmsle, "rmsle_clipped_zero": rmsle_clipped_zero,
            "median_absolute_error": float(np.median(absolute_errors)), "wape_percent": wape,
            "mean_signed_error": float(np.mean(errors)), "p90_absolute_error": float(np.quantile(absolute_errors, .90)),
            "negative_prediction_rate": negative_rate,
            "mae_usd": mae * 1000 if unit_confirmed else None,
            "rmse_usd": rmse * 1000 if unit_confirmed else None,
            "metric_warnings": metric_warnings,
        }
    """),
    md("""
    MAE is the primary metric. It is reported in thousands of dollars and US dollars. MAPE and WAPE are percentages, and mean signed error is positive for overprediction.
    """),
    md("""
    ## 12. Metric Unit Tests

    Small known examples check correct values and error handling. The notebook stops if any assertion fails.
    """),
    code("""
    perfect = evaluate_regression_predictions([1, 2, 3], [1, 2, 3])
    assert perfect["mae"] == 0 and perfect["mse"] == 0 and perfect["rmse"] == 0
    assert perfect["r_squared"] == 1

    known = evaluate_regression_predictions([1, 2, 3], [2, 2, 1])
    assert math.isclose(known["mae"], 1.0)
    assert math.isclose(known["mse"], 5 / 3)
    assert math.isclose(known["rmse"], math.sqrt(5 / 3))
    assert math.isclose(known["mean_signed_error"], -1 / 3)

    true_input = np.array([1., 2., 3.]); negative_input = np.array([-1., 2., 3.])
    negative_case = evaluate_regression_predictions(true_input, negative_input)
    assert math.isclose(negative_case["negative_prediction_rate"], 1 / 3)
    assert negative_case["rmsle"] is None and negative_case["rmsle_clipped_zero"] is not None
    assert np.array_equal(negative_input, np.array([-1., 2., 3.]))

    zero_case = evaluate_regression_predictions([0, 0], [0, 1])
    assert zero_case["mape_percent"] is None and zero_case["wape_percent"] is None
    assert evaluate_regression_predictions([2, 2], [2, 3])["r_squared"] is None

    for bad_true, bad_pred, expected_error in [
        ([1, 2], [1], ValueError), ([1, 2], [1, np.nan], ValueError),
        ([1, 2], [1, np.inf], ValueError), ([1, "x"], [1, 2], TypeError),
        ([], [], ValueError), ([[1, 2]], [[1, 2]], ValueError),
    ]:
        try:
            evaluate_regression_predictions(bad_true, bad_pred)
            raise AssertionError("Invalid metric input did not raise an error.")
        except expected_error:
            pass

    metric_schema = {
        "primary_metric": "mae", "target_unit": CONFIG["target_unit"],
        "definitions": {
            "mae": "mean absolute error in target units", "mse": "mean squared error",
            "rmse": "square root of MSE in target units", "mape_percent": "mean absolute percentage error; unavailable for zero targets",
            "r_squared": "coefficient of determination; unavailable for constant targets or fewer than two rows",
            "rmsle": "RMSLE when targets and predictions are non-negative", "rmsle_clipped_zero": "RMSLE with negative predictions clipped only for this metric",
            "median_absolute_error": "median absolute error", "wape_percent": "sum absolute error divided by sum absolute target, as percent",
            "mean_signed_error": "mean(prediction - target); positive means overprediction", "p90_absolute_error": "90th percentile absolute error",
            "negative_prediction_rate": "share of predictions below zero", "mae_usd": "MAE multiplied by 1000",
            "rmse_usd": "RMSE multiplied by 1000"
        },
        "all_final_metrics_use_original_target_scale": True,
    }
    (ARTIFACT_DIRS["data_contract"] / "metric_schema.json").write_text(json.dumps(metric_schema, indent=2), encoding="utf-8")
    metric_unit_tests_passed = True
    print("Metric unit tests passed:", metric_unit_tests_passed)
    """),
    md("""
    All metric tests passed. Invalid, missing, infinite, empty, and mismatched inputs raise clear errors.
    """),
    md("""
    ## 13. Experiment Result Registry

    The registry gives later prompts one stable result format. Existing valid rows are preserved.
    """),
    code("""
    REGISTRY_COLUMNS = [
        "experiment_id", "timestamp_utc", "model_family", "model_name", "sensitive_mode", "feature_set",
        "target_mode", "evaluation_stage", "fold_number", "training_row_count", "validation_row_count", "test_row_count",
        "parameter_json", "mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero",
        "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate",
        "fit_time_seconds", "prediction_time_seconds", "status", "notes", "model_artifact_path", "prediction_artifact_path"
    ]
    registry_path = ARTIFACT_DIRS["results"] / "experiment_results.csv"

    def validate_result_registry(frame):
        missing = [c for c in REGISTRY_COLUMNS if c not in frame.columns]
        extra = [c for c in frame.columns if c not in REGISTRY_COLUMNS]
        if missing or extra: raise ValueError(f"Registry schema mismatch. Missing={missing}, extra={extra}")
        if frame["experiment_id"].dropna().duplicated().any(): raise ValueError("Experiment IDs must be unique.")
        if not frame.empty:
            if frame["experiment_id"].isna().any(): raise ValueError("Experiment IDs must not be missing.")
            if pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True).isna().any(): raise ValueError("Registry timestamps must be valid UTC timestamps.")
            if (~frame["status"].isin(["success", "failed", "skipped"])).any(): raise ValueError("Registry status is invalid.")
            for count_column in ["training_row_count", "validation_row_count", "test_row_count"]:
                values = pd.to_numeric(frame[count_column], errors="coerce")
                if values.isna().any() or (values < 0).any(): raise ValueError(f"{count_column} must be a non-negative number.")
            for value in frame["parameter_json"]:
                json.loads(value)
            metric_columns = ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"]
            numeric_metrics = frame[metric_columns].apply(pd.to_numeric, errors="coerce")
            if np.isinf(numeric_metrics.to_numpy(dtype=float)).any(): raise ValueError("Registry metrics must not be infinite.")
        return True

    def append_experiment_result(frame, record):
        missing = [c for c in REGISTRY_COLUMNS if c not in record]
        extra = [c for c in record if c not in REGISTRY_COLUMNS]
        if missing or extra: raise ValueError(f"Result record schema mismatch. Missing={missing}, extra={extra}")
        result = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
        validate_result_registry(result)
        return result

    def save_result_registry(frame, path=registry_path):
        validate_result_registry(frame)
        temp_path = path.with_suffix(".tmp")
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)

    if registry_path.exists():
        experiment_results = pd.read_csv(registry_path)
        validate_result_registry(experiment_results)
    else:
        experiment_results = pd.DataFrame(columns=REGISTRY_COLUMNS)
    save_result_registry(experiment_results)
    reloaded_registry = pd.read_csv(registry_path)
    registry_round_trip_passed = validate_result_registry(reloaded_registry) and list(reloaded_registry.columns) == REGISTRY_COLUMNS
    synthetic_record = {column: None for column in REGISTRY_COLUMNS}
    synthetic_record.update({
        "experiment_id": "registry_test_only", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "test", "model_name": "test", "sensitive_mode": "without_sensitive",
        "feature_set": "test", "target_mode": "raw", "evaluation_stage": "unit_test", "fold_number": 0,
        "training_row_count": 2, "validation_row_count": 1, "test_row_count": 0, "parameter_json": "{}",
        "status": "success", "notes": "in-memory registry validation only"
    })
    synthetic_registry = append_experiment_result(pd.DataFrame(columns=REGISTRY_COLUMNS), synthetic_record)
    registry_nonempty_append_test_passed = validate_result_registry(synthetic_registry)
    print({"registry_rows": len(reloaded_registry), "round_trip_passed": registry_round_trip_passed,
           "synthetic_append_test_passed": registry_nonempty_append_test_passed})
    """),
    md("""
    The empty or existing registry saves, reloads, and validates correctly. No experiment row was added because no model was trained.
    """),
    md("""
    ## 14. Saved Artifacts

    This section checks source safety again and confirms that required artifacts exist.
    """),
    code("""
    source_hashes_after = {name: file_fingerprint(path) for name, path in protected_source_paths.items()}
    after_path = ARTIFACT_DIRS["data_contract"] / "source_hashes_after.json"
    after_path.write_text(json.dumps(source_hashes_after, indent=2), encoding="utf-8")
    hashes_unchanged = all(source_hashes_before[name]["sha256"] == source_hashes_after[name]["sha256"] for name in protected_source_paths)
    if not hashes_unchanged:
        raise RuntimeError("CRITICAL: A source CSV hash changed.")

    required_artifacts = [
        before_path, after_path,
        ARTIFACT_DIRS["data_contract"] / "feature_inventory.csv",
        ARTIFACT_DIRS["data_contract"] / "feature_sets.json",
        ARTIFACT_DIRS["data_contract"] / "leakage_and_suspicious_columns.csv",
        ARTIFACT_DIRS["data_contract"] / "metric_schema.json",
        ARTIFACT_DIRS["splits"] / "train_row_ids.csv",
        ARTIFACT_DIRS["splits"] / "test_row_ids.csv",
        ARTIFACT_DIRS["splits"] / "split_config.json",
        ARTIFACT_DIRS["splits"] / "cv_fold_assignments.csv",
        ARTIFACT_DIRS["splits"] / "split_verification.json",
        registry_path,
    ]
    required_artifacts_exist = all(path.is_file() and path.stat().st_size > 0 for path in required_artifacts)
    artifact_table = pd.DataFrame({"artifact": [str(p.relative_to(PROJECT_ROOT)) for p in required_artifacts], "exists": [p.is_file() for p in required_artifacts]})
    display(artifact_table)
    print("Source hashes unchanged:", hashes_unchanged)
    """),
    md("""
    Required data-contract, split, metric, and registry artifacts exist. Both source CSV hashes are unchanged.
    """),
    md("""
    ## 15. Verification Summary

    The final report stores real pass or fail values. Reaching this cell proves that all earlier notebook assertions passed.
    """),
    code("""
    state_files = [PROJECT_ROOT / name for name in ["AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md"]]
    state_text = {path.name: path.read_text(encoding="utf-8") if path.is_file() else "" for path in state_files}
    reviewer_path = ARTIFACT_DIRS["reports"] / "prompt1_reviewer.md"
    state_freshness = (
        "Independent review cycle 1" in state_text["LOG.md"]
        and "First full notebook execution: PASS" in state_text["TASK.md"]
        and "Final execution proof" in state_text["DECISIONS.md"]
    )
    verification_checks = {
        "source_files_found": all(path.is_file() for path in source_paths.values()),
        "source_files_readable": len(df_with_sensitive_raw) > 0 and len(df_without_sensitive_raw) > 0,
        "source_hashes_recorded": before_path.is_file() and after_path.is_file(),
        "source_hashes_unchanged": hashes_unchanged,
        "equal_row_counts": len(df_with_sensitive_raw) == len(df_without_sensitive_raw),
        "equal_row_order": all(common_values_equal.values()),
        "equal_target_values": common_values_equal[TARGET],
        "equal_common_feature_values": all(common_values_equal[c] for c in common_columns if c != TARGET),
        "valid_target": all(target_checks.values()),
        "sensitive_feature_difference_valid": len(sensitive_features) > 0 and TARGET not in sensitive_features,
        "no_duplicate_columns": not duplicate_column_names(df_with_sensitive_raw) and not duplicate_column_names(df_without_sensitive_raw),
        "no_confirmed_target_leakage": len(exact_target_duplicates) == 0,
        "train_test_overlap_zero": split_checks["train_test_overlap_zero"],
        "train_test_coverage_complete": split_checks["train_test_coverage_complete"],
        "cv_fold_coverage_complete": split_checks["cv_coverage_complete"],
        "no_test_row_in_cv": split_checks["no_test_row_in_cv"],
        "split_reproducibility": split_checks["split_reproducible"] and split_checks["folds_reproducible"],
        "metric_unit_tests": metric_unit_tests_passed,
        "result_registry_save_and_reload": registry_round_trip_passed and registry_nonempty_append_test_passed,
        "notebook_reached_final_cell": True,
        "required_artifacts_exist": required_artifacts_exist,
        "required_markdown_files_updated": all(path.is_file() and path.stat().st_size > 0 for path in state_files) and state_freshness,
        "independent_reviewer_completed": reviewer_path.is_file() and reviewer_path.stat().st_size > 0,
        "part1_notebook_unchanged": source_hashes_before["part1_notebook"]["sha256"] == source_hashes_after["part1_notebook"]["sha256"],
        "no_real_model_trained": len(reloaded_registry) == 0,
    }
    verification_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INTERNAL_PASS_PENDING_EXTERNAL_EXECUTION" if all(verification_checks.values()) else "FAIL",
        "checks": verification_checks,
        "dataset_shapes": {"with_sensitive": list(df_with_sensitive_raw.shape), "without_sensitive": list(df_without_sensitive_raw.shape)},
        "target": {"name": TARGET, "unit": CONFIG["target_unit"], **target_summary},
        "feature_counts": {"with_sensitive": len(features_with_sensitive), "without_sensitive": len(features_without_sensitive), "sensitive": len(sensitive_features)},
        "split_counts": {"train": len(train_ids), "test": len(test_ids), "cv_folds": CONFIG["n_cv_folds"]},
        "remaining_risks": [
            "Random row splitting allows the same lenders and geographic groups in train and test; deployment should guide later group or time robustness checks.",
            "The target has a rare extreme tail, so skew and maximum values differ across subsets even when quantile-bin proportions are stable.",
            "The non-sensitive feature set still contains possible proxies for protected attributes."
        ],
    }
    verification_path = ARTIFACT_DIRS["reports"] / "prompt1_verification.json"
    verification_path.write_text(json.dumps(verification_report, indent=2), encoding="utf-8")
    if verification_report["status"] == "FAIL":
        raise AssertionError(f"Final verification failed: {verification_checks}")
    display(pd.DataFrame.from_dict(verification_checks, orient="index", columns=["passed"]))
    print("NOTEBOOK INTERNAL STATUS:", verification_report["status"])
    """),
    md("""
    ### Completion Note

    The notebook work for Prompt 1 is complete. The data contract, shared split, shared CV folds, metric helpers, metric tests, result registry, and safety checks were created and verified. No real model was trained. The test set is now locked. An external check must confirm the saved execution before final PASS. The next stage is baseline and linear regression models.
    """),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "prompt1_owned_sections": [str(i) for i in range(16)],
    },
)
nbf.write(notebook, OUTPUT)
print(f"Created {OUTPUT}")
