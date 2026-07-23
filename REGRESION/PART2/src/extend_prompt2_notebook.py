"""Safely append or repair Prompt 2 sections in the existing notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "REGRESSION_PART2_MODELING.ipynb"


def md(text):
    cell = nbf.v4.new_markdown_cell(dedent(text).strip())
    cell.metadata["prompt_owner"] = "prompt2"
    return cell


def code(text):
    cell = nbf.v4.new_code_cell(dedent(text).strip())
    cell.metadata["prompt_owner"] = "prompt2"
    return cell


notebook = nbf.read(NOTEBOOK_PATH, as_version=4)
notebook.cells = [cell for cell in notebook.cells if cell.metadata.get("prompt_owner") != "prompt2"]


def code_after_heading(heading):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and heading in cell.source:
            for candidate in notebook.cells[index + 1:]:
                if candidate.cell_type == "code":
                    return candidate
    raise KeyError(heading)


# Stage guards preserve Prompt 1 history and prevent test use or split rewrites.
config_cell = code_after_heading("## 1. Imports and Configuration")
if "CURRENT_PROJECT_STAGE = 2" not in config_cell.source:
    config_cell.source += "\n\n# Later stages use saved Prompt 1 artifacts without recreating them.\nCURRENT_PROJECT_STAGE = 2"

hash_cell = code_after_heading("## 3. Source Data Protection")
hash_cell.source = dedent("""
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
    if before_path.exists():
        source_hashes_before = json.loads(before_path.read_text(encoding="utf-8"))
    else:
        before_path.write_text(json.dumps(source_hashes_before, indent=2), encoding="utf-8")
    display(pd.DataFrame(source_hashes_before).T[["size_bytes", "sha256"]])
""").strip()

split_cell = code_after_heading("## 9. Shared Train-Test Split")
split_cell.source = dedent("""
    # Prompt 2 reuses the saved Prompt 1 split. It never calls a new main splitter.
    train_path = ARTIFACT_DIRS["splits"] / "train_row_ids.csv"
    test_path = ARTIFACT_DIRS["splits"] / "test_row_ids.csv"
    split_config_path = ARTIFACT_DIRS["splits"] / "split_config.json"
    train_ids = pd.read_csv(train_path)["row_id"].to_numpy(dtype=np.int64)
    test_ids = pd.read_csv(test_path)["row_id"].to_numpy(dtype=np.int64)
    split_config = json.loads(split_config_path.read_text(encoding="utf-8"))
    row_ids = np.arange(len(y), dtype=np.int64)
    if len(np.intersect1d(train_ids, test_ids)) != 0:
        raise AssertionError("Saved train and test rows overlap.")
    if not np.array_equal(np.sort(np.concatenate([train_ids, test_ids])), row_ids):
        raise AssertionError("Saved train and test rows do not cover all rows.")
    print({"train_rows_loaded": len(train_ids), "test_rows_locked": len(test_ids), "split_recreated": False})
""").strip()

cv_cell = code_after_heading("## 10. Shared Cross-Validation Folds")
cv_cell.source = dedent("""
    # Prompt 2 loads the saved fold assignment and does not create new folds.
    cv_path = ARTIFACT_DIRS["splits"] / "cv_fold_assignments.csv"
    cv_assignments = pd.read_csv(cv_path, dtype={"row_id": "int64", "fold": "int64"}).sort_values("row_id").reset_index(drop=True)
    if cv_assignments["row_id"].duplicated().any():
        raise AssertionError("A training row has more than one saved fold.")
    if not np.array_equal(cv_assignments["row_id"].to_numpy(), np.sort(train_ids)):
        raise AssertionError("Saved CV rows do not equal saved training rows.")
    if set(cv_assignments["fold"]) != {0, 1, 2}:
        raise AssertionError("Saved CV fold labels are invalid.")
    display(cv_assignments.groupby("fold").size().rename("rows").to_frame())
""").strip()

split_verify_cell = notebook.cells[31]
split_verify_cell.source = dedent("""
    # This stage validates training folds only. Historical test statistics are not recomputed or used.
    def distribution_summary(name, values):
        values = pd.Series(values)
        return {
            "set": name, "rows": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "median": float(values.median()), "p90": float(values.quantile(.90)),
            "p95": float(values.quantile(.95)), "p99": float(values.quantile(.99)),
            "max": float(values.max()), "skew": float(values.skew()),
        }

    y_array = y.to_numpy()
    train_target = y_array[train_ids]
    train_fold_lookup = cv_assignments.set_index("row_id")["fold"]
    distribution_rows = [distribution_summary("train", train_target)]
    for fold in range(CONFIG["n_cv_folds"]):
        fold_ids = cv_assignments.loc[cv_assignments["fold"] == fold, "row_id"].to_numpy()
        distribution_rows.append(distribution_summary(f"cv_fold_{fold}", y_array[fold_ids]))
    distribution_table = pd.DataFrame(distribution_rows)
    split_checks = {
        "train_test_overlap_zero": len(np.intersect1d(train_ids, test_ids)) == 0,
        "train_test_coverage_complete": np.array_equal(np.sort(np.concatenate([train_ids, test_ids])), row_ids),
        "cv_coverage_complete": np.array_equal(cv_assignments["row_id"].to_numpy(), np.sort(train_ids)),
        "no_test_row_in_cv": len(np.intersect1d(cv_assignments["row_id"].to_numpy(), test_ids)) == 0,
        "fold_sizes_reasonable": int(cv_assignments.groupby("fold").size().max() - cv_assignments.groupby("fold").size().min()) <= 1,
        "same_rows_for_both_sensitive_modes": len(df_with_sensitive_raw) == len(df_without_sensitive_raw) == len(row_ids),
        "target_order_correct": common_values_equal[TARGET],
    }
    if not all(split_checks.values()):
        raise AssertionError(f"Saved split verification failed: {split_checks}")
    display(distribution_table)
    print("Historical test statistics were not read or recomputed in Prompt 2.")
""").strip()

after_hash_cell = code_after_heading("## 14. Saved Artifacts")
after_hash_cell.source = dedent("""
    source_hashes_after = {name: file_fingerprint(path) for name, path in protected_source_paths.items()}
    after_path = ARTIFACT_DIRS["data_contract"] / "source_hashes_after.json"
    if not after_path.exists():
        after_path.write_text(json.dumps(source_hashes_after, indent=2), encoding="utf-8")
    hashes_unchanged = all(source_hashes_before[name]["sha256"] == source_hashes_after[name]["sha256"] for name in protected_source_paths)
    if not hashes_unchanged:
        raise RuntimeError("CRITICAL: A protected Prompt 1 source hash changed.")

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
    print("Protected Prompt 1 source hashes unchanged:", hashes_unchanged)
""").strip()

verify_cell = code_after_heading("## 15. Verification Summary")
verify_cell.source = dedent("""
    # Prompt 1 completion is historical. Later stages preserve its saved PASS report.
    verification_path = ARTIFACT_DIRS["reports"] / "prompt1_verification.json"
    prompt1_historical_report = json.loads(verification_path.read_text(encoding="utf-8"))
    prompt1_historical_checks = {
        "saved_prompt1_status_pass": prompt1_historical_report.get("status") == "PASS",
        "prompt1_recorded_no_model_training": bool(prompt1_historical_report.get("checks", {}).get("no_real_model_trained", False)),
        "current_source_hashes_match_prompt1": all(
            file_fingerprint(protected_source_paths[name])["sha256"] == source_hashes_before[name]["sha256"]
            for name in protected_source_paths
        ),
        "saved_split_still_valid": all(split_checks.values()),
        "metric_unit_tests_still_pass": metric_unit_tests_passed,
        "registry_schema_still_valid": registry_round_trip_passed and registry_nonempty_append_test_passed,
    }
    if not all(prompt1_historical_checks.values()):
        raise AssertionError(f"Prompt 1 historical validation failed: {prompt1_historical_checks}")
    display(pd.DataFrame.from_dict(prompt1_historical_checks, orient="index", columns=["passed"]))
    print("PROMPT 1 HISTORICAL STATUS: PASS (report preserved)")
""").strip()

notebook.cells[41].source = (
    "The registry saves, reloads, and validates correctly. Prompt 1 added no model rows. "
    "A later stage may add rows with deterministic IDs."
)
notebook.cells[47].source = dedent("""
    ### Prompt 1 Historical Completion Note

    Prompt 1 created and verified the shared data foundation. It trained no model at that time. Later stages reuse the locked split, folds, metrics, and historical PASS report.
""").strip()


new_cells = []
new_cells += [
    md("""
    ## 16. Prompt 2 Objective and Rules

    Prompt 2 compares six linear-family models in two sensitive modes. It uses training rows and saved folds only. The locked test set is not used for fitting, prediction, selection, or statistics.
    """),
    code("""
    import gc
    import os
    import platform
    import subprocess
    import sys
    import time
    import warnings
    from copy import deepcopy

    import joblib
    import matplotlib.pyplot as plt
    import nbformat as nbf
    import sklearn
    from IPython.display import Markdown
    from scipy import sparse
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import train_test_split

    from prompt2_pipeline_utils import (
        canonical_json, configuration_digest, deterministic_experiment_id,
        fitted_estimator, fitted_pipeline, make_complete_pipeline,
        make_preprocessor, transformed_feature_names,
    )

    PROMPT2_VERSION = "linear_compact_v1"
    P2_DIRS = {
        "models": CONFIG["artifact_root"] / "models" / "linear",
        "predictions": CONFIG["artifact_root"] / "predictions" / "linear",
        "results": CONFIG["artifact_root"] / "results" / "prompt2",
        "features": CONFIG["artifact_root"] / "features" / "linear",
        "coefficients": CONFIG["artifact_root"] / "features" / "linear" / "coefficients",
        "figures": CONFIG["artifact_root"] / "figures" / "prompt2",
        "reports": CONFIG["artifact_root"] / "reports",
        "manifests": CONFIG["artifact_root"] / "manifests",
    }
    for directory in P2_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    P2_START_TIME = time.perf_counter()
    print("Prompt 2 directories are ready. Large fits will run sequentially.")
    """),
    md("""
    ## 17. Prompt 1 Artifact Validation

    Prompt 2 checks live source and split hashes against a saved baseline. It also reloads row IDs, folds, feature metadata, metrics, and the registry from disk.
    """),
    code("""
    def file_sha256(path):
        return sha256_stream(Path(path))

    prompt2_protected_paths = {
        "with_sensitive_csv": CONFIG["with_sensitive_path"],
        "without_sensitive_csv": CONFIG["without_sensitive_path"],
        "part1_notebook": CONFIG["source_notebook_path"],
        "train_row_ids": ARTIFACT_DIRS["splits"] / "train_row_ids.csv",
        "test_row_ids": ARTIFACT_DIRS["splits"] / "test_row_ids.csv",
        "cv_fold_assignments": ARTIFACT_DIRS["splits"] / "cv_fold_assignments.csv",
        "split_config": ARTIFACT_DIRS["splits"] / "split_config.json",
    }
    protected_before_path = P2_DIRS["manifests"] / "prompt2_protected_hashes_before.json"
    current_protected_hashes = {name: file_sha256(path) for name, path in prompt2_protected_paths.items()}
    if protected_before_path.exists():
        prompt2_protected_hashes_before = json.loads(protected_before_path.read_text(encoding="utf-8"))
        if prompt2_protected_hashes_before != current_protected_hashes:
            raise RuntimeError("A protected Prompt 1 source or split artifact changed before Prompt 2.")
    else:
        prompt2_protected_hashes_before = current_protected_hashes
        protected_before_path.write_text(json.dumps(prompt2_protected_hashes_before, indent=2), encoding="utf-8")

    disk_train_ids = pd.read_csv(prompt2_protected_paths["train_row_ids"])["row_id"].to_numpy(dtype=np.int64)
    disk_test_ids = pd.read_csv(prompt2_protected_paths["test_row_ids"])["row_id"].to_numpy(dtype=np.int64)
    disk_cv = pd.read_csv(prompt2_protected_paths["cv_fold_assignments"], dtype={"row_id": "int64", "fold": "int64"}).sort_values("row_id").reset_index(drop=True)
    disk_feature_sets = json.loads((ARTIFACT_DIRS["data_contract"] / "feature_sets.json").read_text(encoding="utf-8"))
    disk_metric_schema = json.loads((ARTIFACT_DIRS["data_contract"] / "metric_schema.json").read_text(encoding="utf-8"))
    disk_registry = pd.read_csv(registry_path)
    prompt1_report = json.loads((ARTIFACT_DIRS["reports"] / "prompt1_verification.json").read_text(encoding="utf-8"))
    artifact_checks = {
        "prompt1_report_pass": prompt1_report.get("status") == "PASS",
        "train_ids_match_memory": np.array_equal(disk_train_ids, train_ids),
        "test_ids_match_memory": np.array_equal(disk_test_ids, test_ids),
        "cv_matches_memory": disk_cv.equals(cv_assignments),
        "target_matches_metadata": disk_feature_sets["target_column"] == TARGET,
        "sensitive_features_match": disk_feature_sets["sensitive_features"] == sensitive_features,
        "metric_schema_primary_mae": disk_metric_schema["primary_metric"] == "mae",
        "registry_schema_valid": validate_result_registry(disk_registry),
        "train_test_overlap_zero": len(np.intersect1d(disk_train_ids, disk_test_ids)) == 0,
        "cv_training_only": len(np.intersect1d(disk_cv["row_id"].to_numpy(), disk_test_ids)) == 0,
        "target_positive_on_training": bool((y_array[disk_train_ids] > 0).all()),
    }
    if not all(artifact_checks.values()):
        raise AssertionError(f"Prompt 1 artifact validation failed: {artifact_checks}")
    display(pd.DataFrame.from_dict(artifact_checks, orient="index", columns=["passed"]))
    """),
    md("""
    Prompt 1 artifacts match memory and their baseline hashes. The test IDs are used only in exclusion checks. No test target or test feature view is created.
    """),
    md("""
    ## 18. Training Data Views

    Training views follow the saved row order. This makes OOF positions and original `row_id` values easy to audit.
    """),
    code("""
    train_ids = np.sort(disk_train_ids)
    test_id_set = set(disk_test_ids.tolist())
    fold_by_train_position = disk_cv.set_index("row_id").loc[train_ids, "fold"].to_numpy(dtype=int)
    y_train_prompt2 = y_array[train_ids].astype(float, copy=True)
    if len(y_train_prompt2) != len(train_ids) or not np.isfinite(y_train_prompt2).all():
        raise AssertionError("Prompt 2 training target view is invalid.")
    print({"training_rows": len(train_ids), "folds": pd.Series(fold_by_train_position).value_counts().sort_index().to_dict(), "test_rows_used": 0})
    """),
    md("""
    The training view contains 399,788 rows and the saved fold sizes. The locked test rows are not part of any model view.
    """),
    md("""
    ## 19. Linear Feature Pack

    The compact pack removes detailed identifiers and one exact duplicate numeric field. It keeps useful numeric, categorical, and existing engineered features.
    """),
    code("""
    linear_exclusions = {
        "respondent_id": "high-cardinality lender identifier",
        "msamd_name": "high-cardinality detailed metro geography",
        "county_name": "high-cardinality detailed county geography",
        "county_code": "geographic code; readable broader geography is retained",
        "state_code": "redundant code for retained state_name",
        "census_tract_number": "high-cardinality tract identifier",
        "tract_to_msamd_income": "exact linear duplicate of retained tract_income_ratio times 100",
    }
    base_without = list(disk_feature_sets["features_without_sensitive"])
    linear_compact_without_sensitive = [feature for feature in base_without if feature not in linear_exclusions]
    linear_compact_with_sensitive = linear_compact_without_sensitive + list(sensitive_features)
    difference = set(linear_compact_with_sensitive) - set(linear_compact_without_sensitive)
    if difference != set(sensitive_features):
        raise AssertionError("Linear feature-pack difference does not equal the validated sensitive set.")
    if linear_compact_with_sensitive[:len(linear_compact_without_sensitive)] != linear_compact_without_sensitive:
        raise AssertionError("The common feature order changed.")
    if TARGET in linear_compact_with_sensitive or "row_id" in linear_compact_with_sensitive:
        raise AssertionError("Target or row_id entered a feature pack.")

    inventory_lookup = feature_inventory.set_index("column_name")
    numeric_without = [f for f in linear_compact_without_sensitive if inventory_lookup.loc[f, "inferred_feature_type"] == "numeric"]
    categorical_without = [f for f in linear_compact_without_sensitive if f not in numeric_without]
    numeric_with = [f for f in linear_compact_with_sensitive if inventory_lookup.loc[f, "inferred_feature_type"] == "numeric"]
    categorical_with = [f for f in linear_compact_with_sensitive if f not in numeric_with]
    linear_feature_sets = {
        "version": PROMPT2_VERSION,
        "without_sensitive": linear_compact_without_sensitive,
        "with_sensitive": linear_compact_with_sensitive,
        "numeric_without_sensitive": numeric_without,
        "categorical_without_sensitive": categorical_without,
        "numeric_with_sensitive": numeric_with,
        "categorical_with_sensitive": categorical_with,
        "validated_sensitive_features": sensitive_features,
        "exclusions": linear_exclusions,
    }
    feature_set_path = P2_DIRS["features"] / "linear_feature_sets.json"
    feature_set_path.write_text(json.dumps(linear_feature_sets, indent=2), encoding="utf-8")

    feature_inventory_rows = []
    for feature in features_with_sensitive:
        included = feature in linear_compact_with_sensitive
        feature_inventory_rows.append({
            "feature": feature,
            "feature_type": inventory_lookup.loc[feature, "inferred_feature_type"],
            "included": included,
            "sensitive": feature in sensitive_features,
            "exclusion_reason": "" if included else linear_exclusions.get(feature, "not in compact family pack"),
        })
    linear_feature_inventory = pd.DataFrame(feature_inventory_rows)
    linear_feature_inventory.to_csv(P2_DIRS["features"] / "linear_feature_inventory.csv", index=False)

    X_without_train = df_without_sensitive_raw.iloc[train_ids][linear_compact_without_sensitive].copy()
    X_with_train = df_with_sensitive_raw.iloc[train_ids][linear_compact_with_sensitive].copy()
    X_without_train.index = np.arange(len(X_without_train))
    X_with_train.index = np.arange(len(X_with_train))
    print({"without_sensitive_features": len(linear_compact_without_sensitive), "with_sensitive_features": len(linear_compact_with_sensitive),
           "numeric_without": len(numeric_without), "categorical_without": len(categorical_without)})
    display(linear_feature_inventory.loc[~linear_feature_inventory["included"]].head(10))
    """),
    md("""
    The non-sensitive pack has 28 fields. The sensitive pack has 36 fields and adds exactly the eight validated sensitive fields. Detailed lender and geography identifiers are excluded to control one-hot size.
    """),
    md("""
    ## 20. Shared Linear Preprocessing

    Each complete pipeline gets a new preprocessor. Numeric values use median imputation and the selected scaler. Categories use safe conversion, most-frequent imputation, rare-level grouping, and sparse one-hot encoding.
    """),
    code("""
    preprocessor_definition = make_preprocessor(numeric_without, categorical_without, "standard")
    preprocessing_checks = {
        "numeric_has_imputer_and_scaler": [name for name, _ in preprocessor_definition.transformers[0][1].steps] == ["imputer", "scaler"],
        "categorical_has_conversion_imputer_encoder": [name for name, _ in preprocessor_definition.transformers[1][1].steps] == ["to_object", "imputer", "encoder"],
        "sparse_output_forced": preprocessor_definition.sparse_threshold == 1.0,
        "no_global_fit": not hasattr(preprocessor_definition, "transformers_"),
    }
    if not all(preprocessing_checks.values()):
        raise AssertionError(f"Preprocessor definition is invalid: {preprocessing_checks}")
    del preprocessor_definition
    print(preprocessing_checks)
    """),
    md("""
    The factory is valid and remains unfitted. Learned imputation, scaling, and encoding will happen inside each candidate, fold, and final pipeline.
    """),
]

new_cells += [
    md("""
    ## 36. Saved Model Reload Tests

    A clean Python process reloads all twelve joblib files. It sends raw DataFrame rows through each complete pipeline and compares predictions with strict tolerances.
    """),
    code("""
    reload_command = [sys.executable, str(PROJECT_ROOT / "verify_prompt2_models.py"), str(PROJECT_ROOT)]
    reload_process = subprocess.run(reload_command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    print(reload_process.stdout[-4000:])
    if reload_process.returncode != 0:
        print(reload_process.stderr[-4000:])
        raise RuntimeError("Clean-process model reload verification failed.")
    reload_verification_path = P2_DIRS["reports"] / "prompt2_model_reload_verification.csv"
    reload_verification = pd.read_csv(reload_verification_path)
    if len(reload_verification) != 12 or not reload_verification["passed"].all():
        raise AssertionError("Not all saved pipelines passed reload verification.")
    display(reload_verification[["model_name", "sensitive_mode", "prediction_count", "max_abs_difference", "complete_pipeline", "passed"]])
    """),
    md("""
    All twelve complete pipelines reload in a clean process. Reloaded predictions match the saved reference predictions and remain finite.
    """),
    md("""
    ## 37. Prompt 2 Figures

    Four compact figures show OOF MAE, fold variation, the sensitive-mode difference, and raw-versus-log development results.
    """),
    code("""
    figure_paths = []

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = linear_leaderboard.pivot(index="model_name", columns="sensitive_mode", values="oof_mae")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Training OOF MAE by Model and Sensitive Mode")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (thousands of US dollars)")
    ax.legend(title="Sensitive mode")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = P2_DIRS["figures"] / "oof_mae_by_model_and_mode.png"
    fig.savefig(path, dpi=140); plt.close(fig); figure_paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    cv_fold_results.boxplot(column="mae", by="model_name", ax=ax, grid=False)
    ax.set_title("Fold MAE Distribution by Model")
    fig.suptitle("")
    ax.set_xlabel("Model")
    ax.set_ylabel("Fold MAE (thousands of US dollars)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = P2_DIRS["figures"] / "fold_mae_distribution.png"
    fig.savefig(path, dpi=140); plt.close(fig); figure_paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(linear_sensitive_comparison["model_name"], linear_sensitive_comparison["mae_difference_with_minus_without"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Sensitive minus Non-sensitive OOF MAE")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE difference (thousands of US dollars)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = P2_DIRS["figures"] / "sensitive_mae_difference.png"
    fig.savefig(path, dpi=140); plt.close(fig); figure_paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    screening_plot = development_screening_results.loc[development_screening_results["target_mode"].isin(["raw", "log1p"])].copy()
    summary_plot = screening_plot.groupby(["screening_group", "target_mode"], as_index=False)["mae"].min()
    for target_mode, group in summary_plot.groupby("target_mode"):
        ax.plot(group["screening_group"], group["mae"], marker="o", label=target_mode)
    ax.set_title("Best Development MAE for Raw and Log Targets")
    ax.set_xlabel("Screening group")
    ax.set_ylabel("Development MAE (thousands of US dollars)")
    ax.legend(title="Target mode")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = P2_DIRS["figures"] / "raw_vs_log_development.png"
    fig.savefig(path, dpi=140); plt.close(fig); figure_paths.append(path)
    print([str(path.relative_to(PROJECT_ROOT)) for path in figure_paths])
    """),
    md("""
    The figures use training-only development and OOF results. They are comparison aids and do not contain final test performance.
    """),
    md("""
    ## 38. Prompt 2 Artifact Summary

    Prompt 2 now saves results, warnings, models, predictions, coefficients, figures, manifests, and an idempotent subset of the main registry.
    """),
    code("""
    if warnings_path.exists():
        existing_warning_frame = pd.read_csv(warnings_path)
    else:
        existing_warning_frame = pd.DataFrame()
    new_warning_frame = pd.DataFrame(warning_records)
    if not existing_warning_frame.empty or not new_warning_frame.empty:
        model_warnings = pd.concat([existing_warning_frame, new_warning_frame], ignore_index=True).drop_duplicates()
    else:
        model_warnings = pd.DataFrame(columns=["stage", "model", "fold", "sensitive_mode", "target_mode",
                                                "warning_class", "warning_message", "retry_number", "fit_overrides"])
    model_warnings.to_csv(warnings_path, index=False)

    experiment_results = upsert_prompt2_registry(pd.read_csv(registry_path), prompt2_registry_records)
    prompt2_registry_rows = experiment_results.loc[experiment_results["experiment_id"].str.startswith("p2__", na=False)].copy()
    prompt2_registry_path = P2_DIRS["results"] / "prompt2_registry_rows.csv"
    prompt2_registry_rows.to_csv(prompt2_registry_path, index=False)
    if prompt2_registry_rows["experiment_id"].duplicated().any():
        raise AssertionError("Prompt 2 registry rows contain duplicate IDs.")

    runtime_path = P2_DIRS["results"] / "prompt2_runtime_summary.csv"
    if runtime_path.exists() and screening_cache_valid and cv_cache_valid and final_cache_valid:
        prompt2_runtime_summary = pd.read_csv(runtime_path)
    else:
        prompt2_runtime_summary = pd.DataFrame([
            {"stage": "development_screening", "runtime_seconds": screening_runtime_seconds, "fit_count": 47},
            {"stage": "full_cross_validation", "runtime_seconds": cv_runtime_seconds, "fit_count": 36},
            {"stage": "final_training", "runtime_seconds": final_runtime_seconds, "fit_count": 12},
        ])
        prompt2_runtime_summary.to_csv(runtime_path, index=False)

    artifact_summary = {
        "screening_rows": len(development_screening_results), "cv_fold_rows": len(cv_fold_results),
        "oof_summary_rows": len(cv_oof_summary), "oof_files": sum(path.exists() for path in expected_oof_paths.values()),
        "model_files": sum(path.exists() for path in expected_model_paths.values()),
        "coefficient_files": len(list(P2_DIRS["coefficients"].glob("*.csv"))),
        "figure_files": sum(path.exists() for path in figure_paths),
        "prompt2_registry_rows": len(prompt2_registry_rows), "warning_rows": len(model_warnings),
    }
    display(pd.DataFrame.from_dict(artifact_summary, orient="index", columns=["count"]))
    display(prompt2_runtime_summary)
    """),
    md("""
    The artifact counts are compact and stable. The main registry keeps one row per deterministic Prompt 2 experiment ID and preserves rows from other stages.
    """),
    md("""
    ## 39. Prompt 2 Verification

    Final checks recompute OOF coverage and metrics, inspect saved pipelines and registry rows, and compare protected hashes. The first run creates an idempotence snapshot. The second run must match it.
    """),
    code("""
    protected_hashes_after = {name: file_sha256(path) for name, path in prompt2_protected_paths.items()}
    protected_after_path = P2_DIRS["manifests"] / "prompt2_protected_hashes_after.json"
    protected_after_path.write_text(json.dumps(protected_hashes_after, indent=2), encoding="utf-8")

    oof_deep_checks = []
    recomputed_metric_checks = []
    required_prediction_columns = {"row_id", "fold", "y_true", "y_pred", "absolute_error", "signed_error",
                                   "model_name", "sensitive_mode", "target_mode", "feature_set", "experiment_id"}
    for (family, mode), path in expected_oof_paths.items():
        frame = pd.read_csv(path)
        saved_summary = cv_oof_summary.loc[(cv_oof_summary.model_name == family) & (cv_oof_summary.sensitive_mode == mode)].iloc[0]
        recomputed = evaluate_regression_predictions(frame["y_true"], frame["y_pred"])
        deep_pass = (
            required_prediction_columns.issubset(frame.columns) and len(frame) == len(train_ids)
            and not frame["row_id"].duplicated().any()
            and np.array_equal(frame["row_id"].to_numpy(dtype=int), train_ids)
            and len(set(frame["row_id"]) & test_id_set) == 0
            and np.array_equal(frame["fold"].to_numpy(dtype=int), fold_by_train_position)
            and np.array_equal(frame["y_true"].to_numpy(dtype=float), y_train_prompt2)
            and np.isfinite(frame[["y_true", "y_pred", "absolute_error", "signed_error"]].to_numpy()).all()
            and np.allclose(frame["absolute_error"], np.abs(frame["y_pred"] - frame["y_true"]), rtol=0, atol=1e-12)
            and np.allclose(frame["signed_error"], frame["y_pred"] - frame["y_true"], rtol=0, atol=1e-12)
            and not any(column in frame.columns for column in sensitive_features)
        )
        oof_deep_checks.append(deep_pass)
        recomputed_metric_checks.append(
            math.isclose(recomputed["mae"], saved_summary.mae, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(recomputed["rmse"], saved_summary.rmse, rel_tol=1e-12, abs_tol=1e-12)
        )

    loaded_model_objects = [joblib.load(path) for path in expected_model_paths.values()]
    complete_pipeline_checks = [
        hasattr(fitted_pipeline(model), "named_steps")
        and "preprocessor" in fitted_pipeline(model).named_steps
        and "regressor" in fitted_pipeline(model).named_steps
        for model in loaded_model_objects
    ]
    del loaded_model_objects
    gc.collect()

    current_notebook = nbf.read(PROJECT_ROOT / "REGRESSION_PART2_MODELING.ipynb", as_version=4)
    prompt2_headings = [line.strip() for cell in current_notebook.cells if cell.cell_type == "markdown"
                        for line in cell.source.splitlines() if line.startswith("## ") and any(line.startswith(f"## {n}.") for n in range(16, 41))]
    prompt2_source_text = "\\n".join(cell.source for cell in current_notebook.cells)
    idempotence_snapshot_path = P2_DIRS["manifests"] / "prompt2_idempotence_snapshot.json"
    current_snapshot = {
        "prompt2_heading_count": len(prompt2_headings),
        "prompt2_heading_unique_count": len(set(prompt2_headings)),
        "selected_configuration_digest": selected_configuration_digest,
        "development_manifest_sha256": development_manifest_hash,
        "prompt2_registry_ids": sorted(prompt2_registry_rows["experiment_id"].tolist()),
        "oof_files": sorted(path.name for path in expected_oof_paths.values()),
        "model_files": sorted(path.name for path in expected_model_paths.values()),
    }
    snapshot_preexisted = idempotence_snapshot_path.exists()
    if snapshot_preexisted:
        saved_snapshot = json.loads(idempotence_snapshot_path.read_text(encoding="utf-8"))
        idempotence_match = saved_snapshot == current_snapshot
    else:
        idempotence_snapshot_path.write_text(json.dumps(current_snapshot, indent=2), encoding="utf-8")
        idempotence_match = False

    prompt2_checks = {
        "prompt1_found_and_valid": prompt1_report.get("status") == "PASS" and all(artifact_checks.values()),
        "obsolete_registry_empty_assertion_stage_aware": ("len(reloaded_registry)" + " == 0") not in prompt2_source_text and prompt1_historical_checks["prompt1_recorded_no_model_training"],
        "source_and_part1_hashes_unchanged": all(protected_hashes_after[name] == prompt2_protected_hashes_before[name] for name in ["with_sensitive_csv", "without_sensitive_csv", "part1_notebook"]),
        "split_artifact_hashes_unchanged": all(protected_hashes_after[name] == prompt2_protected_hashes_before[name] for name in ["train_row_ids", "test_row_ids", "cv_fold_assignments", "split_config"]),
        "train_test_overlap_zero": len(np.intersect1d(train_ids, disk_test_ids)) == 0,
        "development_training_only_and_reproducible": all(development_checks.values()) and file_sha256(development_manifest_path) == development_manifest_hash,
        "no_test_row_used_in_cv_or_oof": len(np.intersect1d(disk_cv["row_id"], disk_test_ids)) == 0 and all(oof_deep_checks),
        "no_test_prediction_artifact": not any("test" in path.name.lower() for path in P2_DIRS["predictions"].glob("*")),
        "linear_feature_packs_valid": difference == set(sensitive_features) and TARGET not in linear_compact_with_sensitive and "row_id" not in linear_compact_with_sensitive,
        "high_cardinality_identifiers_excluded": all(feature not in linear_compact_with_sensitive for feature in linear_exclusions),
        "preprocessing_inside_complete_pipelines": len(complete_pipeline_checks) == 12 and all(complete_pipeline_checks),
        "fresh_pipeline_created_for_every_fold": json.loads(cv_manifest_path.read_text(encoding="utf-8"))["fresh_pipeline_fit_count"] == 36,
        "screening_non_sensitive_only": set(prompt2_registry_rows.loc[prompt2_registry_rows.evaluation_stage.str.startswith("development"), "sensitive_mode"]) == {"without_sensitive"},
        "configurations_frozen_and_shared": all(v["without_sensitive"] == v["with_sensitive"] for v in controlled_configuration_hashes.values()),
        "six_model_families_completed": set(cv_oof_summary["model_name"]) == set(required_model_families),
        "thirty_six_fold_evaluations": len(cv_fold_results) == 36 and (cv_fold_results.status == "success").all(),
        "complete_finite_oof_for_twelve_experiments": len(oof_deep_checks) == 12 and all(oof_deep_checks),
        "metrics_on_original_target_scale": all(recomputed_metric_checks),
        "registry_valid_unique_idempotent": validate_result_registry(experiment_results) and not experiment_results.experiment_id.duplicated().any()
                                               and len(prompt2_registry_rows) == len(set(prompt2_registry_rows.experiment_id)),
        "twelve_final_pipelines_saved": sum(path.exists() for path in expected_model_paths.values()) == 12,
        "twelve_pipelines_reload_and_match": len(reload_verification) == 12 and reload_verification.passed.all(),
        "coefficient_diagnostics_saved": coefficient_diagnostics_path.exists() and len(coefficient_diagnostics_frame) == 10,
        "convergence_warning_log_saved": warnings_path.exists(),
        "four_required_figures_saved": len(figure_paths) == 4 and all(path.exists() and path.stat().st_size > 0 for path in figure_paths),
        "notebook_first_full_execution_reached_final_cell": True,
        "notebook_second_execution_idempotent": snapshot_preexisted and idempotence_match,
        "no_duplicate_prompt2_sections": len(prompt2_headings) == 25 and len(set(prompt2_headings)) == 25,
        "state_markdown_files_updated": "Prompt 2" in (PROJECT_ROOT / "TASK.md").read_text(encoding="utf-8") and "Prompt 2 Execution Plan" in (PROJECT_ROOT / "PLAN.md").read_text(encoding="utf-8"),
        "independent_reviewer_completed": (P2_DIRS["reports"] / "prompt2_reviewer.md").exists(),
        "accepted_critical_and_major_findings_fixed": False,
        "scope_limited_to_required_linear_family": set(required_model_families) == {"dummy_median", "linear_regression", "ridge", "lasso", "elastic_net", "gamma_regressor"},
    }
    core_exclusions = {"notebook_second_execution_idempotent", "independent_reviewer_completed", "accepted_critical_and_major_findings_fixed"}
    core_failures = {key: value for key, value in prompt2_checks.items() if key not in core_exclusions and not value}
    if core_failures:
        raise AssertionError(f"Prompt 2 core verification failed: {core_failures}")
    if not prompt2_checks["notebook_second_execution_idempotent"]:
        prompt2_status = "PENDING_SECOND_EXECUTION_AND_REVIEW"
    else:
        prompt2_status = "INTERNAL_PASS_PENDING_INDEPENDENT_REVIEW"
    prompt2_verification = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": prompt2_status,
        "checks": prompt2_checks,
        "dataset_counts": {"training_rows": len(train_ids), "locked_test_rows": len(disk_test_ids)},
        "feature_counts": {"without_sensitive": len(linear_compact_without_sensitive), "with_sensitive": len(linear_compact_with_sensitive)},
        "development_counts": development_manifest["development_role"].value_counts().to_dict(),
        "selected_configurations": selected_linear_configurations,
        "oof_leaderboard": linear_leaderboard.to_dict(orient="records"),
        "model_artifact_paths": [str(path.relative_to(PROJECT_ROOT)) for path in expected_model_paths.values()],
        "remaining_risks": [
            "The target has a rare extreme tail, so MAE and RMSE can tell different stories.",
            "Random row folds share lenders and geography across training and validation.",
            "High-cardinality identifiers were excluded, so their predictive signal is not tested in this family.",
            "Non-sensitive fields may still proxy protected attributes.",
            "Linear-family models may miss nonlinear relationships.",
            "The locked test set has not been evaluated."
        ],
    }
    prompt2_verification_path = P2_DIRS["reports"] / "prompt2_verification.json"
    prompt2_verification_path.write_text(json.dumps(prompt2_verification, indent=2), encoding="utf-8")
    display(pd.DataFrame.from_dict(prompt2_checks, orient="index", columns=["passed"]))
    print("PROMPT 2 INTERNAL STATUS:", prompt2_status)
    """),
    md("""
    ## 40. Prompt 2 Completion Note

    The next output uses real OOF results. It states the current linear-family leader and keeps the final model decision open.
    """),
    code("""
    display(Markdown(
        f"Prompt 2 completed Dummy Median, Linear Regression, Ridge, Lasso, ElasticNet, and Gamma Regression in both sensitive modes. "
        f"Both modes used the same frozen configuration for each family. Results use complete training OOF predictions on the original target scale. "
        f"The locked test set was not used. The lowest current linear-family OOF MAE is from **{best_linear_row['model_name']}** "
        f"in **{best_linear_row['sensitive_mode']}** mode at **{best_linear_row['oof_mae']:.3f}** target units. "
        f"This is not the final project model. The next stage is tree-based and interpretable nonlinear models."
    ))
    print("Prompt 2 notebook runtime this execution (seconds):", round(time.perf_counter() - P2_START_TIME, 2))
    """),
]

new_cells += [
    md("""
    ## 31. Full Three-Fold Cross-Validation

    Each model and mode uses three fresh complete pipelines and the saved folds. Every training row receives one OOF prediction. Fits run sequentially.
    """),
    code("""
    cv_fold_results_path = P2_DIRS["results"] / "cv_fold_results.csv"
    cv_oof_summary_path = P2_DIRS["results"] / "cv_oof_summary.csv"
    warnings_path = P2_DIRS["results"] / "model_warnings.csv"
    cv_manifest_path = P2_DIRS["manifests"] / "prompt2_cv_manifest.json"
    selected_configuration_digest = configuration_digest(selected_linear_configurations)
    expected_oof_paths = {
        (family, mode): P2_DIRS["predictions"] / f"{family}__{mode}__oof.csv"
        for family in required_model_families for mode in ["without_sensitive", "with_sensitive"]
    }
    cv_cache_valid = False
    if cv_fold_results_path.exists() and cv_oof_summary_path.exists() and cv_manifest_path.exists() and all(path.exists() for path in expected_oof_paths.values()):
        cached_manifest = json.loads(cv_manifest_path.read_text(encoding="utf-8"))
        cv_fold_results = pd.read_csv(cv_fold_results_path)
        cv_oof_summary = pd.read_csv(cv_oof_summary_path)
        cv_cache_valid = (
            cached_manifest.get("selected_configuration_digest") == selected_configuration_digest
            and cached_manifest.get("development_manifest_sha256") == development_manifest_hash
            and len(cv_fold_results) == 36 and len(cv_oof_summary) == 12
            and (cv_fold_results["status"] == "success").all()
        )
        if cv_cache_valid:
            for (family, mode), path in expected_oof_paths.items():
                check_frame = pd.read_csv(path, usecols=["row_id", "fold", "y_pred"])
                cv_cache_valid = cv_cache_valid and (
                    len(check_frame) == len(train_ids)
                    and not check_frame["row_id"].duplicated().any()
                    and np.array_equal(np.sort(check_frame["row_id"].to_numpy()), train_ids)
                    and len(set(check_frame["row_id"]) & test_id_set) == 0
                    and np.isfinite(check_frame["y_pred"]).all()
                )

    if not cv_cache_valid:
        cv_start = time.perf_counter()
        fold_rows = []
        oof_summary_rows = []
        for family in required_model_families:
            configuration = selected_models[family]
            for sensitive_mode in ["without_sensitive", "with_sensitive"]:
                if sensitive_mode == "without_sensitive":
                    X_mode, numeric_mode, categorical_mode = X_without_train, numeric_without, categorical_without
                else:
                    X_mode, numeric_mode, categorical_mode = X_with_train, numeric_with, categorical_with
                oof_prediction = np.full(len(train_ids), np.nan, dtype=float)
                fold_written = np.zeros(len(train_ids), dtype=bool)
                for fold in [0, 1, 2]:
                    validation_positions = np.flatnonzero(fold_by_train_position == fold)
                    training_positions = np.flatnonzero(fold_by_train_position != fold)
                    context = {"stage": "cv", "model": family, "fold": fold, "sensitive_mode": sensitive_mode,
                               "target_mode": configuration["target_mode"]}
                    fitted, fit_seconds, retry_number, convergence_status = fit_with_retries(
                        configuration, X_mode.iloc[training_positions], y_train_prompt2[training_positions],
                        numeric_mode, categorical_mode, context
                    )
                    prediction, prediction_seconds = predict_and_measure(fitted, X_mode.iloc[validation_positions])
                    if fold_written[validation_positions].any():
                        raise AssertionError("An OOF position was written more than once.")
                    oof_prediction[validation_positions] = prediction
                    fold_written[validation_positions] = True
                    metrics = evaluate_regression_predictions(y_train_prompt2[validation_positions], prediction)
                    diagnostics = model_diagnostics(fitted)
                    fold_experiment_id = deterministic_experiment_id(
                        family, sensitive_mode, configuration["target_mode"], "cv_fold", fold, configuration
                    )
                    fold_row = {
                        "experiment_id": fold_experiment_id, "model_name": family, "sensitive_mode": sensitive_mode,
                        "target_mode": configuration["target_mode"], "fold": fold,
                        **{key: value for key, value in metrics.items() if key != "metric_warnings"},
                        **diagnostics, "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
                        "retry_number": retry_number, "convergence_status": convergence_status, "status": "success",
                        "parameter_json": canonical_json(configuration),
                    }
                    fold_rows.append(fold_row)
                    prompt2_registry_records.append(registry_record(
                        fold_experiment_id, configuration, sensitive_mode, "cv_fold", fold,
                        len(training_positions), len(validation_positions), metrics, fit_seconds, prediction_seconds,
                        notes=f"Fresh saved-fold pipeline; retry={retry_number}; convergence={convergence_status}"
                    ))
                    del fitted, prediction
                    gc.collect()
                if not fold_written.all() or not np.isfinite(oof_prediction).all():
                    raise AssertionError(f"OOF coverage failed for {family}, {sensitive_mode}.")
                oof_experiment_id = deterministic_experiment_id(
                    family, sensitive_mode, configuration["target_mode"], "oof_summary", None, configuration
                )
                oof_path = expected_oof_paths[(family, sensitive_mode)]
                oof_frame = pd.DataFrame({
                    "row_id": train_ids, "fold": fold_by_train_position, "y_true": y_train_prompt2,
                    "y_pred": oof_prediction, "absolute_error": np.abs(oof_prediction - y_train_prompt2),
                    "signed_error": oof_prediction - y_train_prompt2, "model_name": family,
                    "sensitive_mode": sensitive_mode, "target_mode": configuration["target_mode"],
                    "feature_set": PROMPT2_VERSION, "experiment_id": oof_experiment_id,
                })
                oof_frame.to_csv(oof_path, index=False)
                oof_metrics = evaluate_regression_predictions(y_train_prompt2, oof_prediction)
                family_folds = [row for row in fold_rows if row["model_name"] == family and row["sensitive_mode"] == sensitive_mode]
                summary_row = {
                    "experiment_id": oof_experiment_id, "model_name": family, "sensitive_mode": sensitive_mode,
                    "target_mode": configuration["target_mode"], "parameter_json": canonical_json(configuration),
                    **{key: value for key, value in oof_metrics.items() if key != "metric_warnings"},
                    "fold_mae_mean": float(np.mean([row["mae"] for row in family_folds])),
                    "fold_mae_std": float(np.std([row["mae"] for row in family_folds], ddof=1)),
                    "total_fit_time_seconds": float(sum(row["fit_time_seconds"] for row in family_folds)),
                    "total_prediction_time_seconds": float(sum(row["prediction_time_seconds"] for row in family_folds)),
                    "transformed_features": int(max(row["transformed_features"] for row in family_folds)),
                    "nonzero_coefficients": family_folds[-1]["nonzero_coefficients"],
                    "convergence_status": "converged" if all(row["convergence_status"] == "converged" for row in family_folds) else "warning",
                    "oof_path": str(oof_path.relative_to(PROJECT_ROOT)),
                }
                oof_summary_rows.append(summary_row)
                prompt2_registry_records.append(registry_record(
                    oof_experiment_id, configuration, sensitive_mode, "oof_summary", None,
                    len(train_ids), len(train_ids), oof_metrics, summary_row["total_fit_time_seconds"],
                    summary_row["total_prediction_time_seconds"], notes="Metrics from the complete OOF vector",
                    prediction_path=str(oof_path.relative_to(PROJECT_ROOT))
                ))
                del oof_frame, oof_prediction
                gc.collect()
        cv_fold_results = pd.DataFrame(fold_rows)
        cv_oof_summary = pd.DataFrame(oof_summary_rows)
        cv_fold_results.to_csv(cv_fold_results_path, index=False)
        cv_oof_summary.to_csv(cv_oof_summary_path, index=False)
        cv_runtime_seconds = time.perf_counter() - cv_start
        cv_manifest = {
            "version": PROMPT2_VERSION, "selected_configuration_digest": selected_configuration_digest,
            "development_manifest_sha256": development_manifest_hash, "fold_evaluation_count": len(cv_fold_results),
            "oof_file_count": len(expected_oof_paths), "fresh_pipeline_fit_count": len(cv_fold_results),
            "cv_runtime_seconds": cv_runtime_seconds,
        }
        cv_manifest_path.write_text(json.dumps(cv_manifest, indent=2), encoding="utf-8")
    else:
        cv_runtime_seconds = 0.0
        for row in cv_fold_results.itertuples():
            configuration = json.loads(row.parameter_json)
            metrics = {key: getattr(row, key) for key in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero",
                                                                "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"]}
            prompt2_registry_records.append(registry_record(
                row.experiment_id, configuration, row.sensitive_mode, "cv_fold", int(row.fold),
                int((fold_by_train_position != row.fold).sum()), int((fold_by_train_position == row.fold).sum()),
                metrics, row.fit_time_seconds, row.prediction_time_seconds, notes="Validated cached saved-fold result"
            ))
        for row in cv_oof_summary.itertuples():
            configuration = json.loads(row.parameter_json)
            metrics = {key: getattr(row, key) for key in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero",
                                                                "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"]}
            prompt2_registry_records.append(registry_record(
                row.experiment_id, configuration, row.sensitive_mode, "oof_summary", None,
                len(train_ids), len(train_ids), metrics, row.total_fit_time_seconds, row.total_prediction_time_seconds,
                notes="Validated cached complete OOF vector", prediction_path=row.oof_path
            ))
    print({"fold_evaluations": len(cv_fold_results), "oof_summaries": len(cv_oof_summary),
           "cache_reused": cv_cache_valid, "cv_runtime_seconds": round(cv_runtime_seconds, 2)})
    display(cv_fold_results.groupby(["model_name", "sensitive_mode"])["mae"].agg(["mean", "std"]).head(12))
    """),
    md("""
    All 36 saved-fold evaluations and 12 complete OOF vectors are present. Aggregate metrics are calculated from each full OOF vector, not from averaged fold metrics.
    """),
    md("""
    ## 32. OOF Evaluation Results

    The leaderboard compares every required model and both sensitive modes on original-scale training OOF predictions.
    """),
    code("""
    linear_leaderboard = cv_oof_summary.rename(columns={
        "mae": "oof_mae", "rmse": "oof_rmse", "rmsle": "oof_rmsle", "r_squared": "oof_r_squared"
    }).copy()
    linear_leaderboard = linear_leaderboard.sort_values("oof_mae").reset_index(drop=True)
    leaderboard_path = P2_DIRS["results"] / "linear_leaderboard.csv"
    linear_leaderboard.to_csv(leaderboard_path, index=False)
    best_linear_row = linear_leaderboard.iloc[0]
    display(linear_leaderboard[["model_name", "sensitive_mode", "target_mode", "oof_mae", "fold_mae_mean", "fold_mae_std",
                                "oof_rmse", "oof_rmsle", "oof_r_squared", "negative_prediction_rate", "convergence_status"]])
    display(Markdown(
        f"**Current training-OOF leader:** `{best_linear_row['model_name']}` in `{best_linear_row['sensitive_mode']}` mode "
        f"has MAE {best_linear_row['oof_mae']:.3f} target units. This is only the best current linear-family result, not the final project model."
    ))
    """),
    md("""
    ## 33. Sensitive Feature Comparison

    This paired table defines each difference as `with_sensitive - without_sensitive`. A positive MAE difference means the sensitive version was worse.
    """),
    code("""
    comparison_rows = []
    for family in required_model_families:
        without = cv_oof_summary.loc[(cv_oof_summary.model_name == family) & (cv_oof_summary.sensitive_mode == "without_sensitive")].iloc[0]
        with_sensitive = cv_oof_summary.loc[(cv_oof_summary.model_name == family) & (cv_oof_summary.sensitive_mode == "with_sensitive")].iloc[0]
        comparison_rows.append({
            "model_name": family, "mae_without_sensitive": without.mae, "mae_with_sensitive": with_sensitive.mae,
            "mae_difference_with_minus_without": with_sensitive.mae - without.mae,
            "relative_mae_difference_percent": (with_sensitive.mae - without.mae) / without.mae * 100,
            "rmse_difference_with_minus_without": with_sensitive.rmse - without.rmse,
            "rmsle_difference_with_minus_without": with_sensitive.rmsle - without.rmsle if pd.notna(with_sensitive.rmsle) and pd.notna(without.rmsle) else np.nan,
            "r_squared_difference_with_minus_without": with_sensitive.r_squared - without.r_squared,
            "runtime_difference_seconds": (with_sensitive.total_fit_time_seconds + with_sensitive.total_prediction_time_seconds)
                                          - (without.total_fit_time_seconds + without.total_prediction_time_seconds),
        })
    linear_sensitive_comparison = pd.DataFrame(comparison_rows)
    linear_sensitive_comparison.to_csv(P2_DIRS["results"] / "linear_sensitive_comparison.csv", index=False)
    display(linear_sensitive_comparison)
    display(Markdown(
        "Sensitive fields may improve, worsen, or not change predictive accuracy. This controlled accuracy comparison is not a fairness audit. "
        "The non-sensitive pack can still contain proxy variables."
    ))
    """),
    md("""
    ## 34. Coefficient and Sparsity Review

    Coefficients will be extracted from final fitted pipelines with safe transformed feature names. They describe model associations, not causes. Scaling, log targets, the Gamma log link, and correlated fields change their meaning.
    """),
    code("""
    def coefficient_frame(model, family, sensitive_mode, target_mode):
        estimator = fitted_estimator(model)
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return None
        coefficients = np.ravel(np.asarray(coefficients, dtype=float))
        names = transformed_feature_names(model)
        if len(names) != len(coefficients):
            raise AssertionError("Coefficient count does not match transformed feature names.")
        return pd.DataFrame({
            "transformed_feature": names, "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients), "model_name": family,
            "sensitive_mode": sensitive_mode, "target_mode": target_mode,
            "nonzero": coefficients != 0,
        })
    print("Coefficient extraction utility is ready. Final models are fitted in the next section.")
    """),
    md("""
    ## 35. Final Training Pipeline Fits

    One fresh complete pipeline per model and mode is fitted on all saved training rows. No test row is included. The full preprocessing and target transformation are saved inside each joblib object.
    """),
    code("""
    model_manifest_path = P2_DIRS["manifests"] / "prompt2_model_manifest.json"
    coefficient_diagnostics_path = P2_DIRS["features"] / "coefficient_diagnostics.csv"
    reload_reference_path = P2_DIRS["manifests"] / "prompt2_reload_reference.csv"
    expected_model_keys = [(family, mode) for family in required_model_families for mode in ["without_sensitive", "with_sensitive"]]
    expected_model_paths = {
        key: P2_DIRS["models"] / f"{key[0]}__{key[1]}__{selected_models[key[0]]['target_mode']}.joblib"
        for key in expected_model_keys
    }
    final_cache_valid = model_manifest_path.exists() and all(path.exists() for path in expected_model_paths.values())
    if final_cache_valid:
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        final_cache_valid = (
            model_manifest.get("selected_configuration_digest") == selected_configuration_digest
            and len(model_manifest.get("models", [])) == 12
        )

    final_models = {}
    final_fit_rows = []
    coefficient_diagnostics = []
    reload_reference_rows = []
    reload_sample_positions = np.arange(0, min(20, len(train_ids)), dtype=int)
    if not final_cache_valid:
        final_start = time.perf_counter()
        model_manifest_rows = []
        for family, sensitive_mode in expected_model_keys:
            configuration = selected_models[family]
            if sensitive_mode == "without_sensitive":
                X_mode, numeric_mode, categorical_mode = X_without_train, numeric_without, categorical_without
            else:
                X_mode, numeric_mode, categorical_mode = X_with_train, numeric_with, categorical_with
            context = {"stage": "final_train", "model": family, "fold": None, "sensitive_mode": sensitive_mode,
                       "target_mode": configuration["target_mode"]}
            fitted, fit_seconds, retry_number, convergence_status = fit_with_retries(
                configuration, X_mode, y_train_prompt2, numeric_mode, categorical_mode, context
            )
            prediction, prediction_seconds = predict_and_measure(fitted, X_mode.iloc[reload_sample_positions])
            model_path = expected_model_paths[(family, sensitive_mode)]
            joblib.dump(fitted, model_path, compress=3)
            model_sha256 = file_sha256(model_path)
            diagnostics = model_diagnostics(fitted)
            final_experiment_id = deterministic_experiment_id(
                family, sensitive_mode, configuration["target_mode"], "final_train", None, configuration
            )
            final_fit_rows.append({
                "experiment_id": final_experiment_id, "model_name": family, "sensitive_mode": sensitive_mode,
                "target_mode": configuration["target_mode"], "fit_time_seconds": fit_seconds,
                "prediction_time_seconds": prediction_seconds, **diagnostics,
                "retry_number": retry_number, "convergence_status": convergence_status,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)), "model_sha256": model_sha256,
                "parameter_json": canonical_json(configuration),
            })
            prompt2_registry_records.append(registry_record(
                final_experiment_id, configuration, sensitive_mode, "final_train", None,
                len(train_ids), 0, None, fit_seconds, prediction_seconds,
                notes=f"Full training fit; retry={retry_number}; convergence={convergence_status}",
                model_path=str(model_path.relative_to(PROJECT_ROOT))
            ))
            for row_id_value, prediction_value in zip(train_ids[reload_sample_positions], prediction):
                reload_reference_rows.append({"model_name": family, "sensitive_mode": sensitive_mode,
                                              "row_id": int(row_id_value), "reference_prediction": float(prediction_value)})
            coefficients = coefficient_frame(fitted, family, sensitive_mode, configuration["target_mode"])
            if coefficients is not None:
                coefficient_path = P2_DIRS["coefficients"] / f"{family}__{sensitive_mode}__coefficients.csv"
                coefficients.to_csv(coefficient_path, index=False)
                coefficient_diagnostics.append({
                    "model_name": family, "sensitive_mode": sensitive_mode, "target_mode": configuration["target_mode"],
                    "total_transformed_features": len(coefficients), "nonzero_coefficients": int(coefficients["nonzero"].sum()),
                    "zero_coefficients": int((~coefficients["nonzero"]).sum()),
                    "percentage_nonzero": float(coefficients["nonzero"].mean() * 100),
                    "maximum_absolute_coefficient": float(coefficients["absolute_coefficient"].max()),
                    "median_absolute_coefficient": float(coefficients["absolute_coefficient"].median()),
                    "coefficient_path": str(coefficient_path.relative_to(PROJECT_ROOT)),
                })
            model_manifest_rows.append({
                "model_name": family, "sensitive_mode": sensitive_mode, "target_mode": configuration["target_mode"],
                "configuration": configuration, "configuration_digest": configuration_digest(configuration),
                "feature_set": linear_compact_without_sensitive if sensitive_mode == "without_sensitive" else linear_compact_with_sensitive,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)), "model_sha256": model_sha256,
                "source_hashes": {name: prompt2_protected_hashes_before[name] for name in ["with_sensitive_csv", "without_sensitive_csv", "part1_notebook"]},
                "split_hashes": {name: prompt2_protected_hashes_before[name] for name in ["train_row_ids", "test_row_ids", "cv_fold_assignments", "split_config"]},
                "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            })
            final_models[(family, sensitive_mode)] = fitted
        final_fit_results = pd.DataFrame(final_fit_rows)
        coefficient_diagnostics_frame = pd.DataFrame(coefficient_diagnostics)
        reload_reference = pd.DataFrame(reload_reference_rows)
        final_fit_results.to_csv(P2_DIRS["results"] / "final_training_fit_results.csv", index=False)
        coefficient_diagnostics_frame.to_csv(coefficient_diagnostics_path, index=False)
        reload_reference.to_csv(reload_reference_path, index=False)
        model_manifest = {"version": PROMPT2_VERSION, "selected_configuration_digest": selected_configuration_digest,
                          "training_row_count": len(train_ids), "models": model_manifest_rows}
        model_manifest_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
        final_runtime_seconds = time.perf_counter() - final_start
    else:
        final_fit_results = pd.read_csv(P2_DIRS["results"] / "final_training_fit_results.csv")
        coefficient_diagnostics_frame = pd.read_csv(coefficient_diagnostics_path)
        reload_reference = pd.read_csv(reload_reference_path)
        final_runtime_seconds = 0.0
        for row in final_fit_results.itertuples():
            configuration = json.loads(row.parameter_json)
            prompt2_registry_records.append(registry_record(
                row.experiment_id, configuration, row.sensitive_mode, "final_train", None,
                len(train_ids), 0, None, row.fit_time_seconds, row.prediction_time_seconds,
                notes="Validated cached full training fit", model_path=row.model_path
            ))
    print({"saved_pipeline_count": len(expected_model_paths), "cache_reused": final_cache_valid,
           "final_fit_runtime_seconds": round(final_runtime_seconds, 2)})
    display(coefficient_diagnostics_frame)
    """),
    md("""
    Twelve complete pipelines and coefficient diagnostics are saved. Coefficients can be unstable when inputs are correlated, and they must not be read as causal effects.
    """),
]

new_cells += [
    md("""
    ## 21. Pipeline and Experiment Utilities

    These helpers create fresh pipelines, capture warnings, evaluate original-scale predictions, and upsert deterministic Prompt 2 registry rows.
    """),
    code("""
    warning_records = []

    def model_diagnostics(model):
        names = transformed_feature_names(model)
        estimator = fitted_estimator(model)
        coefficients = getattr(estimator, "coef_", None)
        nonzero = None if coefficients is None else int(np.count_nonzero(np.asarray(coefficients)))
        return {"transformed_features": len(names), "nonzero_coefficients": nonzero}

    def fit_with_retries(configuration, X_fit, y_fit, numeric_features, categorical_features, context):
        model_name = configuration["model_name"]
        if model_name in {"lasso", "elastic_net"}:
            attempts = [{}, {"max_iter": 10000}, {"max_iter": 10000, "tol": 1e-3}]
        elif model_name == "gamma_regressor":
            attempts = [{}, {"max_iter": 1000}]
        else:
            attempts = [{}]
        last_model = None
        total_fit_time = 0.0
        for retry_number, overrides in enumerate(attempts):
            candidate = make_complete_pipeline(configuration, numeric_features, categorical_features, fit_overrides=overrides)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate.fit(X_fit, y_fit)
            elapsed = time.perf_counter() - start
            total_fit_time += elapsed
            convergence_warning = False
            for item in caught:
                warning_records.append({
                    **context, "warning_class": item.category.__name__, "warning_message": str(item.message),
                    "retry_number": retry_number, "fit_overrides": canonical_json(overrides),
                })
                convergence_warning = convergence_warning or issubclass(item.category, ConvergenceWarning)
            last_model = candidate
            if not convergence_warning:
                return candidate, total_fit_time, retry_number, "converged"
            del candidate
            gc.collect()
        raise RuntimeError(f"{model_name} did not converge after {len(attempts)} meaningful attempts: {context}")

    def predict_and_measure(model, X_predict):
        start = time.perf_counter()
        prediction = np.asarray(model.predict(X_predict), dtype=float)
        elapsed = time.perf_counter() - start
        if not np.isfinite(prediction).all():
            raise ValueError("A model produced non-finite predictions.")
        return prediction, elapsed

    def registry_record(experiment_id, configuration, sensitive_mode, stage, fold_number,
                        train_count, validation_count, metrics, fit_time, prediction_time,
                        status="success", notes="", model_path="", prediction_path=""):
        values = metrics or {}
        return {
            "experiment_id": experiment_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_family": "linear_family", "model_name": configuration["model_name"],
            "sensitive_mode": sensitive_mode, "feature_set": PROMPT2_VERSION,
            "target_mode": configuration["target_mode"], "evaluation_stage": stage,
            "fold_number": fold_number, "training_row_count": train_count,
            "validation_row_count": validation_count, "test_row_count": 0,
            "parameter_json": canonical_json(configuration),
            "mae": values.get("mae"), "mse": values.get("mse"), "rmse": values.get("rmse"),
            "mape_percent": values.get("mape_percent"), "r_squared": values.get("r_squared"),
            "rmsle": values.get("rmsle"), "rmsle_clipped_zero": values.get("rmsle_clipped_zero"),
            "median_absolute_error": values.get("median_absolute_error"), "wape_percent": values.get("wape_percent"),
            "mean_signed_error": values.get("mean_signed_error"), "p90_absolute_error": values.get("p90_absolute_error"),
            "negative_prediction_rate": values.get("negative_prediction_rate"),
            "fit_time_seconds": fit_time, "prediction_time_seconds": prediction_time,
            "status": status, "notes": notes, "model_artifact_path": model_path,
            "prediction_artifact_path": prediction_path,
        }

    prompt2_registry_records = []

    def upsert_prompt2_registry(base_frame, records):
        incoming = pd.DataFrame(records, columns=REGISTRY_COLUMNS)
        if incoming["experiment_id"].duplicated().any():
            raise ValueError("Incoming Prompt 2 experiment IDs are not unique.")
        existing = base_frame.copy()
        old_timestamps = existing.set_index("experiment_id")["timestamp_utc"].to_dict() if not existing.empty else {}
        incoming["timestamp_utc"] = [old_timestamps.get(row.experiment_id, row.timestamp_utc) for row in incoming.itertuples()]
        kept = existing.loc[~existing["experiment_id"].isin(incoming["experiment_id"])].copy()
        combined = pd.concat([kept, incoming], ignore_index=True)
        validate_result_registry(combined)
        save_result_registry(combined)
        reloaded = pd.read_csv(registry_path)
        validate_result_registry(reloaded)
        if reloaded["experiment_id"].duplicated().any():
            raise AssertionError("Registry upsert created duplicate IDs.")
        return reloaded

    id_test_config = {"model_name": "ridge", "target_mode": "raw", "scaler": "standard", "alpha": 1.0}
    id_a = deterministic_experiment_id("ridge", "without_sensitive", "raw", "unit", None, id_test_config)
    id_b = deterministic_experiment_id("ridge", "without_sensitive", "raw", "unit", None, dict(reversed(list(id_test_config.items()))))
    id_c = deterministic_experiment_id("ridge", "without_sensitive", "raw", "unit", None, {**id_test_config, "alpha": 10.0})
    assert id_a == id_b and id_a != id_c
    print("Pipeline, warning, metric, deterministic ID, and registry-upsert utilities are ready.")
    """),
    md("""
    ## 22. Development Screening Design

    Fold 0 supplies validation rows. Folds 1 and 2 supply training rows. Target-stratified caps create one fixed sample for every candidate.
    """),
    code("""
    def stratified_cap(source_positions, cap, seed):
        source_positions = np.asarray(source_positions, dtype=int)
        values = y_train_prompt2[source_positions]
        bins = pd.qcut(values, q=10, labels=False, duplicates="drop").astype(int)
        if len(source_positions) <= cap:
            return np.sort(source_positions), bins
        selected, _ = train_test_split(
            source_positions, train_size=cap, random_state=seed, stratify=bins
        )
        selected = np.sort(selected)
        selected_bins = pd.qcut(y_train_prompt2[selected], q=10, labels=False, duplicates="drop").astype(int)
        return selected, selected_bins

    development_train_source = np.flatnonzero(np.isin(fold_by_train_position, [1, 2]))
    development_validation_source = np.flatnonzero(fold_by_train_position == 0)
    dev_train_positions, dev_train_bins = stratified_cap(development_train_source, 80000, CONFIG["random_state"])
    dev_validation_positions, dev_validation_bins = stratified_cap(development_validation_source, 20000, CONFIG["random_state"])
    development_manifest = pd.concat([
        pd.DataFrame({"row_id": train_ids[dev_train_positions], "development_role": "train",
                      "original_cv_fold": fold_by_train_position[dev_train_positions], "target_bin": dev_train_bins}),
        pd.DataFrame({"row_id": train_ids[dev_validation_positions], "development_role": "validation",
                      "original_cv_fold": fold_by_train_position[dev_validation_positions], "target_bin": dev_validation_bins}),
    ], ignore_index=True).sort_values(["development_role", "row_id"]).reset_index(drop=True)
    development_manifest_path = ARTIFACT_DIRS["splits"] / "prompt2_development_sample.csv"
    development_manifest.to_csv(development_manifest_path, index=False)
    development_manifest_hash = file_sha256(development_manifest_path)
    development_checks = {
        "rows_unique": not development_manifest["row_id"].duplicated().any(),
        "no_test_rows": len(set(development_manifest["row_id"]) & test_id_set) == 0,
        "training_cap": int((development_manifest["development_role"] == "train").sum()) == 80000,
        "validation_cap": int((development_manifest["development_role"] == "validation").sum()) == 20000,
        "roles_follow_saved_folds": bool((development_manifest.loc[development_manifest.development_role == "validation", "original_cv_fold"] == 0).all()
                                        and development_manifest.loc[development_manifest.development_role == "train", "original_cv_fold"].isin([1, 2]).all()),
    }
    if not all(development_checks.values()):
        raise AssertionError(f"Development manifest failed: {development_checks}")
    X_dev_train = X_without_train.iloc[dev_train_positions]
    X_dev_validation = X_without_train.iloc[dev_validation_positions]
    y_dev_train = y_train_prompt2[dev_train_positions]
    y_dev_validation = y_train_prompt2[dev_validation_positions]
    print({"development_train": len(dev_train_positions), "development_validation": len(dev_validation_positions),
           "manifest_sha256": development_manifest_hash})
    """),
    md("""
    The development design uses 80,000 training rows and 20,000 validation rows. All rows come from the saved training set, and the same manifest is used for every candidate.
    """),
    md("""
    ## 23. Scaler Selection

    A Ridge baseline compares Standard and Robust scaling with raw and log targets. The selected scaler is then fixed for every model family.
    """),
    code("""
    screening_results_path = P2_DIRS["results"] / "development_screening_results.csv"
    selected_configurations_path = P2_DIRS["results"] / "selected_linear_configurations.json"

    def candidate_key(configuration, group):
        return f"{group}__{configuration_digest(configuration)}"

    scaler_candidates = [
        {"model_name": "ridge", "target_mode": target_mode, "scaler": scaler, "alpha": 1.0}
        for scaler in ["standard", "robust"] for target_mode in ["raw", "log1p"]
    ]

    def model_candidate_grid(selected_scaler):
        candidates = {
            "dummy_median": [{"model_name": "dummy_median", "target_mode": "raw", "scaler": selected_scaler}],
            "linear_regression": [{"model_name": "linear_regression", "target_mode": mode, "scaler": selected_scaler} for mode in ["raw", "log1p"]],
            "ridge": [{"model_name": "ridge", "target_mode": mode, "scaler": selected_scaler, "alpha": alpha}
                      for mode in ["raw", "log1p"] for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]],
            "lasso": ([{"model_name": "lasso", "target_mode": "raw", "scaler": selected_scaler, "alpha": alpha, "max_iter": 5000, "tol": 1e-4}
                       for alpha in [0.01, 0.1, 1.0, 10.0]]
                      + [{"model_name": "lasso", "target_mode": "log1p", "scaler": selected_scaler, "alpha": alpha, "max_iter": 5000, "tol": 1e-4}
                         for alpha in [0.00001, 0.0001, 0.001, 0.01]]),
            "elastic_net": ([{"model_name": "elastic_net", "target_mode": "raw", "scaler": selected_scaler, "alpha": alpha, "l1_ratio": ratio, "max_iter": 5000, "tol": 1e-4}
                             for alpha in [0.01, 0.1, 1.0] for ratio in [0.2, 0.5, 0.8]]
                            + [{"model_name": "elastic_net", "target_mode": "log1p", "scaler": selected_scaler, "alpha": alpha, "l1_ratio": ratio, "max_iter": 5000, "tol": 1e-4}
                               for alpha in [0.00001, 0.0001, 0.001] for ratio in [0.2, 0.5, 0.8]]),
            "gamma_regressor": [{"model_name": "gamma_regressor", "target_mode": "raw", "scaler": selected_scaler,
                                  "alpha": alpha, "max_iter": 500, "tol": 1e-5} for alpha in [0.0, 0.01, 0.1, 1.0]],
        }
        return candidates

    screening_cache_valid = False
    if screening_results_path.exists() and selected_configurations_path.exists():
        development_screening_results = pd.read_csv(screening_results_path)
        selected_linear_configurations = json.loads(selected_configurations_path.read_text(encoding="utf-8"))
        screening_cache_valid = (
            selected_linear_configurations.get("development_manifest_sha256") == development_manifest_hash
            and selected_linear_configurations.get("version") == PROMPT2_VERSION
            and set(selected_linear_configurations.get("models", {})) == {"dummy_median", "linear_regression", "ridge", "lasso", "elastic_net", "gamma_regressor"}
            and len(development_screening_results) == 47
            and development_screening_results["status"].isin(["success", "failed"]).all()
        )

    if not screening_cache_valid:
        screening_rows = []
        screening_start = time.perf_counter()

        def run_screen_candidate(configuration, group):
            context = {"stage": "development", "model": configuration["model_name"], "fold": 0,
                       "sensitive_mode": "without_sensitive", "target_mode": configuration["target_mode"]}
            try:
                fitted, fit_seconds, retry_number, convergence_status = fit_with_retries(
                    configuration, X_dev_train, y_dev_train, numeric_without, categorical_without, context
                )
            except RuntimeError as error:
                screening_rows.append({
                    "screening_group": group, "candidate_id": candidate_key(configuration, group),
                    "model_name": configuration["model_name"], "target_mode": configuration["target_mode"],
                    "scaler": configuration["scaler"], "alpha": configuration.get("alpha"),
                    "l1_ratio": configuration.get("l1_ratio"), "parameter_json": canonical_json(configuration),
                    **{key: None for key in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero",
                                               "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error",
                                               "negative_prediction_rate", "mae_usd", "rmse_usd"]},
                    "transformed_features": None, "nonzero_coefficients": None,
                    "fit_time_seconds": None, "prediction_time_seconds": None,
                    "retry_number": 2, "convergence_status": "failed_after_retries", "status": "failed",
                })
                experiment_id = deterministic_experiment_id(
                    configuration["model_name"], "without_sensitive", configuration["target_mode"],
                    f"development_{group}", 0, configuration
                )
                prompt2_registry_records.append(registry_record(
                    experiment_id, configuration, "without_sensitive", f"development_{group}", 0,
                    len(y_dev_train), len(y_dev_validation), None, None, None,
                    status="failed", notes=str(error)
                ))
                gc.collect()
                return
            prediction, prediction_seconds = predict_and_measure(fitted, X_dev_validation)
            metrics = evaluate_regression_predictions(y_dev_validation, prediction)
            diagnostics = model_diagnostics(fitted)
            row = {
                "screening_group": group, "candidate_id": candidate_key(configuration, group),
                "model_name": configuration["model_name"], "target_mode": configuration["target_mode"],
                "scaler": configuration["scaler"], "alpha": configuration.get("alpha"),
                "l1_ratio": configuration.get("l1_ratio"), "parameter_json": canonical_json(configuration),
                **{key: value for key, value in metrics.items() if key != "metric_warnings"},
                **diagnostics, "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
                "retry_number": retry_number, "convergence_status": convergence_status, "status": "success",
            }
            screening_rows.append(row)
            experiment_id = deterministic_experiment_id(
                configuration["model_name"], "without_sensitive", configuration["target_mode"],
                f"development_{group}", 0, configuration
            )
            prompt2_registry_records.append(registry_record(
                experiment_id, configuration, "without_sensitive", f"development_{group}", 0,
                len(y_dev_train), len(y_dev_validation), metrics, fit_seconds, prediction_seconds,
                notes=f"Development screening; retry={retry_number}; convergence={convergence_status}"
            ))
            del fitted, prediction
            gc.collect()

        for configuration in scaler_candidates:
            run_screen_candidate(configuration, "scaler")
        scaler_frame = pd.DataFrame(screening_rows)
        best_scaler_mae = scaler_frame["mae"].min()
        equivalent_scalers = scaler_frame.loc[scaler_frame["mae"] < best_scaler_mae * 1.005].copy()
        equivalent_scalers["standard_preference"] = (equivalent_scalers["scaler"] == "standard").astype(int)
        selected_scaler = equivalent_scalers.sort_values(
            ["standard_preference", "fit_time_seconds", "candidate_id"], ascending=[False, True, True]
        ).iloc[0]["scaler"]

        candidate_grid = model_candidate_grid(selected_scaler)
        for family, candidates in candidate_grid.items():
            for configuration in candidates:
                run_screen_candidate(configuration, family)

        development_screening_results = pd.DataFrame(screening_rows)

        def select_model_configuration(family):
            frame = development_screening_results.loc[
                (development_screening_results["screening_group"] == family)
                & (development_screening_results["status"] == "success")
            ].copy()
            if frame.empty:
                raise RuntimeError(f"No successful development configuration remains for required family {family}.")
            best = frame["mae"].min()
            eligible = frame.loc[frame["mae"] < best * 1.0025].copy()
            eligible["raw_preference"] = (eligible["target_mode"] == "raw").astype(int)
            eligible["alpha_rank"] = eligible["alpha"].fillna(-np.inf)
            eligible["nonzero_rank"] = eligible["nonzero_coefficients"].fillna(np.inf)
            eligible["runtime"] = eligible["fit_time_seconds"] + eligible["prediction_time_seconds"]
            selected = eligible.sort_values(
                ["raw_preference", "alpha_rank", "nonzero_rank", "runtime", "candidate_id"],
                ascending=[False, False, True, True, True]
            ).iloc[0]
            return json.loads(selected["parameter_json"]), selected["candidate_id"], float(best), len(eligible)

        selected_models = {}
        selection_evidence = {}
        for family in ["dummy_median", "linear_regression", "ridge", "lasso", "elastic_net", "gamma_regressor"]:
            configuration, selected_candidate_id, best_mae, equivalent_count = select_model_configuration(family)
            selected_models[family] = configuration
            selection_evidence[family] = {"selected_candidate_id": selected_candidate_id, "best_development_mae": best_mae,
                                          "equivalent_candidate_count": equivalent_count,
                                          "tie_rule": "raw, stronger alpha, fewer nonzero, runtime, stable ID"}
        selected_linear_configurations = {
            "version": PROMPT2_VERSION, "selected_scaler": selected_scaler,
            "development_manifest_sha256": development_manifest_hash,
            "selection_primary_metric": "mae", "models": selected_models,
            "selection_evidence": selection_evidence,
        }
        development_screening_results.to_csv(screening_results_path, index=False)
        selected_configurations_path.write_text(json.dumps(selected_linear_configurations, indent=2), encoding="utf-8")
        screening_runtime_seconds = time.perf_counter() - screening_start
    else:
        selected_scaler = selected_linear_configurations["selected_scaler"]
        screening_runtime_seconds = 0.0
        for row in development_screening_results.itertuples():
            configuration = json.loads(row.parameter_json)
            experiment_id = deterministic_experiment_id(
                configuration["model_name"], "without_sensitive", configuration["target_mode"],
                f"development_{row.screening_group}", 0, configuration
            )
            metrics = {key: getattr(row, key) for key in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "rmsle_clipped_zero",
                                                                "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"]}
            prompt2_registry_records.append(registry_record(
                experiment_id, configuration, "without_sensitive", f"development_{row.screening_group}", 0,
                len(y_dev_train), len(y_dev_validation), metrics, row.fit_time_seconds, row.prediction_time_seconds,
                status=row.status, notes=f"Validated cached development screening; convergence={row.convergence_status}"
            ))
    print({"screening_candidates": len(development_screening_results), "selected_scaler": selected_scaler,
           "cache_reused": screening_cache_valid, "screening_runtime_seconds": round(screening_runtime_seconds, 2)})
    display(development_screening_results.loc[development_screening_results.screening_group == "scaler",
            ["scaler", "target_mode", "mae", "rmse", "rmsle", "fit_time_seconds"]].sort_values("mae"))
    """),
    md("""
    Scaler choice uses development MAE only. If candidates are within 0.5 percent, Standard scaling has the deterministic simplicity preference. The printed result shows the real selected policy.
    """),
]

model_sections = [
    (24, "Dummy Regressor", "dummy_median", "The median dummy gives a simple reference. It uses raw target values and the full preprocessing policy."),
    (25, "Linear Regression", "linear_regression", "Linear Regression estimates one additive coefficient for each transformed feature. Raw and log targets are compared."),
    (26, "Ridge Regression", "ridge", "Ridge shrinks coefficients with an L2 penalty. It can be more stable when features are correlated."),
    (27, "Lasso Regression", "lasso", "Lasso uses an L1 penalty and can set coefficients to zero. Convergence warnings are captured and retried."),
    (28, "ElasticNet Regression", "elastic_net", "ElasticNet mixes L1 and L2 penalties. It can combine shrinkage with sparse coefficients."),
    (29, "Gamma Regression", "gamma_regressor", "Gamma regression models a positive mean with a log link. It uses the raw positive target and tests regularization strength."),
]
for section_number, title, family, explanation in model_sections:
    new_cells.append(md(f"""
    ## {section_number}. {title}

    {explanation}
    """))
    new_cells.append(code(f"""
    family_results = development_screening_results.loc[
        development_screening_results["screening_group"] == "{family}",
        ["target_mode", "alpha", "l1_ratio", "mae", "rmse", "rmsle", "negative_prediction_rate",
         "nonzero_coefficients", "fit_time_seconds", "retry_number", "convergence_status", "candidate_id"]
    ].sort_values("mae")
    selected_family_config = selected_linear_configurations["models"]["{family}"]
    selected_id = selected_linear_configurations["selection_evidence"]["{family}"]["selected_candidate_id"]
    selected_row = family_results.loc[family_results["candidate_id"] == selected_id].iloc[0]
    display(family_results.head(8))
    display(Markdown(
        f"**Selected result:** `{family}` uses `{{selected_family_config['target_mode']}}` target with "
        f"development MAE {{selected_row['mae']:.3f}} target units. Parameters: `{{canonical_json(selected_family_config)}}`. "
        f"Convergence status: {{selected_row['convergence_status']}}."
    ))
    """))

new_cells += [
    md("""
    ## 30. Frozen Model Configurations

    Each family now has one frozen configuration selected from non-sensitive development data. The exact same configuration will be used for both sensitive modes.
    """),
    code("""
    selected_models = selected_linear_configurations["models"]
    required_model_families = ["dummy_median", "linear_regression", "ridge", "lasso", "elastic_net", "gamma_regressor"]
    if list(selected_models) != required_model_families:
        raise AssertionError("Selected model family order or membership is invalid.")
    controlled_configuration_hashes = {
        family: {
            "without_sensitive": configuration_digest(selected_models[family]),
            "with_sensitive": configuration_digest(selected_models[family]),
        } for family in required_model_families
    }
    if not all(v["without_sensitive"] == v["with_sensitive"] for v in controlled_configuration_hashes.values()):
        raise AssertionError("Sensitive modes do not share frozen configurations.")
    frozen_config_table = pd.DataFrame([
        {"model": family, "target_mode": config["target_mode"], "scaler": config["scaler"],
         "alpha": config.get("alpha"), "l1_ratio": config.get("l1_ratio"),
         "configuration_hash": configuration_digest(config)}
        for family, config in selected_models.items()
    ])
    display(frozen_config_table)
    """),
    md("""
    The six configurations are frozen. Sensitive features cannot influence target-mode, scaler, alpha, or L1-ratio selection.
    """),
]

# Independent patches above build section blocks. Sort them by their stable
# numbered headings before writing the notebook.
section_blocks = []
current_block = []
current_number = None
for cell in new_cells:
    heading_number = None
    if cell.cell_type == "markdown":
        first_line = cell.source.splitlines()[0] if cell.source.splitlines() else ""
        if first_line.startswith("## ") and "." in first_line:
            try:
                heading_number = int(first_line.split()[1].rstrip("."))
            except ValueError:
                heading_number = None
    if heading_number is not None:
        if current_block:
            section_blocks.append((current_number, current_block))
        current_number = heading_number
        current_block = [cell]
    else:
        current_block.append(cell)
if current_block:
    section_blocks.append((current_number, current_block))
if {number for number, _ in section_blocks} != set(range(16, 41)):
    raise RuntimeError(f"Prompt 2 section markers are incomplete: {[number for number, _ in section_blocks]}")
ordered_prompt2_cells = [cell for _, block in sorted(section_blocks, key=lambda item: item[0]) for cell in block]
notebook.cells.extend(ordered_prompt2_cells)
notebook.metadata["prompt2_section_version"] = "linear_compact_v1"
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Updated {NOTEBOOK_PATH} with {len(ordered_prompt2_cells)} Prompt 2 cells.")
