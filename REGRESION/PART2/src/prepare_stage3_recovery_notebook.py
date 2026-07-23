"""Patch known Stage 3 cells to consume validated recovery caches."""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART3_TREE_MODELS.ipynb"


SECTION_17 = r'''
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
    "hist_gradient_boosting": choose_near_best(hist_screening, "hist_gradient_boosting"),
}
selected_path = DIRS["results"] / "selected_tree_configurations.json"
selected_configurations = json.loads(selected_path.read_text(encoding="utf-8"))
repair = selected_configurations.get("bagging_runtime_repair", {})
assert repair.get("status") == "completed" and repair.get("pilot_status") == "PASS"
assert selected_configurations["models"]["decision_tree"] == json.loads(selected_rows["decision_tree"]["configuration_json"])
assert selected_configurations["models"]["hist_gradient_boosting"] == json.loads(selected_rows["hist_gradient_boosting"]["configuration_json"])
assert repair["new_configuration"] == selected_configurations["models"]["bagging"]
assert repair["selection_data"] == "saved non-sensitive development sample only"
assert repair["same_configuration_both_sensitive_modes"] is True

bagging_runtime_repair_screening = pd.read_csv(DIRS["results"] / "bagging_runtime_repair_screening.csv")
bagging_full_fold_pilot = pd.read_csv(DIRS["results"] / "bagging_full_fold_pilot.csv")
assert len(bagging_runtime_repair_screening) <= 8 and bagging_runtime_repair_screening["eligible"].all()
assert bagging_full_fold_pilot["pilot_eligible"].astype(bool).any()
selected_display = pd.DataFrame([{"role": role, **config} for role, config in selected_configurations["models"].items()])
display(bagging_runtime_repair_screening.sort_values(["mae", "fit_time_seconds"])[
    ["candidate_number", "model_name", "mae", "rmse", "fit_time_seconds", "model_size_bytes", "eligible"]
])
display(bagging_full_fold_pilot[["model_name", "development_mae", "mae", "fit_time_seconds", "pilot_eligible"]])
display(selected_display)
print(f"The repaired bagging model is {selected_configurations['models']['bagging']['model_name']}. "
      f"Its full-Fold pilot fit took {repair['full_fold_pilot_fit_seconds']:.1f} seconds. "
      "The same frozen configuration is used in both sensitive modes.")
'''


SECTION_18 = r'''
from stage3_recovery_utils import (
    experiment_digest as recovery_experiment_digest,
    load_all_targets as recovery_load_all_targets,
    validate_fold_checkpoint as recovery_validate_fold_checkpoint,
)

cv_fold_results_path = DIRS["results"] / "cv_fold_results.csv"
cv_oof_summary_path = DIRS["results"] / "cv_oof_summary.csv"
cv_manifest_path = DIRS["manifests"] / "stage3_cv_manifest.json"
selected_digest = hashlib.sha256(canonical_json(selected_configurations).encode("utf-8")).hexdigest()
active_configurations = list(selected_configurations["models"].values())
cv_manifest = json.loads(cv_manifest_path.read_text(encoding="utf-8"))
cv_fold_results = pd.read_csv(cv_fold_results_path)
cv_oof_summary = pd.read_csv(cv_oof_summary_path)
assert cv_manifest.get("fold_fits") == 18 and cv_manifest.get("oof_experiments") == 6
assert len(cv_fold_results) == 18 and cv_fold_results["status"].eq("success").all()
assert len(cv_oof_summary) == 6 and cv_oof_summary["status"].eq("success").all()

y_checkpoint = recovery_load_all_targets()
for configuration in active_configurations:
    for sensitive_mode in ["without_sensitive", "with_sensitive"]:
        key = f"{configuration['model_name']}__{sensitive_mode}"
        assert cv_manifest["per_experiment_digests"][key] == recovery_experiment_digest(
            configuration, sensitive_mode, None, "oof_summary"
        )
        for fold in [0, 1, 2]:
            valid, reason, _, _ = recovery_validate_fold_checkpoint(
                configuration, sensitive_mode, fold, y_all=y_checkpoint
            )
            if not valid:
                raise AssertionError(f"Invalid Fold checkpoint {key} Fold {fold}: {reason}")

for row in cv_oof_summary.itertuples(index=False):
    frame = pd.read_csv(ROOT / row.oof_path, usecols=["row_id", "fold", "y_true", "y_pred"])
    assert len(frame) == len(train_ids) and frame["row_id"].is_unique
    assert set(frame["row_id"]) == set(train_ids) and not set(frame["row_id"]).intersection(test_ids)
    assert set(frame["fold"]) == {0, 1, 2}
    assert np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all()

print("Validated and reused 18 atomic Fold checkpoints and six complete OOF files.")
display(cv_oof_summary.sort_values("mae")[[
    "model_name", "sensitive_mode", "mae", "rmse", "rmsle", "r_squared", "total_fit_time_seconds"
]])
'''


SECTION_24 = r'''
importance_summary_path = DIRS["importance"] / "stage3_feature_importance_summary.csv"
importance_expected = [
    DIRS["importance"] / f"{config['model_name']}__{mode}__permutation.csv"
    for config in active_configurations for mode in ["without_sensitive", "with_sensitive"]
]
assert all(path.exists() and path.stat().st_size > 0 for path in importance_expected)
importance_frames = []
engineered_sources = set(ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES +
                         ["respondent_id", "msamd_name", "county_name", "census_tract_number"])
for path in importance_expected:
    frame = pd.read_csv(path)
    assert frame["sample_rows"].max() <= 10000 and set(frame["n_repeats"]) == {3}
    assert frame["associative_not_causal"].astype(bool).all()
    frame["is_engineered"] = frame["source_feature"].isin(engineered_sources)
    frame["is_sensitive"] = frame["source_feature"].isin(SENSITIVE_FEATURES)
    importance_frames.append(frame)
feature_importance_summary = pd.concat(importance_frames, ignore_index=True)
tree_structure_summary = pd.read_csv(DIRS["importance"] / "stage3_tree_structure.csv")
save_csv(feature_importance_summary, importance_summary_path)
top_importance = feature_importance_summary.sort_values("importance", ascending=False).groupby(
    ["model_name", "sensitive_mode", "method"], as_index=False
).head(15)
display(top_importance[["model_name", "sensitive_mode", "method", "source_feature", "importance", "importance_std", "is_engineered", "is_sensitive"]])
print("Permutation importance uses at most 10,000 training rows and three repeats. It is associative, not causal.")
'''


SECTION_25 = r'''
from stage3_recovery_utils import experiment_digest as recovery_experiment_digest

model_manifest_path = DIRS["manifests"] / "stage3_model_manifest.json"
final_fit_path = DIRS["results"] / "final_training_fit_results.csv"
model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
final_fit_results = pd.read_csv(final_fit_path)
assert model_manifest.get("selected_configuration_digest") == selected_digest
assert len(model_manifest.get("models", [])) == 6 and len(final_fit_results) == 6
for model_row in model_manifest["models"]:
    model_path = ROOT / model_row["model_path"]
    sample_path = ROOT / model_row["reload_sample_path"]
    configuration = selected_configurations["models"][
        "bagging" if model_row["model_name"] in {"random_forest", "extra_trees"} else model_row["model_name"]
    ]
    assert model_row["configuration_digest"] == recovery_experiment_digest(
        configuration, model_row["sensitive_mode"], None, "final_training_fit"
    )
    assert model_path.exists() and sha256_file(model_path) == model_row["model_sha256"]
    assert sample_path.exists() and model_row["reload_prediction_match"] is True

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
        **{name: values.get(name, np.nan) for name in metric_columns() if name not in {"mae_usd", "rmse_usd"}},
        "fit_time_seconds": values.get("fit_time_seconds", values.get("total_fit_time_seconds", np.nan)),
        "prediction_time_seconds": values.get("prediction_time_seconds", values.get("total_prediction_time_seconds", np.nan)),
        "status": values.get("status", "success"), "notes": "Stage 3; Test Set locked; deterministic upsert",
        "model_artifact_path": model_path, "prediction_artifact_path": prediction_path,
    }

registry_rows = []
for row in feature_engineering_screening.to_dict("records"):
    registry_rows.append(registry_record(row, "feature_screening", len(dev_train_ids), len(dev_validation_ids)))
registry_screening = model_screening_results.copy()
old_config_json = canonical_json(selected_configurations["bagging_runtime_repair"]["old_configuration"])
registry_screening.loc[registry_screening["configuration_json"].eq(old_config_json), "status"] = "superseded_for_runtime_feasibility"
for row in registry_screening.to_dict("records"):
    registry_rows.append(registry_record(row, "model_screening", len(dev_train_ids), len(dev_validation_ids)))
for row in bagging_runtime_repair_screening.to_dict("records"):
    configuration = json.loads(row["configuration_json"])
    row["experiment_id"] = deterministic_experiment_id(
        row["model_name"], "without_sensitive", row["target_mode"], "runtime_repair_screening", 0,
        configuration, row["feature_pack"])
    row["status"] = "success" if bool(row["eligible"]) else row.get("status", "failed")
    registry_rows.append(registry_record(row, "runtime_repair_screening", row["training_rows"], row["validation_rows"]))
for row in bagging_full_fold_pilot.to_dict("records"):
    registry_rows.append(registry_record(row, "full_fold_pilot", row["training_rows"], row["validation_rows"],
                                         model_path=row["pilot_model_path"]))
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
       "prior_registry_rows_unchanged": True, "runtime_repair_rows": len(bagging_runtime_repair_screening)})
'''


def patch_notebook(path: Path = NOTEBOOK) -> None:
    notebook = nbf.read(path, as_version=4)
    replacements = {17: SECTION_17, 18: SECTION_18, 24: SECTION_24, 25: SECTION_25}
    seen = set()
    for index, cell in enumerate(notebook.cells[:-1]):
        if cell.cell_type != "markdown":
            continue
        for section, source in replacements.items():
            if cell.source.lstrip().startswith(f"## {section}."):
                target = notebook.cells[index + 1]
                if target.cell_type != "code":
                    raise AssertionError(f"Section {section} is not followed by a code cell.")
                target.source = source.strip()
                target.outputs = []
                target.execution_count = None
                seen.add(section)
    if seen != set(replacements):
        raise AssertionError(f"Missing known Stage 3 sections: {sorted(set(replacements) - seen)}")
    for cell in notebook.cells:
        if cell.cell_type == "code" and "preferred_method = \"impurity\" if model_name != \"decision_tree\" else \"impurity\"" in cell.source:
            cell.source = cell.source.replace(
                'preferred_method = "impurity" if model_name != "decision_tree" else "impurity"',
                'preferred_method = "permutation"',
            )
        elif cell.cell_type == "code" and 'preferred_method = "permutation" if model_name != "decision_tree" else "impurity"' in cell.source:
            cell.source = cell.source.replace(
                'preferred_method = "permutation" if model_name != "decision_tree" else "impurity"',
                'preferred_method = "permutation"',
            )
    temporary = path.with_suffix(path.suffix + ".tmp")
    nbf.write(notebook, temporary)
    temporary.replace(path)


if __name__ == "__main__":
    patch_notebook()
    print(f"Patched {NOTEBOOK.name} to reuse validated Stage 3 recovery caches.")
