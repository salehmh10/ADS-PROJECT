"""Append cache-only Stage 4D-E sections to the Stage 4 CatBoost notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd

import stage4_boosting_utils as s4
import stage4de_catboost_utils as de


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_CATBOOST.ipynb"
PREFIX_CELLS = 76


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def prepare_reports() -> None:
    results = ROOT / "artifacts/results/stage4/catboost/final"
    reports = ROOT / "artifacts/reports"
    confirmation = pd.read_csv(ROOT / "artifacts/results/stage4/catboost/feature_confirmation/catboost_feature_confirmation_results.csv")
    tuning = pd.read_csv(results / "catboost_final_tuning.csv")
    controlled = pd.read_csv(results / "catboost_final_validation_results.csv")
    registry = pd.read_csv(results / "stage4de_registry_rows.csv")
    full = read_json("artifacts/results/stage4/catboost/final/catboost_full_train_manifest.json")
    interpretation = read_json("artifacts/results/stage4/catboost/final/catboost_final_interpretation_manifest.json")
    runtime = read_json("artifacts/reports/stage4de_runtime_report.json")
    protected = de.recheck_protected(ROOT)
    required = [
        results / "catboost_final_feature_pack.json",
        results / "catboost_final_configuration.json",
        results / "catboost_final_sensitive_comparison.csv",
        results / "catboost_final_importance_without_sensitive.csv",
        results / "catboost_final_importance_with_sensitive.csv",
        results / "catboost_final_shap_without_sensitive.csv",
        results / "catboost_final_shap_with_sensitive.csv",
        ROOT / "artifacts/models/catboost/final/catboost_final_without_sensitive.joblib",
        ROOT / "artifacts/models/catboost/final/catboost_final_with_sensitive.joblib",
        ROOT / "artifacts/models/catboost/final/catboost_final_without_sensitive.cbm",
        ROOT / "artifacts/models/catboost/final/catboost_final_with_sensitive.cbm",
    ]
    checks = {
        "stage4c_pass": read_json("artifacts/reports/stage4c_verification.json")["status"] == "PASS",
        "protected_inputs_unchanged": protected["status"] == "PASS",
        "confirmation_fit_count_at_most_three": len(confirmation) <= 3,
        "tuning_fit_count_exactly_three": len(tuning) == 3,
        "controlled_modes_exactly_two": set(controlled["sensitive_mode"]) == {"without_sensitive", "with_sensitive"},
        "full_train_modes_pass": full["status"] == "PASS" and set(full["models"]) == {"without_sensitive", "with_sensitive"},
        "full_train_rows_exact": all(item["training_row_count"] == 399788 for item in full["models"].values()),
        "clean_reload_pass": all(item["reload_status"] == "PASS" for item in full["models"].values()),
        "importance_and_shap_pass": interpretation["status"] == "PASS" and interpretation["same_ids_both_modes"] is True,
        "registry_rows_unique": len(registry) == 16 and registry["experiment_id"].is_unique,
        "runtime_within_budget": runtime["status"] == "PASS" and runtime["within_budget"] is True,
        "required_artifacts_present": all(path.is_file() and path.stat().st_size > 0 for path in required),
    }
    verification = {
        "stage": de.STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(reports / "stage4de_internal_verification.json", verification)
    if verification["status"] != "PASS":
        raise AssertionError(f"Stage 4D-E internal verification failed: {checks}")
    final_config = read_json("artifacts/results/stage4/catboost/final/catboost_final_configuration.json")
    impact = read_json("artifacts/results/stage4/catboost/final/catboost_feature_engineering_impact.json")
    summary = {
        "stage": de.STAGE_ID,
        "stage_name": de.STAGE_NAME,
        "feature_confirmation_rows": len(confirmation),
        "accepted_stage4d_proposals": read_json("artifacts/results/stage4/catboost/final/catboost_final_feature_pack.json")["selected_proposals"],
        "feature_confirmation_selection": impact["selection"],
        "tuning_rows": len(tuning),
        "selected_tuning_fit": final_config["selected_fit_id"],
        "fixed_iteration_count": final_config["fixed_iteration_count"],
        "controlled_validation_mae": dict(zip(controlled["sensitive_mode"], controlled["mae"])),
        "full_train_rows_per_mode": 399788,
        "registry_rows": len(registry),
        "runtime_accounted_seconds": runtime["accounted_seconds"],
        "next_stage": "Stage 4F",
        "test_set": "locked",
        "status": "PASS",
    }
    s4.atomic_write_json(reports / "stage4de_analysis_summary.json", summary)


def md(section: int, title: str, interpretation: str) -> list[Any]:
    metadata = {"tags": ["stage4de_owned"], "stage4de_section": section}
    return [
        nbformat.v4.new_markdown_cell(
            f"## {section}. {title}\n\n**Reason.** This section records the validated evidence for {title.lower()} before the cached code is displayed.",
            metadata=metadata,
        ),
        None,
        nbformat.v4.new_markdown_cell(f"**Interpretation.** {interpretation}", metadata=metadata),
    ]


def build_cells() -> list[Any]:
    blocks: list[list[Any]] = []
    code: dict[int, str] = {
        25: "import stage4de_catboost_utils as de\nSTAGE4DE_CACHE_ONLY = os.environ.get('STAGE4DE_CACHE_ONLY', '0') == '1'\ngate = json.loads((ROOT/'artifacts/reports/stage4c_verification.json').read_text(encoding='utf-8'))\ndisplay({'stage4c_status': gate['status'], 'stage4de_cache_only': STAGE4DE_CACHE_ONLY, 'test_set': 'locked'})",
        26: "display({'stage': 'Stage 4D-E - CatBoost Feature Confirmation and Final Model', 'feature_confirmation': 'non-sensitive only', 'tuning_candidates': 3, 'final_modes': 2, 'test_set': 'locked'})",
        27: "proposal_review = pd.read_csv(ROOT/'artifacts/features/stage4/catboost/catboost_round2_proposal_review.csv')\ndisplay(proposal_review[['feature_name','target_derived','sensitive_derived','formula_stable','approved_for_combined_confirmation']])",
        28: "v2 = json.loads((ROOT/'artifacts/manifests/stage4/catboost/catboost_feature_engineer_v2_manifest.json').read_text(encoding='utf-8'))\ndisplay({'status': v2['status'], 'fit_rows': v2['fit_rows'], 'validation_rows': v2['validation_rows'], **v2['checks']})",
        29: "sample_check = json.loads((ROOT/'artifacts/splits/stage4/stage4_sample_verification.json').read_text(encoding='utf-8'))\ndisplay(sample_check['samples']['feature_confirmation'])",
        30: "confirmation = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/feature_confirmation/catboost_feature_confirmation_results.csv')\nimpact = json.loads((ROOT/'artifacts/results/stage4/catboost/final/catboost_feature_engineering_impact.json').read_text(encoding='utf-8'))\ndisplay(confirmation[['fit_id','feature_pack_id','mae','rmse','rmsle','top_decile_mae','p90_absolute_error','fit_time_seconds','selected_final_pack']])\ndisplay(impact)",
        31: "final_pack = json.loads((ROOT/'artifacts/results/stage4/catboost/final/catboost_final_feature_pack.json').read_text(encoding='utf-8'))\ndisplay({'selection': final_pack['selection'], 'pack_id': final_pack['feature_pack']['pack_id'], 'selected_proposals': final_pack['selected_proposals'], 'acceptance_rule': final_pack['acceptance_rule']})",
        32: "display(sample_check['samples']['final_selection'])",
        33: "tuning = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_tuning.csv')\ndisplay(tuning[['mae_rank','fit_id','depth','learning_rate','l2_leaf_reg','best_iteration','mae','rmse','r_squared','top_decile_mae','fit_time_seconds','selected']])",
        34: "final_config = json.loads((ROOT/'artifacts/results/stage4/catboost/final/catboost_final_configuration.json').read_text(encoding='utf-8'))\ndisplay(pd.Series({'feature_pack': final_config['feature_pack']['pack_id'], 'target_mode': final_config['target_mode'], 'selected_fit_id': final_config['selected_fit_id'], 'iterations': final_config['fixed_iteration_count'], 'depth': final_config['parameters']['depth'], 'learning_rate': final_config['parameters']['learning_rate'], 'l2_leaf_reg': final_config['parameters']['l2_leaf_reg'], 'execution_mode': final_config['execution_mode'], 'threads': final_config['thread_count']}))",
        35: "controlled = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_validation_results.csv')\ndisplay(controlled[['sensitive_mode','feature_pack_id','best_iteration','feature_count','mae','rmse','rmsle','r_squared','mean_signed_error','top_decile_mae','fit_time_seconds']])",
        36: "sensitive_comparison = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_sensitive_comparison.csv')\ndisplay(sensitive_comparison.loc[sensitive_comparison['metric'].isin(['mae','rmse','rmsle','r_squared','mean_signed_error','top_decile_mae','fit_time_seconds'])])",
        37: "full_manifest = json.loads((ROOT/'artifacts/results/stage4/catboost/final/catboost_full_train_manifest.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame([{'sensitive_mode': mode, 'training_rows': item['training_row_count'], 'iterations': item['fixed_iteration_count'], 'features': item['feature_count'], 'fit_seconds': item['fit_seconds'], 'reload_status': item['reload_status']} for mode,item in full_manifest['models'].items()]))",
        38: "display(pd.DataFrame([{'sensitive_mode': mode, 'bundle_sha256': item['model_sha256'], 'native_sha256': item['native_model_sha256'], 'source_digest': item['source_hash_digest']} for mode,item in full_manifest['models'].items()]))",
        39: "reloads = pd.read_csv(ROOT/'artifacts/reports/stage4de_catboost_reload_verification.csv')\ndisplay(reloads[['sensitive_mode','model_bytes','native_model_bytes','wall_seconds','return_code','status']])",
        40: "final_imp0 = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_importance_without_sensitive.csv')\nfinal_imp1 = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_importance_with_sensitive.csv')\ndisplay(final_imp0.head(12))\ndisplay(final_imp1.head(12))",
        41: "shap_ids = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_shap_sample_ids.csv')\nfinal_shap0 = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_shap_without_sensitive.csv')\nfinal_shap1 = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_final_shap_with_sensitive.csv')\ndisplay({'rows': len(shap_ids), 'unique_ids': shap_ids['row_id'].nunique(), 'same_ids_both_modes': True, 'space': 'log1p target'})\ndisplay(final_shap0.head(12))\ndisplay(final_shap1.head(12))",
        42: "previous = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/catboost_previous_stage_reference.csv')\ndisplay(previous)\ndisplay('The locked Test Set will later provide the common final comparison.')",
        43: "stage4de_summary = json.loads((ROOT/'artifacts/reports/stage4de_analysis_summary.json').read_text(encoding='utf-8'))\nruntime = json.loads((ROOT/'artifacts/reports/stage4de_runtime_report.json').read_text(encoding='utf-8'))\ndisplay(stage4de_summary)\ndisplay(runtime)",
        44: "internal = json.loads((ROOT/'artifacts/reports/stage4de_internal_verification.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(internal['checks'], name='pass'))\ndisplay({'status': internal['status']})",
        45: "stage4de_rows = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/final/stage4de_registry_rows.csv')\ndisplay({'model_work': 'complete', 'registry_rows': len(stage4de_rows), 'unique_experiment_ids': stage4de_rows['experiment_id'].nunique(), 'test_predictions': 0, 'next_stage_after_final_pass': 'Stage 4F'})",
    }
    descriptions = {
        25: ("Stage 4C Recovery and Prerequisite Gate", "Stage 4C is finalized and protected before Stage 4D-E evidence is used."),
        26: ("Stage 4D-E Objective", "This stage confirms a small Feature set, tunes exactly three final Candidates, fits both frozen modes, and keeps Test locked."),
        27: ("Stage 4C Feature Proposal Review", "All three proposals passed source, target, sensitive, stability, and leakage review before confirmation."),
        28: ("CatBoost Feature Engineer v2", "The v2 transformer preserved row order, learned its income bands on training rows, and passed saved and clean-process transform tests."),
        29: ("Feature Confirmation Sample", "Feature confirmation used the saved 80,000 Train and 20,000 Validation rows without sensitive Features or Test overlap."),
        30: ("Lean Feature Confirmation", "The combined pack and ratio rescue both missed the fixed acceptance rules. The original pack remained selected."),
        31: ("Final CatBoost Feature Pack", "The final pack is the unchanged Stage 4C native CatBoost pack. No proposed Stage 4D Feature was accepted."),
        32: ("Final Selection Sample", "Tuning and the controlled comparison used the saved 100,000 Train and 25,000 Validation rows. Test remained locked."),
        33: ("Three Final Tuning Candidates", "Candidate C had the lowest MAE, but Candidate B was within the 0.25 percent tie band and was safer in depth, RMSE, tail error, signed error, and runtime."),
        34: ("Final CatBoost Configuration", "The final model uses the original pack, log1p target, depth 6, learning rate 0.05, L2 value 20, and 2,000 trees on CPU with four threads."),
        35: ("Controlled Sensitive Comparison", "Both modes used the same pack, parameters, and 2,000 trees. The sensitive mode improved MAE slightly but had worse RMSE and R-squared."),
        36: ("Final Validation Results", "The displayed difference is an accuracy comparison, not a fairness conclusion. The non-sensitive model remains the selection reference."),
        37: ("Full-Training Model Fits", "Each final pipeline used all 399,788 saved Train rows in a separate sequential fit. No Test row was loaded."),
        38: ("Final Model Manifests", "Each bundle and native model has source, sample, configuration, and file-hash evidence."),
        39: ("Clean-Process Reload Verification", "Both complete pipelines loaded in clean processes and reproduced fixed Train-reference predictions."),
        40: ("Final Feature Importance", "PredictionValuesChange importance is reported for every final model Feature and grouped by Feature origin."),
        41: ("Final Native SHAP", "Both modes use the same 300 Final Selection validation IDs. Values summarize log1p-target contributions and are not raw-dollar effects."),
        42: ("Previous-Stage Reference", "Stage 2 and 3 OOF scores, Stage 4C Discovery scores, and Stage 4D-E Final Selection scores use different rows, so the table is a reference rather than a direct leaderboard."),
        43: ("Stage 4D-E Artifact Summary", "All planned model, interpretation, Registry, and runtime artifacts are present and the accounted runtime is within 210 minutes."),
        44: ("Stage 4D-E Verification", "The pre-notebook safety and completeness checks pass. Final PASS also requires clean notebook runs and independent review."),
        45: ("CatBoost Track Completion Note", "Model work is complete with Test still locked. Stage 4F may start only after notebook, review, and final verification evidence pass."),
    }
    for section in range(25, 46):
        title, interpretation = descriptions[section]
        block = md(section, title, interpretation)
        block[1] = nbformat.v4.new_code_cell(code[section], metadata={"tags": ["stage4de_owned"], "stage4de_section": section})
        blocks.append(block)
    return [cell for block in blocks for cell in block]


def main() -> int:
    prepare_reports()
    backups = sorted((ROOT / "artifacts/backups").glob("REGRESSION_PART4_CATBOOST_before_stage4de_*.ipynb"))
    if not backups:
        raise FileNotFoundError("The required pre-Stage 4D-E notebook backup is missing.")
    backup = nbformat.read(backups[-1], as_version=4)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if len(backup.cells) != PREFIX_CELLS or notebook.cells[:PREFIX_CELLS] != backup.cells:
        raise AssertionError("The finalized Stage 4A-C notebook prefix does not match its backup.")
    unknown_tail = [cell for cell in notebook.cells[PREFIX_CELLS:] if "stage4de_owned" not in cell.metadata.get("tags", [])]
    notebook.cells = list(notebook.cells[:PREFIX_CELLS]) + unknown_tail + build_cells()
    notebook.metadata["stage4de"] = {"sections": list(range(25, 46)), "cache_only": True, "version": de.VERSION}
    nbformat.validate(notebook)
    temp = NOTEBOOK.with_suffix(".stage4de.tmp.ipynb")
    nbformat.write(notebook, temp)
    temp.replace(NOTEBOOK)
    rebuilt = nbformat.read(NOTEBOOK, as_version=4)
    if rebuilt.cells[:PREFIX_CELLS] != backup.cells or len(rebuilt.cells) != PREFIX_CELLS + len(unknown_tail) + 63:
        raise AssertionError("Notebook prefix or Stage 4D-E cell count changed after writing.")
    print(json.dumps({"status": "PASS", "prefix_cells": PREFIX_CELLS, "unknown_tail_cells": len(unknown_tail), "appended_cells": 63, "total_cells": len(rebuilt.cells)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
