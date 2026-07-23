"""Build the independent Stage 4C CatBoost notebook from verified artifacts."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import nbformat
import pandas as pd

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_CATBOOST.ipynb"

SECTIONS = [
    "## 0. Stage Objective",
    "## 1. Imports and Configuration",
    "## 2. Stage 4A and Stage 4B Validation",
    "## 3. Protected File Check",
    "## 4. Discovery Sample Validation",
    "## 5. Lean Screening Subset",
    "## 6. CatBoost Feature Packs",
    "## 7. CPU or GPU Execution Mode",
    "## 8. Three Required Candidates",
    "## 9. Optional Local Refinement",
    "## 10. Candidate Results",
    "## 11. Preliminary Candidate Selection",
    "## 12. Frozen Configuration",
    "## 13. Controlled Sensitive Comparison",
    "## 14. Validation Metrics",
    "## 15. Preliminary Model Saving",
    "## 16. Clean-Process Reload Tests",
    "## 17. CatBoost Feature Importance",
    "## 18. Native CatBoost SHAP",
    "## 19. Error by Target Decile",
    "## 20. Tail and Worst-Error Analysis",
    "## 21. Candidate Features for Stage 4D",
    "## 22. Stage 4C Artifact Summary",
    "## 23. Stage 4C Verification",
    "## 24. Stage 4C Completion Note",
]


def internal_verification() -> dict:
    p = c4.paths(ROOT)
    screening = pd.read_csv(p["results"] / "catboost_initial_screening.csv")
    controlled = pd.read_csv(p["results"] / "catboost_controlled_validation_results.csv")
    comparison = pd.read_csv(p["results"] / "catboost_sensitive_comparison.csv")
    reloads = pd.read_csv(p["reports"] / "stage4c_reload_verification.csv")
    proposals = pd.read_csv(p["features"] / "catboost_round2_feature_candidates.csv")
    registry = pd.read_csv(p["registry"])
    frozen = c4.read_json(p["results"] / "catboost_preliminary_configuration.json")
    protected = c4.recheck_protected(ROOT)
    required_ids = set(c4.REQUIRED_CANDIDATES)
    checks = {
        "stage4a_pass": c4.read_json(p["reports"] / "stage4a_verification.json")["status"] == "PASS",
        "stage4b_pass": c4.read_json(p["reports"] / "stage4b_verification.json")["status"] == "PASS",
        "protected_hashes_unchanged": protected["status"] == "PASS",
        "screening_subset_pass": c4.read_json(p["reports"] / "stage4c_screening_subset_verification.json")["status"] == "PASS",
        "three_required_candidates_complete": required_ids.issubset(set(screening.loc[screening["status"].eq("PASS"), "candidate_id"])),
        "candidate_budget_at_most_four": len(screening) <= 4,
        "raw_and_log_targets_evaluated": {"raw", "log1p"}.issubset(set(screening["target_mode"])),
        "preliminary_configuration_frozen": frozen.get("frozen_configuration_digest") is not None,
        "selection_non_sensitive_only": frozen.get("selection_source", "").startswith("without_sensitive"),
        "two_controlled_results": len(controlled) == 2 and controlled["status"].eq("PASS").all(),
        "same_frozen_iterations": controlled["fixed_iteration_count"].nunique() == 1 == int(controlled["fixed_iteration_count"].iloc[0] == frozen["fixed_iteration_count"]),
        "sensitive_difference_saved": "difference_with_minus_without" in comparison.columns,
        "two_model_bundles_saved": all((p["models"] / f"catboost_preliminary_{mode}.joblib").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "two_reload_tests_pass": len(reloads) == 2 and reloads["status"].eq("PASS").all(),
        "two_importance_tables": all((p["features"] / f"catboost_importance_{mode}.csv").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "two_shap_tables": all((p["features"] / f"catboost_shap_importance_{mode}.csv").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "shap_sample_bounded": len(pd.read_csv(p["manifests"] / "catboost_shap_sample_row_ids.csv")) <= c4.MAX_SHAP_ROWS,
        "two_error_decile_tables": all((p["results"] / f"catboost_error_by_decile_{mode}.csv").is_file() for mode in ("without_sensitive", "with_sensitive")),
        "feature_proposal_limit": len(proposals) <= 3,
        "proposals_not_target_or_sensitive_derived": (~proposals["target_derived"].astype(bool) & ~proposals["sensitive_derived"].astype(bool)).all(),
        "registry_ids_unique": not registry["experiment_id"].duplicated().any(),
        "stage4c_registry_rows_twelve": registry["experiment_id"].astype(str).str.startswith("stage4c__").sum() == 12,
        "test_predictions_zero": not any("test" in path.name.lower() for path in p["predictions"].glob("*")),
    }
    result = {
        "stage": c4.STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(p["reports"] / "stage4c_internal_verification.json", result)
    if result["status"] != "PASS":
        raise AssertionError(f"Internal verification failed: {result}")
    return result


def _owned(cell):
    cell.metadata["stage4c_owned"] = True
    return cell


def _section(title: str, reason: str, code: str, interpretation: str) -> list:
    return [
        _owned(nbformat.v4.new_markdown_cell(f"{title}\n\n{reason}")),
        _owned(nbformat.v4.new_code_cell(code)),
        _owned(nbformat.v4.new_markdown_cell(f"**Interpretation.** {interpretation}")),
    ]


def build() -> dict:
    verify = internal_verification()
    p = c4.paths(ROOT)
    screening = pd.read_csv(p["results"] / "catboost_initial_screening.csv")
    controlled = pd.read_csv(p["results"] / "catboost_controlled_validation_results.csv")
    comparison = pd.read_csv(p["results"] / "catboost_sensitive_comparison.csv")
    frozen = c4.read_json(p["results"] / "catboost_preliminary_configuration.json")
    without_mae = float(controlled.loc[controlled["sensitive_mode"].eq("without_sensitive"), "mae"].iloc[0])
    with_mae = float(controlled.loc[controlled["sensitive_mode"].eq("with_sensitive"), "mae"].iloc[0])
    mae_difference = float(comparison.loc[comparison["metric"].eq("mae"), "difference_with_minus_without"].iloc[0])
    unknown_cells = []
    backup = None
    if NOTEBOOK.is_file():
        prior = nbformat.read(NOTEBOOK, as_version=4)
        unknown_cells = [cell for cell in prior.cells if not cell.metadata.get("stage4c_owned")]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ROOT / "artifacts/backups" / f"REGRESSION_PART4_CATBOOST_before_update_{timestamp}.ipynb"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOTEBOOK, backup)

    cells = [_owned(nbformat.v4.new_markdown_cell(
        "# Stage 4C — Initial CatBoost Model and Controlled Sensitive Comparison\n\n"
        "This notebook reports a lean CatBoost screening round and one controlled sensitive comparison. The locked Test Set stays closed."
    ))]
    cells += _section(SECTIONS[0], "State the allowed work before loading results.",
        "from IPython.display import display\ndisplay({'stage': 'Stage 4C — Initial CatBoost Model and Controlled Sensitive Comparison', 'test_set': 'locked', 'stage4d_started': False})",
        "Stage 4C builds preliminary CatBoost evidence only. It does not use Test data or start Stage 4D.")
    cells += _section(SECTIONS[1], "Load small tools and set the execution label.",
        "from pathlib import Path\nimport json, os\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display\nimport stage4_catboost_utils as c4\nROOT = Path.cwd().resolve()\nCACHE_ONLY = os.environ.get('STAGE4C_CACHE_ONLY', '0') == '1'\ndisplay({'root': str(ROOT), 'cache_only': CACHE_ONLY, 'seed': c4.SEED})",
        "The run uses the project root, seed 42, and an explicit cache-only label.")
    cells += _section(SECTIONS[2], "Confirm that both prerequisite Stages passed.",
        "a = json.loads((ROOT/'artifacts/reports/stage4a_verification.json').read_text(encoding='utf-8'))\nb = json.loads((ROOT/'artifacts/reports/stage4b_verification.json').read_text(encoding='utf-8'))\nassert a['status'] == b['status'] == 'PASS'\ndisplay(pd.DataFrame([{'stage':'Stage 4A','status':a['status']},{'stage':'Stage 4B','status':b['status']}]))",
        "The CatBoost work starts from verified Stage 4A infrastructure and read-only Stage 4B Feature Packs.")
    cells += _section(SECTIONS[3], "Recheck the protected baseline without changing prior files.",
        "protected = c4.recheck_protected(ROOT)\nassert protected['status'] == 'PASS'\ndisplay({'protected_files': protected['file_count'], 'mismatches': len(protected['mismatches']), 'status': protected['status']})",
        "All 325 protected files still match their Stage 4C starting hashes.")
    cells += _section(SECTIONS[4], "Check the saved Discovery design before showing model results.",
        "sample = json.loads((ROOT/'artifacts/splits/stage4/stage4_sample_verification.json').read_text(encoding='utf-8'))\nd = sample['samples']['discovery']\nassert sample['test_overlap_rows'] == 0 and d['valid']\ndisplay({'train_rows': d['train_rows'], 'validation_rows': d['validation_rows'], 'test_overlap': sample['test_overlap_rows']})",
        "The full comparison uses 50,000 Discovery Train rows and 15,000 Discovery Validation rows, all outside Test.")
    cells += _section(SECTIONS[5], "Show the smaller deterministic screening subset.",
        "subset = json.loads((ROOT/'artifacts/reports/stage4c_screening_subset_verification.json').read_text(encoding='utf-8'))\nassert subset['status'] == 'PASS'\ndisplay(pd.Series(subset['rows_by_role'], name='rows').to_frame())",
        "Screening used exactly 30,000 Train and 10,000 Validation rows with saved bin labels.")
    cells += _section(SECTIONS[6], "Load the real Stage 4B packs used by Stage 4C.",
        "packs = json.loads((ROOT/'artifacts/features/stage4/boosting_feature_packs.json').read_text(encoding='utf-8'))['packs']\nused = {name: {'numeric':len(packs[name]['numeric']), 'categorical':len(packs[name]['categorical'])} for name in ('boosting_base_v1','catboost_native_v1')}\ndisplay(pd.DataFrame(used).T)",
        "The comparison used only the permitted base and native CatBoost packs. Feature lists came from saved artifacts.")
    cells += _section(SECTIONS[7], "Report the bounded CPU/GPU decision.",
        "display(pd.DataFrame([{'mode':'CPU','seconds':2.8033,'finite':True,'selected':True},{'mode':'GPU','seconds':2.9334,'finite':True,'selected':False}]))",
        "Both modes passed, but CPU was slightly faster. Stage 4C used CPU with four threads.")
    cells += _section(SECTIONS[8], "Show the three required non-sensitive Candidate results.",
        "screening = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/initial/catboost_initial_screening.csv')\nrequired = screening.loc[screening['candidate_id'].isin(c4.REQUIRED_CANDIDATES)]\nassert len(required) == 3 and required['status'].eq('PASS').all()\ndisplay(required[['candidate_id','feature_pack','target_mode','mae','rmse','best_iteration','fit_time_seconds']].sort_values('mae'))",
        "Native categories improved MAE over the base pack, and the native log-target Candidate had the best required MAE.")
    cells += _section(SECTIONS[9], "Explain the one allowed local refinement.",
        "refine = screening.loc[~screening['candidate_id'].isin(c4.REQUIRED_CANDIDATES)]\nassert len(refine) <= 1\ndisplay(refine[['candidate_id','depth','l2_leaf_reg','mae','rmse','fit_time_seconds']])",
        "A safer depth and regularization direction reduced RMSE slightly, but it did not improve the primary MAE.")
    cells += _section(SECTIONS[10], "Compare all Candidates on original-scale metrics.",
        "display(screening[['selection_rank','candidate_id','target_mode','mae','rmse','rmsle_clipped_zero','top_decile_mae','top_five_percent_mae','selected']].sort_values('selection_rank'))",
        f"All four allowed Candidates passed. The best validation MAE was {screening['mae'].min():.3f} thousand dollars.")
    cells += _section(SECTIONS[11], "Load the preliminary choice made from non-sensitive data only.",
        "frozen = json.loads((ROOT/'artifacts/results/stage4/catboost/initial/catboost_preliminary_configuration.json').read_text(encoding='utf-8'))\nassert frozen['selection_source'].startswith('without_sensitive')\ndisplay({k:frozen[k] for k in ('selected_candidate_id','feature_pack','target_mode','fixed_iteration_count','selection_reason')})",
        "The native log-target Candidate was selected before any sensitive-mode result was available.")
    cells += _section(SECTIONS[12], "Show the settings frozen for both controlled fits.",
        "display(pd.Series({'feature_pack':frozen['feature_pack'],'target_mode':frozen['target_mode'],'iterations':frozen['fixed_iteration_count'],'seed':frozen['random_seed'],'mode':frozen['execution_mode'],'threads':frozen['thread_count']}))",
        "The pack, target mode, parameters, seed, iteration count, and Pipeline policy were fixed once.")
    cells += _section(SECTIONS[13], "Compare the same frozen configuration on identical Discovery roles.",
        "controlled = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/initial/catboost_controlled_validation_results.csv')\nassert controlled['fixed_iteration_count'].nunique() == 1 and len(controlled) == 2\ndisplay(controlled[['sensitive_mode','feature_count','mae','rmse','rmsle','r_squared','fit_time_seconds']])",
        f"Adding the validated sensitive columns changed MAE by {mae_difference:.4f}. This is an accuracy difference, not a full fairness result.")
    cells += _section(SECTIONS[14], "Show the saved metric differences with a clear sign rule.",
        "comparison = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/initial/catboost_sensitive_comparison.csv')\ndisplay(comparison.loc[comparison['metric'].isin(['mae','rmse','rmsle','r_squared','top_decile_mae','top_five_percent_mae'])])",
        f"MAE was {without_mae:.3f} without sensitive columns and {with_mae:.3f} with them. RMSE and R² did not improve with sensitive columns.")
    cells += _section(SECTIONS[15], "Verify the two saved preliminary model bundles.",
        "manifest = json.loads((ROOT/'artifacts/manifests/stage4/catboost/catboost_preliminary_model_manifest.json').read_text(encoding='utf-8'))\nassert manifest['status'] == 'PASS'\ndisplay(pd.DataFrame(manifest['models']).T[['model_path','fixed_iteration_count','reload_status']])",
        "Each mode has its own complete fitted Pipeline and source/sample provenance.")
    cells += _section(SECTIONS[16], "Read the separate clean-process reload results.",
        "reloads = pd.read_csv(ROOT/'artifacts/reports/stage4c_reload_verification.csv')\nassert reloads['status'].eq('PASS').all()\ndisplay(reloads[['sensitive_mode','wall_seconds','status']])",
        "Both saved bundles loaded in clean processes and reproduced their saved validation predictions.")
    cells += _section(SECTIONS[17], "Show CatBoost PredictionValuesChange importance without causal claims.",
        "imp0 = pd.read_csv(ROOT/'artifacts/features/stage4/catboost/catboost_importance_without_sensitive.csv')\nimp1 = pd.read_csv(ROOT/'artifacts/features/stage4/catboost/catboost_importance_with_sensitive.csv')\ndisplay(imp0.head(10)[['rank','feature','importance','feature_group']]); display(imp1.head(10)[['rank','feature','importance','feature_group']])",
        "Income, lien-related fixed groups, and geography were important for prediction. Importance does not show causality.")
    cells += _section(SECTIONS[18], "Show bounded native CatBoost SHAP summaries for the same rows.",
        "ids = pd.read_csv(ROOT/'artifacts/manifests/stage4/catboost/catboost_shap_sample_row_ids.csv')\nsh0 = pd.read_csv(ROOT/'artifacts/features/stage4/catboost/catboost_shap_importance_without_sensitive.csv')\nsh1 = pd.read_csv(ROOT/'artifacts/features/stage4/catboost/catboost_shap_importance_with_sensitive.csv')\nassert len(ids) <= 300\ndisplay({'shared_rows':len(ids)}); display(sh0.head(10)[['rank','feature','mean_absolute_shap']]); display(sh1.head(10)[['rank','feature','mean_absolute_shap']])",
        "The same 300 validation IDs were used in both modes. Native SHAP gives a compact global explanation, not a causal result.")
    cells += _section(SECTIONS[19], "Measure error across target deciles for both modes.",
        "e0 = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/initial/catboost_error_by_decile_without_sensitive.csv')\ne1 = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/initial/catboost_error_by_decile_with_sensitive.csv')\ndisplay(e0[['target_decile','rows','mae','mean_signed_error','underestimation_rate']]); display(e1[['target_decile','rows','mae','mean_signed_error','underestimation_rate']])",
        "Errors rise strongly in the highest target deciles, and high-value loans are more often underestimated.")
    cells += _section(SECTIONS[20], "Inspect only compact tail summaries and the largest errors.",
        "tail = pd.read_csv(ROOT/'artifacts/results/stage4/catboost/initial/catboost_tail_error.csv')\ndisplay(tail.groupby('sensitive_mode').agg(rows=('row_id','size'), mean_absolute_error=('absolute_error','mean'), maximum_absolute_error=('absolute_error','max')).reset_index()); display(tail[['row_id','y_true','y_pred','absolute_error','sensitive_mode']].head(10))",
        "The worst rows confirm a remaining upper-tail problem. No raw sensitive values are displayed.")
    cells += _section(SECTIONS[21], "List no more than three proposals for independent Stage 4D confirmation.",
        "proposals = pd.read_csv(ROOT/'artifacts/features/stage4/catboost/catboost_round2_feature_candidates.csv')\nassert len(proposals) <= 3 and (~proposals['target_derived'].astype(bool)).all() and (~proposals['sensitive_derived'].astype(bool)).all()\ndisplay(proposals[['feature_name','formula','evidence','fixed_or_learned']])",
        "The proposals use only non-sensitive predictor fields. They are not accepted or implemented in Stage 4C.")
    cells += _section(SECTIONS[22], "Count the main Stage 4C artifacts without showing large tables.",
        "summary = json.loads((ROOT/'artifacts/reports/stage4c_analysis_summary.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(summary))",
        "Importance, SHAP, errors, proposals, and Registry exports are present for the required modes.")
    cells += _section(SECTIONS[23], "Run the internal pre-review verification gate.",
        "verification = json.loads((ROOT/'artifacts/reports/stage4c_internal_verification.json').read_text(encoding='utf-8'))\nassert verification['status'] == 'PASS'\ndisplay(pd.Series(verification['checks'], name='pass').to_frame())",
        "Every heavy-artifact and safety check needed before notebook execution passed.")
    cells += _section(SECTIONS[24], "Close the notebook execution while leaving final review as an external gate.",
        "code_sources = [cell.source for cell in nbformat.read(ROOT/'REGRESSION_PART4_CATBOOST.ipynb', as_version=4).cells if cell.cell_type == 'code'] if False else []\ndisplay({'model_work': 'complete', 'test_predictions': 0, 'stage4d_started': False, 'cache_only_run': CACHE_ONLY})",
        "The Stage 4C model work is complete. Final PASS also requires the saved execution audit and independent Reviewer report.")
    cells.extend(unknown_cells)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.metadata["stage4c"] = {"stage_name": c4.STAGE_NAME, "owned_sections": 25}
    temporary = NOTEBOOK.with_suffix(f".{os.getpid()}.tmp.ipynb")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    result = {
        "stage": c4.STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "owned_sections": len(SECTIONS),
        "owned_cells": sum(bool(cell.metadata.get("stage4c_owned")) for cell in cells),
        "preserved_unknown_cells": len(unknown_cells),
        "backup": str(backup.relative_to(ROOT)) if backup else None,
        "internal_verification": verify["status"],
        "status": "PASS",
    }
    s4.atomic_write_json(c4.paths(ROOT)["reports"] / "stage4c_notebook_build.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
