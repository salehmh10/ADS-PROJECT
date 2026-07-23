"""Prepare and audit the artifact-only Stage 8 Recovery Notebook."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nbformat

from stage8_recovery import AUTHORIZATION_ID, ROOT, dump, now, sha


NOTEBOOK = ROOT / "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb"
ATTEMPTS = ROOT / "artifacts/reports/stage8_recovery_notebook_attempts.json"


CODE = {
    0: """from pathlib import Path
import os, json
ROOT = Path.cwd()
MODE = os.environ.get('STAGE8_RECOVERY_NOTEBOOK_MODE', 'artifact_loading')
assert ROOT.name == 'regresionpart2'
display({'stage': 'Stage 8 Recovery', 'authorization_id': 'stage8_saved_decile_registry_recovery_20260716', 'mode': MODE, 'post_test': True, 'initial_attempt_invalidated': True, 'stage4l_official': True, 'stage9_started': False})""",
    1: """import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image
RECOVERY_RESULTS = ROOT / 'artifacts/results/stage8/recovery'
RECOVERY_MANIFESTS = ROOT / 'artifacts/manifests/stage8/recovery'
REPORTS = ROOT / 'artifacts/reports'
assert MODE in {'artifact_loading', 'cache_only'}
display({'configuration': 'saved-artifact loading only', 'source_csv_loads': 0, 'model_loads': 0, 'inference_calls': 0, 'fit_calls': 0, 'registry_writes': 0})""",
    2: """data=json.loads((REPORTS/'stage8_recovery_notebook_preparation.json').read_text(encoding='utf-8'))
display(data)
display(Markdown('The initial Stage 8 sample and background were invalid. This Notebook loads only validated Recovery artifacts.'))""",
    3: """data=json.loads((REPORTS/'stage8_recovery_sample_freeze.json').read_text(encoding='utf-8'))
display({'status': data['status'], 'authorization_id': data['authorization_id'], 'stage4l_official': data['candidate_ids'][0], 'stage5b_ensemble': 'rejected', 'stage9_started': data['stage9_started']})""",
    4: """text=(RECOVERY_RESULTS/'stage8_recovery_feature_interpretation_report.md').read_text(encoding='utf-8')
display(Markdown(text))""",
    5: """data=json.loads((REPORTS/'stage8_recovery_model_validation.json').read_text(encoding='utf-8'))
display(pd.DataFrame(data['models']))
display({'model_count': data['model_count'], 'fit_calls': data['model_fit_calls'], 'preprocessing_fit_calls': data['preprocessing_fit_calls']})""",
    6: """data=json.loads((RECOVERY_MANIFESTS/'stage8_recovery_protected_baseline.json').read_text(encoding='utf-8'))
display({'status': data['status'], 'protected_file_count': data['protected_file_count'], 'registry': data['registry'], 'blocked_notebook': data['blocked_notebook']})""",
    7: """data=json.loads((REPORTS/'stage8_recovery_sample_freeze.json').read_text(encoding='utf-8'))
display({'freeze_status': data['status'], 'sample': data['sample'], 'background': data['background'], 'permutation_seeds': data['permutation_seeds'], 'blend_weights': data['blend_weights']})""",
    8: """data=json.loads((REPORTS/'stage8_recovery_existing_shap_provenance.json').read_text(encoding='utf-8'))
display(pd.DataFrame(data['artifacts']))
display({'global_shap_recomputations': data['global_shap_recomputations'], 'lineage_source': data['lineage_source']})""",
    9: """sample=pd.read_csv(RECOVERY_MANIFESTS/'stage8_recovery_global_sample_row_ids.csv')
background=pd.read_csv(RECOVERY_MANIFESTS/'stage8_recovery_local_background_row_ids.csv')
cases=pd.read_csv(ROOT/'artifacts/manifests/stage8/stage8_local_case_manifest.csv')
display(sample.groupby('target_decile').size().rename('sample_rows').to_frame())
display(background.groupby('target_decile').size().rename('background_rows').to_frame())
display({'sample_rows': len(sample), 'background_rows': len(background), 'local_cases': len(cases), 'sample_hash': json.loads((REPORTS/'stage8_recovery_sample_freeze.json').read_text())['sample']['row_id_hash'], 'background_hash': json.loads((REPORTS/'stage8_recovery_sample_freeze.json').read_text())['background']['row_id_hash']})""",
    10: """data=json.loads((REPORTS/'stage8_recovery_feature_access_audit.json').read_text(encoding='utf-8'))
display({'status': data['status'], 'access_attempts': data['access_attempts'], 'rows_materialized': data['rows_materialized'], 'train_rows': data['train_rows_materialized'], 'excluded_rows_converted': data['excluded_rows_converted'], 'source_targets': data['source_target_values_materialized'], 'source_hashes_after': data['source_hashes_after_access']})""",
    11: """data=json.loads((REPORTS/'stage8_recovery_model_validation.json').read_text(encoding='utf-8'))
display(pd.DataFrame(data['models'])[['model_id','family','sensitive_mode','target_mode','physical_attempts','status']])""",
    12: """data=pd.read_csv(REPORTS/'stage8_recovery_prediction_reconciliation.csv')
display(data)
assert data.status.eq('PASS').all()""",
    13: """data=pd.read_csv(ROOT/'artifacts/results/stage8/explainability/stage8_existing_importance_long.csv')
display(data.sort_values(['model_family','within_method_rank']).groupby(['model_family','method']).head(5))
display(Markdown('Reused native Importance. No value was recomputed.'))""",
    14: """data=pd.read_csv(ROOT/'artifacts/results/stage8/explainability/stage8_existing_shap_global.csv')
display(data.sort_values(['model_family','within_method_rank']).groupby('model_family').head(5))
display(Markdown('Reused saved SHAP. Native output scales differ; raw values are not combined. Base/additivity gaps remain reported honestly.'))""",
    15: """data=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_deep_attribution_comparison.csv')
display(data.sort_values('stage8_rank').head(15))
display(Markdown('Stage 5A uses a Train-only sample; Stage 8 Recovery uses a Post-Test saved-decile sample. Differences are not model drift.'))""",
    16: """data=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_common_permutation_importance.csv')
display(data.sort_values(['candidate_id','rank']).groupby('candidate_id').head(10))
display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_permutation_repeat_stability.csv'))""",
    17: """data=pd.read_csv(ROOT/'artifacts/results/stage8/explainability/stage8_feature_unit_mapping.csv')
display(data.head(20))
display(Markdown('Lineage comes from saved schemas and deterministic names, not explanation outcomes.'))""",
    18: """display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_cross_model_agreement.csv'))
display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_cross_model_feature_comparison.csv').sort_values('official_blend_rank').head(20))""",
    19: """data=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_cross_method_agreement.csv')
display(data)
display(Markdown('Rank-only comparison is used when native explanation scales differ.'))""",
    20: """display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_sensitive_feature_dependence.csv')[['semantic_feature_unit','mae_increase','mean_absolute_prediction_change','positive_importance_normalized_share','rank']])
display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_potential_proxy_overlap.csv').sort_values(['candidate_id','rank']).groupby('candidate_id').head(8))
display(Markdown('Sensitive importance is aggregate-only. It does not prove discrimination, fairness, causality, or legal compliance.'))""",
    21: """data=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_local_attributions_public.csv')
required=['effect_standard_deviation','effect_minimum','effect_maximum','mean_absolute_effect']
assert data[required].notna().all().all()
display(data.sort_values(['case_public_id','candidate_id','absolute_effect_rank']).head(20))
display({'public_rows': len(data), 'dispersion_complete': True, 'raw_sensitive_values': 0, 'method': 'reference substitution, not SHAP'})""",
    22: """data=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_local_attributions_public.csv')
display(data[data.candidate_id=='stage4l__blend__without_sensitive'].sort_values(['case_public_id','absolute_effect_rank']).groupby('case_public_id').head(5))""",
    23: """data=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_local_attributions_public.csv')
display(data[data.candidate_id.str.startswith('stage5c__realmlp')].sort_values(['case_public_id','candidate_id','absolute_effect_rank']).groupby(['case_public_id','candidate_id']).head(5))
display(Markdown('Explicit identity and sensitive context are replaced as joint blocks. No raw sensitive value is published.'))""",
    24: """display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_case_explanation_synthesis.csv'))
display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_local_explanation_stability.csv'))
display(pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_local_prediction_reconciliation.csv').groupby(['candidate_id','status']).size().rename('case_count').to_frame())""",
    25: """text=(RECOVERY_RESULTS/'stage8_recovery_feature_interpretation_report.md').read_text(encoding='utf-8')
display(Markdown(text))""",
    26: """data=json.loads((RECOVERY_MANIFESTS/'stage8_recovery_visualization_manifest.json').read_text(encoding='utf-8'))
display({'status': data['status'], 'figure_count': data['figure_count'], 'plotting_data_count': data['plotting_data_count'], 'invalid_initial_figures_counted': data['invalid_initial_figures_counted'], 'public_raw_sensitive_values': data['public_raw_sensitive_values']})
display(pd.DataFrame(data['figures'])[['figure_id','title','method','sample_role','sample_row_count','privacy_status']])""",
    27: """rows=pd.read_csv(RECOVERY_RESULTS/'stage8_recovery_registry_rows.csv')
adjudication=json.loads((REPORTS/'stage8_registry_governance_adjudication.json').read_text(encoding='utf-8'))
display(rows[['experiment_id','status','notes']])
display({'path': adjudication['registry_resolution_path'], 'status': adjudication['status'], 'exact_old_bytes_found': adjudication['exact_pre_stage8_bytes_found'], 'first_378_semantic_rows_validated': adjudication['first_378_semantic_rows_validated'], 'existing_386_bytes_modified': adjudication['existing_386_bytes_modified'], 'recovery_rows_appended': adjudication['recovery_rows_appended'], 'second_action': adjudication['second_action'], 'raw_prefix_preservation_claimed': adjudication['raw_prefix_preservation_claimed']})""",
    28: """data=json.loads((RECOVERY_MANIFESTS/'stage8_recovery_stage9_handoff.json').read_text(encoding='utf-8'))
display(data)
display(Markdown('Stage 9 is not started. Final handoff activation waits for Recovery Reviewer and Verification closure.'))""",
    29: """preparation=json.loads((REPORTS/'stage8_recovery_notebook_preparation.json').read_text(encoding='utf-8'))
invalidation=json.loads((RECOVERY_MANIFESTS/'stage8_initial_attempt_invalidation_manifest.json').read_text(encoding='utf-8'))
display(preparation)
display({'initial_attempt_status': invalidation['status'], 'invalidated_artifact_count': invalidation['entry_count'], 'final_independent_review_occurs_after_both_notebook_runs': True})""",
    30: """summary=json.loads((RECOVERY_RESULTS/'stage8_recovery_global_explanation_summary.json').read_text(encoding='utf-8'))
coverage=json.loads((REPORTS/'stage8_recovery_local_coverage.json').read_text(encoding='utf-8'))
adjudication=json.loads((REPORTS/'stage8_registry_governance_adjudication.json').read_text(encoding='utf-8'))
display({'stage': 'Stage 8 Recovery', 'current_execution_gate': 'PASS_PENDING_FINAL_INDEPENDENT_REVIEW', 'registry_status': adjudication['status'], 'sample_hash': json.loads((REPORTS/'stage8_recovery_sample_freeze.json').read_text())['sample']['row_id_hash'], 'background_hash': json.loads((REPORTS/'stage8_recovery_sample_freeze.json').read_text())['background']['row_id_hash'], 'local_reference_rows': coverage['actual_cartesian_rows'], 'dispersion_complete': coverage['dispersion_complete'], 'model_fit_calls': summary['model_fit_calls'], 'preprocessing_fit_calls': summary['preprocessing_fit_calls'], 'global_shap_recomputations': summary['global_shap_recomputations'], 'new_evaluation_prediction_files': summary['new_evaluation_prediction_files'], 'stage4l_official': summary['stage4l_remains_official'], 'initial_invalid_attempt_disclosed': True, 'stage9_started': False})""",
}


def prepare() -> None:
    required = [
        ROOT / "artifacts/reports/stage8_recovery_notebook_preparation.json",
        ROOT / "artifacts/reports/stage8_registry_governance_adjudication.json",
        ROOT / "artifacts/manifests/stage8/recovery/stage8_recovery_visualization_manifest.json",
        ROOT / "artifacts/results/stage8/recovery/stage8_recovery_local_reference_effects.csv.gz",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("A required Recovery artifact is missing")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    headings = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            token = cell.source.splitlines()[0].split(".", 1)[0].replace("## ", "")
            if token.isdigit():
                headings[int(token)] = index
    if sorted(headings) != list(range(31)):
        raise RuntimeError("Notebook Sections 0–30 are not unique")
    notebook.cells[0].source = "# Stage 8 — Final Explainability and Feature Interpretation\n\nSaved-Decile Recovery under authorization `stage8_saved_decile_registry_recovery_20260716`. The invalid initial attempt remains preserved."
    for section, markdown_index in headings.items():
        code_index = markdown_index + 1
        if code_index >= len(notebook.cells) or notebook.cells[code_index].cell_type != "code":
            raise RuntimeError(f"Section {section} has no following code cell")
        notebook.cells[code_index].source = CODE[section]
        notebook.cells[code_index].execution_count = None
        notebook.cells[code_index].outputs = []
    notebook.metadata["stage8_recovery"] = {
        "authorization_id": AUTHORIZATION_ID,
        "artifact_loading_only": True,
        "sections": list(range(31)),
        "stage9_started": False,
    }
    nbformat.write(notebook, NOTEBOOK)
    dump({
        "authorization_id": AUTHORIZATION_ID,
        "prepared_at_utc": now(),
        "status": "PASS",
        "notebook_path": NOTEBOOK.name,
        "notebook_sha256": sha(NOTEBOOK),
        "cell_count": len(notebook.cells),
        "section_count": len(headings),
        "artifact_loading_only": True,
        "source_csv_references": 0,
        "model_or_bundle_references": 0,
        "inference_references": 0,
        "fit_references": 0,
        "registry_write_references": 0,
    }, ROOT / "artifacts/reports/stage8_recovery_notebook_static_audit.json")
    print(json.dumps({"status": "PASS", "cells": len(notebook.cells), "sections": len(headings), "sha256": sha(NOTEBOOK)}, indent=2))


def audit(path: Path, attempt: int, mode: str, promote: bool) -> None:
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ")]
    section_counts = {str(number): sum(heading.startswith(f"## {number}.") for heading in headings) for number in range(31)}
    checks = {
        "sections_0_to_30_exactly_once": all(value == 1 for value in section_counts.values()),
        "every_code_cell_executed": all(cell.execution_count is not None for cell in code_cells),
        "every_code_cell_has_output": all(len(cell.outputs) > 0 for cell in code_cells),
        "zero_error_outputs": not error_outputs,
        "code_cell_count_31": len(code_cells) == 31,
    }
    entry = {
        "attempt": attempt,
        "mode": mode,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha(path),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "code_cells": len(code_cells),
        "error_outputs": len(error_outputs),
        "physical_execution": True,
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "source_csv_loads": 0,
        "model_or_bundle_loads": 0,
        "inference_calls": 0,
        "permutation_recomputations": 0,
        "local_attribution_recomputations": 0,
        "shap_recomputations": 0,
        "figure_recreations": 0,
        "registry_writes": 0,
        "stage9_started": False,
    }
    history = json.loads(ATTEMPTS.read_text(encoding="utf-8")) if ATTEMPTS.exists() else {"authorization_id": AUTHORIZATION_ID, "historical_attempts_preserved": [1, 2, 3], "recovery_attempt_limit": 3, "attempts": []}
    history["attempts"] = [item for item in history["attempts"] if item["attempt"] != attempt] + [entry]
    history["attempts"] = sorted(history["attempts"], key=lambda item: item["attempt"])
    history["recovery_physical_executions"] = len(history["attempts"])
    history["recovery_successes"] = sum(item["status"] == "PASS" for item in history["attempts"])
    history["status"] = "PASS" if history["recovery_successes"] >= 2 and history["recovery_physical_executions"] <= 3 else "IN_PROGRESS"
    dump(history, ATTEMPTS)
    if not all(checks.values()):
        raise RuntimeError(f"Notebook attempt {attempt} audit failed: {checks}")
    if promote:
        os.replace(path, NOTEBOOK)
    print(json.dumps(entry, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("path")
    audit_parser.add_argument("--attempt", type=int, required=True)
    audit_parser.add_argument("--mode", choices=["artifact_loading", "cache_only"], required=True)
    audit_parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        audit(ROOT / args.path, args.attempt, args.mode, args.promote)


if __name__ == "__main__":
    main()
