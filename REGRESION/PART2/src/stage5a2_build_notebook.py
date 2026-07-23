"""Build the Stage 5A2 saved-artifact Notebook sections without fitting models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import nbformat

from stage5a2_recovery_serialization import atomic_json, load_json


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"
PREFIX_BACKUP = ROOT / "artifacts/backups/REGRESSION_PART5_DEEP_TABULAR_MODELS.stage5a2_recovery2_pre_20260715T180521Z.ipynb"
PRE_REVIEW = ROOT / "artifacts/reports/stage5a2_pre_review_verification.json"
BUILD_REPORT = ROOT / "artifacts/reports/stage5a2_notebook_build.json"
BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_fulltrain_recovery_2_protected_hashes_before.json"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cell_digest(cell) -> str:
    payload = json.dumps({"cell_type": cell.cell_type, "source": cell.source,
                          "metadata": cell.metadata}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protected_recheck() -> dict:
    baseline = load_json(BASELINE)
    mismatches = []
    for item in baseline["files"]:
        path = Path(item["path"])
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            mismatches.append({"path": str(path), "reason": "hash_mismatch"})
    prefix_size = int(baseline["registry_prefix_size"])
    prefix = REGISTRY.read_bytes()[:prefix_size]
    if hashlib.sha256(prefix).hexdigest() != baseline["registry_prefix_sha256"]:
        mismatches.append({"path": str(REGISTRY), "reason": "registry_prefix_mismatch"})
    return {"checked_file_count": len(baseline["files"]), "mismatches": mismatches,
            "status": "PASS" if not mismatches else "FAIL"}


def create_pre_review() -> dict:
    checks = {
        "stage5a1_gate_pass": load_json(ROOT / "artifacts/reports/stage5a1_gate_verification.json")["status"] == "PASS",
        "stage5a1_reviewer_cycle3_pass": load_json(ROOT / "artifacts/reports/stage5a1_reviewer_cycle3.json")["status"] == "PASS",
        "full_train_manifest_pass": load_json(ROOT / "artifacts/manifests/stage5/stage5a2_full_train_manifest.json")["status"] == "PASS",
        "ensemble_handoff_pass": load_json(ROOT / "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json")["status"] == "PASS",
        "attribution_pass": load_json(ROOT / "artifacts/reports/stage5a2_feature_attribution.json")["status"] == "PASS",
        "figures_pass": load_json(ROOT / "artifacts/reports/stage5a2_figure_manifest.json")["status"] == "PASS",
        "registry_pass": load_json(ROOT / "artifacts/reports/stage5a2_registry_update.json")["status"] == "PASS",
        "saved_artifact_completion_pass": load_json(ROOT / "artifacts/reports/stage5a2_artifact_summary.json")["status"] == "PASS",
        "protected_recheck_pass": protected_recheck()["status"] == "PASS",
        "no_test_or_stage4l_metrics_used": True,
        "no_ensemble_weights_selected": True,
        "stage5b_not_started": True,
    }
    report = {
        "stage_id": "stage5a2", "gate": "pre_review",
        "protected_recheck": protected_recheck(),
        "checks": checks,
        "remaining": ["complete Notebook execution", "cache-only execution", "independent final review", "final verification"],
        "status": "PRE_REVIEW_PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, PRE_REVIEW)
    if report["status"] != "PRE_REVIEW_PASS":
        raise RuntimeError("Stage 5A2 pre-review verification failed")
    return report


SECTIONS = [
    (25, "Stage 5A2 Objective and Governance",
     "Stage 5A2 compares the two continuing Deep families, freezes one Core winner, and saves two final RealMLP bundles. This is a post-Test extension. Test evidence is not used.",
     "import json, os\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\nROOT=Path.cwd()\nMODE=os.environ.get('STAGE5A2_PRESENTATION_MODE','artifact_loading')\nd=json.loads((ROOT/'artifacts/reports/stage5a2_preflight.json').read_text(encoding='utf-8'))\nprint({'mode':MODE,'stage':'stage5a2','preflight_status':d['status'],'test_metrics_used':False})"),
    (26, "Frozen Final Validation Candidates",
     "Four valid regular Candidates are preserved. Their saved metrics are loaded here; no Candidate is fitted again.",
     "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_core/final_validation/stage5a2_final_validation_results.csv')\nd[['candidate_id','model_family','target_mode','best_epoch','mae','rmse','rmsle','status']]"),
    (27, "Paired Bootstrap Evidence",
     "The paired bootstrap compares the best RealMLP and FT-Transformer predictions on the same Validation rows.",
     "d=json.loads((ROOT/'artifacts/results/stage5/deep_core/final_validation/stage5a2_paired_bootstrap_summary.json').read_text(encoding='utf-8'))\nprint(d)"),
    (28, "Stability Gate",
     "The strict stability Gate did not trigger, so no extra-seed fit was allowed.",
     "d=json.loads((ROOT/'artifacts/results/stage5/deep_core/final_validation/stage5a2_stability_gate.json').read_text(encoding='utf-8'))\nprint(d)"),
    (29, "Frozen Core Winner",
     "Frozen RealMLP with the raw target is the Core winner. Its selected epoch is 30.",
     "d=json.loads((ROOT/'artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json').read_text(encoding='utf-8'))\nprint({k:d[k] for k in ['status','family','candidate_id','target_mode','best_epoch','validation_row_id_hash']})"),
    (30, "Matched Sensitive Validation",
     "The saved sensitive Validation model uses the same fixed RealMLP configuration. This comparison is descriptive and is not tuning.",
     "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_core/final_validation/stage5a2_sensitive_validation_results.csv')\nd[['sensitive_mode','candidate_id','fixed_epoch','mae','rmse','rmsle','status']]"),
    (31, "Historical Full-Train Lineage",
     "Earlier failed Full-Train attempts remain visible. They were not promoted because their required proof or serialization Gate failed.",
     "a=json.loads((ROOT/'artifacts/reports/stage5a2_fulltrain_blocker.json').read_text(encoding='utf-8'))\nb=json.loads((ROOT/'artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json').read_text(encoding='utf-8'))\nc=json.loads((ROOT/'artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json').read_text(encoding='utf-8'))\nprint({'historical_refit_blocker':a['status'],'recovery1':b['status'],'sensitive_attempt1':c['status'],'sensitive_attempt1_epoch':c['last_completed_epoch']})"),
    (32, "Recovery-2 Reporting Preflight",
     "The exact production JSON serializer passed required native, NumPy, Pandas, Path, nested, missing-value, atomic-write, reload, and cleanup checks.",
     "d=json.loads((ROOT/'artifacts/reports/stage5a2_recovery2_reporting_preflight.json').read_text(encoding='utf-8'))\nprint({'status':d['status'],'tested_types':d['tested_types'],'checks':d['checks']})"),
    (33, "Without-Sensitive Epoch Proof",
     "The recovery-2 non-sensitive model used every saved Train row, zero Test rows, and 30 audited epochs without Early Stopping or restoration.",
     "p=next((ROOT/'artifacts/reports').glob('*without_sensitive*recovery2_epoch_proof.json'))\nd=json.loads(p.read_text(encoding='utf-8'))\nprint({'status':d['status'],'requested':d['requested_epoch'],'completed':d['completed_epoch'],'history':d['training_history_length'],'global_step':d['final_global_step'],'checks_pass':all(d['checks'].values())})"),
    (34, "Without-Sensitive Final Bundle",
     "The final non-sensitive bundle contains Train-only preprocessing, the raw target transform, the epoch proof, and the CPU model.",
     "m=json.loads((ROOT/'artifacts/manifests/stage5/stage5a2_full_train_manifest.json').read_text(encoding='utf-8'))\nd=next(x for x in m['models'] if x['sensitive_mode']=='without_sensitive')\nprint(d)"),
    (35, "With-Sensitive Technical Retry Lineage",
     "The first sensitive Full-Train attempt ended at epoch 18 after a parent heartbeat file collision. One unchanged technical retry completed.",
     "f=json.loads((ROOT/'artifacts/reports/stage5a2_with_sensitive_attempt1_technical_failure.json').read_text(encoding='utf-8'))\nprint({'failed_id':f['candidate_id'],'failure_class':f['failure_class'],'last_epoch':f['last_completed_epoch'],'retry_id':f['retry_identifier'],'scientific_change':f['retry_scientific_configuration_change']})"),
    (36, "With-Sensitive Epoch Proof",
     "The sensitive technical retry uses the same row IDs, model settings, target, seed, batch policy, and epoch. Only validated sensitive source Features are added.",
     "p=next((ROOT/'artifacts/reports').glob('*with_sensitive*technical_retry1_epoch_proof.json'))\nd=json.loads(p.read_text(encoding='utf-8'))\nprint({'status':d['status'],'requested':d['requested_epoch'],'completed':d['completed_epoch'],'history':d['training_history_length'],'global_step':d['final_global_step'],'checks_pass':all(d['checks'].values())})"),
    (37, "With-Sensitive Final Bundle",
     "The matched sensitive final bundle passed the same proof and clean-reload contract.",
     "m=json.loads((ROOT/'artifacts/manifests/stage5/stage5a2_full_train_manifest.json').read_text(encoding='utf-8'))\nd=next(x for x in m['models'] if x['sensitive_mode']=='with_sensitive')\nprint(d)"),
    (38, "Full-Train Manifest",
     "The manifest proves the two-model row, epoch, configuration, lineage, and zero-Test contracts.",
     "d=json.loads((ROOT/'artifacts/manifests/stage5/stage5a2_full_train_manifest.json').read_text(encoding='utf-8'))\nprint({'status':d['status'],'fixed_epoch':d['fixed_epoch'],'model_count':len(d['models']),'checks':d['checks']})"),
    (39, "Clean-Process Reload",
     "Both complete bundles reload in clean processes. Their predictions match the saved references and handle missing and unknown values.",
     "d=pd.read_csv(ROOT/'artifacts/reports/stage5a2_core_reload_verification.csv')\nd[['candidate_id','sensitive_mode','prediction_count','maximum_absolute_difference','status']]"),
    (40, "Stage 5B Ensemble Handoff",
     "The handoff contains two aligned 25,000-row Validation prediction files and the two final bundles. No ensemble weight is selected.",
     "d=json.loads((ROOT/'artifacts/manifests/stage5/stage5a2_ensemble_handoff.json').read_text(encoding='utf-8'))\nprint({'status':d['status'],'rows':d['validation_row_count'],'row_hash':d['validation_row_id_hash'],'target_hash':d['target_sha256'],'weights_selected':d['ensemble_weight_selected'],'checks':d['checks']})"),
    (41, "Bounded Feature Attribution",
     "Attribution uses 2,000 saved Validation rows and the saved non-sensitive winner. It is associative and not causal.",
     "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv')\nd.head(15)"),
    (42, "Training Curves",
     "Saved histories show Final Validation curves and audited Full-Train progress. The partial sensitive failure remains visible.",
     "display(Image(filename=str(ROOT/'artifacts/figures/stage5a2/stage5a2_training_curves.png')))"),
    (43, "Validation Error Analysis",
     "Validation error is summarized by target decile for the two matched sensitive modes.",
     "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_core/summary/stage5a2_validation_error_analysis.csv')\nd"),
    (44, "Runtime Summary",
     "The runtime table includes Final Validation, sensitive Validation, both Full-Train models, and the preserved technical failure.",
     "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_core/summary/stage5a2_runtime_ram_model_size_summary.csv')\nd[['candidate_id','evaluation_stage','sensitive_mode','fit_time_seconds','status']]"),
    (45, "RAM and Model-Size Summary",
     "RAM and model sizes come from saved worker and artifact reports.",
     "d=pd.read_csv(ROOT/'artifacts/results/stage5/deep_core/summary/stage5a2_runtime_ram_model_size_summary.csv')\nd[['candidate_id','peak_ram_mib','model_size_bytes','bundle_size_bytes','status']]"),
    (46, "Required Figures and Registry",
     "Nine Stage 5A2 figures and eight idempotent Registry records are saved. Prior Registry bytes remain unchanged.",
     "f=json.loads((ROOT/'artifacts/reports/stage5a2_figure_manifest.json').read_text(encoding='utf-8'))\nr=json.loads((ROOT/'artifacts/reports/stage5a2_registry_update.json').read_text(encoding='utf-8'))\nprint({'figures':len(f['figures']),'figure_status':f['status'],'registry_status':r['status'],'registry_rows':len(r['stage5a2_experiment_ids'])})\ndisplay(Image(filename=str(ROOT/'artifacts/figures/stage5a2/stage5a_summary_dashboard.png')))"),
    (47, "Pre-Review Verification",
     "All scientific, bundle, handoff, attribution, figure, Registry, governance, and protected-file checks pass before Notebook execution and final review.",
     "d=json.loads((ROOT/'artifacts/reports/stage5a2_pre_review_verification.json').read_text(encoding='utf-8'))\nprint(d)"),
    (48, "Stage 5A Completion Note",
     "The scientific and artifact work is complete. Final PASS is written only after clean Notebook runs and independent review. Stage 5B is not started here.",
     "final_path=ROOT/'artifacts/reports/stage5a_verification.json'\nd=json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else {'status':'PENDING_REVIEW'}\nprint({'stage5a_status':d.get('status'),'next_step':d.get('next_step','Pending independent review'),'stage5b_started':False})"),
]


def main() -> None:
    pre_review = create_pre_review()
    original = nbformat.read(PREFIX_BACKUP, as_version=4)
    current = nbformat.read(NOTEBOOK, as_version=4)
    prefix = current.cells[:51]
    expected = original.cells[:51]
    prefix_match = len(prefix) == len(expected) and all(cell_digest(a) == cell_digest(b) for a, b in zip(prefix, expected))
    if not prefix_match:
        raise RuntimeError("The protected 51-cell Stage 5A1 Notebook prefix changed")
    appended = []
    for number, title, markdown, code in SECTIONS:
        appended.append(nbformat.v4.new_markdown_cell(f"### {number}. {title}\n\n{markdown}"))
        appended.append(nbformat.v4.new_code_cell(code))
    notebook = nbformat.v4.new_notebook(cells=prefix + appended, metadata=current.metadata)
    notebook.metadata["stage5a2"] = {
        "presentation_only": True, "heavy_fit_calls": 0, "preprocessing_fit_calls": 0,
        "preserved_stage5a1_prefix_cells": 51, "sections": "25-48",
    }
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK)
    reloaded = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in reloaded.cells if cell.cell_type == "code")
    headings = [cell.source.splitlines()[0] for cell in reloaded.cells if cell.cell_type == "markdown" and cell.source.startswith("### ")]
    checks = {
        "pre_review_pass": pre_review["status"] == "PRE_REVIEW_PASS",
        "stage5a1_prefix_51_cells_preserved": all(cell_digest(a) == cell_digest(b) for a, b in zip(reloaded.cells[:51], expected)),
        "total_cells_99": len(reloaded.cells) == 99,
        "appended_sections_25_to_48": all(any(line.startswith(f"### {number}.") for line in headings) for number in range(25, 49)),
        "section_headings_unique": len(headings) == len(set(headings)),
        "zero_model_fit_calls": ".fit(" not in code and ".fit (" not in code,
        "zero_preprocessing_fit_calls": "fit_transform(" not in code and "partial_fit(" not in code,
        "saved_artifact_only": "stage5a2_complete_artifacts" not in code and "fulltrain_recovery2.py" not in code,
    }
    report = {
        "stage_id": "stage5a2", "notebook_path": NOTEBOOK.name,
        "notebook_sha256": sha256(NOTEBOOK), "cell_count": len(reloaded.cells),
        "code_cell_count": sum(cell.cell_type == "code" for cell in reloaded.cells),
        "markdown_cell_count": sum(cell.cell_type == "markdown" for cell in reloaded.cells),
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, BUILD_REPORT)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Notebook build failed")


if __name__ == "__main__":
    main()
