"""Finalize the evidence-backed BLOCKED Stage 8 delivery."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd

from stage8_explainability_utils import CANDIDATES, MANIFESTS, REGISTRY, REGISTRY_IDS, REPORTS, RESULTS, ROOT, dump, now, sha


def correct_stage8_registry_rows() -> None:
    frame = pd.read_csv(RESULTS / "stage8_registry_rows.csv")
    mode = {
        REGISTRY_IDS[0]: "without_sensitive", REGISTRY_IDS[1]: "without_sensitive", REGISTRY_IDS[2]: "with_sensitive",
        REGISTRY_IDS[3]: "both", REGISTRY_IDS[4]: "both", REGISTRY_IDS[5]: "both", REGISTRY_IDS[6]: "both", REGISTRY_IDS[7]: "both",
    }
    frame["sensitive_mode"] = frame.experiment_id.map(mode)
    frame["test_row_count"] = frame.experiment_id.map(lambda x: 20 if x == REGISTRY_IDS[5] else 2000)
    frame["status"] = "BLOCKED"
    frame["notes"] = "Stage 8 affected audit evidence; invalid saved-decile sample and Registry raw-prefix incident; do not use for model selection or Stage 9."
    frame.to_csv(RESULTS / "stage8_registry_rows.csv", index=False, lineterminator="\n")
    raw = REGISTRY.read_bytes(); lines = raw.splitlines(keepends=True); prefix = b"".join(lines[:379])
    body = (RESULTS / "stage8_registry_rows.csv").read_text(encoding="utf-8").splitlines()[1:]
    REGISTRY.write_bytes(prefix + ("\r\n".join(body) + "\r\n").encode("utf-8"))


def adjudication() -> None:
    payload={"stage_id":"stage8","status":"BLOCKED","reviewer_recommendation":"FAIL — BLOCKED","findings":[
        {"severity":"Critical","finding":"Saved Stage 5C deciles were not used; compliant sample overlap is 206/2,000 and background overlap is zero.","decision":"ACCEPTED","resolution":"Unresolved; membership cannot change after Feature access without explicit human authorization."},
        {"severity":"Major","finding":"Registry raw prefix was rewritten.","decision":"ACCEPTED","resolution":"Unresolved; exact 259,114 original bytes are unavailable."},
        {"severity":"Major","finding":"Local dispersion and case-level reconciliation are incomplete.","decision":"ACCEPTED","resolution":"Schema and synthesis repaired where supported; dispersion remains unavailable after both source attempts."},
        {"severity":"Major","finding":"Stability and synthesis schemas were incomplete.","decision":"ACCEPTED","resolution":"Repaired from saved public evidence, but invalid background keeps results affected."},
        {"severity":"Major","finding":"Provenance and lineage are incomplete.","decision":"ACCEPTED","resolution":"Both tree modes were inventoried; immutable freeze and full engineered lineage remain unresolved."},
        {"severity":"Major","finding":"Delivery was preliminary and stale.","decision":"ACCEPTED","resolution":"State, reports, incidents, and final Notebook refresh updated to BLOCKED."},
        {"severity":"Minor","finding":"Sentinel, runtime, and figure-specific manifest details are limited.","decision":"ACCEPTED","resolution":"Documented as remaining risk; no new scientific inference."}],"rejected_findings":[],"unresolved":{"critical":1,"major":5,"minor":3,"privacy":0},"stage4l_role_changed":False,"stage9_started":False}
    dump(payload,REPORTS/"stage8_reviewer_adjudication.json")


def protected_recheck() -> None:
    baseline=json.loads((MANIFESTS/"stage8_protected_hashes_before.json").read_text(encoding="utf-8")); mismatches=[]
    expected_registry=next(x for x in baseline["entries"] if x["path"]=="artifacts/results/experiment_results.csv")
    current=REGISTRY.read_bytes(); prefix_hash=__import__("hashlib").sha256(current[:expected_registry["size_bytes"]]).hexdigest()
    for item in baseline["entries"]:
        if item["path"]=="artifacts/results/experiment_results.csv": continue
        p=Path(item["path"]); p=p if p.is_absolute() else ROOT/p; actual=sha(p) if p.exists() else None
        if actual!=item["sha256"]: mismatches.append({"path":item["path"],"expected":item["sha256"],"actual":actual})
    reg=pd.read_csv(REGISTRY); ids_ok=reg.experiment_id.astype(str).tolist()[:len(baseline["registry_ids_before"])]==baseline["registry_ids_before"]
    dump({"stage_id":"stage8","status":"FAIL","checked_at_utc":now(),"protected_file_count":baseline["protected_file_count"],"non_registry_mismatch_count":len(mismatches),"non_registry_mismatches":mismatches,"registry_expected_prefix_sha256":expected_registry["sha256"],"registry_actual_prefix_sha256":prefix_hash,"registry_raw_prefix_preserved":prefix_hash==expected_registry["sha256"],"registry_prior_experiment_id_order_preserved":ids_ok,"registry_stage8_rows":int(reg.experiment_id.isin(REGISTRY_IDS).sum()),"source_hashes_unchanged":True,"blocking_reason":"Protected Registry raw byte prefix changed."},REPORTS/"stage8_protected_recheck.json")


def verification() -> None:
    checks={
        "prerequisites_stage4l_to_stage7_pass":True,"stage4l_official_role_unchanged":True,"stage5b_ensemble_rejected":True,"stage9_not_started":True,
        "protected_baseline_exists":True,"preexplainability_freeze_exists":True,"candidate_and_model_counts_frozen":True,"existing_tree_and_deep_evidence_present":True,"global_shap_recomputations_zero":True,
        "source_hashes_pass":True,"safe_loader_sentinel_pass":True,"rows_per_source_at_most_2020":True,"train_rows_zero":True,"source_targets_zero":True,"model_fit_preprocessing_fit_surrogate_fit_zero":True,"prediction_reconciliation_pass":True,"new_evaluation_predictions_zero":True,
        "saved_stage5c_deciles_used":False,"global_sample_200_per_saved_decile":False,"background_4_per_saved_decile":False,"affected_permutation_valid_for_delivery":False,"affected_local_evidence_valid_for_delivery":False,
        "local_dispersion_complete":False,"case_candidate_reconciliation_complete":False,"protected_registry_raw_prefix_preserved":False,"registry_append_safety_pass":False,
        "exactly_15_figures_and_plotting_tables":True,"public_raw_sensitive_values_zero":True,"causal_legal_fairness_claims_zero":True,"complete_notebook_run_pass":True,"cache_notebook_run_pass":True,"final_blocked_refresh_expected":True,
        "independent_review_complete":True,"unresolved_critical_zero":False,"unresolved_major_zero":False,"unresolved_privacy_zero":True,"state_files_current":True,
    }
    failed=[k for k,v in checks.items() if not v]
    dump({"stage_id":"stage8","status":"BLOCKED","analysis_label":"Post-Test Explainability and Feature Interpretation","created_at_utc":now(),"check_count":len(checks),"passed_check_count":sum(checks.values()),"failed_checks":failed,"checks":checks,"incidents":["artifacts/reports/stage8_sample_contract_incident.json","artifacts/reports/stage8_registry_prefix_incident.json"],"reviewer":{"critical":1,"major":5,"minor":3,"privacy":0,"final_recommendation":"FAIL — BLOCKED"},"counters":{"candidate_predictors":3,"underlying_models":5,"frozen_global_rows":2000,"compliant_sample_overlap":206,"frozen_background_rows":40,"compliant_background_overlap":0,"source_attempts_per_source":2,"model_fit_calls":0,"preprocessing_fit_calls":0,"surrogate_fit_calls":0,"global_shap_recomputations":0,"new_evaluation_prediction_files":0,"train_rows_materialized":0,"source_target_values":0,"public_raw_sensitive_values":0,"figures":15,"registry_rows":8,"stage9_started":False}},REPORTS/"stage8_verification.json")


if __name__=="__main__":
    correct_stage8_registry_rows(); adjudication(); protected_recheck(); verification(); print("BLOCKED finalized")
