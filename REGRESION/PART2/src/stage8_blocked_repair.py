"""Preserve valid Stage 8 evidence and document the two blocking defects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stage8_explainability_utils import CANDIDATES, EXPECTED, MANIFESTS, MODELS, PREDICTIONS, REGISTRY, REPORTS, RESULTS, ROOT, dump, now, record, sha


def repair_local_public() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local = pd.read_csv(RESULTS / "stage8_local_attributions_public.csv")
    cases = pd.read_csv(ROOT / "artifacts/results/stage7/fairness/stage7_representative_cases_public.csv")
    meta = pd.read_csv(MANIFESTS / "stage8_local_case_manifest.csv")
    cases = cases.merge(meta[["row_id", "semantic_case_type"]], on="row_id", how="inner")
    case_id = cases.semantic_case_type + "__" + cases.case_rank.astype(int).astype(str)
    cases["case_public_id"] = case_id
    maps = {
        CANDIDATES[0]: ("Stage 4L official blend", "official pre-registered primary", "candidate_1_prediction", "candidate_1_signed_error"),
        CANDIDATES[1]: ("RealMLP without sensitive", "Post-Test descriptive extension", "candidate_2_prediction", "candidate_2_signed_error"),
        CANDIDATES[2]: ("RealMLP with sensitive", "Post-Test descriptive sensitive companion", "candidate_3_prediction", "candidate_3_signed_error"),
    }
    additions = []
    lookup = cases.set_index("case_public_id")
    for row in local.itertuples():
        c = lookup.loc[row.case_public_id]; label, role, pred_col, err_col = maps[row.candidate_id]
        additions.append({"row_id": int(c.row_id), "candidate_label": label, "official_role": role, "y_true": float(c.y_true), "frozen_prediction": float(c[pred_col]), "signed_error": float(c[err_col])})
    add = pd.DataFrame(additions)
    local["row_id"] = add.row_id; local["candidate_label"] = add.candidate_label; local["official_role"] = add.official_role; local["y_true"] = add.y_true; local["frozen_prediction"] = add.frozen_prediction; local["signed_error"] = add.signed_error
    local["effect_mean"] = local.reference_substitution_effect
    local["effect_standard_deviation"] = np.nan; local["effect_minimum"] = np.nan; local["effect_maximum"] = np.nan; local["mean_absolute_effect"] = np.nan
    local["direction"] = local.effect_direction; local["absolute_effect_rank"] = local.within_case_candidate_rank
    local["sensitive_block_flag"] = local.semantic_feature_unit.isin(["explicit_sensitive_identity_block", "sensitive_context_block"])
    proxy_families = {"Geography and region", "Lender, respondent, and agency", "Applicant income", "Tract and area income context", "Property and occupancy", "Loan purpose and type", "Loan structure and lien status"}
    local["potential_proxy_category_flag"] = local.feature_family.isin(proxy_families)
    local["analysis_label"] = "Post-Test Explainability and Feature Interpretation"
    local["limitation"] = "Reference substitution is not SHAP, additive, or causal. Dispersion fields are unavailable because the initial bounded batch did not persist per-reference predictions; Stage 8 is BLOCKED."
    columns = ["case_type","case_rank","row_id","case_public_id","candidate_id","candidate_label","official_role","y_true","frozen_prediction","signed_error","semantic_feature_unit","feature_family","effect_mean","effect_standard_deviation","effect_minimum","effect_maximum","mean_absolute_effect","direction","absolute_effect_rank","sensitive_block_flag","potential_proxy_category_flag","method","analysis_label","limitation","raw_sensitive_value_public"]
    local[columns].to_csv(RESULTS / "stage8_local_attributions_public.csv", index=False)

    detail = pd.read_csv(RESULTS / "stage8_local_explanation_stability.csv")
    detail.to_csv(RESULTS / "stage8_local_explanation_stability_detail.csv", index=False)
    summary = []
    for (cid, candidate), group in detail.groupby(["case_public_id", "candidate_id"]):
        rho = spearmanr(group.effect_background_half_a, group.effect_background_half_b).statistic
        topa5 = set(group.assign(a=group.effect_background_half_a.abs()).nlargest(5, "a").semantic_feature_unit); topb5 = set(group.assign(b=group.effect_background_half_b.abs()).nlargest(5, "b").semantic_feature_unit)
        topa10 = set(group.assign(a=group.effect_background_half_a.abs()).nlargest(10, "a").semantic_feature_unit); topb10 = set(group.assign(b=group.effect_background_half_b.abs()).nlargest(10, "b").semantic_feature_unit)
        summary.append({"case_public_id":cid,"candidate_id":candidate,"spearman_rank_correlation":rho,"top_5_overlap":len(topa5&topb5),"top_10_overlap":len(topa10&topb10),"median_absolute_effect_difference":group.absolute_difference.median(),"maximum_absolute_effect_difference":group.absolute_difference.max(),"background_a_rows":20,"background_b_rows":20,"low_stability_flag":bool(pd.isna(rho) or rho<0.7),"method":"background-half local reference-substitution stability","limitation":"Four frozen visualization cases only; not causal."})
    stable = pd.DataFrame(summary); stable.to_csv(RESULTS / "stage8_local_explanation_stability.csv", index=False)

    public = pd.read_csv(RESULTS / "stage8_local_attributions_public.csv")
    synth = []
    for cid, group in public.groupby("case_public_id"):
        c = lookup.loc[cid]; tops = {candidate: g.nsmallest(5, "absolute_effect_rank").semantic_feature_unit.tolist() for candidate, g in group.groupby("candidate_id")}
        sets = [set(tops.get(candidate, [])) for candidate in CANDIDATES]; consensus = sorted(set.intersection(*sets)); disagreement = sorted(set.union(*sets) - set(consensus))
        signed = [float(c.candidate_1_signed_error), float(c.candidate_2_signed_error), float(c.candidate_3_signed_error)]
        s = stable[stable.case_public_id == cid]
        synth.append({"case_type":c.case_type,"case_rank":int(c.case_rank),"row_id":int(c.row_id),"case_public_id":cid,"y_true":float(c.y_true),"stage4l_prediction":float(c.candidate_1_prediction),"realmlp_without_prediction":float(c.candidate_2_prediction),"realmlp_with_prediction":float(c.candidate_3_prediction),"stage4l_signed_error":signed[0],"realmlp_without_signed_error":signed[1],"realmlp_with_signed_error":signed[2],"stage4l_top_feature_units":"|".join(tops.get(CANDIDATES[0],[])),"realmlp_without_top_feature_units":"|".join(tops.get(CANDIDATES[1],[])),"realmlp_with_top_feature_units":"|".join(tops.get(CANDIDATES[2],[])),"consensus_feature_units":"|".join(consensus),"disagreement_feature_units":"|".join(disagreement),"all_models_underpredict":all(x<0 for x in signed),"all_models_overpredict":all(x>0 for x in signed),"sensitive_mode_changes_top_explanation":tops.get(CANDIDATES[1],[None])[0]!=tops.get(CANDIDATES[2],[None])[0],"local_stability_summary":"not a visualization case" if s.empty else f"minimum Spearman={s.spearman_rank_correlation.min():.3f}; low_stability={bool(s.low_stability_flag.any())}","interpretation":"The models associate this prediction with the listed Feature units and may emphasize different signals.","limitation":"Reference substitution is not SHAP, additive, realistic intervention evidence, or causality."})
    synthesis = pd.DataFrame(synth); synthesis.to_csv(RESULTS / "stage8_case_explanation_synthesis.csv", index=False)
    return local, stable, synthesis


def repair_proxy() -> None:
    p=pd.read_csv(RESULTS/"stage8_potential_proxy_overlap.csv"); p["candidate"]=p.candidate_id; p["feature"]=p.semantic_feature_unit; p["importance_rank"]=p["rank"]; p["importance_share"]=p.positive_importance_normalized_share; p["potential_proxy_category"]=p.feature_family; p["limitation"]="Potential proxy category only; not a confirmed proxy, causal finding, or removal recommendation."; p.to_csv(RESULTS/"stage8_potential_proxy_overlap.csv",index=False)


def summaries(local: pd.DataFrame, stability: pd.DataFrame, synthesis: pd.DataFrame) -> None:
    perm=pd.read_csv(RESULTS/"stage8_common_permutation_importance.csv"); family=pd.read_csv(RESULTS/"stage8_feature_family_summary.csv"); cross=pd.read_csv(RESULTS/"stage8_cross_model_agreement.csv"); cm=pd.read_csv(RESULTS/"stage8_cross_method_agreement.csv"); sens=pd.read_csv(RESULTS/"stage8_sensitive_feature_dependence.csv"); proxy=pd.read_csv(RESULTS/"stage8_potential_proxy_overlap.csv"); inv=pd.read_csv(RESULTS/"stage8_explainability_inventory.csv")
    summary={"stage_status":"BLOCKED","analysis_label":"Post-Test Explainability and Feature Interpretation","stage4l_official_status":"official pre-registered primary unchanged","candidate_ids":CANDIDATES,"model_identities":MODELS,"existing_importance_summary":{"artifact_count":int((inv.method=="importance").sum()),"modes":["without_sensitive","with_sensitive"],"status":"PASS"},"existing_shap_summary":{"artifact_count":6,"maximum_rows":300,"output_scales":["log1p model output","raw original target scale"],"recomputation_count":0,"status":"PASS"},"existing_deep_attribution_summary":{"path":"artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv","sample_rows":2000,"sample_role":"Train-only","status":"PASS"},"common_permutation_summary":{"candidate_count":3,"sample_rows":2000,"repeats":2,"seeds":[42,43],"original_target_scale":True,"status":"PASS"},"top_features_by_candidate":{c:perm[perm.candidate_id==c].nsmallest(10,"rank").semantic_feature_unit.tolist() for c in CANDIDATES},"top_feature_families_by_candidate":{c:family[family.candidate_id==c].nlargest(5,"positive_permutation_importance_share").feature_family.tolist() for c in CANDIDATES},"cross_model_rank_agreement":cross.to_dict("records"),"cross_method_agreement":cm.to_dict("records"),"sensitive_identity_importance_share":sens[sens.feature_or_block=="explicit_sensitive_identity_block"][["model_family","evidence_method","positive_importance_normalized_share"]].to_dict("records"),"sensitive_context_importance_share":sens[sens.feature_or_block=="sensitive_context_block"][["model_family","evidence_method","positive_importance_normalized_share"]].to_dict("records"),"potential_proxy_overlap":proxy.groupby("candidate_id").size().astype(int).to_dict(),"global_explanation_limitations":["Stage 8 is Post-Test and descriptive.","Importance is not causality.","Correlated Features can divide or mask importance.","Native SHAP output scales differ.","Sensitive importance is not fairness or discrimination evidence."],"causal_conclusion":"none","model_selection_effect":"none","blockers":["Frozen Registry 378-row raw byte prefix was reserialized and the original bytes are unavailable.","Required local per-reference dispersion was not persisted before both permitted source attempts were exhausted."]}
    dump(summary,RESULTS/"stage8_global_explanation_summary.json")
    sections=[("Scope","Post-Test descriptive explanation of frozen model behavior."),("Models explained","Three Candidate predictors and five underlying frozen models."),("Methods used","Reused native importance, reused native SHAP, saved Deep attribution, grouped permutation, and local reference substitution."),("Methods not used","No global SHAP recomputation, full-Test SHAP, LIME, surrogate, fitting, tuning, or selection."),("Official blend global explanation","Original-scale grouped permutation is primary; exact native blend SHAP is unavailable."),("RealMLP global explanation","Saved Train-only attribution and bounded Stage 8 permutation are compared with their sample-role difference visible."),("Sensitive versus non-sensitive interpretation","Sensitive identity and context are aggregate-only and do not establish fairness."),("Cross-model similarities","Applicant income and lien status are common leading units."),("Cross-model differences","Geography, income context, and occupancy ranks differ across predictors."),("Cross-method agreement and disagreement","Rank agreement varies; disagreement is a finding, not an error."),("Local case findings","Models often emphasize different Feature units for the same frozen case."),("Explanation stability","Four visualization cases have background-half evidence; low stability remains visible."),("Potential proxy limitations","Geography, lender, income, property, and loan categories are potential proxies only."),("Fairness limitations","Sensitive importance neither proves nor disproves fairness, discrimination, or compliance."),("Causal limitations","Importance and substitution effects explain model behavior, not the data-generating process."),("Post-Test disclosure","Test was already consumed; this Stage cannot select or promote a model."),("Inputs for Stage 9","Stage 9 must reuse saved outputs and must not rerun inference or explanation.")]
    text=["# Stage 8 Feature Interpretation Report","","> Status: BLOCKED. Valid scientific artifacts are preserved, but the Registry prefix and local-dispersion delivery contracts are unresolved.",""]
    for i,(title,body) in enumerate(sections,1): text.extend([f"## {i}. {title}","",body,""])
    (RESULTS/"stage8_feature_interpretation_report.md").write_text("\n".join(text),encoding="utf-8")


def handoff_and_incident() -> None:
    refs={"existing_explainability_inventory":"artifacts/results/stage8/explainability/stage8_explainability_inventory.csv","existing_shap_validation":"artifacts/reports/stage8_existing_shap_validation.json","common_permutation":"artifacts/results/stage8/explainability/stage8_common_permutation_importance.csv","cross_model_comparison":"artifacts/results/stage8/explainability/stage8_cross_model_feature_comparison.csv","cross_method_comparison":"artifacts/results/stage8/explainability/stage8_cross_method_agreement.csv","feature_family_summary":"artifacts/results/stage8/explainability/stage8_feature_family_summary.csv","sensitive_feature_dependence":"artifacts/results/stage8/explainability/stage8_sensitive_feature_dependence.csv","potential_proxy_overlap":"artifacts/results/stage8/explainability/stage8_potential_proxy_overlap.csv","public_local_attribution":"artifacts/results/stage8/explainability/stage8_local_attributions_public.csv","case_synthesis":"artifacts/results/stage8/explainability/stage8_case_explanation_synthesis.csv","explanation_stability":"artifacts/results/stage8/explainability/stage8_local_explanation_stability.csv","global_summary":"artifacts/results/stage8/explainability/stage8_global_explanation_summary.json","feature_interpretation_report":"artifacts/results/stage8/explainability/stage8_feature_interpretation_report.md","visualization_manifest":"artifacts/manifests/stage8/stage8_visualization_manifest.json"}
    handoff={"stage_id":"stage8","status":"BLOCKED","analysis_label":"Post-Test Explainability and Feature Interpretation","stage4l_official_candidate_id":CANDIDATES[0],"stage4l_official_role_unchanged":True,"candidate_predictor_ids":CANDIDATES,"underlying_model_identities":MODELS,"prediction_artifacts":PREDICTIONS,"test_row_count":EXPECTED["rows"],"test_row_id_hash":EXPECTED["row_hash"],"target_hash":EXPECTED["target_hash"],"artifacts":{k:record(v) for k,v in refs.items()},"stage6_representative_cases":record("artifacts/results/stage6/error_analysis/stage6_representative_cases.csv"),"stage7_public_representative_cases":record("artifacts/results/stage7/fairness/stage7_representative_cases_public.csv"),"stage7_fairness_summary":record("artifacts/results/stage7/fairness/stage7_fairness_summary.json"),"stage7_restricted_data_public_exposure_prohibited":True,"existing_shap_recomputation_count":0,"model_fit_count":0,"preprocessing_fit_count":0,"new_candidate_prediction_count":0,"model_selection_performed":False,"causal_explanation_performed":False,"fairness_certification_performed":False,"stage9_must_use_saved_stage8_artifacts":True,"stage9_must_not_rerun_explainability":True,"stage9_must_not_rerun_model_inference":True,"stage9_must_preserve_post_test_labels":True,"recommended_final_report_visuals":[2,5,10,11,14,15],"recommended_model_card_explainability_statements":["Stage 4L remains official.","Importance is not causality.","Sensitive importance is not fairness certification.","Native SHAP scales differ.","Local substitution is non-additive."],"next_stage":"BLOCKED — do not begin Stage 9","stage9_started":False,"blockers":["Exact pre-Stage8 Registry byte prefix is unavailable after full serialization.","A third bounded source read needs explicit human approval to recover required local dispersion statistics."]}
    dump(handoff,MANIFESTS/"stage8_stage9_handoff.json")
    baseline=json.loads((MANIFESTS/"stage8_protected_hashes_before.json").read_text(encoding="utf-8")); current=REGISTRY.read_bytes(); prefix=current[:baseline["entries"][[x["path"] for x in baseline["entries"]].index("artifacts/results/experiment_results.csv")]["size_bytes"]]
    incident={"stage_id":"stage8","status":"BLOCKED","detected_at_utc":now(),"protected_registry_expected_sha256":baseline["registry_sha256_before"],"protected_registry_expected_size":next(x["size_bytes"] for x in baseline["entries"] if x["path"]=="artifacts/results/experiment_results.csv"),"current_prefix_sha256":__import__("hashlib").sha256(prefix).hexdigest(),"prior_semantic_experiment_ids_preserved":pd.read_csv(REGISTRY).experiment_id.astype(str).tolist()[:len(baseline["registry_ids_before"])]==baseline["registry_ids_before"],"cause":"Stage 8 pandas full-file serialization instead of raw append.","exact_raw_recovery_available":False,"stage8_rows_present":8,"stage4l_role_changed":False,"required_resolution":"Provide the exact pre-Stage8 Registry bytes or explicitly adjudicate the protected-byte incident; do not claim append safety."}
    dump(incident,REPORTS/"stage8_registry_prefix_incident.json")


if __name__=="__main__":
    local,stable,synthesis=repair_local_public(); repair_proxy(); summaries(local,stable,synthesis); handoff_and_incident(); print("BLOCKED evidence preserved")
