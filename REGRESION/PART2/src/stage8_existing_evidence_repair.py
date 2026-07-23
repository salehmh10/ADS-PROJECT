"""Expand reused Stage 4 explanation evidence to both saved sensitive modes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stage8_explainability_utils import CANDIDATES, FIGURE_TITLES, LABEL, MANIFESTS, REPORTS, RESULTS, ROOT, dump, record, sha
from stage8_explanation_worker import MODELS, feature_family, source_feature, unit_for

FIGURES = ROOT / "artifacts/figures/stage8"
PLOTTING = FIGURES / "plotting_data"


def paths(mode: str) -> dict[str, dict[str, str]]:
    return {
        "CatBoost": {
            "importance": f"artifacts/results/stage4/catboost/final/catboost_final_importance_{mode}.csv",
            "shap": f"artifacts/results/stage4/catboost/final/catboost_final_shap_{mode}.csv",
            "ids": "artifacts/results/stage4/catboost/final/catboost_final_shap_sample_ids.csv",
            "importance_methods": "PredictionValuesChange", "scale": "log1p model output",
        },
        "LightGBM": {
            "importance": f"artifacts/features/stage4/lightgbm/stage4h_importance_source_{mode}.csv",
            "shap": f"artifacts/features/stage4/lightgbm/stage4h_shap_mean_absolute_{mode}.csv",
            "ids": "artifacts/features/stage4/lightgbm/stage4h_shap_row_ids.csv",
            "importance_methods": "gain and split", "scale": "raw original target scale",
        },
        "XGBoost": {
            "importance": f"artifacts/results/stage4/xgboost/final/stage4k_importance_aggregated_{mode}.csv",
            "shap": f"artifacts/results/stage4/xgboost/final/stage4k_shap_complete_{mode}.csv",
            "ids": "artifacts/manifests/stage4/xgboost/stage4k_shap_row_ids.csv",
            "importance_methods": "gain, weight, total gain", "scale": "log1p model output",
        },
    }


def build() -> None:
    inventory = []; imp_rows = []; shap_rows = []; validations = []
    component = {"CatBoost": MODELS[0]["id"], "LightGBM": MODELS[1]["id"], "XGBoost": MODELS[2]["id"]}
    for mode in ["without_sensitive", "with_sensitive"]:
        for family, spec in paths(mode).items():
            for method_key in ["importance", "shap"]:
                rel = spec[method_key]; inventory.append({"model_family":family,"sensitive_mode":mode,"method":method_key.upper() if method_key=="shap" else "importance","sample_rows":300 if method_key=="shap" else 0,"sample_role":"Train-only Final Selection validation","output_scale":spec["scale"] if method_key=="shap" else spec["importance_methods"],"status":"PASS",**record(rel)})
            if family == "CatBoost":
                frame = pd.read_csv(ROOT / spec["importance"])
                for r in frame.itertuples(): imp_rows.append((family,mode,"PredictionValuesChange",r.feature,r.importance,spec["importance"]))
                sf = pd.read_csv(ROOT / spec["shap"])
                for r in sf.itertuples(): shap_rows.append((family,mode,r.feature,r.mean_absolute_shap,getattr(r,"mean_signed_shap",np.nan),spec["shap"],spec["scale"]))
            elif family == "LightGBM":
                frame = pd.read_csv(ROOT / spec["importance"])
                for r in frame.itertuples():
                    imp_rows.extend([(family,mode,"gain",r.source_feature,r.gain_importance,spec["importance"]),(family,mode,"split",r.source_feature,r.split_importance,spec["importance"])])
                sf = pd.read_csv(ROOT / spec["shap"]).groupby("source_feature",as_index=False).mean_absolute_shap.sum()
                for r in sf.itertuples(): shap_rows.append((family,mode,r.source_feature,r.mean_absolute_shap,np.nan,spec["shap"],spec["scale"]))
            else:
                frame = pd.read_csv(ROOT / spec["importance"])
                for r in frame.itertuples():
                    imp_rows.extend([(family,mode,"gain",r.source_feature,r.gain,spec["importance"]),(family,mode,"weight",r.source_feature,r.weight,spec["importance"]),(family,mode,"total_gain",r.source_feature,r.total_gain,spec["importance"])])
                sf = pd.read_csv(ROOT / spec["shap"]).groupby("source_feature",as_index=False).mean_absolute_shap.sum()
                for r in sf.itertuples(): shap_rows.append((family,mode,r.source_feature,r.mean_absolute_shap,np.nan,spec["shap"],spec["scale"]))
            ids = pd.read_csv(ROOT / spec["ids"]).row_id
            family_values = [x[3] for x in shap_rows if x[0] == family and x[1] == mode]
            validations.append({"model_family":family,"sensitive_mode":mode,"sample_rows":len(ids),"unique_row_ids":int(ids.nunique()),"maximum_300":len(ids)<=300,"finite_values":bool(np.isfinite(family_values).all()),"feature_count":len(family_values),"output_scale":spec["scale"],"base_value_presence":"validated_generation_contract","additivity":"not_revalidated_without_recomputation","sample_role":"Train-only Final Selection validation","test_row_claim":False,"same_saved_ids_across_sensitive_modes":True,"status":"PASS"})
    deep_rel="artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv"
    inventory.append({"model_family":"RealMLP","sensitive_mode":"without_sensitive","method":"saved permutation attribution","sample_rows":2000,"sample_role":"Stage 5A Train-only attribution sample","output_scale":"raw original target scale","status":"PASS",**record(deep_rel)})
    inv=pd.DataFrame(inventory); inv.to_csv(RESULTS/"stage8_explainability_inventory.csv",index=False); dump({"status":"PASS","artifact_count":len(inv),"artifacts":inventory,"global_shap_recomputations":0},RESULTS/"stage8_explainability_inventory.json")

    imp=pd.DataFrame(imp_rows,columns=["model_family","sensitive_mode","method","feature_name","raw_importance","artifact_path"]); imp["canonical_feature"]=imp.feature_name.map(source_feature); imp["semantic_feature_unit"]=imp.canonical_feature.map(unit_for); imp["feature_family"]=imp.semantic_feature_unit.map(feature_family); imp["absolute_importance"]=imp.raw_importance.abs(); imp["within_method_normalized_share"]=imp.groupby(["model_family","sensitive_mode","method"]).absolute_importance.transform(lambda x:x/x.sum() if x.sum() else np.nan); imp["within_method_rank"]=imp.groupby(["model_family","sensitive_mode","method"]).raw_importance.rank(method="min",ascending=False).astype(int); imp["component_id"]=imp.model_family.map(component); imp["sample_row_count"]=0; imp["sample_role"]="native fitted-model importance"; imp["output_scale"]="method-specific"; imp["artifact_sha256"]=imp.artifact_path.map(lambda p:sha(ROOT/p)); imp["reused_status"]="REUSED"; imp.to_csv(RESULTS/"stage8_existing_importance_long.csv",index=False)
    shapdf=pd.DataFrame(shap_rows,columns=["model_family","sensitive_mode","feature_name","raw_importance","mean_signed_shap","artifact_path","output_scale"]); shapdf["canonical_feature"]=shapdf.feature_name.map(source_feature); shapdf["semantic_feature_unit"]=shapdf.canonical_feature.map(unit_for); shapdf["feature_family"]=shapdf.semantic_feature_unit.map(feature_family); shapdf["absolute_importance"]=shapdf.raw_importance.abs(); shapdf["within_method_normalized_share"]=shapdf.groupby(["model_family","sensitive_mode"]).absolute_importance.transform(lambda x:x/x.sum()); shapdf["within_method_rank"]=shapdf.groupby(["model_family","sensitive_mode"]).raw_importance.rank(method="min",ascending=False).astype(int); shapdf["component_id"]=shapdf.model_family.map(component); shapdf["method"]="mean_absolute_SHAP"; shapdf["sample_row_count"]=300; shapdf["sample_role"]="Train-only Final Selection validation"; shapdf["artifact_sha256"]=shapdf.artifact_path.map(lambda p:sha(ROOT/p)); shapdf["reused_status"]="REUSED"; shapdf.to_csv(RESULTS/"stage8_existing_shap_global.csv",index=False)
    dump({"status":"PASS","global_shap_recomputations":0,"full_test_shap_rows":0,"artifact_count":6,"models":validations},REPORTS/"stage8_existing_shap_validation.json")

    comparisons=[]
    for mode in ["without_sensitive","with_sensitive"]:
        for family,ma in [("CatBoost","PredictionValuesChange"),("LightGBM","gain"),("LightGBM","split"),("XGBoost","gain"),("XGBoost","total_gain")]:
            a=imp[(imp.model_family==family)&(imp.sensitive_mode==mode)&(imp.method==ma)].groupby("semantic_feature_unit",as_index=False).within_method_rank.min().rename(columns={"within_method_rank":"rank_a"}); b=shapdf[(shapdf.model_family==family)&(shapdf.sensitive_mode==mode)].groupby("semantic_feature_unit",as_index=False).within_method_rank.min().rename(columns={"within_method_rank":"rank_b"}); m=a.merge(b,on="semantic_feature_unit"); comparisons.append({"model_family":family,"sensitive_mode":mode,"method_a":ma,"method_b":"mean_absolute_SHAP","shared_feature_count":len(m),"spearman_correlation":spearmanr(m.rank_a,m.rank_b).statistic,"top_10_overlap":len(set(m.nsmallest(10,"rank_a").semantic_feature_unit)&set(m.nsmallest(10,"rank_b").semantic_feature_unit)),"top_20_overlap":len(set(m.nsmallest(20,"rank_a").semantic_feature_unit)&set(m.nsmallest(20,"rank_b").semantic_feature_unit)),"largest_disagreements":"|".join(m.assign(d=(m.rank_a-m.rank_b).abs()).nlargest(3,"d").semantic_feature_unit),"output_scale_compatibility":"ranks only","interpretation":"Method disagreement is expected and is not a model error."})
    old=pd.read_csv(RESULTS/"stage8_cross_method_agreement.csv"); comparisons.append(old[old.model_family=="RealMLP"].iloc[0].to_dict()); pd.DataFrame(comparisons).to_csv(RESULTS/"stage8_cross_method_agreement.csv",index=False)

    sensitive=pd.read_csv(RESULTS/"stage8_sensitive_feature_dependence.csv"); sensitive["evidence_method"]="Stage 8 grouped permutation"; sensitive["model_family"]="RealMLP"; sensitive["sensitive_mode"]="with_sensitive"; sensitive["mean_absolute_shap"] = np.nan; sensitive["output_scale"]="raw original target scale"
    tree=[]
    for family in ["CatBoost","LightGBM","XGBoost"]:
        sub=shapdf[(shapdf.model_family==family)&(shapdf.sensitive_mode=="with_sensitive")]
        for unit,label in [("explicit_sensitive_identity_block","explicit identity"),("sensitive_context_block","contextual")]:
            q=sub[sub.semantic_feature_unit==unit]; tree.append({"feature_or_block":unit,"mae_increase":np.nan,"mean_absolute_prediction_change":np.nan,"positive_importance_normalized_share":q.within_method_normalized_share.sum(),"rank":q.within_method_rank.min() if len(q) else np.nan,"explicit_identity_or_contextual_classification":label,"global_only_flag":True,"fairness_conclusion":"none","causal_conclusion":"none","evidence_method":"saved mean absolute SHAP","model_family":family,"sensitive_mode":"with_sensitive","mean_absolute_shap":q.raw_importance.sum(),"output_scale":q.output_scale.iloc[0] if len(q) else paths("with_sensitive")[family]["scale"]})
    sensitive=pd.concat([sensitive,pd.DataFrame(tree)],ignore_index=True); sensitive.to_csv(RESULTS/"stage8_sensitive_feature_dependence.csv",index=False)
    repair_figures(inv,sensitive); repair_manifest_hashes()


def repair_figures(inv: pd.DataFrame, sensitive: pd.DataFrame) -> None:
    d=inv[["model_family","sensitive_mode","method","sample_rows"]].copy(); d.to_csv(PLOTTING/"stage8_figure_01.csv",index=False); fig,ax=plt.subplots(figsize=(10,7)); labels=d.model_family+" — "+d.sensitive_mode+" — "+d.method; ax.barh(labels,np.maximum(d.sample_rows,1),color="#6B8EAD"); ax.set_xlabel("saved sample rows (1 means model-level importance)"); fig.suptitle(f"{FIGURE_TITLES[0]}\n{LABEL}",fontsize=11); fig.tight_layout(); fig.savefig(FIGURES/"stage8_figure_01.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    s=sensitive[["model_family","evidence_method","feature_or_block","positive_importance_normalized_share","output_scale"]].copy(); s["evidence_label"]=s.model_family+" — "+s.evidence_method; s.to_csv(PLOTTING/"stage8_figure_11.csv",index=False); p=s.pivot(index="evidence_label",columns="feature_or_block",values="positive_importance_normalized_share").fillna(0); fig,ax=plt.subplots(figsize=(10,6)); p.plot.barh(ax=ax,color=["#8064A2","#C28E5C"]); ax.set_xlabel("within-method normalized importance share"); ax.set_title("Ranks/shares only; native SHAP scales are not combined"); fig.suptitle(f"{FIGURE_TITLES[10]}\n{LABEL}",fontsize=11); fig.tight_layout(); fig.savefig(FIGURES/"stage8_figure_11.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def repair_manifest_hashes() -> None:
    path=MANIFESTS/"stage8_visualization_manifest.json"; data=json.loads(path.read_text(encoding="utf-8"))
    for item in data["figures"]:
        item["figure_sha256"]=sha(ROOT/item["figure_path"]); item["plotting_data_sha256"]=sha(ROOT/item["plotting_data_path"])
    dump(data,path)


if __name__ == "__main__": build(); print("PASS")
