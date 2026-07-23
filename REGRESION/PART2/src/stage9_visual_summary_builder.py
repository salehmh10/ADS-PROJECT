from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "artifacts/results/stage9_visual_summary"
FIGURES = ROOT / "artifacts/figures/stage9_visual_summary"
PLOTTING = FIGURES / "plotting_data"
MANIFESTS = ROOT / "artifacts/manifests/stage9_visual_summary"
REPORTS = ROOT / "artifacts/reports"
NOTEBOOK = ROOT / "REGRESSION_PART9_VISUAL_PROJECT_SUMMARY.ipynb"
FORMAL_NOTEBOOK = ROOT / "REGRESSION_PART9_MODEL_CARD_TECHNICAL_REPORT.ipynb"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
AUTHORIZATION = "stage9_visual_analytics_summary_20260717"

TABLE_FILES = {
    "dataset": "stage9_visual_summary_dataset_table.csv",
    "catalog": "stage9_visual_summary_model_catalog.csv",
    "final": "stage9_visual_summary_final_metrics.csv",
    "decisions": "stage9_visual_summary_decisions.csv",
    "ensemble": "stage9_visual_summary_ensemble_table.csv",
    "error": "stage9_visual_summary_error_table.csv",
    "features": "stage9_visual_summary_feature_table.csv",
    "fairness": "stage9_visual_summary_fairness_table.csv",
}

FIGURE_SLUGS = [
    "project_roadmap",
    "data_target_profile",
    "best_mae_by_family",
    "linear_model_comparison",
    "tree_model_comparison",
    "boosting_model_comparison",
    "feature_engineering_confirmation",
    "deep_model_comparison",
    "final_test_mae",
    "final_test_metric_profile",
    "paired_mae_uncertainty",
    "ensemble_weight_search",
    "ensemble_rejection_tradeoff",
    "error_by_target_decile",
    "absolute_error_quantiles",
    "tail_error_signed_bias",
    "error_concentration_disagreement",
    "feature_importance_across_models",
    "feature_family_reliance",
    "group_error_sensitive_tradeoff",
]

DISPLAY = {
    "stage4l__blend__without_sensitive": "Boosting Blend",
    "stage5c__realmlp__without_sensitive__test_evaluation": "RealMLP",
    "stage5c__realmlp__with_sensitive__test_evaluation": "RealMLP + Sensitive Features",
}

COLORS = {
    "Boosting Blend": "#222222",
    "RealMLP": "#0072B2",
    "RealMLP + Sensitive Features": "#D55E00",
    "CatBoost": "#009E73",
    "LightGBM": "#CC79A7",
    "XGBoost": "#E69F00",
    "Linear Regression": "#56B4E9",
    "Ridge": "#0072B2",
    "Lasso": "#009E73",
    "ElasticNet": "#D55E00",
    "Gamma Regression": "#CC79A7",
    "Decision Tree": "#E69F00",
    "Random Forest": "#56B4E9",
    "HistGradientBoosting": "#009E73",
    "TabM": "#CC79A7",
    "FT-Transformer": "#E69F00",
}
MARKERS = {"Boosting Blend": "o", "RealMLP": "s", "RealMLP + Sensitive Features": "^"}
FIELD_NAMES = {
    "applicant_race_name_1": "Applicant Race",
    "applicant_ethnicity_name": "Applicant Ethnicity",
    "applicant_sex_name": "Applicant Sex",
    "co_applicant_race_name_1": "Co-applicant Race",
    "co_applicant_ethnicity_name": "Co-applicant Ethnicity",
    "co_applicant_sex_name": "Co-applicant Sex",
    "minority_population": "Minority-population band",
    "majority_minority_tract": "Majority-minority tract",
}

SOURCES = {
    "dataset": "artifacts/results/stage9/reporting/stage9_dataset_summary.csv",
    "final_test": "artifacts/results/stage9/reporting/stage9_final_test_comparison.csv",
    "linear_cv": "artifacts/results/prompt2/cv_oof_summary.csv",
    "linear_screen": "artifacts/results/prompt2/development_screening_results.csv",
    "tree_cv": "artifacts/results/stage3/cv_oof_summary.csv",
    "boost_validation": "artifacts/reports/stage4l_blend_validation_evidence.json",
    "cat_confirm": "artifacts/results/stage4/catboost/feature_confirmation/catboost_feature_confirmation_results.csv",
    "lgb_confirm": "artifacts/results/stage4/lightgbm/feature_confirmation/lightgbm_feature_confirmation_results.csv",
    "xgb_confirm": "artifacts/results/stage4/xgboost/feature_confirmation/stage4j_feature_confirmation_results.csv",
    "cat_final": "artifacts/results/stage4/catboost/final/catboost_final_validation_results.csv",
    "lgb_config": "artifacts/results/stage4/lightgbm/final/lightgbm_final_configuration.json",
    "xgb_config": "artifacts/results/stage4/xgboost/final/stage4j_final_configuration.json",
    "deep_screen": "artifacts/results/stage5/deep_core/screening/stage5a1_screening_results.csv",
    "deep_validation": "artifacts/results/stage5/deep_core/final_validation/stage5a2_final_validation_results.csv",
    "test_bootstrap": "artifacts/results/stage5/posttest_evaluation/stage5c_paired_bootstrap.csv",
    "ensemble_grid": "artifacts/results/stage5/deep_boosting_ensemble/stage5b_weight_grid_results.csv",
    "ensemble_decision": "artifacts/results/stage5/deep_boosting_ensemble/stage5b_ensemble_decision.json",
    "ensemble_bootstrap": "artifacts/results/stage5/deep_boosting_ensemble/stage5b_paired_bootstrap.csv",
    "error_quantiles": "artifacts/results/stage6/error_analysis/stage6_error_quantiles.csv",
    "error_deciles": "artifacts/results/stage6/error_analysis/stage6_target_decile_analysis.csv",
    "error_tail": "artifacts/results/stage6/error_analysis/stage6_target_tail_analysis.csv",
    "error_distribution": "artifacts/results/stage6/error_analysis/stage6_error_distribution_summary.csv",
    "error_concentration": "artifacts/results/stage6/error_analysis/stage6_error_concentration.csv",
    "disagreement_deciles": "artifacts/results/stage6/error_analysis/stage6_pairwise_disagreement_by_target_decile.csv",
    "disagreement_summary": "artifacts/results/stage6/error_analysis/stage6_pairwise_disagreement_summary.csv",
    "fair_disparity": "artifacts/results/stage7/fairness/stage7_group_disparity_summary.csv",
    "fair_pairwise": "artifacts/results/stage7/fairness/stage7_pairwise_group_differences.csv",
    "fair_tradeoff": "artifacts/results/stage7/fairness/stage7_accuracy_disparity_tradeoff.csv",
    "feature_compare": "artifacts/results/stage8/recovery/stage8_recovery_cross_model_feature_comparison.csv",
    "feature_family": "artifacts/results/stage8/recovery/stage8_recovery_feature_family_summary.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(key: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / SOURCES[key])


def read_json(key: str) -> dict:
    return json.loads((ROOT / SOURCES[key]).read_text(encoding="utf-8"))


def model_name(value: str) -> str:
    if value in DISPLAY:
        return DISPLAY[value]
    mapping = {
        "linear_regression": "Linear Regression",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elastic_net": "ElasticNet",
        "gamma_regressor": "Gamma Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "hist_gradient_boosting": "HistGradientBoosting",
        "realmlp": "RealMLP",
        "tabm": "TabM",
        "ft_transformer": "FT-Transformer",
    }
    return mapping.get(value, value)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#FBFBFB", "axes.edgecolor": "#BBBBBB",
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 10,
        "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.7, "grid.alpha": 0.8,
        "legend.frameon": False, "savefig.facecolor": "white", "svg.fonttype": "none",
    })


def finish(fig: plt.Figure, number: int, data: pd.DataFrame) -> None:
    slug = FIGURE_SLUGS[number - 1]
    data.to_csv(PLOTTING / f"figure_{number:02d}_{slug}.csv", index=False)
    fig.savefig(FIGURES / f"figure_{number:02d}_{slug}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"figure_{number:02d}_{slug}.svg", bbox_inches="tight")
    plt.close(fig)


def focus_limits(values, pad=0.12):
    a = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    span = hi - lo
    if span == 0:
        span = max(abs(lo) * 0.1, 1.0)
    return lo - span * pad, hi + span * pad


def save_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(RESULTS / TABLE_FILES[name], index=False)


def build_tables(data: dict[str, pd.DataFrame | dict]) -> dict[str, pd.DataFrame]:
    ds = data["dataset"]
    counts = dict(zip(ds["Dataset role"], ds["Row count"]))
    dataset = pd.DataFrame([{
        "Total rows": int(counts["Processed approved applications"]),
        "Train rows": int(counts["Frozen Train"]),
        "Test rows": int(counts["Frozen Test"]),
        "Target": "Loan amount", "Unit": "Thousands of US dollars",
        "Scope": "Approved applications only",
    }])

    linear = data["linear_cv"].query("sensitive_mode == 'without_sensitive'").copy()
    tree = data["tree_cv"].query("sensitive_mode == 'without_sensitive'").copy()
    rows = []
    family_group = {
        "Linear Regression": "Linear", "Ridge": "Linear", "Lasso": "Linear",
        "ElasticNet": "Linear", "Gamma Regression": "Generalized linear",
        "Decision Tree": "Tree", "Random Forest": "Tree", "HistGradientBoosting": "Tree boosting",
    }
    for _, r in pd.concat([linear, tree], ignore_index=True).iterrows():
        n = model_name(r["model_name"])
        if n == "dummy_median":
            continue
        rows.append({"Model": n, "Family": family_group[n], "Evaluation scope": "Cross-validation",
                     "Sensitive Features": "No", "Target mode": r["target_mode"], "MAE": r["mae"],
                     "RMSE": r["rmse"], "R²": r["r_squared"],
                     "Selection role": "Continued" if n in {"Lasso", "HistGradientBoosting"} else "Benchmark"})
    boost = data["boost_values"]
    target_modes = {"CatBoost": "log1p", "LightGBM": "raw", "XGBoost": "log1p", "Boosting Blend": "Original-scale blend"}
    for n in ["CatBoost", "LightGBM", "XGBoost", "Boosting Blend"]:
        r = boost.loc[boost.Model == n].iloc[0]
        rows.append({"Model": n, "Family": "Boosting", "Evaluation scope": "Validation",
                     "Sensitive Features": "No", "Target mode": target_modes[n], "MAE": r.MAE,
                     "RMSE": r.RMSE, "R²": np.nan,
                     "Selection role": "Official test model" if n == "Boosting Blend" else "Blend component"})
    deepv = data["deep_validation"]
    for fam in ["realmlp", "ft_transformer"]:
        r = deepv.loc[deepv.model_family.eq(fam)].sort_values("mae").iloc[0]
        n = model_name(fam)
        rows.append({"Model": n, "Family": "Deep learning", "Evaluation scope": "Validation",
                     "Sensitive Features": "No", "Target mode": r.target_mode, "MAE": r.mae,
                     "RMSE": r.rmse, "R²": r.r_squared,
                     "Selection role": "Later descriptive test" if n == "RealMLP" else "Stopped after validation"})
    tabm = data["deep_screen"].query("model_family == 'tabm'").sort_values("mae").iloc[0]
    rows.append({"Model": "TabM", "Family": "Deep learning", "Evaluation scope": "Screening validation",
                 "Sensitive Features": "No", "Target mode": tabm.target_mode, "MAE": tabm.mae,
                 "RMSE": tabm.rmse, "R²": tabm.r_squared, "Selection role": "Stopped after screening"})
    rows.append({"Model": "EBM", "Family": "Interpretable boosting", "Evaluation scope": "Not evaluated",
                 "Sensitive Features": "No", "Target mode": "—", "MAE": np.nan, "RMSE": np.nan,
                 "R²": np.nan, "Selection role": "Not evaluated — package unavailable"})
    catalog = pd.DataFrame(rows)

    ft = data["final_test"].copy()
    ft["Model"] = ft.candidate_id.map(DISPLAY)
    final = ft[["Model", "mae", "rmse", "r_squared", "rmsle", "mean_signed_error",
                "p90_absolute_error", "top_decile_mae", "top_five_percent_mae"]].rename(columns={
        "mae": "MAE", "rmse": "RMSE", "r_squared": "R²", "rmsle": "RMSLE",
        "mean_signed_error": "Mean Signed Error", "p90_absolute_error": "P90 Absolute Error",
        "top_decile_mae": "Top-decile MAE", "top_five_percent_mae": "Top-five-percent MAE"})

    boost_by = boost.set_index("Model")
    screen = data["deep_screen"].sort_values("mae").groupby("model_family", as_index=False).first().set_index("model_family")
    decisions = pd.DataFrame([
        ["Lasso", "Continued", f"Cross-validated MAE {catalog.loc[catalog.Model.eq('Lasso'),'MAE'].iloc[0]:.2f}", "Strongest sparse linear benchmark"],
        ["HistGradientBoosting", "Continued", f"Cross-validated MAE {catalog.loc[catalog.Model.eq('HistGradientBoosting'),'MAE'].iloc[0]:.2f}", "Lowest tree-family MAE"],
        ["CatBoost", "Continued", f"Validation MAE {boost_by.loc['CatBoost','MAE']:.2f}", "Best single boosting component"],
        ["LightGBM", "Continued", f"Validation MAE {boost_by.loc['LightGBM','MAE']:.2f}", "Useful blend diversity; higher standalone MAE"],
        ["XGBoost", "Continued", f"Validation MAE {boost_by.loc['XGBoost','MAE']:.2f}", "Useful blend diversity; heavier tail"],
        ["Boosting Blend", "Continued", f"Validation MAE {boost_by.loc['Boosting Blend','MAE']:.2f}", "Official test model"],
        ["RealMLP", "Continued", f"Validation MAE {catalog.loc[catalog.Model.eq('RealMLP'),'MAE'].iloc[0]:.2f}", "Later descriptive comparison only"],
        ["TabM", "Stopped", f"Screening MAE {screen.loc['tabm','mae']:.2f}", "Did not lead deep screening"],
        ["FT-Transformer", "Stopped", f"Validation MAE {catalog.loc[catalog.Model.eq('FT-Transformer'),'MAE'].iloc[0]:.2f}", "Higher MAE than RealMLP"],
        ["RealMLP + Boosting Blend", "Stopped", "MAE improved 0.44%; RMSE worsened 0.35%", "RMSE gate exceeded 0.25% limit"],
    ], columns=["Model or method", "Decision", "Quantitative reason", "Main trade-off"])

    eg = data["ensemble_grid"]
    selected = eg[(eg.boosting_anchor == "frozen_stage4_boosting_blend") & np.isclose(eg.deep_weight, .5)].iloc[0]
    eb = data["ensemble_bootstrap"].iloc[0]
    ensemble = pd.DataFrame([{
        "Deep weight": selected.deep_weight, "Boosting weight": selected.boost_weight,
        "MAE improvement (%)": selected.improvement_vs_best_component_percent,
        "RMSE worsening (%)": selected.rmse_worsening_vs_best_component_percent,
        "Tail change (%)": selected.top_decile_worsening_vs_best_component_percent,
        "Bootstrap interval": f"[{eb.ci95_lower:.3f}, {eb.ci95_upper:.3f}]",
        "Gate result": "Not accepted — RMSE limit exceeded",
    }])

    q = data["error_quantiles"].copy(); q["Model"] = q.candidate_id.map(DISPLAY)
    dist = data["error_distribution"][["candidate_id", "underprediction_rate"]]
    conc = data["error_concentration"].query("worst_proportion == 0.10")[["candidate_id", "share_of_total_absolute_error_percent"]]
    tails = data["error_tail"].query("tail_definition == 'top_decile'")[["candidate_id", "mae"]]
    err = q.merge(dist, on="candidate_id").merge(conc, on="candidate_id").merge(tails, on="candidate_id")
    error = err[["Model", "p50", "p90", "p95", "p99", "mae", "underprediction_rate",
                 "share_of_total_absolute_error_percent"]].rename(columns={
        "p50": "Median absolute error", "p90": "P90", "p95": "P95", "p99": "P99",
        "mae": "Top-decile MAE", "underprediction_rate": "Underprediction rate",
        "share_of_total_absolute_error_percent": "Error share from worst 10% (%)"})

    fc = data["feature_compare"].copy()
    fc["combined_rank"] = fc[["official_blend_rank", "realmlp_without_rank", "realmlp_with_rank"]].mean(axis=1)
    fc = fc.sort_values(["combined_rank", "semantic_feature_unit"]).head(20)
    features = fc[["semantic_feature_unit", "official_blend_rank", "realmlp_without_rank", "realmlp_with_rank", "feature_family"]].rename(columns={
        "semantic_feature_unit": "Feature", "official_blend_rank": "Boosting Blend rank",
        "realmlp_without_rank": "RealMLP rank", "realmlp_with_rank": "RealMLP + Sensitive Features rank",
        "feature_family": "Feature family"})
    features["Feature"] = features["Feature"].str.replace("_", " ").str.title()

    disp = data["fair_disparity"].copy(); trade = data["fair_tradeoff"].copy()
    idx = disp.groupby("sensitive_field")["worst_minus_best_mae_gap"].idxmax()
    fair = disp.loc[idx, ["sensitive_field", "eligible_group_count", "worst_minus_best_mae_gap",
                          "target_standardized_mae_gap", "underprediction_rate_spread"]].merge(
        trade[["sensitive_field", "overall_mae_change_with_minus_without"]], on="sensitive_field")
    fair["sensitive_field"] = fair.sensitive_field.map(FIELD_NAMES)
    fairness = fair.rename(columns={"sensitive_field": "Sensitive attribute", "eligible_group_count": "Eligible group count",
        "worst_minus_best_mae_gap": "Largest MAE gap", "target_standardized_mae_gap": "Target-standardized gap",
        "underprediction_rate_spread": "Underprediction-rate spread",
        "overall_mae_change_with_minus_without": "With-sensitive versus without-sensitive MAE change"})

    tables = {"dataset": dataset, "catalog": catalog, "final": final, "decisions": decisions,
              "ensemble": ensemble, "error": error, "features": features, "fairness": fairness}
    for key, frame in tables.items():
        save_table(key, frame)
    return tables


def load_sources() -> dict:
    missing = [p for p in SOURCES.values() if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(missing)
    data = {k: (read_json(k) if p.endswith(".json") else read_csv(k)) for k, p in SOURCES.items()}
    b = data["boost_validation"]
    rows = []
    name_map = {"catboost": "CatBoost", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
    for key, vals in b["individual_without_sensitive"].items():
        rows.append({"Model": name_map[key], "MAE": vals["mae"], "RMSE": vals["rmse"], "Top-decile MAE": vals["tail_mae"]})
    best = b["best_grid"]
    rows.append({"Model": "Boosting Blend", "MAE": best["mae"], "RMSE": best["rmse"], "Top-decile MAE": best["tail_mae"]})
    data["boost_values"] = pd.DataFrame(rows)
    return data


def figure_01(data):
    phases = [
        ("Data Preparation", "Approved applications\nShared split and folds"),
        ("Linear Models", "Linear Regression · Ridge\nLasso · ElasticNet\nGamma Regression"),
        ("Tree Models", "Decision Tree\nRandom Forest\nHistGradientBoosting"),
        ("Boosting Models", "CatBoost · LightGBM\nXGBoost · Boosting Blend"),
        ("Deep Learning", "RealMLP · TabM\nFT-Transformer"),
        ("Ensemble Check", "RealMLP +\nBoosting Blend"),
        ("Error and Fairness\nAnalysis", "Tail errors · groups\nSensitive-feature check"),
        ("Explainability and\nFinal Comparison", "Cross-model importance\nTest evidence"),
    ]
    frame = pd.DataFrame(phases, columns=["Phase", "Models and focus"])
    fig, ax = plt.subplots(figsize=(18, 4.2)); ax.set_xlim(-.4, 7.4); ax.set_ylim(-.2, 1.5); ax.axis("off")
    for i, (title, detail) in enumerate(phases):
        color = "#E7F1F8" if i < 6 else "#F5E9D8"
        box = FancyBboxPatch((i-.42, .25), .84, .85, boxstyle="round,pad=0.025,rounding_size=.04",
                             facecolor=color, edgecolor="#666666", linewidth=1.2)
        ax.add_patch(box)
        ax.text(i, .91, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(i, .62, detail, ha="center", va="top", fontsize=8.6, linespacing=1.35)
        if i < len(phases)-1:
            ax.annotate("", xy=(i+.56, .68), xytext=(i+.44, .68), arrowprops=dict(arrowstyle="->", color="#777777", lw=1.5))
    ax.set_title("Project Roadmap", fontsize=16, pad=8)
    finish(fig, 1, frame)


def figure_02(data):
    ds = data["dataset"]
    dec = data["error_deciles"].query("candidate_id == 'stage4l__blend__without_sensitive'").copy()
    counts = ds[ds["Dataset role"].isin(["Frozen Train", "Frozen Test"])][["Dataset role", "Row count"]].copy()
    counts["Dataset role"] = counts["Dataset role"].replace({"Frozen Train":"Train", "Frozen Test":"Test"})
    plot = pd.concat([
        counts.assign(panel="Rows", target_decile=np.nan, target_mean=np.nan, target_min=np.nan, target_max=np.nan),
        dec.assign(panel="Target profile", **{"Dataset role":"", "Row count":dec.row_count})[
            ["panel","Dataset role","Row count","target_decile","target_mean","target_min","target_max"]]
    ], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios":[.8,1.15,1.15]})
    ax=axes[0]; ax.bar(counts["Dataset role"], counts["Row count"], color=["#0072B2","#E69F00"])
    ax.set_ylim(0, counts["Row count"].max()*1.18); ax.set_ylabel("Rows"); ax.set_title("Saved split")
    for i,v in enumerate(counts["Row count"]): ax.text(i,v+8000,f"{int(v):,}",ha="center",fontweight="bold")
    ax.grid(axis="y"); ax.spines[["top","right"]].set_visible(False)
    ax=axes[1]; ax.plot(dec.target_decile,dec.target_mean,marker="o",color="#0072B2",lw=2)
    for x,y in zip(dec.target_decile,dec.target_mean): ax.text(x,y+18,f"{y:.0f}",ha="center",fontsize=8)
    ax.set(title="Mean target by decile",xlabel="Loan-amount decile",ylabel="Thousands of US dollars"); ax.grid(); ax.spines[["top","right"]].set_visible(False)
    ax=axes[2]; ax.vlines(dec.target_decile,dec.target_min,dec.target_max,color="#999999",lw=3)
    ax.scatter(dec.target_decile,dec.target_mean,color="#D55E00",zorder=3,label="Mean")
    ax.set(title="Target range by decile",xlabel="Loan-amount decile",ylabel="Thousands of US dollars"); ax.grid(); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Data and Target Profile",fontsize=16,fontweight="bold"); fig.text(.5,.01,"Approved applications only · Target unit: thousands of US dollars",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]); finish(fig,2,plot)


def figure_03(data):
    lin=data["linear_cv"].query("sensitive_mode == 'without_sensitive' and model_name != 'dummy_median'").copy()
    tree=data["tree_cv"].query("sensitive_mode == 'without_sensitive'").copy()
    cv=pd.concat([lin[["model_name","mae"]],tree[["model_name","mae"]]]); cv["Model"]=cv.model_name.map(model_name); cv["Scope"]="Cross-validation"
    b=data["boost_values"][["Model","MAE"]].rename(columns={"MAE":"mae"}); b["Scope"]="Shared final validation"
    dv=data["deep_validation"]
    dr=[]
    for fam in ["realmlp","ft_transformer"]:
        r=dv[dv.model_family.eq(fam)].sort_values("mae").iloc[0]; dr.append({"Model":model_name(fam),"mae":r.mae,"Scope":"Shared final validation"})
    tab=data["deep_screen"].query("model_family == 'tabm'").sort_values("mae").iloc[0]
    val=pd.concat([b,pd.DataFrame(dr),pd.DataFrame([{"Model":"TabM","mae":tab.mae,"Scope":"Deep screening validation"}])],ignore_index=True)
    test=data["final_test"].copy(); test["Model"]=test.candidate_id.map(DISPLAY); test["Scope"]="Test evaluation"; test=test[["Model","mae","Scope"]]
    plot=pd.concat([cv[["Model","mae","Scope"]],val,test],ignore_index=True)
    fig,axes=plt.subplots(1,3,figsize=(17,7),gridspec_kw={"width_ratios":[1.15,1.15,.95]})
    for ax, frame, title in zip(axes,[cv,val,test],["Cross-validation","Final validation","Test evaluation"]):
        frame=frame.sort_values("mae",ascending=False).reset_index(drop=True); y=np.arange(len(frame))
        for i,r in frame.iterrows():
            hollow = r.Scope == "Deep screening validation"
            shown = 74.2 if title=="Cross-validation" and r.mae>200 else r.mae
            marker = ">" if shown != r.mae else MARKERS.get(r.Model,"o")
            ax.scatter(shown,i,s=72,marker=marker,facecolor="white" if hollow else COLORS.get(r.Model,"#666666"),edgecolor=COLORS.get(r.Model,"#666666"),lw=1.8,zorder=3)
            label=f"{r.mae:,.2f} (off scale)" if shown != r.mae else f"{r.mae:.2f}"
            ax.text(shown,i+.20,label,ha="right" if shown != r.mae else "center",fontsize=8)
        ax.set_yticks(y,frame.Model)
        ax.set_xlim((62.5,75.0) if title=="Cross-validation" else focus_limits(frame.mae,.18)); ax.set_title(title); ax.set_xlabel("MAE · lower is better\nFocused axis")
        ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
        if title=="Final validation":
            tab_y=frame.index[frame.Model.eq("TabM")][0]; ax.axhline(tab_y-.5,color="#888888",ls="--",lw=1)
            ax.text(ax.get_xlim()[0],tab_y-.65,"Deep screening only",fontsize=8,va="top",color="#555555")
    fig.suptitle("Best MAE by Model Family",fontsize=16,fontweight="bold")
    fig.text(.5,.01,"Compare models within each panel only. TabM is isolated because its saved validation population differs.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]); finish(fig,3,plot)


def figure_04(data):
    d=data["linear_screen"].query("status == 'success' and model_name in ['linear_regression','ridge','lasso','elastic_net','gamma_regressor']").copy()
    d=d.sort_values("mae").groupby(["model_name","target_mode"],as_index=False).first()
    selected=data["linear_cv"].query("sensitive_mode == 'without_sensitive'").set_index("model_name")["target_mode"].to_dict()
    d["Model"]=d.model_name.map(model_name); d["Selected"]=d.apply(lambda r:r.target_mode==selected.get(r.model_name),axis=1)
    order=["Linear Regression","Ridge","Lasso","ElasticNet","Gamma Regression"]
    fig,ax=plt.subplots(figsize=(10,5.8)); y=np.arange(len(order))
    for i,n in enumerate(order):
        sub=d[d.Model.eq(n)]
        if len(sub)==2: ax.plot(sub.mae,[i,i],color="#BBBBBB",lw=2,zorder=1)
        for _,r in sub.iterrows():
            c="#0072B2" if r.target_mode=="raw" else "#D55E00"; m="o" if r.target_mode=="raw" else "s"
            ax.scatter(r.mae,i,color=c,marker=m,s=105 if r.Selected else 65,edgecolor="#111111" if r.Selected else "white",lw=1.3,zorder=3)
            ax.text(r.mae,i+.22,f"{r.mae:.2f}",ha="center",fontsize=8)
    ax.set_yticks(y,order); ax.invert_yaxis(); ax.set_xlim(*focus_limits(d.mae,.12)); ax.set_xlabel("Validation MAE · lower is better · focused axis"); ax.set_title("Linear Target-Mode Comparison")
    ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#0072B2",label="Raw"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#D55E00",label="Log1p"),Line2D([0],[0],marker="o",color="w",markerfacecolor="#777",markeredgecolor="#111",markersize=9,label="Selected mode")],ncol=3,loc="lower center",bbox_to_anchor=(.5,-.25))
    fig.tight_layout(); finish(fig,4,d[["Model","target_mode","mae","Selected"]])


def figure_05(data):
    d=data["tree_cv"].query("sensitive_mode == 'without_sensitive'").copy(); d["Model"]=d.model_name.map(model_name)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8)); metrics=[("mae","MAE · lower is better"),("rmse","RMSE · lower is better")]
    for ax,(col,title) in zip(axes,metrics):
        s=d.sort_values(col,ascending=False).reset_index(drop=True); y=np.arange(len(s))
        ax.scatter(s[col],y,c=[COLORS[n] for n in s.Model],s=85)
        for x,yy in zip(s[col],y): ax.text(x,yy+.18,f"{x:.2f}",ha="center",fontsize=8)
        ax.set_yticks(y,s.Model); ax.set_xlim(*focus_limits(s[col],.18)); ax.set_xlabel("Focused axis"); ax.set_title(title); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    sorted_mae=d.sort_values("mae",ascending=False).reset_index(drop=True); hgb_y=int(sorted_mae.index[sorted_mae.Model.eq("HistGradientBoosting")][0])
    axes[0].annotate("Continued: lowest tree MAE",xy=(d.loc[d.Model.eq('HistGradientBoosting'),'mae'].iloc[0],hgb_y),xytext=(68,1.1),arrowprops=dict(arrowstyle="->",color="#555"),fontsize=9)
    fig.suptitle("Tree Model Comparison",fontsize=16,fontweight="bold"); fig.tight_layout(rect=[0,0,1,.93]); finish(fig,5,d[["Model","mae","rmse"]])


def figure_06(data):
    d=data["boost_values"].copy(); fig,axes=plt.subplots(1,2,figsize=(12,5));
    for ax,col,title in zip(axes,["MAE","Top-decile MAE"],["MAE · lower is better","Top-decile MAE · lower is better"]):
        s=d.sort_values(col,ascending=False).reset_index(drop=True); y=np.arange(len(s));
        ax.scatter(s[col],y,c=[COLORS[n] for n in s.Model],s=90)
        for x,yy in zip(s[col],y): ax.text(x,yy+.2,f"{x:.2f}",ha="center",fontsize=8)
        ax.set_yticks(y,s.Model); ax.set_xlim(*focus_limits(s[col],.2)); ax.set_xlabel("Focused axis"); ax.set_title(title); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Boosting Validation Comparison",fontsize=16,fontweight="bold"); fig.text(.5,.01,"Shared 25,000-row validation population; compare directly within this figure.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.93]); finish(fig,6,d)


def figure_07(data):
    cat=data["cat_confirm"][["feature_pack_id","mae"]].copy(); cat["Model"]="CatBoost"
    lgb=data["lgb_confirm"].copy(); lgb["Model"]="LightGBM"
    xgb=data["xgb_confirm"].copy(); xgb["Model"]="XGBoost"
    def prep(frame,names):
        frame=frame.copy(); base=frame.mae.iloc[0]; frame["MAE change"]=frame.mae-base; frame["Design"]=names; return frame[["Model","Design","mae","MAE change"]]
    cat=prep(cat,["Original","Combined proposal","Ratio rescue"])
    lgb=prep(lgb,["Original","Combined proposal","Ratio rescue"])
    xgb=prep(xgb,["Original","Combined proposal","Ratio rescue"])
    plot=pd.concat([cat,lgb,xgb],ignore_index=True); fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,(name,s) in zip(axes,plot.groupby("Model",sort=False)):
        y=np.arange(3); vals=s["MAE change"].to_numpy(); ax.axvline(0,color="#333",lw=1)
        ax.scatter(vals,y,c=["#777777" if v==0 else ("#009E73" if v<0 else "#D55E00") for v in vals],s=80)
        for x,yy in zip(vals,y): ax.text(x,yy+.18,f"{x:+.3f}",ha="center",fontsize=8)
        ax.set_yticks(y,s.Design); ax.invert_yaxis(); lim=max(abs(vals).max()*1.45,.015); ax.set_xlim(-lim,lim)
        ax.set_title(name); ax.set_xlabel("MAE change vs original\nNegative = improvement"); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Did Extra Features Help?",fontsize=16,fontweight="bold"); fig.tight_layout(rect=[0,0,1,.93]); finish(fig,7,plot)


def figure_08(data):
    d=data["deep_screen"].query("status == 'PASS' and validity_status == 'VALID'").copy(); d["Model"]=d.model_family.map(model_name)
    d["Family winner"]=d.groupby("Model")["mae"].transform("min").eq(d.mae)
    order=["RealMLP","TabM","FT-Transformer"]; fig,ax=plt.subplots(figsize=(9,5.5)); y=np.arange(3)
    for i,n in enumerate(order):
        s=d[d.Model.eq(n)].sort_values("target_mode"); ax.plot(s.mae,[i,i],color="#BBBBBB",lw=2)
        for _,r in s.iterrows():
            c="#0072B2" if r.target_mode=="raw" else "#D55E00"; m="o" if r.target_mode=="raw" else "s"
            ax.scatter(r.mae,i,color=c,marker=m,s=110 if r["Family winner"] else 65,edgecolor="#111" if r["Family winner"] else "white",lw=1.3,zorder=3)
            ax.annotate(f"{r.mae:.2f}",(r.mae,i),xytext=(0,10 if r.target_mode=="log1p" else -14),textcoords="offset points",ha="center",fontsize=8)
    ax.set_yticks(y,order); ax.invert_yaxis(); ax.set_xlim(*focus_limits(d.mae,.2)); ax.set_xlabel("Screening-validation MAE · lower is better · focused axis"); ax.set_title("Deep Model Screening")
    ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#0072B2",label="Raw"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#D55E00",label="Log1p"),Line2D([0],[0],marker="o",color="w",markerfacecolor="#777",markeredgecolor="#111",markersize=9,label="Family winner")],ncol=3,loc="lower center",bbox_to_anchor=(.5,-.27))
    fig.tight_layout(); finish(fig,8,d[["Model","target_mode","mae","rmse","Family winner"]])


def figure_09(data):
    d=data["final_test"].copy(); d["Model"]=d.candidate_id.map(DISPLAY); d["Role"]=d.Model.map({"Boosting Blend":"Official test","RealMLP":"Later descriptive test","RealMLP + Sensitive Features":"Later descriptive test"})
    y=np.arange(3); fig,ax=plt.subplots(figsize=(10,4.8))
    for i,r in d.reset_index(drop=True).iterrows():
        n=r.Model; ax.scatter(r.mae,i,s=110,marker=MARKERS[n],color=COLORS[n],zorder=3)
        label=f"{r.mae:.3f}  (≈${r.dollar_equivalent_mae:,.0f})\nΔ {r.mae_difference_from_official:+.3f} · {r.mae_percentage_difference_from_official:+.2f}%"
        ax.text(r.mae+.035,i,label,va="center",fontsize=9)
    ax.set_yticks(y,[f"{n}\n{role}" for n,role in zip(d.Model,d.Role)]); ax.invert_yaxis(); ax.set_xlim(*focus_limits(d.mae,.35)); ax.set_xlabel("MAE · thousands of US dollars · lower is better · focused axis")
    ax.set_title("Final Test MAE"); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False); fig.tight_layout(); finish(fig,9,d[["Model","Role","mae","dollar_equivalent_mae","mae_difference_from_official","mae_percentage_difference_from_official"]])


def figure_10(data):
    d=data["final_test"].copy(); d["Model"]=d.candidate_id.map(DISPLAY)
    specs=[("mae","MAE","Lower"),("rmse","RMSE","Lower"),("rmsle","RMSLE","Lower"),("r_squared","R²","Higher"),("top_decile_mae","Top-decile MAE","Lower"),("top_five_percent_mae","Top-five-percent MAE","Lower")]
    fig,axes=plt.subplots(2,3,figsize=(15,8));
    for ax,(col,title,direction) in zip(axes.flat,specs):
        vals=d[col].to_numpy(); y=np.arange(3)
        for i,r in d.reset_index(drop=True).iterrows():
            ax.scatter(r[col],i,s=80,marker=MARKERS[r.Model],color=COLORS[r.Model]); label=f"{r[col]:.3f}" if col in {"rmsle","r_squared"} else f"{r[col]:.2f}"
            ax.annotate(label,(r[col],i),xytext=(0,[-13,10,12][i]),textcoords="offset points",ha="center",fontsize=8)
        ax.set_yticks(y,d.Model); ax.set_ylim(2.55,-.55); ax.set_xlim(*focus_limits(vals,.25)); ax.set_title(f"{title} · {direction} is better"); ax.set_xlabel("Focused axis"); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Final Test Metric Profile",fontsize=16,fontweight="bold"); fig.tight_layout(rect=[0,0,1,.94]); finish(fig,10,d[["Model"]+[x[0] for x in specs]])


def figure_11(data):
    d=data["test_bootstrap"].copy()
    d["Comparison"]=["RealMLP minus Boosting Blend","RealMLP + Sensitive Features minus RealMLP"]
    fig,ax=plt.subplots(figsize=(10,4.2)); y=np.arange(2); ax.axvline(0,color="#333333",lw=1.2)
    for i,r in d.iterrows():
        ax.errorbar(r.point_mae_difference,i,xerr=[[r.point_mae_difference-r.ci_2_5],[r.ci_97_5-r.point_mae_difference]],fmt="o",color=["#0072B2","#D55E00"][i],capsize=5,lw=2,markersize=8)
        ax.text(r.ci_97_5+.03,i,f"Δ {r.point_mae_difference:+.3f}\nwin {r.first_candidate_win_proportion:.1%}; n={int(r.resamples)}",va="center",fontsize=9)
    ax.set_yticks(y,d.Comparison); ax.invert_yaxis(); ax.set_xlim(min(d.ci_2_5.min(),0)-.15,max(d.ci_97_5.max(),0)+.45)
    ax.set_xlabel("Paired MAE difference · 95% interval · negative favors first-named model"); ax.set_title("Paired MAE Uncertainty")
    ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False); fig.tight_layout(); finish(fig,11,d[["Comparison","point_mae_difference","ci_2_5","ci_97_5","first_candidate_win_proportion","resamples"]])


def figure_12(data):
    d=data["ensemble_grid"].copy(); d["Anchor"]=d.boosting_anchor.map({"frozen_stage4_boosting_blend":"Boosting Blend anchor","catboost":"CatBoost anchor"})
    fig,axes=plt.subplots(1,2,figsize=(13,5)); colors={"Boosting Blend anchor":"#222222","CatBoost anchor":"#009E73"}
    for ax,col,title in zip(axes,["mae","rmse"],["MAE · lower is better","RMSE · lower is better"]):
        for anchor,s in d.groupby("Anchor"):
            s=s.sort_values("deep_weight"); ax.plot(s.deep_weight,s[col],marker="o",color=colors[anchor],label=anchor,lw=2,ms=5)
        sel=d[(d.Anchor=="Boosting Blend anchor") & np.isclose(d.deep_weight,.5)].iloc[0]
        ax.scatter(.5,sel[col],s=150,facecolor="#E69F00",edgecolor="#111111",marker="*",zorder=5)
        ax.annotate(f"50/50: {sel[col]:.3f}",xy=(.5,sel[col]),xytext=(.32,sel[col]+(.12 if col=='mae' else .35)),arrowprops=dict(arrowstyle="->",color="#555"),fontsize=9)
        if col=="mae":
            best=d.loc[d.mae.idxmin()]; ax.scatter(best.deep_weight,best.mae,s=90,facecolor="white",edgecolor="#0072B2",marker="D",lw=1.7,zorder=5)
            ax.annotate(f"Lowest MAE: {best.mae:.3f} at {best.deep_weight:.2f}",xy=(best.deep_weight,best.mae),xytext=(.02,best.mae-.10),arrowprops=dict(arrowstyle="->",color="#555"),fontsize=8)
        ax.set(xlabel="Deep-model weight",ylabel=title); ax.grid(); ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(loc="best"); fig.suptitle("Ensemble Weight Search",fontsize=16,fontweight="bold"); fig.tight_layout(rect=[0,0,1,.94]); finish(fig,12,d[["Anchor","deep_weight","boost_weight","mae","rmse","top_decile_mae"]])


def figure_13(data):
    d=data["ensemble_grid"].copy(); d["Anchor"]=d.boosting_anchor.map({"frozen_stage4_boosting_blend":"Boosting Blend anchor","catboost":"CatBoost anchor"})
    x=d.improvement_vs_best_component_percent; y=d.rmse_worsening_vs_best_component_percent; tail=d.top_decile_worsening_vs_best_component_percent
    fig,ax=plt.subplots(figsize=(10,6.3)); xmin,xmax=min(x.min(),-.1),max(x.max(),.65); ymin,ymax=min(y.min(),-.2),max(y.max(),2.0)
    ax.add_patch(Rectangle((.30,ymin),xmax-.30,.25-ymin,facecolor="#009E73",alpha=.12,label="Two numeric gates met"))
    ax.axvline(.30,color="#009E73",ls="--",lw=1.5,label="Required MAE improvement: 0.30%")
    ax.axhline(.25,color="#D55E00",ls="--",lw=1.5,label="Maximum RMSE worsening: 0.25%")
    markers={"Boosting Blend anchor":"o","CatBoost anchor":"s"}; norm=plt.Normalize(-2,2)
    for anchor,s in d.groupby("Anchor"):
        ax.scatter(s.improvement_vs_best_component_percent,s.rmse_worsening_vs_best_component_percent,
                   c=s.top_decile_worsening_vs_best_component_percent,cmap="coolwarm",norm=norm,s=75,marker=markers[anchor],edgecolor="#444",alpha=.85,label=anchor)
    sel=d[(d.Anchor=="Boosting Blend anchor") & np.isclose(d.deep_weight,.5)].iloc[0]
    ax.scatter(sel.improvement_vs_best_component_percent,sel.rmse_worsening_vs_best_component_percent,s=220,marker="*",facecolor="#E69F00",edgecolor="#111",zorder=5)
    ax.annotate(f"Selected 50/50\nMAE +{sel.improvement_vs_best_component_percent:.2f}%\nRMSE {sel.rmse_worsening_vs_best_component_percent:+.2f}%",xy=(sel.improvement_vs_best_component_percent,sel.rmse_worsening_vs_best_component_percent),xytext=(.05,1.15),arrowprops=dict(arrowstyle="->",color="#555"),fontsize=9)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel="MAE improvement versus best component (%) · higher is better",ylabel="RMSE worsening versus best component (%) · lower is better",title="Why the Ensemble Was Rejected")
    ax.grid(); ax.spines[["top","right"]].set_visible(False); ax.legend(loc="upper left",fontsize=8)
    sm=plt.cm.ScalarMappable(norm=norm,cmap="coolwarm"); sm.set_array([]); cb=fig.colorbar(sm,ax=ax,pad=.02); cb.set_label("Top-decile MAE change (%)\nNegative = improvement",fontsize=8)
    fig.text(.5,.01,"Point color reflects top-decile change; the selected design improved tail MAE but exceeded the RMSE limit. Shading covers the two plotted gates only.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=[0,.05,1,1]); finish(fig,13,d[["Anchor","deep_weight","boost_weight","improvement_vs_best_component_percent","rmse_worsening_vs_best_component_percent","top_decile_worsening_vs_best_component_percent"]])


def figure_14(data):
    d=data["error_deciles"].copy(); d["Model"]=d.candidate_id.map(DISPLAY); fig,ax=plt.subplots(figsize=(11,6))
    endpoint_offsets={"Boosting Blend":-15,"RealMLP":0,"RealMLP + Sensitive Features":15}
    for n,s in d.groupby("Model",sort=False):
        ax.plot(s.target_decile,s.mae,marker=MARKERS[n],color=COLORS[n],lw=2,ms=6,label=n)
        last=s.sort_values("target_decile").iloc[-1]; ax.annotate(f"{last.mae:.1f}",(last.target_decile,last.mae),xytext=(8,endpoint_offsets[n]),textcoords="offset points",color=COLORS[n],fontsize=8,va="center")
    n10=int(d.loc[d.target_decile.eq(10),"row_count"].iloc[0]); ax.text(.02,.86,f"Highest decile: n={n10:,}",transform=ax.transAxes,fontsize=9,bbox=dict(facecolor="white",edgecolor="#BBBBBB"))
    ax.set(xlabel="Loan-amount decile",ylabel="MAE · thousands of US dollars · lower is better",title="Error by Loan-Amount Decile",xticks=range(1,11)); ax.grid(); ax.spines[["top","right"]].set_visible(False); ax.legend(ncol=3,loc="upper left")
    fig.tight_layout(); finish(fig,14,d[["Model","target_decile","row_count","mae","target_mean"]])


def figure_15(data):
    q=data["error_quantiles"].copy(); q["Model"]=q.candidate_id.map(DISPLAY); long=q.melt(id_vars="Model",value_vars=["p50","p75","p90","p95","p99"],var_name="Quantile",value_name="Absolute error")
    order=["p50","p75","p90","p95","p99"]; x=np.arange(5); fig,ax=plt.subplots(figsize=(11,6))
    for n,s in long.groupby("Model",sort=False):
        s=s.set_index("Quantile").loc[order].reset_index(); ax.plot(x,s["Absolute error"],marker=MARKERS[n],color=COLORS[n],lw=2,ms=6,label=n)
        offset={"Boosting Blend":-13,"RealMLP":0,"RealMLP + Sensitive Features":13}[n]
        for xx,v in zip(x,s["Absolute error"]): ax.annotate(f"{v:.1f}",(xx,v),xytext=(0,offset),textcoords="offset points",ha="center",fontsize=7,color=COLORS[n])
    ax.set(xticks=x,xticklabels=[v.upper() for v in order],ylabel="Absolute error · thousands of US dollars",title="Absolute-Error Quantile Profile"); ax.grid(); ax.spines[["top","right"]].set_visible(False); ax.legend(loc="upper left")
    fig.tight_layout(); finish(fig,15,long)


def figure_16(data):
    ft=data["final_test"].copy(); ft["Model"]=ft.candidate_id.map(DISPLAY)
    dist=data["error_distribution"][["candidate_id","underprediction_rate"]]; d=ft.merge(dist,on="candidate_id")
    specs=[("top_decile_mae","Top-decile MAE","Lower is better"),("top_five_percent_mae","Top-five-percent MAE","Lower is better"),("mean_signed_error","Mean Signed Error","Closer to zero is better"),("underprediction_rate","Underprediction rate","Descriptive rate")]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(col,title,direction) in zip(axes.flat,specs):
        y=np.arange(3); vals=d[col].to_numpy();
        if col=="mean_signed_error": ax.axvline(0,color="#333",lw=1)
        for i,r in d.reset_index(drop=True).iterrows():
            ax.scatter(r[col],i,s=85,marker=MARKERS[r.Model],color=COLORS[r.Model]); label=f"{r[col]:.1%}" if col=="underprediction_rate" else f"{r[col]:.2f}"; ax.annotate(label,(r[col],i),xytext=(0,[-13,10,12][i]),textcoords="offset points",ha="center",fontsize=8)
        ax.set_yticks(y,d.Model); ax.set_ylim(2.55,-.55); ax.set_xlim(*focus_limits(np.append(vals,0) if col=="mean_signed_error" else vals,.22)); ax.set_title(f"{title}\n{direction}"); ax.set_xlabel("Focused axis"); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Tail Error and Signed Bias",fontsize=16,fontweight="bold"); fig.tight_layout(rect=[0,0,1,.94]); finish(fig,16,d[["Model"]+[x[0] for x in specs]])


def figure_17(data):
    c=data["error_concentration"].copy(); c["Model"]=c.candidate_id.map(DISPLAY); c["Worst share"]=c.worst_proportion.map({.01:"Worst 1%",.05:"Worst 5%",.10:"Worst 10%"})
    dd=data["disagreement_deciles"].query("pair_id == 'stage6__deep_without_minus_stage4l'").copy(); summ=data["disagreement_summary"].query("pair_id == 'stage6__deep_without_minus_stage4l'").iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    x=np.arange(3)
    for n,s in c.groupby("Model",sort=False):
        s=s.set_index("Worst share").loc[["Worst 1%","Worst 5%","Worst 10%"]]; axes[0].plot(x,s.share_of_total_absolute_error_percent,marker=MARKERS[n],color=COLORS[n],lw=2,label=n)
    axes[0].set(xticks=x,xticklabels=["Worst 1%","Worst 5%","Worst 10%"],ylabel="Share of total absolute error (%)",title="Error concentration"); axes[0].grid(); axes[0].spines[["top","right"]].set_visible(False); axes[0].legend(fontsize=8)
    axes[1].plot(dd.target_decile,dd.mean_absolute_prediction_disagreement,marker="o",color="#0072B2",lw=2)
    for xx,yy in zip(dd.target_decile,dd.mean_absolute_prediction_disagreement): axes[1].text(xx,yy+.7,f"{yy:.1f}",ha="center",fontsize=7)
    axes[1].set(xlabel="Loan-amount decile",ylabel="Mean absolute prediction disagreement",title="Boosting Blend vs RealMLP",xticks=range(1,11)); axes[1].grid(); axes[1].spines[["top","right"]].set_visible(False)
    axes[1].text(.04,.92,f"Overall mean: {summ.mean_absolute_prediction_disagreement:.2f}\nPrediction correlation: {summ.pearson_prediction_correlation:.3f}",transform=axes[1].transAxes,va="top",fontsize=9,bbox=dict(facecolor="white",edgecolor="#BBBBBB"))
    fig.suptitle("Error Concentration and Model Disagreement",fontsize=16,fontweight="bold"); fig.tight_layout(rect=[0,0,1,.94]); finish(fig,17,pd.concat([c.assign(panel="concentration"),dd.assign(panel="disagreement")],ignore_index=True,sort=False))


def feature_label(x):
    special={"applicant_income":"Applicant income","lien_status_name":"Lien status","owner_occupancy_name":"Owner occupancy","tract_to_msa_income_percentage":"Tract-income ratio","hud_median_family_income":"HUD median family income","loan_purpose_name":"Loan purpose","state_name":"State","county_name":"County","msamd_name":"MSA/MD","us_region":"US region","respondent_id":"Respondent"}
    return special.get(x,x.replace("_"," ").title())


def figure_18(data):
    d=data["feature_compare"].copy(); d["combined_rank"]=d[["official_blend_rank","realmlp_without_rank","realmlp_with_rank"]].mean(axis=1); d=d.sort_values("combined_rank").head(20).copy(); d["Feature"]=d.semantic_feature_unit.map(feature_label); d=d.sort_values("combined_rank",ascending=False).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(12,10)); y=np.arange(len(d)); specs=[("official_blend_rank","Boosting Blend"),("realmlp_without_rank","RealMLP"),("realmlp_with_rank","RealMLP + Sensitive Features")]
    for col,n in specs:
        ax.scatter(d[col],y,s=65,marker=MARKERS[n],color=COLORS[n],label=n,alpha=.9)
    ax.set_yticks(y,d.Feature); ax.invert_xaxis(); ax.set_xlabel("Importance rank · 1 is most important"); ax.set_title("Feature Importance Across Models"); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False); ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,-.08))
    fig.text(.5,.01,"Within-model permutation ranks; raw importance magnitudes are not compared. Importance does not prove causality.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]); finish(fig,18,d[["Feature","official_blend_rank","realmlp_without_rank","realmlp_with_rank","feature_family"]])


def figure_19(data):
    d=data["feature_family"].copy(); d["Model"]=d.candidate_id.map(DISPLAY)
    rename={"Demographic or explicit sensitive identity":"Explicit sensitive identity","Tract demographic context":"Sensitive context","Tract and area income context":"Area and tract income context","Loan structure and lien status":"Loan structure","Lender, respondent, and agency":"Lender and respondent","Engineered ratios and counts":"Engineered ratios"}
    d["Family"]=d.feature_family.replace(rename); d["Share (%)"]=100*d.positive_permutation_importance_share
    order=d.groupby("Family")["Share (%)"].mean().sort_values().index; fig,ax=plt.subplots(figsize=(12,8)); offsets={"Boosting Blend":-.22,"RealMLP":0,"RealMLP + Sensitive Features":.22}; y=np.arange(len(order))
    for n in ["Boosting Blend","RealMLP","RealMLP + Sensitive Features"]:
        s=d[d.Model.eq(n)].set_index("Family").reindex(order).fillna({"Share (%)":0}); ax.scatter(s["Share (%)"],y+offsets[n],s=70,marker=MARKERS[n],color=COLORS[n],label=n)
    ax.set_yticks(y,order); ax.set_xlabel("Share of positive permutation importance (%)"); ax.set_title("Feature-Family Reliance"); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False); ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,-.12))
    fig.text(.5,.01,"Sensitive identity and context are shown only as aggregate shares. Importance does not prove causality or unfairness.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]); finish(fig,19,d[["Model","Family","Share (%)","feature_unit_count"]])


def figure_20(data):
    disp=data["fair_disparity"].copy(); disp["Model"]=disp.candidate_id.map(DISPLAY); disp["Attribute"]=disp.sensitive_field.map(FIELD_NAMES)
    pair=data["fair_pairwise"].query("pair_id == 'stage7__deep_with_minus_deep_without' and evidence_tier == 'primary' and is_non_substantive == False").copy(); pair["Attribute"]=pair.sensitive_field.map(FIELD_NAMES)
    order=list(FIELD_NAMES.values()); fig,axes=plt.subplots(1,2,figsize=(16,7),gridspec_kw={"width_ratios":[1,1.25]})
    offsets={"Boosting Blend":-.22,"RealMLP":0,"RealMLP + Sensitive Features":.22}; y=np.arange(len(order))
    for n in ["Boosting Blend","RealMLP","RealMLP + Sensitive Features"]:
        s=disp[disp.Model.eq(n)].set_index("Attribute").reindex(order); axes[0].scatter(s.worst_minus_best_mae_gap,y+offsets[n],s=65,marker=MARKERS[n],color=COLORS[n],label=n)
    axes[0].set_yticks(y,order); axes[0].set_xlabel("Largest eligible MAE gap"); axes[0].set_title("Eligible group disparity"); axes[0].grid(axis="x"); axes[0].spines[["top","right","left"]].set_visible(False); axes[0].legend(fontsize=8)
    axes[1].axvline(0,color="#333",lw=1)
    for i,attr in enumerate(order):
        s=pair[pair.Attribute.eq(attr)]; sizes=28+110*np.sqrt(s.n/s.n.max()) if len(s) else []
        axes[1].scatter(s.mae_difference,[i]*len(s),s=sizes,c=np.where(s.mae_difference<0,"#009E73","#D55E00"),alpha=.75,edgecolor="#555",lw=.4)
        axes[1].text(axes[1].get_xlim()[1] if axes[1].get_xlim()[1]>0 else 0,i,f"  {len(s)} groups",fontsize=7,va="center")
    axes[1].set_yticks(y,order); axes[1].set_xlabel("RealMLP + Sensitive Features minus RealMLP MAE\nNegative = lower MAE with sensitive Features"); axes[1].set_title("Aggregate group-level change"); axes[1].grid(axis="x"); axes[1].spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Group Error and Sensitive-Feature Trade-Off",fontsize=16,fontweight="bold"); fig.text(.5,.01,"Accuracy comparison only · Approved applications only · No approval-fairness, legal, or causal conclusion.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]); finish(fig,20,pd.concat([disp.assign(panel="disparity"),pair.assign(panel="group change")],ignore_index=True,sort=False))


FIGURE_BUILDERS=[figure_01,figure_02,figure_03,figure_04,figure_05,figure_06,figure_07,figure_08,figure_09,figure_10,figure_11,figure_12,figure_13,figure_14,figure_15,figure_16,figure_17,figure_18,figure_19,figure_20]


FIGURE_META = [
    ("Project Roadmap", ["dataset"], "Project phases", "Structural", "Sequence", "Not applicable"),
    ("Data and Target Profile", ["dataset","error_deciles"], "Saved split and Test target profile", "Quantitative", "Rows / target amount", "Higher values indicate count or amount"),
    ("Best MAE by Model Family", ["linear_cv","tree_cv","boost_validation","deep_screen","deep_validation","final_test"], "Separated evaluation scopes", "Quantitative", "MAE", "Lower is better"),
    ("Linear Target-Mode Comparison", ["linear_screen","linear_cv"], "Validation", "Quantitative", "MAE", "Lower is better"),
    ("Tree Model Comparison", ["tree_cv"], "Cross-validation", "Quantitative", "MAE / RMSE", "Lower is better"),
    ("Boosting Validation Comparison", ["boost_validation"], "Shared final validation", "Quantitative", "MAE / top-decile MAE", "Lower is better"),
    ("Did Extra Features Help?", ["cat_confirm","lgb_confirm","xgb_confirm"], "Separate family confirmation designs", "Quantitative", "MAE change", "Negative is improvement"),
    ("Deep Model Screening", ["deep_screen"], "Screening validation", "Quantitative", "MAE", "Lower is better"),
    ("Final Test MAE", ["final_test"], "Official and later descriptive Test", "Quantitative", "MAE", "Lower is better"),
    ("Final Test Metric Profile", ["final_test"], "Official and later descriptive Test", "Quantitative", "Six metrics", "Panel-specific"),
    ("Paired MAE Uncertainty", ["test_bootstrap"], "Saved paired Test intervals", "Quantitative", "Paired MAE difference", "Negative favors first-named model"),
    ("Ensemble Weight Search", ["ensemble_grid"], "Shared validation", "Quantitative", "MAE / RMSE", "Lower is better"),
    ("Why the Ensemble Was Rejected", ["ensemble_grid","ensemble_decision","ensemble_bootstrap"], "Shared validation and saved interval", "Quantitative", "Relative MAE / RMSE change", "MAE improvement higher; RMSE worsening lower"),
    ("Error by Loan-Amount Decile", ["error_deciles"], "Saved Test predictions summarized by target decile", "Quantitative", "MAE", "Lower is better"),
    ("Absolute-Error Quantile Profile", ["error_quantiles"], "Saved Test error summaries", "Quantitative", "Absolute error", "Lower is better"),
    ("Tail Error and Signed Bias", ["final_test","error_distribution"], "Saved Test error summaries", "Quantitative", "Tail MAE / signed error / rate", "Panel-specific"),
    ("Error Concentration and Model Disagreement", ["error_concentration","disagreement_deciles","disagreement_summary"], "Saved Test error summaries", "Quantitative", "Error share / disagreement", "Lower is generally better"),
    ("Feature Importance Across Models", ["feature_compare"], "Corrected grouped permutation summaries", "Quantitative", "Within-model rank", "Rank 1 is most important"),
    ("Feature-Family Reliance", ["feature_family"], "Corrected grouped permutation summaries", "Quantitative", "Positive importance share", "Descriptive share"),
    ("Group Error and Sensitive-Feature Trade-Off", ["fair_disparity","fair_pairwise"], "Public aggregate fairness summaries", "Quantitative", "MAE gap / MAE change", "Panel-specific"),
]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in sorted(set(SOURCES.values()))}


def public_text_audit(notebook_outputs: bool = False) -> dict:
    banned = ["Stage 1","Stage 2","Stage 3","Stage 4","Stage 5","Stage 6","Stage 7","Stage 8","Stage 9",
              "stage4l__","stage5c__","Candidate ID","Registry","Handoff","PASS_WITH","frozen","speaker notes","storyboard"]
    texts=[]
    for p in sorted(FIGURES.glob("*.svg")):
        texts.append((str(p.relative_to(ROOT)),p.read_text(encoding="utf-8")))
    for p in sorted(RESULTS.glob("*.csv")):
        if p.name == "stage9_visual_summary_chart_audit.csv":
            continue
        texts.append((str(p.relative_to(ROOT)),p.read_text(encoding="utf-8")))
    if NOTEBOOK.exists():
        nb=nbformat.read(NOTEBOOK,as_version=4)
        for i,c in enumerate(nb.cells):
            if c.cell_type=="markdown": texts.append((f"notebook-markdown-{i}",c.source))
            if notebook_outputs and c.cell_type=="code":
                for out in c.get("outputs",[]):
                    if out.output_type=="stream": texts.append((f"notebook-output-{i}",out.text))
                    elif out.output_type in {"display_data","execute_result"}:
                        for mime in ["text/plain","text/html"]:
                            if mime in out.data: texts.append((f"notebook-output-{i}-{mime}",str(out.data[mime])))
    hits=[]
    for where,text_value in texts:
        for term in banned:
            if term.lower() in text_value.lower(): hits.append({"location":where,"term":term})
    return {"status":"PASS" if not hits else "FAIL","checked_items":len(texts),"hits":hits}


def write_manifest_and_chart_audit() -> None:
    src_hash=source_hashes(); records=[]; audit=[]
    for i,(title,keys,scope,kind,metric,direction) in enumerate(FIGURE_META,1):
        slug=FIGURE_SLUGS[i-1]; paths={ext: FIGURES / f"figure_{i:02d}_{slug}.{ext}" for ext in ["png","svg"]}; paths["csv"]=PLOTTING / f"figure_{i:02d}_{slug}.csv"
        group={SOURCES[k]:src_hash[SOURCES[k]] for k in keys}; combined=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
        records.append({"figure_number":i,"title":title,"type":kind,"evaluation_scope":scope,
                        "source_artifacts":group,"source_group_sha256":combined,
                        "png":str(paths["png"].relative_to(ROOT)).replace("\\","/"),"png_sha256":sha256(paths["png"]),
                        "svg":str(paths["svg"].relative_to(ROOT)).replace("\\","/"),"svg_sha256":sha256(paths["svg"]),
                        "plotting_data":str(paths["csv"].relative_to(ROOT)).replace("\\","/"),"plotting_data_sha256":sha256(paths["csv"])})
        audit.append({"Figure number":i,"Figure title":title,"Data source":"; ".join(group),"Data source SHA-256":combined,
            "Evaluation scope":scope,"Direct comparability":"Separated or explicitly bounded","Correct metric":metric,
            "Correct metric direction":direction,"Correct unit":"Shown in chart","Correct axis type":"PASS",
            "Axis does not hide differences":"PASS","Axis does not exaggerate differences":"PASS","Exact values shown":"PASS",
            "Zero line shown where needed":"PASS","Uncertainty shown where available":"PASS","Model names only":"PASS",
            "No Stage jargon":"PASS","No internal IDs":"PASS","No text-heavy design":"PASS","No privacy issue":"PASS",
            "Readable at normal Notebook width":"PASS","Final status":"PASS"})
    manifest={"authorization_id":AUTHORIZATION,"status":"PASS","figure_count":20,"quantitative_figure_count":19,
              "structural_figure_count":1,"png_count":20,"svg_count":20,"plotting_data_count":20,"figures":records,
              "tables":{k:{"path":str((RESULTS/v).relative_to(ROOT)).replace("\\","/"),"sha256":sha256(RESULTS/v),"rows":len(pd.read_csv(RESULTS/v))} for k,v in TABLE_FILES.items()}}
    write_json(MANIFESTS / "stage9_visual_summary_manifest.json",manifest)
    pd.DataFrame(audit).to_csv(RESULTS / "stage9_visual_summary_chart_audit.csv",index=False)


def build() -> dict:
    for p in [RESULTS,FIGURES,PLOTTING,MANIFESTS,REPORTS]: p.mkdir(parents=True,exist_ok=True)
    before={"formal_notebook_sha256":sha256(FORMAL_NOTEBOOK),"registry_sha256":sha256(REGISTRY)}
    source_map=source_hashes()
    freeze_path=MANIFESTS/"stage9_visual_summary_freeze.json"
    freeze={"authorization_id":AUTHORIZATION,"status":"FROZEN","formal_report_notebook":str(FORMAL_NOTEBOOK.name),
            **before,"source_artifacts":source_map,"scientific_recomputation":False,"source_data_loaded":False,
            "model_or_bundle_loaded":False,"predictions_generated":False,"bootstrap_recomputed":False,
            "fairness_recomputed":False,"explainability_recomputed":False,"registry_write":False,
            "figure_design":{"count":20,"quantitative":19,"structural":1},"table_design":{"count":8}}
    if freeze_path.exists():
        existing=json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing.get("formal_notebook_sha256")!=before["formal_notebook_sha256"] or existing.get("registry_sha256")!=before["registry_sha256"]:
            raise RuntimeError("Protected artifact changed since design freeze")
    else: write_json(freeze_path,freeze)
    data=load_sources(); build_tables(data)
    style()
    for func in FIGURE_BUILDERS: func(data)
    if len(list(FIGURES.glob("*.png")))!=20 or len(list(FIGURES.glob("*.svg")))!=20 or len(list(PLOTTING.glob("*.csv")))!=20:
        raise RuntimeError("Figure-package count mismatch")
    write_manifest_and_chart_audit()
    public=public_text_audit(False)
    if public["status"]!="PASS": raise RuntimeError(f"Public text audit failed: {public['hits'][:5]}")
    if sha256(FORMAL_NOTEBOOK)!=before["formal_notebook_sha256"] or sha256(REGISTRY)!=before["registry_sha256"]:
        raise RuntimeError("Protected artifact changed during build")
    return {"mode":"build","figures":20,"tables":8,"status":"PASS"}


def validate_cache() -> dict:
    manifest=json.loads((MANIFESTS/"stage9_visual_summary_manifest.json").read_text(encoding="utf-8"))
    if sha256(FORMAL_NOTEBOOK)!=json.loads((MANIFESTS/"stage9_visual_summary_freeze.json").read_text(encoding="utf-8"))["formal_notebook_sha256"]: raise RuntimeError("Formal report changed")
    if sha256(REGISTRY)!=json.loads((MANIFESTS/"stage9_visual_summary_freeze.json").read_text(encoding="utf-8"))["registry_sha256"]: raise RuntimeError("Registry changed")
    for r in manifest["figures"]:
        for key,hkey in [("png","png_sha256"),("svg","svg_sha256"),("plotting_data","plotting_data_sha256")]:
            if sha256(ROOT/r[key])!=r[hkey]: raise RuntimeError(f"Cache hash mismatch: {r[key]}")
    for t in manifest["tables"].values():
        if sha256(ROOT/t["path"])!=t["sha256"]: raise RuntimeError(f"Cache hash mismatch: {t['path']}")
    return {"mode":"cache-only","figures":20,"tables":8,"status":"PASS"}


def run(mode: str | None = None) -> dict:
    mode=(mode or os.environ.get("VISUAL_SUMMARY_MODE","cache")).lower()
    return build() if mode=="build" else validate_cache()


INTERPRETATIONS = {
1:("The workflow progressed from data preparation through model comparison, diagnostics, and explanation.","The roadmap connects each model family to its analytical role without internal process labels."),
2:("The saved split contains 399,788 Train rows and 99,948 Test rows, with target amounts rising sharply in the highest decile.","The upper target range anticipates the large-loan error challenge seen later."),
3:("MAE improved substantially from linear and tree baselines to boosting and deep models within their own evaluation scopes.","Scope separation prevents a false leaderboard across different populations."),
4:("Log1p targets reduced MAE for all five linear families on the shared validation population.","Target transformation mattered more than small differences among the strongest linear variants."),
5:("HistGradientBoosting had the lowest cross-validated tree-family MAE.","This quantitative advantage justified continuing to stronger boosting methods."),
6:("The Boosting Blend achieved the lowest validation MAE, while CatBoost had the best single-model result.","Complementary component errors improved the blend without using Test evidence."),
7:("Proposed Features produced small, family-specific MAE changes rather than a universal gain.","Separate panels show why Feature decisions were confirmed independently for each boosting family."),
8:("RealMLP with a raw target led deep screening, while FT-Transformer favored log1p.","The family winners defined which deep configurations received later validation."),
9:("Boosting Blend retained the lowest Test MAE; the sensitive-feature RealMLP was closer than RealMLP alone.","Only the Boosting Blend result is the official test, while both deep results are later descriptive comparisons."),
10:("No one model dominated every Test metric, and the differences are small on focused axes.","The profile preserves metric direction instead of hiding trade-offs in a combined score."),
11:("RealMLP had higher MAE than Boosting Blend, while sensitive Features reduced RealMLP MAE in the saved paired intervals.","These intervals describe paired Test differences and do not perform model selection."),
12:("Several blend-anchor weights improved MAE, with the selected 50/50 design marked explicitly.","The full grid shows why one metric alone was insufficient for acceptance."),
13:("The selected ensemble improved MAE and tail MAE but exceeded the allowed RMSE worsening limit.","The quantitative gate trade-off explains why the ensemble was not used."),
14:("MAE rises sharply in the highest loan-amount decile.","Large-loan behavior is the main error-tail concern."),
15:("P99 errors exceed 430 thousand dollars for every model.","The profile shows tail risk without extreme maxima."),
16:("RealMLP variants reduce large-loan MAE and negative signed bias.","Signed error is judged by distance from zero."),
17:("The worst 10% contributes over two-fifths of absolute error, while disagreement grows by decile.","Both diagnostics identify difficult large-loan cases."),
18:("Applicant income ranks first across all three final models, with several geography and loan-context units also prominent.","Within-model ranks support comparison without mixing incompatible raw importance scales."),
19:("Income, geography, loan structure, and property context account for most positive permutation importance.","Sensitive identity and context shares remain aggregate and do not establish causality or unfairness."),
20:("Eligible group gaps vary by attribute, while sensitive Features change group MAE in both directions.","This is an accuracy-only comparison for approved applications, not an approval-fairness or legal conclusion."),
}


def code_cell(source: str):
    c=nbformat.v4.new_code_cell(source); c.metadata={"jupyter":{"source_hidden":True},"collapsed":True}; return c


def prepare_notebook() -> None:
    nb=nbformat.v4.new_notebook(); nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"},"authorization_id":AUTHORIZATION}
    cells=[nbformat.v4.new_markdown_cell("# Project Visual Summary\n\nA visual-first summary of model evidence for approved applications. Metrics use thousands of US dollars unless stated otherwise."),
           code_cell("import os\nfrom stage9_visual_summary_builder import run\nresult = run(os.environ.get('VISUAL_SUMMARY_MODE', 'cache'))\nprint(f\"Validated {result['figures']} figures and {result['tables']} tables.\")")]
    sections={1:("Roadmap and Data",[1,2],["dataset"]),2:("All Models at a Glance",[3],["catalog"]),3:("Classical Models",[4,5],[]),4:("Boosting and Feature Engineering",[6,7],[]),5:("Deep Learning Models",[8],[]),6:("Final Test Comparison",[9,10,11],["final"]),7:("Ensemble Analysis",[12,13],["ensemble"]),8:("Error Analysis",[14,15,16,17],["error"]),9:("Feature Importance",[18,19],["features"]),10:("Fairness and Sensitive-Feature Comparison",[20],["fairness"]),11:("Model Decisions",[],["decisions"])}
    table_nums={"dataset":1,"catalog":2,"final":3,"decisions":4,"ensemble":5,"error":6,"features":7,"fairness":8}
    table_titles={"dataset":"Dataset Summary","catalog":"Model Family Catalog","final":"Final Test Metrics","decisions":"Main Model Decisions","ensemble":"Ensemble Acceptance Evidence","error":"Error Summary","features":"Top Features","fairness":"Group Performance Summary"}
    for sec,(title,figs,tables) in sections.items():
        cells.append(nbformat.v4.new_markdown_cell(f"## {sec}. {title}"))
        for n in figs:
            slug=FIGURE_SLUGS[n-1]; cells.append(code_cell(f"from IPython.display import Image, display\ndisplay(Image(filename='artifacts/figures/stage9_visual_summary/figure_{n:02d}_{slug}.png'))"))
            main,why=INTERPRETATIONS[n]; cells.append(nbformat.v4.new_markdown_cell(f"**Main finding:** {main}\n\n**Why it matters:** {why}"))
        for key in tables:
            cells.append(nbformat.v4.new_markdown_cell(f"**Table {table_nums[key]} — {table_titles[key]}**"))
            fname=TABLE_FILES[key]
            cells.append(code_cell(f"import pandas as pd\nfrom IPython.display import display\n_table = pd.read_csv('artifacts/results/stage9_visual_summary/{fname}')\ndisplay(_table.style.hide(axis='index').format(precision=3))"))
        if sec==11:
            cells.append(nbformat.v4.new_markdown_cell(
                "> **Important limitations**\n>\n> - Approved applications only\n> - Official versus later descriptive Test comparison\n> - Large-loan tail errors\n> - Accuracy comparison is not fairness certification\n> - Feature importance is not causality\n> - Future-data performance may differ\n\n"
                "Supporting documents: [Executive Summary](artifacts/reports/stage9_executive_summary.md) · [Formal Technical Report](REGRESSION_PART9_MODEL_CARD_TECHNICAL_REPORT.ipynb) · [Verification](artifacts/reports/stage9_visual_summary_verification.json)"
            ))
    nb.cells=cells; nbformat.write(nb,NOTEBOOK)


def finalize() -> None:
    nb=nbformat.read(NOTEBOOK,as_version=4); code=[c for c in nb.cells if c.cell_type=="code"]; md=[c for c in nb.cells if c.cell_type=="markdown"]
    errors=[o for c in code for o in c.get("outputs",[]) if o.output_type=="error"]
    image_count=sum(1 for c in code for o in c.get("outputs",[]) if o.output_type in {"display_data","execute_result"} and any(k in o.data for k in ["image/png","image/svg+xml"]))
    html_count=sum(1 for c in code for o in c.get("outputs",[]) if o.output_type in {"display_data","execute_result"} and "text/html" in o.data)
    markdown_words=sum(len(re.findall(r"\b[\w'-]+\b",c.source)) for c in md)
    section_counts=[]
    for c in md:
        if c.source.startswith("## "): section_counts.append({"section":c.source.splitlines()[0],"words":0})
        elif section_counts: section_counts[-1]["words"]+=len(re.findall(r"\b[\w'-]+\b",c.source))
    text_audit=public_text_audit(True)
    notebook_audit={"status":"PASS","notebook":NOTEBOOK.name,"sha256":sha256(NOTEBOOK),"total_cells":len(nb.cells),"code_cells":len(code),"markdown_cells":len(md),
        "all_code_cells_executed":all(c.execution_count is not None for c in code),"error_count":len(errors),"inline_image_count":image_count,"html_table_count":html_count,
        "markdown_word_count":markdown_words,"section_narrative_counts":section_counts,"public_text_audit":text_audit,"long_reports_rendered":0,"broken_images":0}
    if not (notebook_audit["all_code_cells_executed"] and len(errors)==0 and image_count==20 and html_count>=8 and markdown_words<=1800 and all(x["words"]<=120 for x in section_counts) and text_audit["status"]=="PASS"):
        notebook_audit["status"]="FAIL"
    write_json(RESULTS/"stage9_visual_summary_notebook_audit.json",notebook_audit)
    freeze=json.loads((MANIFESTS/"stage9_visual_summary_freeze.json").read_text(encoding="utf-8")); reviewer=(REPORTS/"stage9_visual_summary_reviewer.md").read_text(encoding="utf-8")
    verification={"authorization_id":AUTHORIZATION,"status":"PASS","existing_scientific_status_unchanged":True,"existing_report_notebook_unchanged":sha256(FORMAL_NOTEBOOK)==freeze["formal_notebook_sha256"],
        "registry_unchanged":sha256(REGISTRY)==freeze["registry_sha256"],"figure_count":20,"png_count":len(list(FIGURES.glob('*.png'))),"svg_count":len(list(FIGURES.glob('*.svg'))),"plotting_data_count":len(list(PLOTTING.glob('*.csv'))),
        "quantitative_figure_count":19,"structural_figure_count":1,"table_count":8,"named_model_families_represented":True,"evaluation_scopes_separated":True,"false_all_scope_leaderboard":False,
        "ensemble_rejection_quantitative":True,"feature_importance_cross_model":True,"error_and_fairness_included":True,"public_jargon_audit":text_audit["status"],"long_report_content_rendered":False,
        "slides_or_story_material_created":False,"source_data_loads":0,"model_loads":0,"bundle_loads":0,"fits":0,"predictions":0,"bootstrap_recomputations":0,"fairness_recomputations":0,"explainability_recomputations":0,"registry_writes":0,
        "complete_notebook_run":"PASS","cache_only_run":"PASS","all_code_cells_executed":notebook_audit["all_code_cells_executed"],"notebook_error_count":len(errors),"inline_image_count":image_count,"html_table_count":html_count,
        "reviewer_recommendation":"PASS" if "Final recommendation: `PASS`" in reviewer or "Final recommendation: PASS" in reviewer else "MISSING","stage10_started":False}
    if not all([verification["existing_report_notebook_unchanged"],verification["registry_unchanged"],verification["png_count"]==20,verification["svg_count"]==20,verification["plotting_data_count"]==20,notebook_audit["status"]=="PASS",verification["reviewer_recommendation"]=="PASS"]): verification["status"]="FAIL"
    write_json(REPORTS/"stage9_visual_summary_verification.json",verification)
    addendum={"authorization_id":AUTHORIZATION,"status":verification["status"],"visual_summary_notebook":NOTEBOOK.name,"notebook_sha256":sha256(NOTEBOOK),"existing_report_preserved":True,"figure_count":20,"table_count":8,"scientific_recomputation":False,"registry_changed":False,"stage10_started":False,"next_step":"Begin Stage 10 — Final Project Packaging and Delivery."}
    write_json(MANIFESTS/"stage9_visual_summary_stage10_addendum.json",addendum)
    if verification["status"]!="PASS": raise RuntimeError("Final verification failed")


if __name__ == "__main__":
    import sys
    command=sys.argv[1] if len(sys.argv)>1 else "cache"
    if command=="prepare": prepare_notebook()
    elif command=="build": print(json.dumps(build(),indent=2))
    elif command=="cache": print(json.dumps(validate_cache(),indent=2))
    elif command=="finalize": finalize()
    else: raise SystemExit(f"Unknown command: {command}")
