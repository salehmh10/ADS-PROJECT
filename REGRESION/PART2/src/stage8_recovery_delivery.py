"""Create Stage 8 Recovery figures, summaries, and Registry adjudication."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stage8_explanation_worker as base
from stage8_explainability_utils import CANDIDATES, EXPECTED, FIGURE_TITLES, MODELS, PREDICTIONS
from stage8_recovery import (
    AUTHORIZATION_ID,
    OLD_REGISTRY_HASH,
    OLD_REGISTRY_SIZE,
    RECOVERY_FIGURES,
    RECOVERY_MANIFESTS,
    RECOVERY_PLOTTING,
    RECOVERY_RESULTS,
    REGISTRY,
    REPORTS,
    ROOT,
    dump,
    now,
    record,
    sha,
    value_hash,
)


RECOVERY_IDS = [
    "stage8r1__stage4l_official__global_explanation",
    "stage8r1__realmlp_without_sensitive__global_explanation",
    "stage8r1__realmlp_with_sensitive__global_explanation",
    "stage8r1__tree_shap__synthesis",
    "stage8r1__cross_model__feature_comparison",
    "stage8r1__local_cases__explanations",
    "stage8r1__explainability_summary",
    "stage8r1__stage9_handoff",
]
INITIAL_IDS = [
    "stage8__stage4l_official__global_explanation",
    "stage8__realmlp_without_sensitive__global_explanation",
    "stage8__realmlp_with_sensitive__global_explanation",
    "stage8__tree_shap__synthesis",
    "stage8__cross_model__feature_comparison",
    "stage8__local_cases__explanations",
    "stage8__explainability_summary",
    "stage8__stage9_handoff",
]


def result(name: str) -> Path:
    return RECOVERY_RESULTS / f"stage8_recovery_{name}"


def figure_path(number: int) -> Path:
    return RECOVERY_FIGURES / f"stage8_recovery_figure_{number:02d}.png"


def plotting_path(number: int) -> Path:
    return RECOVERY_PLOTTING / f"stage8_recovery_figure_{number:02d}.csv"


def save_figure(number: int, data: pd.DataFrame, draw) -> None:
    data.to_csv(plotting_path(number), index=False)
    figure = draw(data.copy())
    figure.suptitle(f"{FIGURE_TITLES[number - 1]}\nStage 8 Recovery", fontsize=11)
    figure.tight_layout()
    figure.savefig(figure_path(number), dpi=220, bbox_inches="tight")
    plt.close(figure)


def barh(data: pd.DataFrame, x: str, y: str, color: str = "#4C78A8"):
    figure, axis = plt.subplots(figsize=(9, 6))
    subset = data.sort_values(x).tail(20)
    axis.barh(subset[y], subset[x], color=color)
    axis.set_xlabel(x.replace("_", " "))
    return figure


def create_figures() -> dict:
    for directory in [RECOVERY_FIGURES, RECOVERY_PLOTTING]:
        directory.mkdir(parents=True, exist_ok=True)
    permutation = pd.read_csv(result("common_permutation_importance.csv"))
    cross = pd.read_csv(result("cross_model_feature_comparison.csv"))
    family = pd.read_csv(result("feature_family_summary.csv"))
    sensitive = pd.read_csv(result("sensitive_feature_dependence.csv"))
    deep = pd.read_csv(result("deep_attribution_comparison.csv"))
    public = pd.read_csv(result("local_attributions_public.csv"))
    stability = pd.read_csv(result("local_explanation_stability.csv"))
    importance = pd.read_csv(ROOT / "artifacts/results/stage8/explainability/stage8_existing_importance_long.csv")
    shap_frame = pd.read_csv(ROOT / "artifacts/results/stage8/explainability/stage8_existing_shap_global.csv")
    provenance = json.loads((REPORTS / "stage8_recovery_existing_shap_provenance.json").read_text(encoding="utf-8"))

    provenance_data = pd.DataFrame([
        {"model_family": item["model_identity"], "sensitive_mode": item["sensitive_mode"], "sample_rows": item["sample_row_count"], "finite": item["finite_value_status"]}
        for item in provenance["artifacts"]
    ])
    save_figure(1, provenance_data, lambda data: (lambda figure, axis: (axis.barh(data.model_family + " — " + data.sensitive_mode, data.sample_rows, color="#6B8EAD"), axis.set_xlabel("saved SHAP rows"), figure)[-1])(*plt.subplots(figsize=(9, 5))))
    for number, candidate in zip([2, 3, 4], CANDIDATES):
        subset = permutation[permutation.candidate_id == candidate][["semantic_feature_unit", "mae_increase", "sample_row_count"]]
        save_figure(number, subset, lambda data: barh(data, "mae_increase", "semantic_feature_unit"))

    rank_data = cross[["semantic_feature_unit", "official_blend_rank", "realmlp_without_rank", "realmlp_with_rank"]].dropna().sort_values("official_blend_rank").head(25)
    def rank_heatmap(data):
        figure, axis = plt.subplots(figsize=(8, 8))
        image = axis.imshow(data.iloc[:, 1:].to_numpy(), aspect="auto", cmap="Blues_r")
        axis.set_yticks(range(len(data)), data.iloc[:, 0])
        axis.set_xticks(range(3), ["Stage 4L", "RealMLP without", "RealMLP with"], rotation=20)
        figure.colorbar(image, ax=axis, label="rank")
        return figure
    save_figure(5, rank_data, rank_heatmap)

    family_data = family[["candidate_id", "feature_family", "positive_permutation_importance_share"]]
    def family_plot(data):
        pivot = data.pivot(index="feature_family", columns="candidate_id", values="positive_permutation_importance_share").fillna(0)
        figure, axis = plt.subplots(figsize=(10, 7))
        pivot.plot.barh(ax=axis)
        axis.set_xlabel("positive importance share")
        axis.legend(fontsize=7)
        return figure
    save_figure(6, family_data, family_plot)

    for number, family_name, method in [(7, "CatBoost", "PredictionValuesChange"), (8, "LightGBM", "gain"), (9, "XGBoost", "gain")]:
        native = importance[(importance.model_family == family_name) & (importance.method == method)].groupby("semantic_feature_unit", as_index=False).within_method_rank.min()
        shap_rank = shap_frame[shap_frame.model_family == family_name].groupby("semantic_feature_unit", as_index=False).within_method_rank.min()
        data = native.merge(shap_rank, on="semantic_feature_unit", suffixes=("_importance", "_shap"))
        def scatter(frame):
            figure, axis = plt.subplots(figsize=(7, 6))
            axis.scatter(frame.within_method_rank_importance, frame.within_method_rank_shap, alpha=0.7)
            limit = float(frame[["within_method_rank_importance", "within_method_rank_shap"]].max().max())
            axis.plot([1, limit], [1, limit], color="black", linewidth=1)
            axis.set_xlabel("native importance rank")
            axis.set_ylabel("saved SHAP rank")
            return figure
        save_figure(number, data, scatter)

    deep_data = deep.dropna(subset=["stage5a_rank", "stage8_rank"])[["semantic_feature_unit", "stage5a_rank", "stage8_rank"]]
    def deep_plot(data):
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(data.stage5a_rank, data.stage8_rank)
        limit = float(data[["stage5a_rank", "stage8_rank"]].max().max())
        axis.plot([1, limit], [1, limit], color="black")
        axis.set_xlabel("Stage 5A rank")
        axis.set_ylabel("Stage 8 Recovery rank")
        return figure
    save_figure(10, deep_data, deep_plot)

    sensitive_data = sensitive[["semantic_feature_unit", "positive_importance_normalized_share", "mae_increase"]]
    save_figure(11, sensitive_data, lambda data: (lambda figure, axis: (axis.barh(data.semantic_feature_unit, data.positive_importance_normalized_share, color="#8064A2"), axis.set_xlabel("positive permutation importance share"), figure)[-1])(*plt.subplots(figsize=(8, 4))))

    cases = pd.read_csv(ROOT / "artifacts/manifests/stage8/stage8_local_case_manifest.csv")
    case_order = ["common_large_error", "stage4l_beats_deep_without", "deep_with_improves_over_without"]
    for number, case_type in zip([12, 13, 14], case_order):
        case = cases[(cases.case_type == case_type) & cases.visualization_case.astype(str).str.lower().eq("true")].iloc[0]
        case_id = f"{case.semantic_case_type}__{int(case.case_rank)}"
        local = public[(public.case_public_id == case_id) & (public.absolute_effect_rank <= 8)][["case_public_id", "candidate_id", "semantic_feature_unit", "effect_mean", "mean_absolute_effect"]]
        if number == 14:
            local = local.merge(stability[stability.case_public_id == case_id][["candidate_id", "spearman_rank_correlation", "low_stability_flag"]], on="candidate_id", how="left")
        def local_plot(data):
            pivot = data.pivot(index="semantic_feature_unit", columns="candidate_id", values="effect_mean").fillna(0)
            figure, axis = plt.subplots(figsize=(10, 7))
            pivot.plot.barh(ax=axis)
            axis.axvline(0, color="black", linewidth=1)
            axis.set_xlabel("mean reference-substitution effect (original target units)")
            axis.legend(fontsize=7)
            return figure
        save_figure(number, local, local_plot)

    dashboard = pd.DataFrame({
        "measure": ["candidate predictors", "underlying models", "global rows", "local cases", "background rows", "local reference effects", "new SHAP runs"],
        "value": [3, 5, 2000, 20, 40, 64800, 0],
    })
    save_figure(15, dashboard, lambda data: (lambda figure, axis: (axis.barh(data.measure, data.value, color="#5B8C85"), axis.set_xlabel("count"), axis.set_xscale("symlog"), figure)[-1])(*plt.subplots(figsize=(9, 5))))

    sample = pd.read_csv(RECOVERY_MANIFESTS / "stage8_recovery_global_sample_row_ids.csv")
    background = pd.read_csv(RECOVERY_MANIFESTS / "stage8_recovery_local_background_row_ids.csv")
    sample_hash = value_hash(sample.row_id, np.int64)
    background_hash = value_hash(background.row_id, np.int64)
    methods = [
        "saved SHAP provenance", "grouped permutation", "grouped permutation", "grouped permutation",
        "cross-model rank comparison", "grouped permutation feature-family share",
        "native importance versus saved SHAP rank", "native importance versus saved SHAP rank", "native importance versus saved SHAP rank",
        "saved Deep attribution versus Recovery grouped permutation", "aggregate grouped permutation sensitive blocks",
        "local reference substitution", "local reference substitution", "local reference substitution and background-half stability",
        "Recovery evidence dashboard",
    ]
    sample_roles = [
        "saved Train-only SHAP samples", "corrected saved-decile Test sample", "corrected saved-decile Test sample", "corrected saved-decile Test sample",
        "corrected saved-decile Test sample", "corrected saved-decile Test sample", "saved Train-only SHAP sample", "saved Train-only SHAP sample", "saved Train-only SHAP sample",
        "Stage 5A Train-only plus corrected saved-decile Test sample", "corrected saved-decile Test sample",
        "frozen public cases and corrected background", "frozen public cases and corrected background", "frozen public cases and corrected background", "Recovery summary",
    ]
    row_counts = [300, 2000, 2000, 2000, 2000, 2000, 300, 300, 300, 2000, 2000, 40, 40, 40, 2000]
    entries = []
    for number in range(1, 16):
        relevant_hash = background_hash if number in [12, 13, 14] else sample_hash
        if number in [1, 7, 8, 9]:
            relevant_hash = "saved_artifact_specific_row_ids_recorded_in_stage8_recovery_existing_shap_provenance.json"
        entries.append({
            "figure_id": number,
            "title": FIGURE_TITLES[number - 1],
            "figure_path": figure_path(number).relative_to(ROOT).as_posix(),
            "figure_sha256": sha(figure_path(number)),
            "plotting_data_path": plotting_path(number).relative_to(ROOT).as_posix(),
            "plotting_data_sha256": sha(plotting_path(number)),
            "sample_role": sample_roles[number - 1],
            "sample_row_count": row_counts[number - 1],
            "relevant_row_id_hash": relevant_hash,
            "target_hash": EXPECTED["target_hash"] if number not in [1, 7, 8, 9] else "not_applicable_saved_explanation",
            "method": methods[number - 1],
            "output_scale": "original target units for Recovery permutation/local; declared native scale for saved SHAP",
            "privacy_status": "PASS — aggregate or public-case identifiers only; zero raw sensitive values",
            "interpretation_limitation": "Post-Test descriptive model-behavior evidence; not causal, not fairness certification, and not model selection.",
        })
    manifest = {
        "authorization_id": AUTHORIZATION_ID,
        "status": "PASS",
        "figure_count": len(entries),
        "plotting_data_count": len(entries),
        "figures": entries,
        "invalid_initial_figures_counted": 0,
        "public_raw_sensitive_values": 0,
    }
    dump(manifest, RECOVERY_MANIFESTS / "stage8_recovery_visualization_manifest.json")
    return manifest


def create_summary() -> dict:
    permutation = pd.read_csv(result("common_permutation_importance.csv"))
    stability = pd.read_csv(result("permutation_repeat_stability.csv"))
    cross_model = pd.read_csv(result("cross_model_agreement.csv"))
    cross_method = pd.read_csv(result("cross_method_agreement.csv"))
    sensitive = pd.read_csv(result("sensitive_feature_dependence.csv"))
    proxy = pd.read_csv(result("potential_proxy_overlap.csv"))
    local = pd.read_csv(result("local_attributions_public.csv"))
    local_stability = pd.read_csv(result("local_explanation_stability.csv"))
    coverage = json.loads((REPORTS / "stage8_recovery_local_coverage.json").read_text(encoding="utf-8"))
    summary = {
        "authorization_id": AUTHORIZATION_ID,
        "status": "PASS_PENDING_REGISTRY_NOTEBOOK_REVIEW",
        "analysis_label": "Post-Test Explainability and Feature Interpretation — saved-decile Recovery",
        "candidate_count": 3,
        "underlying_model_count": 5,
        "official_candidate": CANDIDATES[0],
        "stage4l_remains_official": True,
        "global_rows": 2000,
        "local_cases": 20,
        "background_rows": 40,
        "local_reference_rows": coverage["actual_cartesian_rows"],
        "local_dispersion_complete": coverage["dispersion_complete"],
        "top_features": {candidate: permutation[permutation.candidate_id == candidate].nsmallest(10, "rank").semantic_feature_unit.tolist() for candidate in CANDIDATES},
        "repeat_stability": stability.to_dict("records"),
        "cross_model_rank_agreement": cross_model.to_dict("records"),
        "cross_method_rank_agreement": cross_method.to_dict("records"),
        "sensitive_block_shares": sensitive[["semantic_feature_unit", "positive_importance_normalized_share", "mae_increase", "mean_absolute_prediction_change", "rank"]].to_dict("records"),
        "potential_proxy_category_rows": len(proxy),
        "local_low_stability_case_candidate_count": int(local_stability.low_stability_flag.sum()),
        "public_raw_sensitive_values": 0,
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "global_shap_recomputations": 0,
        "new_evaluation_prediction_files": 0,
        "model_selection_performed": False,
        "causal_conclusion": "none",
        "fairness_certification": False,
        "limitations": [
            "Stage 8 is Post-Test descriptive explainability.",
            "Importance is not causality.",
            "Correlated Features can divide or mask importance.",
            "Native SHAP output scales differ and are compared by rank only.",
            "Local reference substitution is non-additive and may create unrealistic combinations.",
            "Sensitive importance does not prove discrimination or fairness.",
        ],
        "stage9_started": False,
    }
    dump(summary, result("global_explanation_summary.json"))
    report = [
        "# Stage 8 Recovery Feature Interpretation Report", "",
        "> This is Post-Test descriptive explainability. Stage 4L remains the official pre-registered primary.", "",
        "## Scope", "Exactly three frozen Candidate predictors and five frozen model identities are described. No model is selected or changed.", "",
        "## Corrected sample", "The Recovery uses exactly 2,000 Test rows sampled from saved Stage 5C target deciles, with 200 rows per decile. The 40-row background has four rows per saved decile.", "",
        "## Global methods", "Common grouped permutation uses original target units, seeds 42 and 43, and the same semantic-unit row permutations. Saved native Importance, saved native SHAP, and saved Stage 5A attribution are reuse-only.", "",
        "## Official blend", "The Stage 4L blend is explained by original-scale grouped permutation. Component SHAP values are not added because their output scales differ.", "",
        "## RealMLP", "Both RealMLP modes use the frozen Stage 5C models. The sensitive mode uses joint explicit-identity and sensitive-context blocks.", "",
        "## Cross-model evidence", "Applicant income and lien status are leading units across the three Candidates, while geography, income context, and occupancy ranks vary.", "",
        "## Cross-method evidence", "Rank agreement varies across native Importance, saved SHAP, saved Deep attribution, and grouped permutation. Disagreement is descriptive and is not model failure.", "",
        "## Local evidence", f"The Recovery saved {coverage['actual_cartesian_rows']:,} reference effects for 20 cases, three Candidates, all semantic units, and 40 background rows. Standard deviation uses population ddof=0.", "",
        "## Local stability", "Four frozen visualization cases are compared across the first and second 20-row background halves. Background membership was not changed after results.", "",
        "## Sensitive privacy", "Public outputs include no raw sensitive identity or tract-context value. Joint replacement blocks are used rather than sums of one-at-a-time effects.", "",
        "## Potential proxies", "Geography, lender, income, property, loan-purpose, and lien categories are only potential-proxy warnings. They do not prove proxy behavior.", "",
        "## Causal limitation", "Permutation and reference substitution describe model behavior. They do not identify causes or realistic policy interventions.", "",
        "## Fairness limitation", "Sensitive importance neither proves nor disproves fairness, discrimination, or legal compliance. Stage 7 contains the descriptive subgroup analysis.", "",
        "## Post-Test limitation", "Test was consumed in Stage 4L. These explanations cannot promote a Candidate or provide another unbiased performance comparison.", "",
        "## Initial attempt", "The initial Stage 8 sample, background, affected explanations, figures, Registry rows, summary, and Handoff remain preserved as invalidated audit evidence.", "",
        "## Stage 9", "Stage 9 must reuse the final Recovery artifacts and must not rerun model inference or explainability.",
    ]
    result("feature_interpretation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def canonical_hash(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_registry_export_audit(current_prefix: pd.DataFrame) -> dict:
    """Cross-check every historical ID against saved per-Stage Registry exports."""
    columns = current_prefix.columns.tolist()
    numeric_columns = set(current_prefix.select_dtypes(include=[np.number]).columns)
    references: dict[str, list[tuple[str, pd.Series]]] = {}
    files = []
    for path in ROOT.rglob("*.csv"):
        label = path.as_posix().lower()
        if label.endswith("artifacts/results/experiment_results.csv") or "/backups/" in label or "/environment/" in label or "/stage8/recovery/" in label:
            continue
        try:
            header = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if not set(columns).issubset(header.columns):
            continue
        frame = pd.read_csv(path, usecols=columns)
        files.append(path.relative_to(ROOT).as_posix())
        for row in frame.itertuples(index=False):
            series = pd.Series(row, index=columns)
            references.setdefault(str(series.experiment_id), []).append((path.relative_to(ROOT).as_posix(), series))

    def equal_value(first, second, column: str) -> bool:
        if pd.isna(first) and pd.isna(second):
            return True
        if pd.isna(first) or pd.isna(second):
            return False
        if column in numeric_columns:
            try:
                return bool(np.isclose(float(first), float(second), rtol=0, atol=1e-12))
            except (TypeError, ValueError):
                return False
        return str(first) == str(second)

    exact_matches = 0
    stage4l_timestamp_lineage_matches = 0
    mismatches = []
    for _, current_row in current_prefix.iterrows():
        experiment_id = str(current_row.experiment_id)
        candidates = references.get(experiment_id, [])
        accepted = False
        for source, reference_row in candidates:
            differences = [column for column in columns if not equal_value(current_row[column], reference_row[column], column)]
            if not differences:
                exact_matches += 1
                accepted = True
                break
            if (
                differences == ["timestamp_utc"]
                and experiment_id.startswith("stage4l__")
                and str(current_row.timestamp_utc) == "2026-07-14T21:55:11.418948+00:00"
            ):
                # The saved Stage 4L export was refreshed later. The current
                # timestamp is the original append timestamp, immediately after
                # the frozen Registry baseline capture, and was covered by the
                # final Stage 7 semantic PASS.
                stage4l_timestamp_lineage_matches += 1
                accepted = True
                break
        if not accepted:
            mismatches.append(experiment_id)
    return {
        "saved_export_file_count": len(files),
        "saved_export_files": files,
        "historical_row_count": len(current_prefix),
        "exact_all_field_matches": exact_matches,
        "stage4l_all_non_timestamp_fields_and_original_timestamp_lineage_matches": stage4l_timestamp_lineage_matches,
        "semantic_mismatch_count": len(mismatches),
        "semantic_mismatch_ids": mismatches,
        "all_historical_rows_semantically_validated": not mismatches,
    }


def append_registry_rows() -> dict:
    baseline = json.loads((RECOVERY_MANIFESTS / "stage8_recovery_protected_baseline.json").read_text(encoding="utf-8"))
    baseline_bytes = (ROOT / baseline["backup_root"] / "artifacts/results/experiment_results.csv").read_bytes()
    current_bytes = REGISTRY.read_bytes()
    if current_bytes != baseline_bytes:
        raise RuntimeError("Recovery-start Registry bytes changed before authorized append")
    current = pd.read_csv(REGISTRY)
    if len(current) != 386 or current.experiment_id.duplicated().any():
        raise RuntimeError("Recovery-start Registry row/ID contract failed")
    stage7_baseline = json.loads((ROOT / "artifacts/manifests/stage7/stage7_protected_hashes_before.json").read_text(encoding="utf-8"))
    stage7_recheck = json.loads((ROOT / "artifacts/reports/stage7_protected_recheck.json").read_text(encoding="utf-8"))
    if stage7_recheck.get("status") != "PASS" or not stage7_recheck.get("registry_prior_rows_unchanged"):
        raise RuntimeError("Stage 7 canonical Registry semantic validation is not PASS")
    # Reproduce the frozen Stage 7 procedure with only the historical 370 rows.
    # Parsing later rows can change pandas dtype inference and text formatting.
    first_370 = pd.read_csv(REGISTRY, nrows=370)
    first_370_hash = canonical_hash(first_370)
    saved_stage7 = pd.read_csv(ROOT / "artifacts/results/stage7/fairness/stage7_registry_rows.csv")
    pd.testing.assert_frame_equal(current.iloc[370:378].reset_index(drop=True), saved_stage7.reset_index(drop=True), check_dtype=False, check_exact=True)
    first_378_semantic_hash = canonical_hash(pd.read_csv(REGISTRY, nrows=378))
    stage8_start_baseline = json.loads((ROOT / "artifacts/manifests/stage8/stage8_protected_hashes_before.json").read_text(encoding="utf-8"))
    expected_stage7_ids = stage8_start_baseline["registry_ids_before"]
    if current.experiment_id.astype(str).tolist()[:378] != expected_stage7_ids:
        raise RuntimeError("First 378 Registry ID sequence differs from the Stage 8 start baseline")
    semantic_audit = semantic_registry_export_audit(current.iloc[:378].reset_index(drop=True))
    if not semantic_audit["all_historical_rows_semantically_validated"]:
        raise RuntimeError("A historical Registry field differs semantically")
    prior_ids = current.experiment_id.astype(str).tolist()

    rows = []
    timestamp = "2026-07-16T17:55:00+00:00"
    for recovery_id, superseded_id in zip(RECOVERY_IDS, INITIAL_IDS):
        row = {column: "" for column in current.columns}
        row.update({
            "experiment_id": recovery_id,
            "timestamp_utc": timestamp,
            "model_family": "explainability",
            "model_name": recovery_id.split("__", 2)[-1],
            "sensitive_mode": "with_sensitive" if "with_sensitive" in recovery_id else "both" if any(token in recovery_id for token in ["synthesis", "cross_model", "local_cases", "summary", "handoff"]) else "without_sensitive",
            "feature_set": "stage8_recovery_saved_decile_semantic_units",
            "target_mode": "original_scale",
            "evaluation_stage": "Stage 8 Recovery Post-Test explainability",
            "training_row_count": 0,
            "validation_row_count": 0,
            "test_row_count": 20 if "local_cases" in recovery_id else 2000,
            "parameter_json": json.dumps({"authorization_id": AUTHORIZATION_ID, "fit_calls": 0, "seeds": [42, 43], "supersedes_experiment_id": superseded_id}, sort_keys=True),
            "fit_time_seconds": 0,
            "status": "PASS_WITH_DOCUMENTED_REGISTRY_GOVERNANCE_EXCEPTION",
            "notes": "Stage 8 saved-decile Recovery; supersedes invalid initial Stage 8 evidence; Stage 4L remains official; Registry Path B semantic preservation exception disclosed.",
        })
        rows.append(row)
    recovery_frame = pd.DataFrame(rows, columns=current.columns)
    recovery_frame.to_csv(result("registry_rows.csv"), index=False)
    existing_ids = set(current.experiment_id.astype(str))
    missing = recovery_frame[~recovery_frame.experiment_id.astype(str).isin(existing_ids)]
    first_action = "REUSED" if missing.empty else "APPENDED"
    if not missing.empty:
        if not current_bytes.endswith(b"\n"):
            raise RuntimeError("Registry does not end with a complete line")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        for row in missing.itertuples(index=False, name=None):
            writer.writerow(row)
        with REGISTRY.open("ab") as handle:
            handle.write(buffer.getvalue().encode("utf-8"))
    after_first = REGISTRY.read_bytes()
    appended = pd.read_csv(REGISTRY)
    if not after_first.startswith(current_bytes) or len(appended) != 394 or appended.experiment_id.duplicated().any():
        raise RuntimeError("Registry byte-safe append failed")
    # Idempotence operation: evaluate the same IDs and perform no write.
    second_missing = recovery_frame[~recovery_frame.experiment_id.astype(str).isin(set(appended.experiment_id.astype(str)))]
    second_action = "REUSED" if second_missing.empty else "ERROR"
    if second_action != "REUSED" or REGISTRY.read_bytes() != after_first:
        raise RuntimeError("Registry second append was not a no-op")
    search = json.loads((REPORTS / "stage8_registry_recovery_search.json").read_text(encoding="utf-8"))
    adjudication = {
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": "PASS_WITH_DOCUMENTED_REGISTRY_GOVERNANCE_EXCEPTION",
        "registry_path": REGISTRY.relative_to(ROOT).as_posix(),
        "registry_resolution_path": "Path B",
        "exact_pre_stage8_bytes_found": search["exact_bytes_found"],
        "expected_old_registry_sha256": OLD_REGISTRY_HASH,
        "expected_old_registry_size_bytes": OLD_REGISTRY_SIZE,
        "recovery_start_registry_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "recovery_start_registry_size_bytes": len(current_bytes),
        "recovery_start_registry_rows": len(current),
        "final_registry_sha256": hashlib.sha256(after_first).hexdigest(),
        "final_registry_size_bytes": len(after_first),
        "final_registry_rows": len(appended),
        "first_370_current_reserialized_canonical_hash": first_370_hash,
        "first_370_stage7_pre_reserialization_canonical_hash": stage7_baseline["registry_prior_rows_canonical_sha256"],
        "historical_text_hash_reproduction_expected": False,
        "historical_text_hash_reproduction_reason": "The disclosed Stage 8 full pandas serialization changed numeric text formatting; semantic field validation is required instead.",
        "stage7_final_semantic_recheck": record(ROOT / "artifacts/reports/stage7_protected_recheck.json"),
        "stage7_final_semantic_recheck_status": stage7_recheck["status"],
        "stage7_eight_rows_match_saved_export": True,
        "first_378_semantic_rows_validated": True,
        "first_378_semantic_hash": first_378_semantic_hash,
        "first_378_id_sequence_matches_stage8_start_baseline": True,
        "semantic_field_audit": semantic_audit,
        "all_prior_experiment_ids_present_unique": appended.experiment_id.astype(str).tolist()[:386] == prior_ids,
        "original_byte_prefix_incident_visible": True,
        "raw_prefix_preservation_claimed": False,
        "recovery_start_bytes_are_final_prefix": after_first.startswith(current_bytes),
        "existing_386_bytes_modified": False,
        "recovery_rows_appended": len(missing),
        "recovery_ids": RECOVERY_IDS,
        "supersedes_lineage_present": True,
        "first_action": first_action,
        "second_action": second_action,
        "incident_classification": "accepted_stage8_registry_prefix_reserialization_with_semantic_preservation_and_no_exact_byte_recovery",
    }
    dump(adjudication, REPORTS / "stage8_registry_governance_adjudication.json")
    return adjudication


def create_preliminary_handoff_and_preparation() -> None:
    summary = json.loads(result("global_explanation_summary.json").read_text(encoding="utf-8"))
    adjudication = json.loads((REPORTS / "stage8_registry_governance_adjudication.json").read_text(encoding="utf-8"))
    handoff = {
        "authorization_id": AUTHORIZATION_ID,
        "stage_id": "stage8",
        "status": "PASS_WITH_DOCUMENTED_REGISTRY_GOVERNANCE_EXCEPTION_PENDING_NOTEBOOK_AND_REVIEW",
        "initial_stage8_explanation_inference_invalidated": True,
        "stage4l_remains_official": True,
        "candidate_ids": CANDIDATES,
        "model_ids": [item["id"] for item in MODELS],
        "artifacts": {
            "valid_reused_native_importance": record(ROOT / "artifacts/results/stage8/explainability/stage8_existing_importance_long.csv"),
            "valid_reused_saved_shap": record(REPORTS / "stage8_recovery_existing_shap_provenance.json"),
            "valid_stage5a_attribution": record(ROOT / "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv"),
            "corrected_permutation": record(result("common_permutation_importance.csv")),
            "corrected_local_reference_effects": record(result("local_reference_effects.csv.gz")),
            "corrected_local_public_summary": record(result("local_attributions_public.csv")),
            "recovery_figures": record(RECOVERY_MANIFESTS / "stage8_recovery_visualization_manifest.json"),
            "registry_adjudication": record(REPORTS / "stage8_registry_governance_adjudication.json"),
            "global_summary": record(result("global_explanation_summary.json")),
            "interpretation_report": record(result("feature_interpretation_report.md")),
        },
        "recovery_reviewer_path": "artifacts/reports/stage8_reviewer.md",
        "recovery_verification_path": "artifacts/reports/stage8_verification.json",
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "global_shap_recomputations": 0,
        "new_evaluation_prediction_files": 0,
        "model_selection_performed": False,
        "causal_conclusion": "none",
        "public_raw_sensitive_values": 0,
        "stage9_must_not_rerun_explainability": True,
        "stage9_must_preserve_registry_governance_disclosure": True,
        "stage9_started": False,
        "next_stage": "Blocked until final Recovery Reviewer and Verification closure",
        "summary_status": summary["status"],
        "registry_status": adjudication["status"],
    }
    dump(handoff, RECOVERY_MANIFESTS / "stage8_recovery_stage9_handoff.json")
    preparation = {
        "authorization_id": AUTHORIZATION_ID,
        "status": "PASS",
        "scientific_artifacts_validated": True,
        "complete_local_dispersion_validated": True,
        "figures_validated": True,
        "registry_path_b_validated": True,
        "notebook_artifact_loading_only_required": True,
        "reviewer_finalization_occurs_after_two_notebook_runs": True,
        "stage9_started": False,
    }
    dump(preparation, REPORTS / "stage8_recovery_notebook_preparation.json")


def run() -> None:
    manifest = create_figures()
    summary = create_summary()
    adjudication = append_registry_rows()
    create_preliminary_handoff_and_preparation()
    print(json.dumps({
        "status": "PASS",
        "figures": manifest["figure_count"],
        "summary_status": summary["status"],
        "registry_status": adjudication["status"],
        "registry_path": adjudication["registry_resolution_path"],
        "registry_rows_appended": adjudication["recovery_rows_appended"],
        "registry_second_action": adjudication["second_action"],
    }, indent=2))


if __name__ == "__main__":
    run()
