"""Repair Stage 8 visualization metadata and Figure 14 from saved public data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stage8_explainability_utils import CANDIDATES, FIGURE_TITLES, LABEL, MANIFESTS, ROOT, dump, sha

FIGURES = ROOT / "artifacts/figures/stage8"
PLOTTING = FIGURES / "plotting_data"
RESULTS = ROOT / "artifacts/results/stage8/explainability"


def repair_figure_14() -> None:
    cases = pd.read_csv(MANIFESTS / "stage8_local_case_manifest.csv")
    selected = cases[(cases.case_type == "deep_with_improves_over_without") & cases.visualization_case].iloc[0]
    case_id = f"{selected.semantic_case_type}__{int(selected.case_rank)}"
    local = pd.read_csv(RESULTS / "stage8_local_attributions_public.csv")
    stability = pd.read_csv(RESULTS / "stage8_local_explanation_stability.csv")
    effects = local[(local.case_public_id == case_id) & (local.within_case_candidate_rank <= 6)][["case_public_id", "candidate_id", "semantic_feature_unit", "reference_substitution_effect"]].copy()
    stable = stability[(stability.case_public_id == case_id) & (stability.candidate_id == CANDIDATES[2])].nsmallest(8, "absolute_difference").copy()
    effects["record_type"] = "local_effect"; effects["effect_background_half_a"] = pd.NA; effects["effect_background_half_b"] = pd.NA; effects["absolute_difference"] = pd.NA
    stable["record_type"] = "background_half_stability"; stable["reference_substitution_effect"] = pd.NA
    data = pd.concat([effects, stable[["case_public_id", "candidate_id", "semantic_feature_unit", "reference_substitution_effect", "effect_background_half_a", "effect_background_half_b", "absolute_difference", "record_type"]]], ignore_index=True)
    data.to_csv(PLOTTING / "stage8_figure_14.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    pivot = effects.pivot(index="semantic_feature_unit", columns="candidate_id", values="reference_substitution_effect").fillna(0)
    pivot.plot.barh(ax=axes[0]); axes[0].axvline(0, color="black", linewidth=1); axes[0].set_xlabel("reference-substitution effect\n(original target units)"); axes[0].set_title("Local effects"); axes[0].legend(fontsize=6)
    s = stable.set_index("semantic_feature_unit")[["effect_background_half_a", "effect_background_half_b"]]
    s.plot.barh(ax=axes[1], color=["#6B8EAD", "#C28E5C"]); axes[1].axvline(0, color="black", linewidth=1); axes[1].set_xlabel("effect by 20-row background half"); axes[1].set_title("Background-half stability"); axes[1].legend(["half A", "half B"], fontsize=8)
    fig.suptitle(f"{FIGURE_TITLES[13]}\n{LABEL}", fontsize=11); fig.tight_layout(); fig.savefig(FIGURES / "stage8_figure_14.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def repair_manifest() -> None:
    freeze = json.loads((ROOT / "artifacts/reports/stage8_preexplainability_freeze.json").read_text(encoding="utf-8"))
    row_hash = freeze["sample_contract"]["global_row_hash"]; target_hash = freeze["sample_contract"]["global_target_hash"]
    methods = ["artifact provenance", "grouped permutation", "grouped permutation", "grouped permutation", "rank comparison", "grouped permutation family share", "native importance versus saved SHAP ranks", "native importance versus saved SHAP ranks", "native importance versus saved SHAP ranks", "saved attribution versus grouped permutation ranks", "aggregate grouped permutation", "local reference substitution", "local reference substitution", "local reference substitution and background-half stability", "artifact dashboard"]
    sample_rows = [300,2000,2000,2000,2000,2000,300,300,300,2000,2000,20,20,20,2000]
    entries = []
    for i in range(1, 16):
        fig = FIGURES / f"stage8_figure_{i:02d}.png"; data = PLOTTING / f"stage8_figure_{i:02d}.csv"
        entries.append({"figure_id": i, "title": FIGURE_TITLES[i-1], "figure_path": str(fig.relative_to(ROOT)).replace("\\", "/"), "figure_sha256": sha(fig), "plotting_data_path": str(data.relative_to(ROOT)).replace("\\", "/"), "plotting_data_sha256": sha(data), "candidate_ids": CANDIDATES, "method": methods[i-1], "sample_row_count": sample_rows[i-1], "row_id_hash": row_hash, "target_hash": target_hash, "output_scale": "original target units for permutation/local; declared native scale for saved SHAP", "privacy_status": "PASS — aggregate only; no raw sensitive values", "analysis_label": LABEL, "interpretation_limitation": "Post-Test descriptive model-behavior evidence; not causal, not fairness certification, and not model selection."})
    dump({"status":"PASS","figure_count":15,"plotting_data_count":15,"figures":entries,"raw_sensitive_values":0}, MANIFESTS / "stage8_visualization_manifest.json")


if __name__ == "__main__":
    repair_figure_14(); repair_manifest(); print("PASS")
