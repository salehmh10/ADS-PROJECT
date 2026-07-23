"""Audit the frozen Stage 8 sample against saved Stage 5C deciles."""

from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd

from stage8_explainability_utils import MANIFESTS, PREDICTIONS, REPORTS, RESULTS, ROOT, dump, now, value_hash


def main() -> None:
    saved = pd.read_csv(ROOT / PREDICTIONS[1]["path"], usecols=["row_id", "y_true", "target_decile"])
    recomputed = pd.qcut(saved.y_true.rank(method="first"), 10, labels=False) + 1
    label_mismatch = int((saved.target_decile.astype(int).to_numpy() != recomputed.astype(int).to_numpy()).sum())
    required = pd.concat([group.sample(n=200, random_state=42) for _, group in saved.groupby("target_decile", sort=True)]).sort_values("row_id")
    frozen = pd.read_csv(MANIFESTS / "stage8_global_explanation_sample_row_ids.csv")
    overlap = len(set(required.row_id.astype(int)) & set(frozen.row_id.astype(int)))
    frozen_saved_deciles = saved.set_index("row_id").loc[frozen.row_id.astype(int), "target_decile"].value_counts().sort_index().astype(int).to_dict()
    cases = set(pd.read_csv(MANIFESTS / "stage8_local_case_manifest.csv").row_id.astype(int))
    expected_bg = pd.concat([group.sample(n=4, random_state=42) for _, group in required[~required.row_id.isin(cases)].groupby("target_decile", sort=True)]).sort_values("row_id")
    frozen_bg = pd.read_csv(MANIFESTS / "stage8_local_background_row_ids.csv")
    payload = {
        "stage_id": "stage8", "status": "CRITICAL_FAIL", "checked_at_utc": now(),
        "saved_stage5c_target_decile_used_by_frozen_sampler": False,
        "incorrect_recomputed_decile_method": "pandas qcut of rank(method=first)",
        "test_row_decile_label_mismatch_count": label_mismatch,
        "frozen_global_sample_rows": len(frozen), "required_saved_decile_sample_rows": len(required),
        "global_sample_membership_overlap": overlap, "global_sample_membership_difference": 2000 - overlap,
        "frozen_sample_counts_under_saved_stage5c_deciles": {str(k): v for k, v in frozen_saved_deciles.items()},
        "frozen_global_row_hash": value_hash(frozen.row_id, np.int64),
        "required_saved_decile_row_hash_audit_only": value_hash(required.row_id, np.int64),
        "background_membership_overlap": len(set(expected_bg.row_id.astype(int)) & set(frozen_bg.row_id.astype(int))),
        "affected_artifacts": ["stage8_global_explanation_sample_row_ids.csv", "stage8_local_background_row_ids.csv", "stage8_common_permutation_importance.csv", "stage8_permutation_repeat_stability.csv", "stage8_local_attributions_public.csv", "stage8_local_explanation_stability.csv", "stage8_case_explanation_synthesis.csv", "Stage 8 Figures 2–15"],
        "post_access_membership_change_permitted": False,
        "required_resolution": "Explicit human authorization for a new frozen saved-decile sample, renewed bounded source/model inference, and superseding affected Stage 8 outputs.",
    }
    dump(payload, REPORTS / "stage8_sample_contract_incident.json")
    for path in [RESULTS / "stage8_global_explanation_summary.json", MANIFESTS / "stage8_stage9_handoff.json"]:
        data = json.loads(path.read_text(encoding="utf-8")); data.setdefault("blockers", []).append("Critical: frozen global/background sampling recomputed deciles instead of using saved Stage 5C target_decile; affected explanation inference is not valid for completion."); data["status" if "status" in data else "stage_status"] = "BLOCKED"; dump(data, path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
