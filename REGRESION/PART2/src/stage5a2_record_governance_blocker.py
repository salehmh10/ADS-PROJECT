"""Record the independently reviewed Stage 5A2 Test-access governance blocker."""

from __future__ import annotations

import json
from pathlib import Path

from stage5a2_finalize_verification import (
    BLOCKED_SNAPSHOT,
    NOTEBOOK_EXECUTIONS,
    PROTECTED_REPORT,
    ROOT,
    VERIFICATION,
    no_stage5b_outputs,
    notebook_execution_summary,
    preserve_blocked_verification,
    protected_recheck,
    sha256,
)
from stage5a2_recovery_serialization import atomic_json, load_json


INCIDENT = ROOT / "artifacts/reports/stage5a2_test_access_governance_incident.json"
ADJUDICATION_JSON = ROOT / "artifacts/reports/stage5a_reviewer_adjudication.json"
ADJUDICATION_MD = ROOT / "artifacts/reports/stage5a_reviewer_adjudication.md"
BLOCKER = ROOT / "artifacts/reports/stage5a2_final_governance_blocker.json"
REVIEWER = ROOT / "artifacts/reports/stage5a_reviewer.md"


def main() -> None:
    preserve_blocked_verification()
    protected = protected_recheck()
    atomic_json(protected, PROTECTED_REPORT)
    notebook = notebook_execution_summary()
    run3 = ROOT / "artifacts/reports/stage5a2_notebook_run3_final_refresh.json"
    major2_fixed = run3.exists() and load_json(run3).get("status") == "PASS"

    incident_checks = {
        "train_ids_399788": True,
        "test_ids_99948": True,
        "ids_partition_499736_source_positions": True,
        "loader_scans_full_source_before_mask": True,
        "test_rows_selected_into_modeling_arrays_zero": True,
        "train_test_overlap_zero": True,
        "stage4l_test_metrics_used_zero": True,
        "test_feature_rows_transiently_materialized_zero": False,
        "test_target_rows_transiently_materialized_zero": False,
        "strict_zero_test_loading_governance_met": False,
    }
    incident = {
        "stage_id": "stage5a2",
        "incident_code": "stage5a2_test_columns_transiently_materialized_before_train_only_filter",
        "reviewer_finding": "Major 1",
        "loader_path": "stage5a2_deep_utils.py",
        "loader_function": "_load_source_rows",
        "mechanism": (
            "pandas.read_csv materialized complete 50,000-row source chunks with requested Feature and target "
            "columns before the wanted row-position mask was applied."
        ),
        "source_row_count": 499736,
        "saved_train_row_count": 399788,
        "saved_test_row_count": 99948,
        "test_feature_rows_transiently_materialized": 99948,
        "test_target_rows_transiently_materialized": 99948,
        "test_rows_selected_into_preprocessing_or_fitting": 0,
        "test_rows_selected_into_validation_or_handoff": 0,
        "test_rows_used_for_metrics_or_selection": 0,
        "stage4l_test_metrics_used": False,
        "statistical_test_leakage_demonstrated": False,
        "procedural_test_access_violation": True,
        "retrospective_technical_repair_can_undo_access": False,
        "new_fit_or_reselection_authorized": False,
        "human_adjudication_required": True,
        "checks": incident_checks,
        "status": "BLOCKED",
    }
    atomic_json(incident, INCIDENT)

    adjudication = {
        "stage_id": "stage5a2",
        "reviewer_report_path": str(REVIEWER.relative_to(ROOT)),
        "reviewer_report_sha256": sha256(REVIEWER),
        "reviewer_verdict": "FAIL pending adjudication",
        "critical_findings": 0,
        "major_findings": 2,
        "minor_findings": 1,
        "findings": [
            {
                "id": "Major 1", "accepted": True, "fixed": False,
                "disposition": "BLOCKER_PENDING_EXPLICIT_HUMAN_ADJUDICATION",
                "reason": "The procedural Test access already occurred and cannot be undone; no statistical use was found.",
            },
            {
                "id": "Major 2", "accepted": True, "fixed": major2_fixed,
                "disposition": "STATE_AND_FINAL_OUTPUT_REPAIRED" if major2_fixed else "FINAL_NOTEBOOK_REFRESH_PENDING",
                "reason": "State and verification now report the real governance blocker; attempt 3 refreshes the saved Notebook output.",
            },
            {
                "id": "Minor 1", "accepted": True, "fixed": True,
                "disposition": "CLARIFIED_WITHOUT_HASHED_ARTIFACT_REWRITE",
                "reason": "The payload was constructed before fit; timestamps prove the effective JSON was written after model/reference/history.",
            },
        ],
        "remaining_critical_findings": 0,
        "remaining_major_findings": 1,
        "accepted_critical_major_fixes_complete": major2_fixed,
        "human_adjudication_required": True,
        "status": "BLOCKED",
    }
    atomic_json(adjudication, ADJUDICATION_JSON)
    ADJUDICATION_MD.write_text(
        "# Stage 5A2 Reviewer Adjudication\n\n"
        "- Critical findings: 0.\n"
        "- Major findings: 2; both accepted.\n"
        "- Minor findings: 1; accepted as a field-label clarification.\n\n"
        "## Major 1\n\nAccepted and unresolved. Full-source chunk parsing transiently materialized 99,948 "
        "Test Feature/target rows before the Train-only mask. Zero Test rows entered modeling or metrics, but strict "
        "access governance failed. Explicit human adjudication is required.\n\n"
        "## Major 2\n\nAccepted. State files and final verification now record BLOCKED. The final Notebook refresh "
        f"is {'PASS' if major2_fixed else 'pending'} within the three-attempt budget.\n\n"
        "## Minor 1\n\nAccepted clarification. The effective payload was built before fit; final JSON file timestamps "
        "show the required post-model write order. Hashed model and bundle artifacts were not rewritten.\n",
        encoding="utf-8",
    )

    passed_technical = {
        "stage5a1_gate": load_json(ROOT / "artifacts/reports/stage5a1_gate_verification.json")["status"] == "PASS",
        "two_full_train_models": load_json(ROOT / "artifacts/manifests/stage5/stage5a2_full_train_manifest.json")["status"] == "PASS",
        "two_clean_reloads": set(__import__("pandas").read_csv(ROOT / "artifacts/reports/stage5a2_core_reload_verification.csv")["status"]) == {"PASS"},
        "handoff": load_json(ROOT / "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json")["status"] == "PASS",
        "attribution": load_json(ROOT / "artifacts/reports/stage5a2_feature_attribution.json")["status"] == "PASS",
        "figures": load_json(ROOT / "artifacts/reports/stage5a2_figure_manifest.json")["status"] == "PASS",
        "registry": load_json(ROOT / "artifacts/reports/stage5a2_registry_update.json")["status"] == "PASS",
        "notebook_complete_and_cache_only": notebook["status"] == "PASS",
        "protected_recheck": protected["status"] == "PASS",
        "independent_review_complete": REVIEWER.exists(),
        "stage5b_not_started": no_stage5b_outputs(),
    }
    failed_governance = {
        "zero_test_feature_rows_loaded": False,
        "zero_test_target_rows_loaded": False,
        "strict_test_access_governance": False,
        "zero_remaining_major_findings": False,
        "human_adjudication_complete": False,
    }
    verification = {
        "stage_id": "stage5a",
        "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "technical_artifact_checks": passed_technical,
        "governance_checks": failed_governance,
        "governance_incident_path": str(INCIDENT.relative_to(ROOT)),
        "reviewer_report_path": str(REVIEWER.relative_to(ROOT)),
        "reviewer_adjudication_path": str(ADJUDICATION_JSON.relative_to(ROOT)),
        "protected_recheck_path": str(PROTECTED_REPORT.relative_to(ROOT)),
        "old_blocked_verification_snapshot_path": str(BLOCKED_SNAPSHOT.relative_to(ROOT)),
        "notebook_execution_summary_path": str(NOTEBOOK_EXECUTIONS.relative_to(ROOT)),
        "test_feature_rows_transiently_materialized": 99948,
        "test_target_rows_transiently_materialized": 99948,
        "test_rows_selected_or_used": 0,
        "stage4l_test_metrics_used": False,
        "statistical_test_leakage_demonstrated": False,
        "procedural_test_access_violation": True,
        "stage5b_started": False,
        "next_step": "Wait for explicit human adjudication; do not begin Stage 5B.",
        "blocker_code": "stage5a2_test_columns_transiently_materialized_before_train_only_filter_requires_human_adjudication",
        "status": "BLOCKED",
    }
    atomic_json(verification, VERIFICATION)
    blocker = {
        "stage_id": "stage5a2", "status": "BLOCKED",
        "blocker_code": verification["blocker_code"],
        "incident_path": str(INCIDENT.relative_to(ROOT)), "incident_sha256": sha256(INCIDENT),
        "reviewer_path": str(REVIEWER.relative_to(ROOT)), "reviewer_sha256": sha256(REVIEWER),
        "adjudication_path": str(ADJUDICATION_JSON.relative_to(ROOT)), "adjudication_sha256": sha256(ADJUDICATION_JSON),
        "valid_artifacts_preserved": True, "new_model_fit_required_or_authorized": False,
        "stage5b_started": False, "human_adjudication_required": True,
    }
    atomic_json(blocker, BLOCKER)
    print(json.dumps({"verification": verification, "adjudication": adjudication, "blocker": blocker}, indent=2))


if __name__ == "__main__":
    main()
