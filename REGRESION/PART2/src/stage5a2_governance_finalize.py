"""Finalize Stage 5A after independent governance review and cache execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts/reports"
BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_governance_protected_hashes_before.json"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
VERIFICATION = REPORTS / "stage5a_verification.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def protected_recheck() -> dict[str, Any]:
    baseline = load(BASELINE)
    mismatches = []
    for entry in baseline["files"]:
        path = Path(entry["path"])
        actual = sha256(path) if path.exists() and path.is_file() else None
        if actual != entry["sha256"] or (path.exists() and path.stat().st_size != entry["size"]):
            mismatches.append({
                "path": entry["path"], "expected_sha256": entry["sha256"],
                "actual_sha256": actual, "expected_size": entry["size"],
                "actual_size": path.stat().st_size if path.exists() and path.is_file() else None,
            })
    registry_bytes = REGISTRY.read_bytes()
    prefix_size = int(baseline["registry_protected_prefix_size_bytes"])
    prefix_hash = hashlib.sha256(registry_bytes[:prefix_size]).hexdigest()
    notebook_backup = ROOT / baseline["notebook_backup_path"]
    return {
        "adjudication_id": "stage5a2_governance_adjudication_1",
        "recorded_at": now(),
        "status": "PASS" if not mismatches and prefix_hash == baseline["registry_protected_prefix_sha256"] else "FAIL",
        "baseline_path": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": sha256(BASELINE),
        "baseline_file_count": len(baseline["files"]),
        "checked_file_count": len(baseline["files"]),
        "mismatches": mismatches,
        "registry_protected_prefix_size_bytes": prefix_size,
        "registry_protected_prefix_sha256": prefix_hash,
        "registry_protected_prefix_preserved": prefix_hash == baseline["registry_protected_prefix_sha256"],
        "notebook_backup_path": baseline["notebook_backup_path"],
        "notebook_backup_sha256": sha256(notebook_backup),
        "notebook_backup_unchanged": sha256(notebook_backup) == baseline["notebook_backup_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["pre_notebook", "final"])
    phase = parser.parse_args().phase

    governance = load(REPORTS / "stage5a2_governance_adjudication.json")
    evidence = load(REPORTS / "stage5a2_governance_evidence_audit.json")
    membership = load(REPORTS / "stage5a2_learned_membership_audit.json")
    supersession = load(REPORTS / "stage5a2_zero_test_loading_claim_supersession.json")
    smoke = load(REPORTS / "stage5a2_future_safe_loader_smoke.json")
    registry_report = load(REPORTS / "stage5a2_governance_registry_update.json")
    reviewer = load(REPORTS / "stage5a_governance_reviewer.json")
    reviewer_counts = reviewer.get("current_findings", {})
    reviewer_critical = int(reviewer_counts.get("critical", reviewer.get("critical_findings", 0)))
    reviewer_major = int(reviewer_counts.get("major", reviewer.get("major_findings", 0)))
    reviewer_minor = int(reviewer_counts.get("minor", reviewer.get("minor_findings", 0)))
    full_train = load(ROOT / "artifacts/manifests/stage5/stage5a2_full_train_manifest.json")
    handoff = load(ROOT / "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json")
    stage5a1 = load(REPORTS / "stage5a1_gate_verification.json")
    reviewer3 = load(REPORTS / "stage5a1_reviewer_cycle3.json")
    protected = protected_recheck()
    atomic_json(protected, REPORTS / "stage5a2_governance_protected_recheck.json")

    notebook = nbformat.read(ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    notebook_report = (
        load(REPORTS / "stage5a2_governance_notebook_execution.json")
        if phase == "final" else {"status": "AUTHORIZED_PENDING", "checks": {}}
    )
    registry = pd.read_csv(REGISTRY)
    stage5b_ids = [value for value in registry["experiment_id"].astype(str) if value.startswith("stage5b__")]

    scientific_checks = {
        "stage5a1_gate_pass": stage5a1["status"] == "PASS",
        "stage5a1_reviewer_cycle3_pass": reviewer3["status"] == "PASS",
        "core_winner_and_sensitive_validation_unchanged": evidence["checks"]["core_winner_unchanged"],
        "selected_train_rows_399788": all(item["training_rows"] == 399_788 for item in full_train["models"]),
        "test_rows_in_preprocessing_fit_zero": all(
            item["test_row_count_in_fit_input"] == 0 for item in membership["learned_operations"]
        ),
        "test_rows_in_model_fit_zero": governance["model_fit_test_row_count"] == 0,
        "test_rows_in_selection_zero": governance["model_selection_test_row_count"] == 0,
        "test_metrics_used_zero": governance["test_metric_use_count"] == 0,
        "stage5_test_predictions_zero": governance["stage5_test_prediction_count"] == 0,
        "stage4l_test_artifacts_used_zero": governance["stage4l_test_artifact_use_count"] == 0,
        "no_statistical_test_leakage_demonstrated": governance["classifications"]["statistical_test_leakage"] == "not_demonstrated",
        "no_refit_required": governance["classifications"]["refit_required"] is False,
    }
    artifact_checks = {
        "two_full_train_models_and_bundles_pass": full_train["status"] == "PASS" and len(full_train["models"]) == 2,
        "model_and_bundle_hashes_match": evidence["checks"]["model_and_bundle_hashes_match"],
        "both_reloads_pass": evidence["checks"]["both_reloads_pass"],
        "handoff_aligned_and_unweighted": handoff["status"] == "PASS" and not handoff["ensemble_weight_selected"],
        "all_stage5_saved_prediction_rows_zero_test_overlap": membership["membership"]["checks"]["all_saved_stage5_prediction_rows_have_zero_test_overlap"],
        "registry_324_unique_one_governance_row": len(registry) == registry["experiment_id"].nunique() == 324
        and int((registry["experiment_id"] == "stage5a2_governance_adjudication_1").sum()) == 1,
        "registry_prior_323_row_prefix_preserved": registry_report["prior_323_row_byte_prefix_preserved"],
        "protected_recheck_pass": protected["status"] == "PASS",
        "stage5b_not_started": len(stage5b_ids) == 0,
    }
    exception_checks = {
        "literal_zero_test_loading_is_false": governance["classifications"]["literal_zero_test_loading"] is False,
        "literal_failure_visible": governance["literal_rule_violated"] is True,
        "human_adjudication_recorded": governance["human_adjudication_result"]
        == "accepted_procedural_test_row_materialization_without_demonstrated_statistical_leakage",
        "procedural_compliance_accepted_exception": governance["classifications"]["procedural_compliance"] == "accepted_exception",
        "false_claims_superseded": supersession["status"] == "PASS" and len(supersession["superseded_claims"]) > 0,
        "future_loader_contract_and_smoke_pass": smoke["status"] == "PASS",
        "reviewer_pass_zero_unresolved_critical_major": reviewer["status"] == "PASS"
        and reviewer_critical == 0 and reviewer_major == 0
        and int(reviewer_counts.get("unresolved_critical", 0)) == 0
        and int(reviewer_counts.get("unresolved_major", 0)) == 0,
    }
    notebook_checks = {
        "notebook_cells_99": len(notebook.cells) == 99,
        "notebook_code_cells_49": len(code_cells) == 49,
        "governance_section_present": "Governance Incident and Human Adjudication" in notebook.cells[95].source,
        "final_completion_section_present": "Final Verification and Completion" in notebook.cells[97].source,
        "final_cache_only_status": notebook_report["status"],
        "final_cache_only_pass": notebook_report["status"] == "PASS" if phase == "final" else True,
    }
    all_checks = [*scientific_checks.values(), *artifact_checks.values(), *exception_checks.values()]
    all_checks.extend(value for key, value in notebook_checks.items() if key != "final_cache_only_status")
    if not all(all_checks):
        raise RuntimeError("A final Stage 5A governance verification check failed")

    reviewer_adjudication = {
        "adjudication_id": "stage5a2_governance_adjudication_1", "recorded_at": now(),
        "status": "PASS" if reviewer["status"] == "PASS" else "FAIL",
        "reviewer_cycle": reviewer["reviewer_cycle"], "maximum_reviewer_cycles": 2,
        "critical_findings": reviewer_critical,
        "major_findings": reviewer_major,
        "minor_findings": reviewer_minor,
        "accepted_fixes": reviewer.get("accepted_fixes", reviewer.get("prior_findings_adjudication", [])),
        "conclusion": reviewer.get("adjudication_conclusion", reviewer["conclusion"]),
        "remaining_risks": reviewer.get("remaining_risks", []),
        "reviewer_report_path": "artifacts/reports/stage5a_governance_reviewer.md",
        "reviewer_report_sha256": sha256(REPORTS / "stage5a_governance_reviewer.md"),
    }
    atomic_json(reviewer_adjudication, REPORTS / "stage5a_governance_reviewer_adjudication.json")

    verification = {
        "adjudication_id": "stage5a2_governance_adjudication_1",
        "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "recorded_at": now(), "phase": phase,
        "status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "stage5a2_status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "stage5a_status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "literal_zero_test_loading": False,
        "literal_zero_test_loading_check": "FAIL",
        "procedural_exception": "accepted_procedural_test_row_materialization_without_demonstrated_statistical_leakage",
        "statistical_test_leakage": "not_demonstrated",
        "refit_required": False,
        "scientific_checks": scientific_checks,
        "artifact_checks": artifact_checks,
        "accepted_exception_checks": exception_checks,
        "reviewer_result": reviewer["status"],
        "reviewer_findings": {
            "critical": reviewer_critical,
            "major": reviewer_major,
            "minor": reviewer_minor,
        },
        "notebook_result": notebook_checks,
        "model_fits_in_governance_task": 0,
        "preprocessing_fits_in_governance_task": 0,
        "prediction_generations_in_governance_task": 0,
        "model_artifact_modifications": 0,
        "bundle_artifact_modifications": 0,
        "prediction_artifact_modifications": 0,
        "protected_recheck_path": "artifacts/reports/stage5a2_governance_protected_recheck.json",
        "stage5b_started": False,
        "next_step": "Begin Stage 5B — Frozen Deep and Boosting Ensemble.",
    }
    atomic_json(verification, VERIFICATION)
    print(json.dumps({
        "status": verification["status"], "phase": phase,
        "literal_zero_test_loading": verification["literal_zero_test_loading"],
        "reviewer": reviewer["status"], "protected": protected["status"],
        "notebook": notebook_report["status"],
    }))


if __name__ == "__main__":
    main()
