"""Independent Stage 8 Recovery review, protected recheck, and verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from stage8_explainability_utils import CANDIDATES, EXPECTED, MODELS, PREDICTIONS
from stage8_recovery import (
    AUTHORIZATION_ID,
    RECOVERY_MANIFESTS,
    RECOVERY_RESULTS,
    REGISTRY,
    REPORTS,
    ROOT,
    SOURCE_HASHES,
    dump,
    now,
    record,
    sha,
    value_hash,
)


FINAL_STATUS = "PASS_WITH_DOCUMENTED_REGISTRY_GOVERNANCE_EXCEPTION"
NOTEBOOK = ROOT / "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb"


def result(name: str) -> Path:
    return RECOVERY_RESULTS / f"stage8_recovery_{name}"


def prepare_final() -> None:
    summary_path = result("global_explanation_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = FINAL_STATUS
    summary["registry_resolution_path"] = "Path B"
    summary["registry_governance_exception"] = "accepted_stage8_registry_prefix_reserialization_with_semantic_preservation_and_no_exact_byte_recovery"
    summary["stage9_started"] = False
    dump(summary, summary_path)

    invalidation_path = RECOVERY_MANIFESTS / "stage8_initial_attempt_invalidation_manifest.json"
    invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    replacements = {
        "stage8_common_permutation_importance.csv": result("common_permutation_importance.csv"),
        "stage8_permutation_repeat_stability.csv": result("permutation_repeat_stability.csv"),
        "stage8_cross_model_feature_comparison.csv": result("cross_model_feature_comparison.csv"),
        "stage8_cross_model_agreement.csv": result("cross_model_agreement.csv"),
        "stage8_deep_attribution_comparison.csv": result("deep_attribution_comparison.csv"),
        "stage8_cross_method_agreement.csv": result("cross_method_agreement.csv"),
        "stage8_feature_family_summary.csv": result("feature_family_summary.csv"),
        "stage8_sensitive_feature_dependence.csv": result("sensitive_feature_dependence.csv"),
        "stage8_potential_proxy_overlap.csv": result("potential_proxy_overlap.csv"),
        "stage8_prediction_reconciliation.csv": REPORTS / "stage8_recovery_prediction_reconciliation.csv",
        "stage8_local_attributions_public.csv": result("local_attributions_public.csv"),
        "stage8_local_prediction_reconciliation.csv": result("local_prediction_reconciliation.csv"),
        "stage8_local_explanation_stability.csv": result("local_explanation_stability.csv"),
        "stage8_case_explanation_synthesis.csv": result("case_explanation_synthesis.csv"),
        "stage8_global_explanation_summary.json": result("global_explanation_summary.json"),
        "stage8_feature_interpretation_report.md": result("feature_interpretation_report.md"),
        "stage8_registry_rows.csv": result("registry_rows.csv"),
        "stage8_visualization_manifest.json": RECOVERY_MANIFESTS / "stage8_recovery_visualization_manifest.json",
        "stage8_stage9_handoff.json": RECOVERY_MANIFESTS / "stage8_recovery_stage9_handoff.json",
        "stage8_global_explanation_sample_row_ids.csv": RECOVERY_MANIFESTS / "stage8_recovery_global_sample_row_ids.csv",
        "stage8_local_background_row_ids.csv": RECOVERY_MANIFESTS / "stage8_recovery_local_background_row_ids.csv",
    }
    for entry in invalidation["entries"]:
        replacement = replacements.get(Path(entry["path"]).name)
        if replacement and replacement.exists():
            entry["superseding_recovery_artifact_path"] = replacement.relative_to(ROOT).as_posix()
    invalidation["superseding_paths_updated_at_utc"] = now()
    dump(invalidation, invalidation_path)

    handoff = {
        "authorization_id": AUTHORIZATION_ID,
        "stage_id": "stage8",
        "status": FINAL_STATUS,
        "initial_stage8_explanation_inference_invalidated": True,
        "stage4l_remains_official": True,
        "candidate_ids": CANDIDATES,
        "model_ids": [item["id"] for item in MODELS],
        "valid_reused_native_importance": record(ROOT / "artifacts/results/stage8/explainability/stage8_existing_importance_long.csv"),
        "valid_reused_saved_shap": record(REPORTS / "stage8_recovery_existing_shap_provenance.json"),
        "valid_stage5a_attribution": record(ROOT / "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv"),
        "corrected_saved_decile_permutation": record(result("common_permutation_importance.csv")),
        "corrected_background_local_evidence": record(result("local_attributions_public.csv")),
        "complete_per_reference_dispersion_evidence": record(result("local_reference_effects.csv.gz")),
        "recovery_visualization_manifest": record(RECOVERY_MANIFESTS / "stage8_recovery_visualization_manifest.json"),
        "registry_governance_adjudication": record(REPORTS / "stage8_registry_governance_adjudication.json"),
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
        "next_stage": "Begin Stage 9 — Model Card and Final Technical Report.",
    }
    dump(handoff, RECOVERY_MANIFESTS / "stage8_recovery_stage9_handoff.json")
    print(json.dumps({"status": FINAL_STATUS, "summary": record(summary_path), "handoff": record(RECOVERY_MANIFESTS / "stage8_recovery_stage9_handoff.json")}, indent=2))


def protected_recheck() -> dict:
    baseline = json.loads((RECOVERY_MANIFESTS / "stage8_recovery_protected_baseline.json").read_text(encoding="utf-8"))
    authorized_mutable = {
        "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb",
        "artifacts/results/experiment_results.csv",
        "artifacts/manifests/stage8/recovery/stage8_initial_attempt_invalidation_manifest.json",
        "artifacts/reports/stage8_reviewer.md",
        "artifacts/reports/stage8_reviewer_adjudication.json",
        "artifacts/reports/stage8_verification.json",
    }
    unexpected = []
    missing = []
    authorized_changes = []
    checked = 0
    for entry in baseline["entries"]:
        label = entry["path"]
        path = Path(label) if Path(label).is_absolute() else ROOT / label
        if not path.exists():
            missing.append(label)
            continue
        actual = sha(path)
        checked += 1
        if actual != entry["sha256"]:
            if label in authorized_mutable:
                authorized_changes.append(label)
            else:
                unexpected.append(label)

    invalidation = json.loads((RECOVERY_MANIFESTS / "stage8_initial_attempt_invalidation_manifest.json").read_text(encoding="utf-8"))
    initial_mismatches = []
    for entry in invalidation["entries"]:
        path = ROOT / entry["path"]
        if not path.exists() or sha(path) != entry["sha256"]:
            initial_mismatches.append(entry["path"])

    backup_root = ROOT / baseline["backup_root"]
    blocked_notebook_backup = backup_root / "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb"
    initial_reviewer_backup = backup_root / "artifacts/reports/stage8_reviewer.md"
    initial_verification_backup = backup_root / "artifacts/reports/stage8_verification.json"
    initial_handoff_backup = backup_root / "artifacts/manifests/stage8/stage8_stage9_handoff.json"
    backup_checks = {
        "blocked_notebook_backup": blocked_notebook_backup.exists() and sha(blocked_notebook_backup) == baseline["blocked_notebook"]["sha256"],
        "initial_reviewer_backup": initial_reviewer_backup.exists(),
        "initial_verification_backup": initial_verification_backup.exists(),
        "initial_handoff_backup": initial_handoff_backup.exists(),
    }
    registry_start = (backup_root / "artifacts/results/experiment_results.csv").read_bytes()
    registry_final = REGISTRY.read_bytes()
    registry_prefix_pass = registry_final.startswith(registry_start)
    adjudication = json.loads((REPORTS / "stage8_registry_governance_adjudication.json").read_text(encoding="utf-8"))
    result_payload = {
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": "PASS" if not unexpected and not missing and not initial_mismatches and all(backup_checks.values()) and registry_prefix_pass else "FAIL",
        "protected_file_count": baseline["protected_file_count"],
        "checked_file_count": checked,
        "missing_file_count": len(missing),
        "missing_paths": missing,
        "unexpected_mismatch_count": len(unexpected),
        "unexpected_mismatch_paths": unexpected,
        "authorized_changed_paths": authorized_changes,
        "initial_stage8_artifact_count": len(invalidation["entries"]),
        "initial_stage8_mismatch_count": len(initial_mismatches),
        "initial_stage8_mismatch_paths": initial_mismatches,
        "backup_checks": backup_checks,
        "source_hashes_unchanged": all(sha(ROOT / relative) == expected for relative, expected in SOURCE_HASHES.items()),
        "model_and_bundle_hashes_unchanged": all(sha(ROOT / item["path"]) == item["sha256"] for item in MODELS),
        "prediction_hashes_unchanged": all(sha(ROOT / item["path"]) == item["sha256"] for item in PREDICTIONS),
        "registry_path": "Path B",
        "historical_byte_mismatch_exception_visible": adjudication["original_byte_prefix_incident_visible"],
        "recovery_start_registry_bytes_are_final_prefix": registry_prefix_pass,
        "raw_prefix_preservation_claimed": adjudication["raw_prefix_preservation_claimed"],
        "stage9_started": False,
    }
    dump(result_payload, REPORTS / "stage8_recovery_protected_recheck.json")
    if result_payload["status"] != "PASS":
        raise RuntimeError(f"Recovery protected recheck failed: {result_payload}")
    print(json.dumps(result_payload, indent=2))
    return result_payload


def reproduce_local_dispersion() -> dict:
    references = pd.read_csv(result("local_reference_effects.csv.gz"))
    public = pd.read_csv(result("local_attributions_public.csv"))
    keys = ["case_public_id", "candidate_id", "semantic_feature_unit"]
    reproduced = references.groupby(keys, as_index=False).agg(
        effect_mean_reproduced=("effect", "mean"),
        effect_standard_deviation_reproduced=("effect", lambda values: float(np.std(values, ddof=0))),
        effect_minimum_reproduced=("effect", "min"),
        effect_maximum_reproduced=("effect", "max"),
        mean_absolute_effect_reproduced=("effect", lambda values: float(np.mean(np.abs(values)))),
        background_rows_reproduced=("background_row_id", "count"),
    )
    merged = public.merge(reproduced, on=keys, validate="one_to_one")
    differences = {
        "effect_mean": float(np.max(np.abs(merged.effect_mean - merged.effect_mean_reproduced))),
        "effect_standard_deviation": float(np.max(np.abs(merged.effect_standard_deviation - merged.effect_standard_deviation_reproduced))),
        "effect_minimum": float(np.max(np.abs(merged.effect_minimum - merged.effect_minimum_reproduced))),
        "effect_maximum": float(np.max(np.abs(merged.effect_maximum - merged.effect_maximum_reproduced))),
        "mean_absolute_effect": float(np.max(np.abs(merged.mean_absolute_effect - merged.mean_absolute_effect_reproduced))),
    }
    return {
        "reference_rows": len(references),
        "public_rows": len(public),
        "background_rows_each": sorted(references.groupby(keys).size().unique().tolist()),
        "maximum_absolute_differences": differences,
        "status": "PASS" if len(references) == 64800 and len(public) == 1620 and max(differences.values()) <= 1e-12 and sorted(references.groupby(keys).size().unique().tolist()) == [40] else "FAIL",
    }


def review() -> dict:
    sample = pd.read_csv(RECOVERY_MANIFESTS / "stage8_recovery_global_sample_row_ids.csv")
    background = pd.read_csv(RECOVERY_MANIFESTS / "stage8_recovery_local_background_row_ids.csv")
    cases = pd.read_csv(ROOT / "artifacts/manifests/stage8/stage8_local_case_manifest.csv")
    saved_decile = json.loads((REPORTS / "stage8_recovery_saved_decile_validation.json").read_text(encoding="utf-8"))
    access = json.loads((REPORTS / "stage8_recovery_feature_access_audit.json").read_text(encoding="utf-8"))
    model_validation = json.loads((REPORTS / "stage8_recovery_model_validation.json").read_text(encoding="utf-8"))
    reconciliation = pd.read_csv(REPORTS / "stage8_recovery_prediction_reconciliation.csv")
    local_reconciliation = pd.read_csv(result("local_prediction_reconciliation.csv"))
    permutation = pd.read_csv(result("common_permutation_importance.csv"))
    local = pd.read_csv(result("local_attributions_public.csv"))
    local_dispersion = reproduce_local_dispersion()
    provenance = json.loads((REPORTS / "stage8_recovery_existing_shap_provenance.json").read_text(encoding="utf-8"))
    invalidation = json.loads((RECOVERY_MANIFESTS / "stage8_initial_attempt_invalidation_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((REPORTS / "stage8_registry_governance_adjudication.json").read_text(encoding="utf-8"))
    figures = json.loads((RECOVERY_MANIFESTS / "stage8_recovery_visualization_manifest.json").read_text(encoding="utf-8"))
    notebook_attempts = json.loads((REPORTS / "stage8_recovery_notebook_attempts.json").read_text(encoding="utf-8"))
    protected = json.loads((REPORTS / "stage8_recovery_protected_recheck.json").read_text(encoding="utf-8"))
    runtime = json.loads((REPORTS / "stage8_recovery_runtime.json").read_text(encoding="utf-8"))
    handoff = json.loads((RECOVERY_MANIFESTS / "stage8_recovery_stage9_handoff.json").read_text(encoding="utf-8"))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    checks = {
        "saved_stage5c_target_decile_reused": saved_decile["status"] == "PASS" and saved_decile["invalid_recomputed_decile_mismatch_count"] == 878,
        "sample_200_per_decile": len(sample) == 2000 and sample.groupby("target_decile").size().eq(200).all(),
        "background_4_per_decile": len(background) == 40 and background.groupby("target_decile").size().eq(4).all(),
        "sample_and_background_hashes": value_hash(sample.row_id, np.int64) == "d7791b5e12cae7d6069419efdb1e4a9bc28c586cb1bc5ae2584d1b08c3a0358f" and value_hash(background.row_id, np.int64) == "80a30da8f60aa3b5db6ec82847bfc0490359bc2412c3329ceb1f9915bfeb2efd",
        "local_cases_unchanged": len(cases) == 20 and sha(ROOT / "artifacts/manifests/stage8/stage8_local_case_manifest.csv") == json.loads((RECOVERY_MANIFESTS / "stage8_recovery_protected_baseline.json").read_text(encoding="utf-8"))["entries"][[item["path"] for item in json.loads((RECOVERY_MANIFESTS / "stage8_recovery_protected_baseline.json").read_text(encoding="utf-8"))["entries"]].index("artifacts/manifests/stage8/stage8_local_case_manifest.csv")]["sha256"],
        "parser_boundary_source_access": access["loader_path"] == "stage5_safe_row_loader.py" and access["excluded_rows_converted"] == 0,
        "maximum_2020_rows_per_source": max(access["rows_materialized"].values()) <= 2020,
        "zero_train_rows": access["train_rows_materialized"] == 0,
        "zero_source_targets": access["source_target_values_materialized"] == 0,
        "five_model_identities_unchanged": model_validation["model_count"] == 5 and all(item["status"] == "PASS" for item in model_validation["models"]),
        "prediction_reconciliation": len(reconciliation) == 6 and reconciliation.status.eq("PASS").all(),
        "permutation_correct_sample_only": permutation.sample_row_count.eq(2000).all() and permutation.row_id_hash.eq(value_hash(sample.row_id, np.int64)).all(),
        "local_correct_background_only": local.background_rows.eq(40).all() and set(background.row_id).isdisjoint(set(cases.row_id)),
        "per_reference_evidence_complete": local_dispersion["status"] == "PASS",
        "dispersion_reproduces": local_dispersion["status"] == "PASS",
        "global_shap_recomputation_zero": provenance["global_shap_recomputations"] == 0 and runtime["global_shap_recomputations"] == 0,
        "shap_provenance_honest": len(provenance["artifacts"]) == 6 and all(item["base_value_presence"] in [True, "not_available_from_saved_artifact"] and item["additivity_evidence_presence"] in [True, "not_revalidated_without_recomputation"] for item in provenance["artifacts"]),
        "feature_lineage_saved_schema_only": provenance["lineage_source"] == "saved schemas and manifests only" and not provenance["explanation_outcomes_used_for_mapping"],
        "public_raw_sensitive_values_zero": local.raw_sensitive_value_public.eq(False).all() and not any(column in local.columns for column in base_sensitive_columns()),
        "initial_invalid_artifacts_preserved": invalidation["entry_count"] == 53 and protected["initial_stage8_mismatch_count"] == 0,
        "registry_path_b_valid": registry["status"] == FINAL_STATUS and registry["first_378_semantic_rows_validated"] and not registry["raw_prefix_preservation_claimed"],
        "prior_registry_semantics_unchanged": registry["semantic_field_audit"]["semantic_mismatch_count"] == 0,
        "registry_append_safe_idempotent": registry["existing_386_bytes_modified"] is False and registry["recovery_start_bytes_are_final_prefix"] and registry["second_action"] == "REUSED" and registry["recovery_rows_appended"] == 8,
        "exactly_15_recovery_figures": figures["figure_count"] == 15 and figures["plotting_data_count"] == 15 and len(list((ROOT / "artifacts/figures/stage8/recovery").glob("stage8_recovery_figure_*.png"))) == 15 and len(list((ROOT / "artifacts/figures/stage8/recovery/plotting_data").glob("stage8_recovery_figure_*.csv"))) == 15,
        "notebook_complete_and_cache_runs": notebook_attempts["status"] == "PASS" and [item["attempt"] for item in notebook_attempts["attempts"]] == [4, 5] and all(item["status"] == "PASS" for item in notebook_attempts["attempts"]),
        "notebook_outputs_complete": len(code_cells) == 31 and all(cell.execution_count is not None and cell.outputs for cell in code_cells) and not any(output.output_type == "error" for cell in code_cells for output in cell.outputs),
        "zero_fit_tuning_selection_new_prediction": runtime["model_fit_calls"] == runtime["preprocessing_fit_calls"] == runtime["surrogate_fit_calls"] == runtime["new_evaluation_prediction_files"] == 0,
        "stage4l_remains_official": handoff["stage4l_remains_official"] and handoff["candidate_ids"][0] == CANDIDATES[0],
        "stage9_unstarted": not handoff["stage9_started"] and not runtime["stage9_started"],
        "protected_recheck_pass": protected["status"] == "PASS" and protected["unexpected_mismatch_count"] == 0,
        "case_candidate_reconciliation_complete": len(local_reconciliation) == 60 and local_reconciliation.status.eq("PASS").all(),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    counts = {"critical": 0 if not failed else 1, "major": 0 if not failed else len(failed), "minor": 0, "privacy": 0}
    recommendation = FINAL_STATUS if not failed else "FAIL — BLOCKED"
    report_lines = [
        "# Stage 8 Independent Recovery Reviewer Report", "",
        f"Final recommendation: **{recommendation}**", "",
        "Unresolved counts:", "",
        f"- Critical: **{counts['critical']}**", f"- Major: **{counts['major']}**", f"- Minor: **{counts['minor']}**", f"- Privacy: **{counts['privacy']}**", "",
        "## Recovery scope and sample", "",
        "The Reviewer reproduced the saved Stage 5C target-decile contract, all 878 initial mismatches, the 200-per-decile Recovery sample, the four-per-decile background, both hashes, and unchanged 20-case membership.", "",
        "## Bounded access and frozen models", "",
        "Parser-boundary access stayed below 2,020 rows per source with zero Train rows, excluded conversions, or source targets. All five frozen identities and all six component/Candidate reconciliation rows pass. No fit, tuning, selection, SHAP recomputation, or evaluation prediction occurred.", "",
        "## Global and local evidence", "",
        f"Grouped permutation uses only the corrected sample. Local substitution uses only the corrected background. The Reviewer recomputed mean, population standard deviation, minimum, maximum, and mean absolute effect from all {local_dispersion['reference_rows']:,} saved reference rows; maximum reproduction differences are {local_dispersion['maximum_absolute_differences']}.", "",
        "## Provenance and privacy", "",
        "Six saved tree-SHAP mode artifacts have metadata/value/row-ID hashes. Missing base or additivity evidence is labelled unavailable or not revalidated. Feature lineage uses saved schemas only. Public raw sensitive values remain zero and joint blocks are preserved.", "",
        "## Registry governance", "",
        "The exhaustive search did not recover the exact pre-Stage8 bytes. Path B is accepted: all 378 Stage 7-and-earlier semantic rows validate, the original incident stays visible, no raw-prefix preservation is claimed, all Recovery-start 386 bytes are the exact prefix of the final Registry, eight rows were appended, and the second action is REUSED.", "",
        "## Notebook and protection", "",
        "Recovery attempts 4 and 5 passed as the complete artifact-loading and cache-only runs. Sections 0–30 appear once; all 31 code cells have outputs and zero errors. The protected recheck has zero unexpected mismatch and all 53 invalid initial artifacts remain byte-preserved.", "",
        "## Methodology risks", "",
        "Correlated Features can divide importance; native SHAP scales differ; reference substitution is non-additive and may create unrealistic combinations; sensitive importance is not fairness, discrimination, causality, or legal evidence; future-data behavior may differ.", "",
        "## Check results", "",
    ]
    report_lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    report_lines.extend(["", "Stage 4L remains official. Stage 9 has not started."])
    (REPORTS / "stage8_reviewer.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    payload = {
        "authorization_id": AUTHORIZATION_ID,
        "review_cycle": 2,
        "cycle_1_technical_failure": "Machine-readable report serialization stopped before promotion on a NumPy/Pandas Boolean; no evidence changed.",
        "reviewed_at_utc": now(),
        "status": recommendation,
        "counts": counts,
        "checks": checks,
        "failed_checks": failed,
        "local_dispersion_reproduction": local_dispersion,
        "final_recommendation": recommendation,
        "methodology_risks": ["correlated Features", "native SHAP scale differences", "non-additive local substitution", "sensitive importance is not fairness", "future-data shift"],
        "stage9_started": False,
    }
    dump(payload, REPORTS / "stage8_reviewer_adjudication.json")
    if failed:
        raise RuntimeError(f"Independent Recovery review failed: {failed}")
    print(json.dumps(payload, indent=2))
    return payload


def base_sensitive_columns() -> list[str]:
    return [
        "applicant_ethnicity_name", "co_applicant_ethnicity_name", "applicant_race_name_1",
        "co_applicant_race_name_1", "applicant_sex_name", "co_applicant_sex_name",
        "minority_population", "majority_minority_tract",
    ]


def verify() -> dict:
    reviewer = json.loads((REPORTS / "stage8_reviewer_adjudication.json").read_text(encoding="utf-8"))
    freeze = json.loads((REPORTS / "stage8_recovery_sample_freeze.json").read_text(encoding="utf-8"))
    access = json.loads((REPORTS / "stage8_recovery_feature_access_audit.json").read_text(encoding="utf-8"))
    runtime = json.loads((REPORTS / "stage8_recovery_runtime.json").read_text(encoding="utf-8"))
    coverage = json.loads((REPORTS / "stage8_recovery_local_coverage.json").read_text(encoding="utf-8"))
    figures = json.loads((RECOVERY_MANIFESTS / "stage8_recovery_visualization_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((REPORTS / "stage8_registry_governance_adjudication.json").read_text(encoding="utf-8"))
    notebook = json.loads((REPORTS / "stage8_recovery_notebook_attempts.json").read_text(encoding="utf-8"))
    protected = json.loads((REPORTS / "stage8_recovery_protected_recheck.json").read_text(encoding="utf-8"))
    checks = {
        "correct_sample_and_background": freeze["status"] == "PASS" and freeze["sample"]["row_count"] == 2000 and freeze["background"]["row_count"] == 40,
        "complete_local_dispersion": coverage["status"] == "PASS" and coverage["dispersion_complete"],
        "all_affected_evidence_regenerated": all((result(name)).exists() for name in ["common_permutation_importance.csv", "local_reference_effects.csv.gz", "cross_model_feature_comparison.csv", "sensitive_feature_dependence.csv"]),
        "exact_registry_bytes_not_found_after_search": not registry["exact_pre_stage8_bytes_found"],
        "first_378_semantic_rows_validate": registry["first_378_semantic_rows_validated"] and registry["semantic_field_audit"]["semantic_mismatch_count"] == 0,
        "recovery_start_registry_prefix_preserved": registry["recovery_start_bytes_are_final_prefix"] and not registry["existing_386_bytes_modified"],
        "recovery_rows_append_only": registry["recovery_rows_appended"] == 8 and registry["second_action"] == "REUSED",
        "governance_adjudication_exists": registry["status"] == FINAL_STATUS,
        "reviewer_recommends_exception": reviewer["final_recommendation"] == FINAL_STATUS and reviewer["counts"] == {"critical": 0, "major": 0, "minor": 0, "privacy": 0},
        "notebook_two_successes": notebook["status"] == "PASS" and notebook["recovery_successes"] == 2,
        "protected_recheck": protected["status"] == "PASS" and protected["unexpected_mismatch_count"] == 0,
        "stage9_unstarted": not runtime["stage9_started"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    status = FINAL_STATUS if not failed else "BLOCKED"
    payload = {
        "stage_id": "stage8",
        "analysis_label": "Post-Test Explainability and Feature Interpretation",
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": status,
        "original_blocked_attempt_lineage": record(ROOT / "artifacts/backups/stage8_recovery_20260716T221406/artifacts/reports/stage8_verification.json"),
        "superseding_recovery_freeze_sha256": sha(REPORTS / "stage8_recovery_sample_freeze.json"),
        "correct_sample_hash": freeze["sample"]["row_id_hash"],
        "correct_background_hash": freeze["background"]["row_id_hash"],
        "source_access_counts": access["access_attempts"],
        "source_rows_materialized": access["rows_materialized"],
        "model_load_counts": runtime["model_attempts"],
        "model_fit_count": runtime["model_fit_calls"],
        "preprocessing_fit_count": runtime["preprocessing_fit_calls"],
        "surrogate_fit_count": runtime["surrogate_fit_calls"],
        "global_shap_recomputation_count": runtime["global_shap_recomputations"],
        "new_evaluation_prediction_files": runtime["new_evaluation_prediction_files"],
        "local_reference_row_count": coverage["actual_cartesian_rows"],
        "dispersion_completeness": coverage["dispersion_complete"],
        "figure_count": figures["figure_count"],
        "registry_path_used": REGISTRY.relative_to(ROOT).as_posix(),
        "registry_recovery_path": "Path B",
        "registry_adjudication_path": "artifacts/reports/stage8_registry_governance_adjudication.json",
        "reviewer_status": reviewer["final_recommendation"],
        "notebook_attempts": notebook["attempts"],
        "protected_recheck": protected["status"],
        "checks": checks,
        "failed_checks": failed,
        "stage4l_official_role_unchanged": True,
        "stage9_started": False,
    }
    dump(payload, REPORTS / "stage8_verification.json")
    if failed:
        raise RuntimeError(f"Stage 8 Recovery Verification failed: {failed}")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-final", "protected-recheck", "review", "verify"])
    args = parser.parse_args()
    if args.command == "prepare-final":
        prepare_final()
    elif args.command == "protected-recheck":
        protected_recheck()
    elif args.command == "review":
        review()
    else:
        verify()


if __name__ == "__main__":
    main()
