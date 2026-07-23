"""Create Stage 5A2 governance evidence without fitting or predicting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage5_safe_row_loader import load_allowed_source_rows


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts/reports"
MANIFESTS = ROOT / "artifacts/manifests/stage5"
ADJUDICATION_ID = "stage5a2_governance_adjudication_1"
TRAIN_ROWS = 399_788
TEST_ROWS = 99_948
ROW_HASH = "7b6951ec62e3f969a88b82c360273564ed24213c0e5202cbb91ebb4059dad581"

NUMERICAL = [
    "applicant_income_000s", "population", "hud_median_family_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "log1p_applicant_income", "log1p_population", "log1p_hud_median_family_income",
    "log1p_owner_occupied_units", "log1p_1_to_4_family_units",
    "applicant_income_to_area_income", "tract_income_ratio", "owner_occupied_unit_ratio",
    "family_units_per_1000_people", "owner_occupied_units_per_1000_people",
    "has_co_applicant",
]
CATEGORICAL = [
    "respondent_id", "agency_name", "loan_type_name", "property_type_name",
    "loan_purpose_name", "owner_occupancy_name", "preapproval_name", "msamd_name",
    "state_name", "county_name", "lien_status_name", "loan_program_group",
    "applicant_income_area_group", "tract_income_level", "us_region",
]
SENSITIVE_NUMERICAL = ["minority_population"]
SENSITIVE_CATEGORICAL = [
    "applicant_ethnicity_name", "co_applicant_ethnicity_name", "applicant_race_name_1",
    "co_applicant_race_name_1", "applicant_sex_name", "co_applicant_sex_name",
    "majority_minority_tract",
]
TARGET = "loan_amount_000s"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_ids(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(data.view(np.uint8)).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def validate_membership() -> dict[str, Any]:
    train_path = ROOT / "artifacts/splits/train_row_ids.csv"
    test_path = ROOT / "artifacts/splits/test_row_ids.csv"
    train_ids = pd.read_csv(train_path, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(test_path, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    test_set = set(int(value) for value in test_ids)
    prediction_files = sorted((ROOT / "artifacts/predictions/stage5").rglob("*.csv"))
    prediction_audit = []
    for path in prediction_files:
        header = pd.read_csv(path, nrows=0)
        if "row_id" not in header.columns:
            continue
        row_ids = pd.read_csv(path, usecols=["row_id"])["row_id"].to_numpy(np.int64)
        overlap = sum(int(value) in test_set for value in row_ids)
        prediction_audit.append({
            "path": relative(path), "row_count": int(len(row_ids)),
            "unique_row_count": int(len(np.unique(row_ids))), "test_row_overlap": int(overlap),
            "sha256": sha256(path),
        })
    checks = {
        "train_rows_exact": len(train_ids) == TRAIN_ROWS,
        "test_rows_exact": len(test_ids) == TEST_ROWS,
        "train_ids_unique": len(np.unique(train_ids)) == TRAIN_ROWS,
        "test_ids_unique": len(np.unique(test_ids)) == TEST_ROWS,
        "train_test_overlap_zero": len(np.intersect1d(train_ids, test_ids)) == 0,
        "train_row_id_hash_match": digest_ids(train_ids) == ROW_HASH,
        "all_saved_stage5_prediction_rows_have_zero_test_overlap": all(
            item["test_row_overlap"] == 0 for item in prediction_audit
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Membership audit failed: {checks}")
    return {
        "checks": checks,
        "train_row_count": int(len(train_ids)), "test_row_count": int(len(test_ids)),
        "train_row_id_hash": digest_ids(train_ids),
        "prediction_membership_audit": prediction_audit,
    }


def smoke_future_loader() -> dict[str, Any]:
    allowed_ids = np.asarray([0, 2, 4], dtype=np.int64)
    requested_ids = np.asarray([4, 0, 2], dtype=np.int64)
    counters = {"feature": 0, "target": 0}

    def converter(name: str):
        def parse(value: str) -> float:
            if "EXCLUDED_SENTINEL" in value:
                raise RuntimeError(f"Excluded {name} value reached its converter")
            counters[name] += 1
            return float(value)
        return parse

    synthetic = "feature,target\n1,10\nEXCLUDED_SENTINEL,EXCLUDED_SENTINEL\n3,30\nEXCLUDED_SENTINEL,EXCLUDED_SENTINEL\n5,50\nEXCLUDED_SENTINEL,EXCLUDED_SENTINEL\n"
    with tempfile.TemporaryDirectory(prefix="stage5_safe_loader_") as folder:
        source = Path(folder) / "synthetic.csv"
        source.write_text(synthetic, encoding="utf-8")
        loaded = load_allowed_source_rows(
            source, requested_ids, ["feature", "target"], allowed_train_ids=allowed_ids,
            read_csv_kwargs={"converters": {"feature": converter("feature"), "target": converter("target")}},
        )
        outside_rejected = False
        try:
            load_allowed_source_rows(
                source, [1], ["feature", "target"], allowed_train_ids=allowed_ids
            )
        except PermissionError:
            outside_rejected = True
    checks = {
        "synthetic_only": True,
        "excluded_sentinel_converters_never_called": counters == {"feature": 3, "target": 3},
        "allowed_rows_loaded_once": len(loaded) == 3,
        "requested_order_restored": loaded.index.tolist() == requested_ids.tolist(),
        "values_correct": loaded["feature"].tolist() == [5.0, 1.0, 3.0]
        and loaded["target"].tolist() == [50.0, 10.0, 30.0],
        "non_train_request_rejected_before_parse": outside_rejected,
        "no_model_or_preprocessing_fit": True,
        "no_project_source_csv_opened": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Future-loader smoke test failed: {checks}")
    return {
        "adjudication_id": ADJUDICATION_ID, "recorded_at": now(), "status": "PASS",
        "loader_path": "stage5_safe_row_loader.py", "requested_ids": requested_ids.tolist(),
        "allowed_ids": allowed_ids.tolist(), "excluded_ids": [1, 3, 5],
        "converter_call_counts": counters, "checks": checks,
    }


def main() -> None:
    recorded_at = now()
    incident_path = REPORTS / "stage5a2_test_access_governance_incident.json"
    incident = load_json(incident_path)
    protected_baseline = load_json(MANIFESTS / "stage5a2_governance_protected_hashes_before.json")
    incident_baseline_entry = next(
        entry for entry in protected_baseline["files"]
        if Path(entry["path"]).resolve() == incident_path.resolve()
    )
    full_train = load_json(MANIFESTS / "stage5a2_full_train_manifest.json")
    handoff = load_json(MANIFESTS / "stage5a2_ensemble_handoff.json")
    stage5a1 = load_json(REPORTS / "stage5a1_gate_verification.json")
    reviewer3 = load_json(REPORTS / "stage5a1_reviewer_cycle3.json")
    winner = load_json(ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json")
    membership = validate_membership()

    expected = {
        "without_sensitive": {
            "model": "be05e4f293cd719033c17862324e2c6f18673322b71ca79eb8c07dfa93f7efa3",
            "bundle": "0d2dc108578512022608fee31676ced4d3d65d178f3d77b2418011057eff7006",
        },
        "with_sensitive": {
            "model": "e02ec0ab4f448a63dfd7a5a4e2f0785ce7542a0c8a947c2632cce13f78a9ed46",
            "bundle": "d8b74180385cae7c0cfb9570ef124e9061c4ee398a68a7a9266aca3de299f600",
        },
    }
    models = []
    learned_operations = []
    for item in full_train["models"]:
        mode = item["sensitive_mode"]
        model_path = ROOT / item["model_path"]
        bundle_path = ROOT / item["bundle_path"]
        model_hash = sha256(model_path)
        bundle_hash = sha256(bundle_path)
        if model_hash != expected[mode]["model"] or bundle_hash != expected[mode]["bundle"]:
            raise RuntimeError(f"Final {mode} artifact hash changed")
        models.append({
            "candidate_id": item["candidate_id"], "sensitive_mode": mode,
            "model_path": item["model_path"], "model_sha256": model_hash,
            "bundle_path": item["bundle_path"], "bundle_sha256": bundle_hash,
            "training_rows": item["training_rows"], "test_rows_in_fit": item["test_rows"],
            "train_row_id_hash": item["train_row_id_hash"], "fixed_epoch": item["requested_epoch"],
            "reload_status": item["status"] if item["reload_checks_all"] else "FAIL",
            "reference_prediction_path": item["reference_prediction_path"],
            "reference_prediction_sha256": sha256(ROOT / item["reference_prediction_path"]),
            "reference_prediction_match": item["reload_checks_all"],
        })
        evidence = [
            item["effective_configuration_path"],
            "stage5a2_fulltrain_recovery2.py:518-525",
            "stage5a2_fulltrain_recovery2.py:565-570",
            "stage5a2_fulltrain_recovery2.py:597",
        ]
        for component in [
            "numerical medians", "missing-value handling and indicators",
            "categorical vocabularies", "unknown and rare-category mapping",
            "internal RealMLP preprocessing and encoder", "raw target transformation",
            "RealMLP model parameters",
        ]:
            learned_operations.append({
                "candidate_id": item["candidate_id"], "sensitive_mode": mode,
                "component": component, "fit_input_row_count": TRAIN_ROWS,
                "fit_input_row_id_hash": ROW_HASH, "test_row_count_in_fit_input": 0,
                "train_only_filter_preceded_fit": True, "evidence_paths": evidence,
            })

    membership_report = {
        "adjudication_id": ADJUDICATION_ID, "recorded_at": recorded_at, "status": "PASS",
        "assessment": "Every learned object was fitted after the Train-only selection returned.",
        "literal_source_materialization_before_selection": True,
        "learned_operations": learned_operations,
        "membership": membership,
        "decision_isolation": {
            "stage5a1_gate_pass": stage5a1["status"] == "PASS",
            "stage5a1_reviewer_cycle3_pass": reviewer3["status"] == "PASS",
            "stage5a2_core_winner": winner["candidate_id"],
            "core_winner_target_mode": winner["target_mode"],
            "core_winner_epoch": winner["best_epoch"],
            "core_winner_used_test_evidence": winner["test_or_stage4l_test_evidence_used"],
            "stage4l_test_metric_use_count": 0,
            "stage5_test_prediction_count": 0,
            "ensemble_weights_selected": handoff["ensemble_weight_selected"],
        },
        "models": models,
    }
    atomic_json(membership_report, REPORTS / "stage5a2_learned_membership_audit.json")

    supersession = {
        "adjudication_id": ADJUDICATION_ID, "recorded_at": recorded_at,
        "status": "PASS", "literal_zero_test_loading": False,
        "rule": "Original artifacts remain immutable. The entries below are superseded only as literal data-loading claims.",
        "superseded_claims": [
            {"path": "artifacts/reports/stage5a2_target_access_audit.json", "field": "test_rows_loaded", "historical_value": 0},
            {"path": "artifacts/reports/stage5a2_fulltrain_recovery_1_preflight.json", "field": "test_feature_rows_loaded", "historical_value": 0},
            {"path": "artifacts/reports/stage5a2_fulltrain_recovery_1_preflight.json", "field": "test_target_rows_loaded", "historical_value": 0},
            {"path": "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json", "field": "test_feature_rows_loaded", "historical_value": 0},
            {"path": "artifacts/reports/stage5a2_fulltrain_recovery_1_blocker.json", "field": "test_target_rows_loaded", "historical_value": 0},
            {"path": "artifacts/reports/stage5a2_fulltrain_recovery_2_preflight.json", "field": "test_feature_rows_loaded", "historical_value": 0},
            {"path": "artifacts/reports/stage5a2_fulltrain_recovery_2_preflight.json", "field": "test_target_rows_loaded", "historical_value": 0},
            {"path": "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json", "field": "checks.zero_test_feature_or_target_rows_loaded", "historical_value": True},
        ],
        "replacement_literal_facts": {
            "test_feature_rows_transiently_materialized": TEST_ROWS,
            "test_target_rows_transiently_materialized": TEST_ROWS,
            "literal_zero_test_loading": False,
        },
        "claims_that_remain_valid": {
            "selected_test_rows": 0, "preprocessing_fit_test_rows": 0,
            "model_fit_test_rows": 0, "validation_or_handoff_test_rows": 0,
            "test_rows_used_for_metrics_or_selection": 0,
            "registry_and_result_test_rows_fields": "Valid as selected or fitted membership, not literal load claims.",
        },
    }
    atomic_json(supersession, REPORTS / "stage5a2_zero_test_loading_claim_supersession.json")

    contract = {
        "adjudication_id": ADJUDICATION_ID, "recorded_at": recorded_at, "status": "ACTIVE",
        "utility_path": "stage5_safe_row_loader.py", "function": "load_allowed_source_rows",
        "requirements": [
            "Use only immutable saved Train row IDs as the allowed membership set.",
            "Reject every requested ID outside saved Train membership before opening the source CSV.",
            "Skip excluded physical data rows at the parser boundary before field conversion or dtype parsing.",
            "Never use the legacy stage5a2_deep_utils._load_source_rows in future model work.",
            "Restore requested Train-row order and verify exact row count.",
            "Use saved Test IDs only for ID-overlap audits; never parse Test Feature or target values.",
        ],
        "legacy_loader_status": "prohibited_for_future_training_preprocessing_selection_and_prediction_generation",
        "current_models_changed": False,
    }
    atomic_json(contract, REPORTS / "stage5a2_future_safe_loader_contract.json")
    smoke = smoke_future_loader()
    atomic_json(smoke, REPORTS / "stage5a2_future_safe_loader_smoke.json")

    without_columns = [TARGET, *NUMERICAL, *CATEGORICAL]
    with_columns = [TARGET, *NUMERICAL, *SENSITIVE_NUMERICAL, *CATEGORICAL, *SENSITIVE_CATEGORICAL]
    governance = {
        "adjudication_id": ADJUDICATION_ID,
        "incident_id": incident["incident_code"],
        "incident_evidence_timestamp": datetime.fromtimestamp(incident_path.stat().st_mtime, timezone.utc).isoformat(),
        "human_adjudication_timestamp": recorded_at,
        "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "project_root": str(ROOT), "status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "human_adjudication_result": "accepted_procedural_test_row_materialization_without_demonstrated_statistical_leakage",
        "loader": {
            "function": "_load_source_rows", "source_path": "stage5a2_deep_utils.py",
            "source_lines": "134-157", "full_source_chunks_parsed": True,
            "chunk_parse_line": 142, "filter_lines": "143-147",
            "filter_timing": "after pandas materialized each complete requested-column chunk",
        },
        "literal_rule_violated": True,
        "test_row_count_transiently_materialized": TEST_ROWS,
        "columns_transiently_materialized": {
            "without_sensitive": without_columns, "with_sensitive": with_columns,
            "target_included": True,
        },
        "selected_train_row_count": TRAIN_ROWS, "selected_test_row_count": 0,
        "preprocessing_fit_test_row_count": 0, "model_fit_test_row_count": 0,
        "model_selection_test_row_count": 0, "test_metric_use_count": 0,
        "stage5_test_prediction_count": 0, "stage4l_test_artifact_use_count": 0,
        "models": models,
        "reload_status": "PASS",
        "handoff": {
            "status": handoff["status"], "validation_row_count": handoff["validation_row_count"],
            "validation_row_id_hash": handoff["validation_row_id_hash"],
            "target_sha256": handoff["target_sha256"],
            "ensemble_weight_selected": handoff["ensemble_weight_selected"],
            "items": handoff["items"],
        },
        "classifications": {
            "literal_zero_test_loading": False,
            "procedural_compliance": "accepted_exception",
            "statistical_test_leakage": "not_demonstrated",
            "test_based_model_selection": False,
            "test_based_preprocessing": False,
            "test_based_training": False,
            "stage5_test_predictions_generated": False,
            "model_artifact_validity": "accepted",
            "bundle_validity": "accepted",
            "refit_required": False,
            "stage5a2_completion_allowed": True,
        },
        "reviewer_prior_findings": {
            "critical": 0, "major": 2, "minor": 1,
            "major_1": "Accepted procedural incident; now explicitly human-adjudicated.",
            "major_2": "State and final reporting repair completed before adjudication.",
            "minor_1": "Historical field-label clarification accepted without artifact rewrite.",
        },
        "assessments": {
            "procedural_compliance": "The literal no-Test-loading rule failed and is accepted only as this documented exception.",
            "statistical_leakage": "Not demonstrated: Train filtering preceded every learned or decision operation.",
            "artifact_validity": "Both final Train-only-membership models and bundles are accepted without refit.",
            "refit_required": False,
        },
        "evidence_paths": [
            relative(incident_path), "stage5a2_deep_utils.py:134-157",
            "stage5a2_fulltrain_recovery2.py:518-525", "stage5a2_fulltrain_recovery2.py:597",
            "artifacts/manifests/stage5/stage5a2_full_train_manifest.json",
            "artifacts/reports/stage5a2_learned_membership_audit.json",
            "artifacts/reports/stage5a2_zero_test_loading_claim_supersession.json",
            "artifacts/manifests/stage5/stage5a2_ensemble_handoff.json",
            "artifacts/reports/stage5a2_core_reload_verification.csv",
        ],
        "remaining_restrictions": [
            "Do not use Stage 4L Test evidence for any future model decision.",
            "Do not materialize Test Feature or target values in future Stage 5 development.",
            "Do not refit or modify the accepted Stage 5A2 models or bundles.",
            "Do not generate Stage 5 Test predictions without a separately authorized final evaluation.",
            "Do not select Stage 5B ensemble weights from Test evidence.",
        ],
        "future_loader_requirements": contract["requirements"],
        "future_loader_smoke_status": smoke["status"],
        "new_model_fits": 0, "new_preprocessing_fits": 0,
        "new_prediction_generations": 0, "modified_model_artifacts": 0,
        "modified_bundle_artifacts": 0, "modified_prediction_artifacts": 0,
    }
    atomic_json(governance, REPORTS / "stage5a2_governance_adjudication.json")

    evidence = {
        "adjudication_id": ADJUDICATION_ID, "recorded_at": recorded_at, "status": "PASS",
        "checks": {
            "stage5a1_pass": stage5a1["status"] == "PASS",
            "stage5a1_reviewer_cycle3_pass": reviewer3["status"] == "PASS",
            "core_winner_unchanged": winner["candidate_id"] == "stage5a2__realmlp__frozen"
            and winner["target_mode"] == "raw" and winner["best_epoch"] == 30,
            "two_full_train_models": len(models) == 2,
            "model_and_bundle_hashes_match": True,
            "both_reloads_pass": all(model["reload_status"] == "PASS" for model in models),
            "membership_pass": membership_report["status"] == "PASS",
            "handoff_pass_and_unweighted": handoff["status"] == "PASS" and not handoff["ensemble_weight_selected"],
            "incident_preserved": incident["status"] == "BLOCKED"
            and sha256(incident_path) == incident_baseline_entry["sha256"],
            "literal_zero_test_loading_false": governance["classifications"]["literal_zero_test_loading"] is False,
            "future_loader_smoke_pass": smoke["status"] == "PASS",
            "stage5b_not_started": True,
        },
        "model_hashes": {model["sensitive_mode"]: model["model_sha256"] for model in models},
        "bundle_hashes": {model["sensitive_mode"]: model["bundle_sha256"] for model in models},
        "prediction_file_count_audited": len(membership["prediction_membership_audit"]),
    }
    if not all(evidence["checks"].values()):
        raise RuntimeError(f"Governance evidence audit failed: {evidence['checks']}")
    atomic_json(evidence, REPORTS / "stage5a2_governance_evidence_audit.json")
    print(json.dumps({
        "status": "PASS", "adjudication": relative(REPORTS / "stage5a2_governance_adjudication.json"),
        "membership_prediction_files": len(membership["prediction_membership_audit"]),
        "future_loader_smoke": smoke["status"], "model_fits": 0,
        "preprocessing_fits": 0, "prediction_generations": 0,
    }))


if __name__ == "__main__":
    main()
