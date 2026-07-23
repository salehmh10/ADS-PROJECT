"""Create the one human-approved Stage 5A2 RealMLP freeze amendment."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ORIGINAL_FREEZE = ROOT / "artifacts/reports/stage5a2_prevalidation_freeze.json"
ORIGINAL_FREEZE_SHA256 = "5d7406e179bf8554ea12c2ee2d9cc052b58bee4495b12cec331382a87b1ee4c4"
FAILED_ID = "stage5a2__realmlp__refined"
REPLACEMENT_ID = "stage5a2__realmlp__refined__dropout020__replacement1"
FAILED_CHECKPOINT = ROOT / f"artifacts/checkpoints/stage5/deep_core/final_validation/{FAILED_ID}.json"
FAILED_PARENT = ROOT / f"artifacts/reports/stage5a2_parent_{FAILED_ID}.json"
FAILED_LOG = ROOT / f"artifacts/reports/stage5a2_parent_{FAILED_ID}.log"
CLASSIFICATION = ROOT / "artifacts/reports/stage5a2_realmlp_failed_refinement_classification.json"
AMENDMENT = ROOT / "artifacts/reports/stage5a2_prevalidation_freeze_amendment_realmlp_dropout020.json"
LOCK = ROOT / "artifacts/manifests/stage5/stage5a2_freeze_amendment_lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if sha256_file(ORIGINAL_FREEZE) != ORIGINAL_FREEZE_SHA256:
        raise RuntimeError("Original pre-validation freeze changed")
    if AMENDMENT.exists() or LOCK.exists() or CLASSIFICATION.exists():
        raise RuntimeError("Amendment evidence already exists; validate and resume without rewriting")
    for path in (FAILED_CHECKPOINT, FAILED_PARENT, FAILED_LOG):
        if not path.exists():
            raise RuntimeError(f"Required failed-attempt evidence is missing: {path}")
    failure = json.loads(FAILED_CHECKPOINT.read_text(encoding="utf-8"))
    parent = json.loads(FAILED_PARENT.read_text(encoding="utf-8"))
    expected_text = "Providing a validation split requires n_repeats=1, but got n_repeats=2"
    if failure.get("status") != "FAIL" or expected_text not in failure.get("error", ""):
        raise RuntimeError("Failed refinement evidence does not match the approved incompatibility")
    original = json.loads(ORIGINAL_FREEZE.read_text(encoding="utf-8"))
    frozen = next(item for item in original["candidates"] if item["candidate_id"] == "stage5a2__realmlp__frozen")
    failed = next(item for item in original["candidates"] if item["candidate_id"] == FAILED_ID)
    replacement = copy.deepcopy(frozen)
    replacement.update({
        "candidate_id": REPLACEMENT_ID,
        "candidate_type": "refined",
        "candidate_role": "authorized_technical_protocol_amendment_replacement",
        "replacement_of": FAILED_ID,
        "refinement_reason": (
            "Human-approved next pre-specified RealMLP refinement fallback. The installed official API "
            "rejected n_repeats=2 with the unchanged external Validation split before epoch 1. "
            "Keep n_repeats=1 and change only p_drop from 0.15 to 0.20. Final Selection performance "
            "did not select this direction."
        ),
    })
    replacement["architecture_and_training"]["p_drop"] = 0.20
    differences = {
        key: {"frozen": frozen["architecture_and_training"].get(key),
              "replacement": replacement["architecture_and_training"].get(key)}
        for key in sorted(set(frozen["architecture_and_training"]) | set(replacement["architecture_and_training"]))
        if frozen["architecture_and_training"].get(key) != replacement["architecture_and_training"].get(key)
    }
    if differences != {"p_drop": {"frozen": 0.15, "replacement": 0.2}}:
        raise RuntimeError(f"Replacement has unauthorized scientific differences: {differences}")
    classification = {
        "stage_id": "stage5a2", "candidate_id": FAILED_ID,
        "classification": "failed_before_training_unsupported_n_repeats_with_external_validation",
        "configured_n_repeats": failed["architecture_and_training"]["n_repeats"],
        "external_validation_required": True, "training_epochs_started": 0,
        "refined_model_weights_created": False, "status": "PRESERVED_FAILURE",
        "failed_checkpoint_path": str(FAILED_CHECKPOINT.relative_to(ROOT)),
        "failed_checkpoint_sha256": sha256_file(FAILED_CHECKPOINT),
        "failed_parent_path": str(FAILED_PARENT.relative_to(ROOT)),
        "failed_parent_sha256": sha256_file(FAILED_PARENT),
        "failed_log_path": str(FAILED_LOG.relative_to(ROOT)),
        "failed_log_sha256": sha256_file(FAILED_LOG),
    }
    atomic_json(classification, CLASSIFICATION)
    amendment = {
        "stage_id": "stage5a2",
        "amendment_id": "stage5a2__freeze_amendment__realmlp_dropout020__replacement1",
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "authorization_scope": "one RealMLP without_sensitive raw targeted-refinement technical protocol amendment fit",
        "original_freeze_path": str(ORIGINAL_FREEZE.relative_to(ROOT)),
        "original_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "original_freeze_unchanged": sha256_file(ORIGINAL_FREEZE) == ORIGINAL_FREEZE_SHA256,
        "failed_candidate_id": FAILED_ID,
        "failed_candidate_classification": classification["classification"],
        "failed_candidate_evidence": str(CLASSIFICATION.relative_to(ROOT)),
        "failed_candidate_evidence_sha256": sha256_file(CLASSIFICATION),
        "replacement_candidate_id": REPLACEMENT_ID,
        "replacement_candidate_definition": replacement,
        "only_scientific_parameter_change_from_frozen_realmlp": differences,
        "n_repeats_proof": replacement["architecture_and_training"]["n_repeats"] == 1,
        "p_drop_proof": replacement["architecture_and_training"]["p_drop"] == 0.20,
        "external_validation_row_id_hash": original["final_selection_validation_row_id_hash"],
        "training_row_id_hash": original["final_selection_train_row_id_hash"],
        "feature_schema": original["feature_schema"],
        "target_mode": "raw", "sensitive_mode": "without_sensitive",
        "seed": 42, "device": "cpu", "precision": "float32",
        "direction_source": "pre-specified RealMLP refinement fallback order plus official API incompatibility",
        "final_selection_performance_used_to_choose_replacement": False,
        "test_or_stage4l_test_evidence_used": False,
        "replacement_fit_budget": 1,
        "no_other_post_freeze_amendment_authorized": True,
        "status": "PASS",
    }
    atomic_json(amendment, AMENDMENT)
    lock = {
        "stage_id": "stage5a2", "amendment_path": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256_file(AMENDMENT),
        "original_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "failed_evidence_sha256": sha256_file(CLASSIFICATION),
        "replacement_candidate_id": REPLACEMENT_ID, "status": "LOCKED",
    }
    atomic_json(lock, LOCK)
    print(json.dumps({"classification": classification, "amendment": amendment, "lock": lock}, indent=2))


if __name__ == "__main__":
    main()
