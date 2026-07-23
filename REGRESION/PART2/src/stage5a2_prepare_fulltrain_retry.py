"""Preserve Stage 5A2 full-Train attempt 1 and authorize one technical retry."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CID = "stage5a2__realmlp__core_final__without_sensitive"
CHECKPOINT = ROOT / f"artifacts/checkpoints/stage5/deep_core/full_train/{CID}.json"
PARENT = ROOT / f"artifacts/reports/stage5a2_parent_{CID}.json"
LOG = ROOT / f"artifacts/reports/stage5a2_parent_{CID}.log"
PROOF = ROOT / f"artifacts/reports/{CID}_fixed_epoch_proof.json"
RETRY = ROOT / f"artifacts/reports/{CID}_technical_retry.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if RETRY.exists():
        raise RuntimeError("Technical retry manifest already exists")
    failure = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if failure.get("status") != "FAIL" or "fixed-epoch/refit proof failed" not in failure.get("error", ""):
        raise RuntimeError("Attempt 1 is not the expected post-fit proof failure")
    archive = ROOT / "artifacts/checkpoints/stage5/deep_core/full_train/attempts"
    report_archive = ROOT / "artifacts/reports/stage5a2_fulltrain_attempts"
    archive.mkdir(parents=True, exist_ok=True); report_archive.mkdir(parents=True, exist_ok=True)
    copies = {
        "checkpoint": archive / f"{CID}__attempt1_failure.json",
        "parent": report_archive / f"{CID}__attempt1_parent.json",
        "log": report_archive / f"{CID}__attempt1.log",
        "proof": report_archive / f"{CID}__attempt1_prefit_proof.json",
    }
    for source, destination in [(CHECKPOINT, copies["checkpoint"]), (PARENT, copies["parent"]),
                                (LOG, copies["log"]), (PROOF, copies["proof"])]:
        shutil.copy2(source, destination)
        if sha(source) != sha(destination):
            raise RuntimeError(f"Attempt 1 preservation hash mismatch: {source}")
    payload = {
        "stage_id": "stage5a2", "candidate_id": CID, "retry_number": 1,
        "authorized_by_original_policy": "Maximum one technical retry per Full-Train mode",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "attempt1_training_completed": True,
        "attempt1_failure_phase": "post_fit_proof_before_bundle_serialization",
        "attempt1_log_proves_cv_and_refit_max_epochs_30": True,
        "technical_root_cause": "Proof combined deployed-refit invariants with a non-deployment CV setup progress equality.",
        "technical_correction": "Record all observed values and require exact fixed-epoch invariants on the deployed all-row refit interface; keep the CV setup value as audit metadata only.",
        "scientific_configuration_changed": False, "training_rows_changed": False,
        "fixed_epoch_changed": False, "sensitive_mode_changed": False,
        "model_family_changed": False, "test_evidence_used": False,
        "maximum_retry_count": 1,
        "preserved_attempt1": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for key, path in copies.items()},
        "status": "RETRY_AUTHORIZED",
    }
    atomic_json(payload, RETRY)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
