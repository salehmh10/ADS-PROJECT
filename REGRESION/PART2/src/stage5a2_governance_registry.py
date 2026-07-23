"""Append or reuse the single Stage 5A2 governance Registry row."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_governance_protected_hashes_before.json"
GOVERNANCE = ROOT / "artifacts/reports/stage5a2_governance_adjudication.json"
REPORT = ROOT / "artifacts/reports/stage5a2_governance_registry_update.json"
EXPORT = ROOT / "artifacts/results/stage5/deep_core/stage5a2_governance_registry_row.csv"
EXPERIMENT_ID = "stage5a2_governance_adjudication_1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8-sig"))
    governance = json.loads(GOVERNANCE.read_text(encoding="utf-8"))
    starting_size = int(baseline["registry_protected_prefix_size_bytes"])
    starting_hash = baseline["registry_protected_prefix_sha256"]
    current_bytes = REGISTRY.read_bytes()
    if len(current_bytes) < starting_size or sha256_bytes(current_bytes[:starting_size]) != starting_hash:
        raise RuntimeError("The protected 323-row Registry prefix changed")

    frame = pd.read_csv(REGISTRY)
    matches = frame.loc[frame["experiment_id"] == EXPERIMENT_ID]
    action = "REUSED"
    if len(matches) == 0:
        header = list(frame.columns)
        row = {column: "" for column in header}
        row.update({
            "experiment_id": EXPERIMENT_ID,
            "timestamp_utc": governance["human_adjudication_timestamp"],
            "model_family": "deep_governance_adjudication",
            "model_name": "accepted_procedural_test_row_materialization",
            "sensitive_mode": "both",
            "feature_set": "deep_core_v1",
            "target_mode": "raw",
            "evaluation_stage": "Stage 5A2 Governance Adjudication",
            "training_row_count": "399788",
            "validation_row_count": "25000",
            "test_row_count": "0",
            "parameter_json": json.dumps(governance["classifications"], sort_keys=True, separators=(",", ":")),
            "status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
            "notes": (
                "literal_zero_test_loading=false; procedural exception accepted; "
                "statistical Test leakage not demonstrated; no refit; Stage 5B not started."
            ),
            "model_artifact_path": "artifacts/reports/stage5a2_governance_adjudication.json",
        })
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\r\n")
        writer.writerow(row)
        with REGISTRY.open("ab") as handle:
            handle.write(buffer.getvalue().encode("utf-8"))
        action = "APPENDED"

    updated = pd.read_csv(REGISTRY)
    matches = updated.loc[updated["experiment_id"] == EXPERIMENT_ID]
    if len(matches) != 1 or updated["experiment_id"].nunique() != len(updated):
        raise RuntimeError("Governance Registry ID is missing, duplicated, or Registry IDs are not unique")
    if len(updated) != 324:
        raise RuntimeError(f"Expected 324 Registry rows, found {len(updated)}")
    final_bytes = REGISTRY.read_bytes()
    if sha256_bytes(final_bytes[:starting_size]) != starting_hash:
        raise RuntimeError("Registry append changed the protected prefix")

    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(EXPORT, index=False, lineterminator="\n")
    previous = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    report = {
        "adjudication_id": EXPERIMENT_ID,
        "status": "PASS",
        "initial_action": previous.get("initial_action", action),
        "last_action": action,
        "idempotent_reuse_observed": previous.get("last_action") == "APPENDED" and action == "REUSED"
        or previous.get("idempotent_reuse_observed", False),
        "registry_path": str(REGISTRY.relative_to(ROOT)),
        "registry_row_count": int(len(updated)),
        "unique_experiment_ids": int(updated["experiment_id"].nunique()),
        "governance_row_count": int(len(matches)),
        "prior_323_row_byte_prefix_preserved": True,
        "prior_prefix_size_bytes": starting_size,
        "prior_prefix_sha256": starting_hash,
        "registry_sha256": sha256(REGISTRY),
        "governance_export_path": str(EXPORT.relative_to(ROOT)),
        "governance_export_sha256": sha256(EXPORT),
        "model_fits": 0,
        "preprocessing_fits": 0,
        "prediction_generations": 0,
    }
    atomic_json(report, REPORT)
    print(json.dumps({"status": "PASS", "action": action, "rows": len(updated)}))


if __name__ == "__main__":
    main()

