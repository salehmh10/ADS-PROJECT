"""No-training production serializer preflight required before recovery 2."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage5a2_deep_utils import ROOT, sha256_file
from stage5a2_recovery_serialization import atomic_json, load_json, normalize_json


REPORT = ROOT / "artifacts/reports/stage5a2_recovery2_reporting_preflight.json"
SAMPLE_DIR = ROOT / "artifacts/checkpoints/stage5/deep_core/full_train/recovery2/reporting_preflight"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if REPORT.exists():
        existing = load_json(REPORT)
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "REUSED", "path": str(REPORT.relative_to(ROOT))}))
            return

    representative = {
        "python_bool": True,
        "numpy_bool_true": np.bool_(True),
        "numpy_bool_false": np.bool_(False),
        "pandas_nullable_boolean": pd.Series([True, False, pd.NA], dtype="boolean"),
        "numpy_integer": np.int64(46830),
        "numpy_floating": np.float32(30.0),
        "numpy_array": np.asarray([1, 2, 3], dtype=np.int32),
        "pandas_series": pd.Series([1.5, np.nan, 3.5], dtype="Float64"),
        "pandas_timestamp": pd.Timestamp("2026-07-15T17:30:00Z"),
        "pathlib_path": Path("artifacts/models/deep/core_final/model.joblib"),
        "nested": {
            "list": [np.bool_(True), {"count": np.int32(30), "score": np.float64(1.25)}],
            "array": np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64),
        },
        "permitted_missing_values": [None, pd.NA, pd.NaT, np.nan, np.float64(np.inf), np.float64(-np.inf)],
    }
    normalized = normalize_json(representative)
    payload_path = SAMPLE_DIR / "representative_payload.json"
    simulated_proof_path = SAMPLE_DIR / "simulated_epoch_proof.json"
    simulated_blocker_path = SAMPLE_DIR / "simulated_blocker_report.json"
    atomic_json(representative, payload_path)
    atomic_json({"status": np.bool_(True), "epoch": np.int64(30), "checks": representative}, simulated_proof_path)
    atomic_json({"status": "SIMULATED", "blocked": np.bool_(False), "details": representative}, simulated_blocker_path)

    reloaded = load_json(payload_path)
    proof = load_json(simulated_proof_path)
    blocker = load_json(simulated_blocker_path)
    checks = {
        "python_bool": isinstance(reloaded["python_bool"], bool) and reloaded["python_bool"] is True,
        "numpy_bool": isinstance(reloaded["numpy_bool_true"], bool) and reloaded["numpy_bool_false"] is False,
        "pandas_nullable_boolean": reloaded["pandas_nullable_boolean"] == [True, False, None],
        "numpy_integer": isinstance(reloaded["numpy_integer"], int) and reloaded["numpy_integer"] == 46830,
        "numpy_floating": isinstance(reloaded["numpy_floating"], float) and reloaded["numpy_floating"] == 30.0,
        "numpy_array": reloaded["numpy_array"] == [1, 2, 3],
        "pandas_series": reloaded["pandas_series"] == [1.5, None, 3.5],
        "pandas_timestamp": reloaded["pandas_timestamp"].startswith("2026-07-15T17:30:00"),
        "pathlib_path": reloaded["pathlib_path"].endswith("model.joblib"),
        "nested_values": reloaded["nested"]["list"] == [True, {"count": 30, "score": 1.25}],
        "missing_values": reloaded["permitted_missing_values"] == [None, None, None, None, None, None],
        "simulated_proof_written": proof["status"] is True and proof["epoch"] == 30,
        "simulated_blocker_written": blocker["status"] == "SIMULATED" and blocker["blocked"] is False,
        "representative_temp_removed": not payload_path.with_suffix(payload_path.suffix + ".tmp").exists(),
        "proof_temp_removed": not simulated_proof_path.with_suffix(simulated_proof_path.suffix + ".tmp").exists(),
        "blocker_temp_removed": not simulated_blocker_path.with_suffix(simulated_blocker_path.suffix + ".tmp").exists(),
        "normalization_idempotent": normalize_json(normalized) == normalized,
    }
    report = {
        "stage_id": "stage5a2", "recovery_id": "stage5a2_fulltrain_recovery_2",
        "recorded_at": utc_now(),
        "production_serializer": "stage5a2_recovery_serialization.atomic_json",
        "production_normalizer": "stage5a2_recovery_serialization.normalize_json",
        "tested_types": [
            "bool", "numpy.bool_", "pandas nullable Boolean", "numpy integer", "numpy floating-point",
            "numpy.ndarray", "pandas.Series", "pandas.Timestamp", "pathlib.Path",
            "nested dictionaries and lists", "permitted missing values",
        ],
        "sample_payload_path": str(payload_path.relative_to(ROOT)),
        "sample_payload_sha256": sha256_file(payload_path),
        "simulated_proof_path": str(simulated_proof_path.relative_to(ROOT)),
        "simulated_proof_sha256": sha256_file(simulated_proof_path),
        "simulated_blocker_path": str(simulated_blocker_path.relative_to(ROOT)),
        "simulated_blocker_sha256": sha256_file(simulated_blocker_path),
        "checks": {key: bool(value) for key, value in checks.items()},
        "type_error_occurred": False,
        "atomic_write_pass": bool(all(checks[key] for key in checks if "temp_removed" in key)),
        "reload_pass": bool(all(checks.values())),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, REPORT)
    verified = load_json(REPORT)
    if verified.get("status") != "PASS" or not all(verified["checks"].values()):
        raise RuntimeError("Recovery 2 reporting preflight failed after reload")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
