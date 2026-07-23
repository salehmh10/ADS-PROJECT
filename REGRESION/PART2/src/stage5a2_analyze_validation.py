"""Aggregate four valid Stage 5A2 Candidates, bootstrap, Gate, and Core winner."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stage5a2_deep_utils import (
    AMENDMENT, EXPECTED_FREEZE_SHA256, FREEZE, REPLACEMENT_CANDIDATE_ID,
    ROOT, atomic_csv, atomic_json, digest_values, sha256_file, validate_freeze,
)


VALID = [
    "stage5a2__realmlp__frozen",
    REPLACEMENT_CANDIDATE_ID,
    "stage5a2__ft_transformer__frozen",
    "stage5a2__ft_transformer__refined",
]
FAILED = "stage5a2__realmlp__refined"
RESULT_ROOT = ROOT / "artifacts/results/stage5/deep_core/final_validation"
PRED_ROOT = ROOT / "artifacts/predictions/stage5/deep_core/final_validation"


def result_path(candidate_id: str) -> Path:
    return RESULT_ROOT / f"candidates/{candidate_id}.json"


def prediction_path(candidate_id: str) -> Path:
    return PRED_ROOT / f"{candidate_id}.csv"


def main() -> None:
    validate_freeze()
    lock = json.loads((ROOT / "artifacts/manifests/stage5/stage5a2_freeze_amendment_lock.json").read_text(encoding="utf-8"))
    if sha256_file(AMENDMENT) != lock["amendment_sha256"]:
        raise RuntimeError("Freeze amendment changed")
    failure_class = json.loads((ROOT / "artifacts/reports/stage5a2_realmlp_failed_refinement_classification.json").read_text(encoding="utf-8"))
    if failure_class["classification"] != "failed_before_training_unsupported_n_repeats_with_external_validation":
        raise RuntimeError("Failed refinement classification changed")

    results = []
    predictions = {}
    reference_rows = None
    reference_target = None
    for candidate_id in VALID:
        result = json.loads(result_path(candidate_id).read_text(encoding="utf-8"))
        reload_report = json.loads((ROOT / f"artifacts/reports/stage5a2_reload_{candidate_id}.json").read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or reload_report.get("status") != "PASS":
            raise RuntimeError(f"Candidate or reload is not PASS: {candidate_id}")
        prediction = pd.read_csv(prediction_path(candidate_id))
        rows = prediction["row_id"].to_numpy(np.int64)
        targets = prediction["y_true"].to_numpy(np.float64)
        if len(prediction) != 25_000 or not prediction["row_id"].is_unique or not np.isfinite(prediction["y_pred"]).all():
            raise RuntimeError(f"Invalid predictions: {candidate_id}")
        if reference_rows is None:
            reference_rows, reference_target = rows, targets
        elif not np.array_equal(rows, reference_rows) or not np.array_equal(targets, reference_target):
            raise RuntimeError("Four-Candidate row or target alignment mismatch")
        predictions[candidate_id] = prediction
        row = {
            "candidate_id": candidate_id,
            "candidate_type": "refined" if "refined" in candidate_id else "frozen",
            "model_family": result["model_family"], "target_mode": result["target_mode"],
            "sensitive_mode": result["sensitive_mode"], "feature_schema": result["feature_schema"],
            "architecture": json.dumps(result["architecture"], sort_keys=True),
            "preprocessing": "realmlp_train_only_vocab_plus_official" if result["model_family"] == "realmlp" else "ft_quantile_normal_train_only_vocab",
            "best_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"],
            **result["metrics"],
            "fit_time_seconds": result["fit_time_seconds"],
            "prediction_time_seconds": result["prediction_time_seconds"],
            "peak_process_tree_ram_mib": result.get("peak_process_tree_ram_mib", result["peak_ram_mib"]),
            "model_size_bytes": result["model_size_bytes"], "status": "PASS",
        }
        results.append(row)
    table = pd.DataFrame(results)
    if len(table) != 4 or table["candidate_id"].nunique() != 4:
        raise RuntimeError("Exactly four valid regular Candidates are required")
    atomic_csv(table, RESULT_ROOT / "stage5a2_final_validation_results.csv")

    best_real_id = table.loc[table.query("model_family == 'realmlp'")["mae"].idxmin(), "candidate_id"]
    best_ft_id = table.loc[table.query("model_family == 'ft_transformer'")["mae"].idxmin(), "candidate_id"]
    real = predictions[best_real_id]["y_pred"].to_numpy(np.float64)
    ft = predictions[best_ft_id]["y_pred"].to_numpy(np.float64)
    y = reference_target
    point = float(np.mean(np.abs(ft - y)) - np.mean(np.abs(real - y)))
    rng = np.random.default_rng(42)
    bootstrap_rows = []
    for index in range(300):
        sampled = rng.integers(0, len(y), size=len(y))
        ft_mae = float(np.mean(np.abs(ft[sampled] - y[sampled])))
        real_mae = float(np.mean(np.abs(real[sampled] - y[sampled])))
        bootstrap_rows.append({"bootstrap_id": index + 1, "ft_transformer_mae": ft_mae,
                               "realmlp_mae": real_mae, "mae_difference_ft_minus_realmlp": ft_mae - real_mae})
    bootstrap = pd.DataFrame(bootstrap_rows)
    atomic_csv(bootstrap, RESULT_ROOT / "stage5a2_paired_bootstrap.csv")
    values = bootstrap["mae_difference_ft_minus_realmlp"].to_numpy()
    ci_low, ci_high = [float(value) for value in np.quantile(values, [0.025, 0.975])]
    win_proportion_realmlp = float(np.mean(values > 0))
    best_real_mae = float(table.loc[table["candidate_id"] == best_real_id, "mae"].iloc[0])
    best_ft_mae = float(table.loc[table["candidate_id"] == best_ft_id, "mae"].iloc[0])
    gap_percent = abs(best_ft_mae - best_real_mae) / min(best_ft_mae, best_real_mae) * 100.0
    bootstrap_summary = {
        "stage_id": "stage5a2", "comparison": "FT-Transformer MAE - RealMLP MAE",
        "best_realmlp_candidate_id": best_real_id, "best_ft_transformer_candidate_id": best_ft_id,
        "point_difference": point, "ci_95_percentile_low": ci_low, "ci_95_percentile_high": ci_high,
        "realmlp_win_proportion": win_proportion_realmlp, "resamples": 300, "seed": 42,
        "same_row_indices_per_resample": True, "original_target_scale": True, "status": "PASS",
    }
    atomic_json(bootstrap_summary, RESULT_ROOT / "stage5a2_paired_bootstrap_summary.json")
    gate_triggered = bool(gap_percent < 0.25 and ci_low <= 0.0 <= ci_high)
    gate = {
        "stage_id": "stage5a2", "absolute_regular_fit_mae_gap_percent": gap_percent,
        "gap_condition_less_than_0_25_percent": gap_percent < 0.25,
        "bootstrap_interval_includes_zero": ci_low <= 0.0 <= ci_high,
        "stability_gate_triggered": gate_triggered,
        "extra_seed": 2026, "stability_fit_count": 2 if gate_triggered else 0,
        "reason": "Both strict conditions passed." if gate_triggered else "Strict stability Gate did not satisfy both conditions; zero extra-seed fits are authorized.",
        "status": "PASS",
    }
    atomic_json(gate, RESULT_ROOT / "stage5a2_stability_gate.json")
    if gate_triggered:
        raise RuntimeError("Stability Gate triggered; run exactly two frozen seed-2026 fits before selecting a winner")

    # The strict Gate did not trigger. Primary MAE selects RealMLP frozen.
    winner_id = best_real_id if best_real_mae < best_ft_mae else best_ft_id
    winner_result = json.loads(result_path(winner_id).read_text(encoding="utf-8"))
    winner_bundle = ROOT / winner_result["bundle_path"]
    winner_prediction = prediction_path(winner_id)
    winner = {
        "stage_id": "stage5a2", "status": "FROZEN",
        "family": winner_result["model_family"], "candidate_id": winner_id,
        "feature_schema": winner_result["feature_schema"], "target_mode": winner_result["target_mode"],
        "sensitive_mode": "without_sensitive", "preprocessing": winner_result.get("preprocessing_version", "bundle-contained training-only preprocessing"),
        "architecture": winner_result["architecture"], "training_configuration": winner_result["training_configuration"],
        "best_epoch": winner_result["best_epoch"], "seed": winner_result["seed"],
        "device": winner_result["device"], "precision": winner_result["precision"],
        "metrics": winner_result["metrics"],
        "selection_reason": "Lowest original-scale Validation MAE; the strict stability Gate did not trigger. Paired bootstrap and tail metrics were retained as supporting evidence.",
        "paired_bootstrap": bootstrap_summary, "stability_gate": gate,
        "validation_bundle_path": str(winner_bundle.relative_to(ROOT)),
        "validation_bundle_sha256": sha256_file(winner_bundle),
        "validation_prediction_path": str(winner_prediction.relative_to(ROOT)),
        "validation_prediction_sha256": sha256_file(winner_prediction),
        "validation_row_id_hash": digest_values(reference_rows),
        "non_sensitive_validation_model_reused": True, "non_sensitive_refit_count": 0,
        "test_or_stage4l_test_evidence_used": False,
        "original_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "amendment_sha256": sha256_file(AMENDMENT),
    }
    atomic_json(winner, RESULT_ROOT / "stage5a_core_winner_configuration.json")
    atomic_json({
        "stage_id": "stage5a2", "valid_regular_candidate_count": 4,
        "valid_candidate_ids": VALID, "failed_audit_candidate_id": FAILED,
        "failed_candidate_counted_as_valid": False,
        "failed_candidate_classification": failure_class["classification"],
        "row_alignment_pass": True, "validation_rows": 25_000,
        "validation_row_id_hash": digest_values(reference_rows),
        "target_hash": hashlib.sha256(np.ascontiguousarray(reference_target).view(np.uint8)).hexdigest(),
        "status": "PASS",
    }, RESULT_ROOT / "stage5a2_four_candidate_validation.json")
    print(json.dumps({"results": results, "bootstrap": bootstrap_summary, "gate": gate, "winner": winner}, indent=2))


if __name__ == "__main__":
    import hashlib
    main()
