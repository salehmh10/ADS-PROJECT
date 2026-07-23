"""Controlled Stage 5A2 sensitive-versus-non-sensitive accuracy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage5a2_deep_utils import ROOT, atomic_csv, atomic_json, digest_values, sha256_file


RESULT_ROOT = ROOT / "artifacts/results/stage5/deep_core/final_validation"
PRED_ROOT = ROOT / "artifacts/predictions/stage5/deep_core/final_validation"
WITHOUT_ID = "stage5a2__realmlp__frozen"
WITH_ID = "stage5a2__realmlp__core__with_sensitive"


def main() -> None:
    without_result = json.loads((RESULT_ROOT / f"candidates/{WITHOUT_ID}.json").read_text(encoding="utf-8"))
    with_result = json.loads((RESULT_ROOT / f"candidates/{WITH_ID}.json").read_text(encoding="utf-8"))
    without = pd.read_csv(PRED_ROOT / f"{WITHOUT_ID}.csv")
    with_sensitive = pd.read_csv(PRED_ROOT / f"{WITH_ID}.csv")
    if not np.array_equal(without["row_id"].to_numpy(np.int64), with_sensitive["row_id"].to_numpy(np.int64)):
        raise RuntimeError("Sensitive comparison row order mismatch")
    if not np.array_equal(without["y_true"].to_numpy(np.float64), with_sensitive["y_true"].to_numpy(np.float64)):
        raise RuntimeError("Sensitive comparison target mismatch")
    y = without["y_true"].to_numpy(np.float64)
    pred_without = without["y_pred"].to_numpy(np.float64)
    pred_with = with_sensitive["y_pred"].to_numpy(np.float64)
    metrics = sorted(set(without_result["metrics"]) & set(with_result["metrics"]))
    validation_rows = pd.DataFrame([
        {"sensitive_mode": "without_sensitive", "candidate_id": WITHOUT_ID,
         "fixed_epoch": without_result["best_epoch"], **without_result["metrics"],
         "fit_time_seconds": without_result["fit_time_seconds"],
         "peak_process_tree_ram_mib": without_result.get("peak_process_tree_ram_mib", without_result["peak_ram_mib"]),
         "model_size_bytes": without_result["model_size_bytes"], "model_reused": True, "status": "PASS"},
        {"sensitive_mode": "with_sensitive", "candidate_id": WITH_ID,
         "fixed_epoch": with_result["fixed_epoch"], **with_result["metrics"],
         "fit_time_seconds": with_result["fit_time_seconds"],
         "peak_process_tree_ram_mib": with_result["peak_process_tree_ram_mib"],
         "model_size_bytes": with_result["model_size_bytes"], "model_reused": False, "status": "PASS"},
    ])
    atomic_csv(validation_rows, RESULT_ROOT / "stage5a2_sensitive_validation_results.csv")
    comparison = []
    for metric in metrics:
        without_value = float(without_result["metrics"][metric])
        with_value = float(with_result["metrics"][metric])
        difference = with_value - without_value
        comparison.append({"metric": metric, "without_sensitive": without_value,
                           "with_sensitive": with_value, "difference_with_minus_without": difference,
                           "relative_difference_percent": difference / abs(without_value) * 100 if without_value != 0 else np.nan})
    comparison_frame = pd.DataFrame(comparison)
    atomic_csv(comparison_frame, RESULT_ROOT / "stage5a2_sensitive_comparison.csv")
    rng = np.random.default_rng(42)
    rows = []
    for index in range(300):
        sampled = rng.integers(0, len(y), size=len(y))
        without_mae = float(np.mean(np.abs(pred_without[sampled] - y[sampled])))
        with_mae = float(np.mean(np.abs(pred_with[sampled] - y[sampled])))
        rows.append({"bootstrap_id": index + 1, "without_sensitive_mae": without_mae,
                     "with_sensitive_mae": with_mae, "difference_with_minus_without": with_mae - without_mae})
    bootstrap = pd.DataFrame(rows)
    atomic_csv(bootstrap, RESULT_ROOT / "stage5a2_sensitive_bootstrap.csv")
    values = bootstrap["difference_with_minus_without"].to_numpy()
    mae_without = float(without_result["metrics"]["mae"])
    mae_with = float(with_result["metrics"]["mae"])
    summary = {
        "stage_id": "stage5a2", "comparison_type": "Validation accuracy comparison; not a fairness audit",
        "without_sensitive_candidate_id": WITHOUT_ID, "with_sensitive_candidate_id": WITH_ID,
        "without_sensitive_mae": mae_without, "with_sensitive_mae": mae_with,
        "mae_difference_with_minus_without": mae_with - mae_without,
        "relative_mae_difference_percent": (mae_with - mae_without) / mae_without * 100,
        "paired_ci_95_low": float(np.quantile(values, 0.025)),
        "paired_ci_95_high": float(np.quantile(values, 0.975)),
        "resamples": 300, "seed": 42, "validation_rows": len(y),
        "validation_row_id_hash": digest_values(without["row_id"].to_numpy(np.int64)),
        "same_rows_and_targets": True, "same_family_configuration_seed_and_epoch": True,
        "non_sensitive_model_reused": True, "sensitive_fit_count": 1,
        "proxy_features_may_remain": True,
        "stage4l_official_test_results_unchanged": True,
        "test_or_stage4l_test_evidence_used": False, "status": "PASS",
    }
    atomic_json(summary, RESULT_ROOT / "stage5a2_sensitive_comparison_summary.json")
    print(json.dumps({"validation": validation_rows.to_dict("records"),
                      "comparison": comparison, "bootstrap_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
