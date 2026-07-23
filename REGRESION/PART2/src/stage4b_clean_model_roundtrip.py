"""Validate Stage 4B fitted smoke Pipelines in separate clean processes."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import stage4_boosting_utils as s4
import stage4b_feature_builder as builder


FACTORIES = {
    "catboost": builder._catboost_pipeline,
    "lightgbm": builder._lightgbm_pipeline,
    "xgboost": builder._xgboost_pipeline,
}


def _feature_names(pipeline: Any, model_name: str) -> list[str]:
    if model_name == "catboost":
        return list(pipeline.named_steps["select"].get_feature_names_out())
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def fit_worker(root: Path, model_name: str, directory: Path) -> None:
    root = root.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    X_fit, y_fit, X_validation, _ = builder._raw_smoke_data(root)
    pipeline = FACTORIES[model_name]()
    start = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    prediction = np.asarray(pipeline.predict(X_validation), dtype=float)
    fit_seconds = time.perf_counter() - start
    if not np.isfinite(prediction).all():
        raise AssertionError("Fit worker produced non-finite predictions.")
    joblib.dump(pipeline, directory / "pipeline.joblib", compress=3)
    joblib.dump(X_validation, directory / "validation.joblib", compress=3)
    np.save(directory / "reference.npy", prediction)
    s4.atomic_write_json(directory / "fit.json", {
        "model": model_name,
        "fit_rows": len(X_fit),
        "validation_rows": len(X_validation),
        "fit_seconds": fit_seconds,
        "feature_names": _feature_names(pipeline, model_name),
        "status": "PASS",
    })


def reload_worker(root: Path, model_name: str, directory: Path) -> None:
    root = root.resolve()
    s4.activate_local_packages(root)
    pipeline = joblib.load(directory / "pipeline.joblib")
    validation = joblib.load(directory / "validation.joblib")
    reference = np.load(directory / "reference.npy")
    prediction = np.asarray(pipeline.predict(validation), dtype=float)
    fit_evidence = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    names = _feature_names(pipeline, model_name)
    custom_classes = sorted({
        type(step).__name__ for step in pipeline.named_steps.values()
        if type(step).__module__ == "stage4_boosting_utils"
    })
    checks = {
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "prediction_count_matches": len(prediction) == len(validation) == len(reference),
        "prediction_values_match": bool(np.allclose(prediction, reference, rtol=0, atol=1e-10)),
        "feature_names_stable": names == fit_evidence["feature_names"],
        "custom_transformer_imported": bool(custom_classes),
    }
    s4.atomic_write_json(directory / "reload.json", {
        "model": model_name,
        "clean_process": True,
        "custom_transformer_classes": custom_classes,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    })
    if not all(checks.values()):
        raise AssertionError(f"Clean reload checks failed: {checks}")


def run(root: str | Path = ".") -> dict[str, Any]:
    project = Path(root).resolve()
    checkpoint_root = project / "artifacts/checkpoints/stage4"
    reports = project / "artifacts/reports"
    features = project / "artifacts/features/stage4"
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stage4b_clean_models_", dir=checkpoint_root) as temporary:
        temporary_root = Path(temporary)
        for model_name in FACTORIES:
            model_dir = temporary_root / model_name
            fit_result = s4.run_worker_process(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--fit-worker", model_name, str(model_dir)],
                timeout_seconds=120,
                cwd=project,
            )
            if fit_result["status"] != "success":
                raise RuntimeError(f"{model_name} bounded fit worker failed: {fit_result}")
            reload_result = s4.run_worker_process(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--reload-worker", model_name, str(model_dir)],
                timeout_seconds=120,
                cwd=project,
            )
            if reload_result["status"] != "success":
                raise RuntimeError(f"{model_name} clean reload worker failed: {reload_result}")
            fit_evidence = json.loads((model_dir / "fit.json").read_text(encoding="utf-8"))
            reload_evidence = json.loads((model_dir / "reload.json").read_text(encoding="utf-8"))
            rows.append({
                "model": model_name,
                "fit_rows": fit_evidence["fit_rows"],
                "validation_rows": fit_evidence["validation_rows"],
                "fit_seconds": fit_evidence["fit_seconds"],
                "fit_parent_wall_seconds": fit_result["wall_seconds"],
                "reload_parent_wall_seconds": reload_result["wall_seconds"],
                "fit_parent_timeout_seconds": 120,
                "reload_parent_timeout_seconds": 120,
                "clean_process_reload_predict": reload_evidence["status"] == "PASS",
                "finite_predictions": reload_evidence["checks"]["finite_predictions"],
                "prediction_count_matches": reload_evidence["checks"]["prediction_count_matches"],
                "prediction_values_match": reload_evidence["checks"]["prediction_values_match"],
                "feature_names_stable": reload_evidence["checks"]["feature_names_stable"],
                "custom_transformer_imported": reload_evidence["checks"]["custom_transformer_imported"],
                "custom_transformer_classes": "|".join(reload_evidence["custom_transformer_classes"]),
                "test_rows": 0,
                "screening_performed": False,
                "status": reload_evidence["status"],
            })
    frame = pd.DataFrame(rows)
    s4.atomic_write_csv(frame, features / "clean_model_roundtrip_results.csv")
    checks = {
        "three_available_models_checked": len(frame) == 3,
        "all_clean_process_reload_predict": frame["clean_process_reload_predict"].all(),
        "all_predictions_finite_and_equal": (
            frame["finite_predictions"].all()
            and frame["prediction_count_matches"].all()
            and frame["prediction_values_match"].all()
        ),
        "all_feature_names_stable": frame["feature_names_stable"].all(),
        "all_custom_transformers_imported": frame["custom_transformer_imported"].all(),
        "all_fit_and_reload_workers_bounded": (
            frame["fit_parent_wall_seconds"].le(120).all()
            and frame["reload_parent_wall_seconds"].le(120).all()
        ),
        "maximum_5000_training_only_rows": (frame["fit_rows"] + frame["validation_rows"]).le(5000).all(),
        "test_rows_zero": frame["test_rows"].eq(0).all(),
        "no_screening": (~frame["screening_performed"]).all(),
    }
    result = {
        "stage": s4.STAGE4B_ID,
        "version": builder.VERSION,
        "created_at_utc": s4.utc_now(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(reports / "stage4b_clean_model_roundtrip.json", result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-worker", nargs=2, metavar=("MODEL", "DIRECTORY"))
    parser.add_argument("--reload-worker", nargs=2, metavar=("MODEL", "DIRECTORY"))
    arguments = parser.parse_args()
    if arguments.fit_worker:
        fit_worker(Path.cwd(), arguments.fit_worker[0], Path(arguments.fit_worker[1]))
        return 0
    if arguments.reload_worker:
        reload_worker(Path.cwd(), arguments.reload_worker[0], Path(arguments.reload_worker[1]))
        return 0
    result = run()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
