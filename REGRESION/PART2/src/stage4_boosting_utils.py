"""Shared safety and execution tools for Stage 4A boosting work.

Stage 4A builds infrastructure only. This module does not fit a boosting model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


STAGE_NAME = "Stage 4A — Boosting Infrastructure and Experiment Foundation"
STAGE_ID = "stage4a"
STAGE4A_RECOVERY_ID = "stage4a-recovery-20260714"
RANDOM_SEED = 42
TARGET_COLUMN = "loan_amount_000s"
STAGE4B_ID = "stage4b"
STAGE4B_VERSION = "stage4b_initial_boosting_packs_v1_20260714"

STAGE4B_FIXED_FEATURES = (
    "estimated_tract_family_income_000s",
    "applicant_vs_area_income_gap_000s",
    "purpose_lien_status_group",
    "occupancy_lien_status_group",
    "loan_type_lien_status_group",
    "state_lien_status_group",
    "property_purpose_group",
    "agency_lien_status_group",
)

SAMPLE_SPECS = {
    "discovery": {"train": 50_000, "validation": 15_000},
    "feature_confirmation": {"train": 80_000, "validation": 20_000},
    "final_selection": {"train": 100_000, "validation": 25_000},
}

REGISTRY_COLUMNS = [
    "experiment_id", "timestamp_utc", "model_family", "model_name", "sensitive_mode",
    "feature_set", "target_mode", "evaluation_stage", "fold_number",
    "training_row_count", "validation_row_count", "test_row_count", "parameter_json",
    "mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle",
    "rmsle_clipped_zero", "median_absolute_error", "wape_percent",
    "mean_signed_error", "p90_absolute_error", "negative_prediction_rate",
    "fit_time_seconds", "prediction_time_seconds", "status", "notes",
    "model_artifact_path", "prediction_artifact_path",
]


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by its permanent state and split files."""
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "artifacts/splits/train_row_ids.csv"
        ).is_file():
            return candidate
    raise FileNotFoundError("Could not find the project root from the current path.")


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for IDs and digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def configuration_digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def stage4a_implementation_digest(
    root: str | Path,
    notebook_path: str | Path | None = None,
) -> str:
    """Hash notebook sources and the shared helper without notebook outputs."""
    import nbformat

    project = Path(root).resolve()
    notebook_file = Path(notebook_path or project / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb").resolve()
    notebook = nbformat.read(notebook_file, as_version=4)
    payload = {
        "notebook_cells": [
            {"cell_type": cell.cell_type, "source": cell.source}
            for cell in notebook.cells
        ],
        "utility_sha256": sha256_file(project / "stage4_boosting_utils.py"),
    }
    return configuration_digest(payload, length=64)


def deterministic_experiment_id(
    model_name: str,
    sensitive_mode: str,
    target_mode: str,
    evaluation_stage: str,
    fold_number: int | None,
    configuration: dict[str, Any],
    feature_pack: str,
    provenance: dict[str, Any] | None = None,
    stage_id: str = STAGE_ID,
) -> str:
    """Build a deterministic later-Stage experiment ID with provenance."""
    if not re.fullmatch(r"stage[0-9]+[a-z0-9]*", stage_id):
        raise ValueError(f"Invalid Stage identifier: {stage_id}")
    provenance = dict(provenance or {})
    required_provenance = {"data_digest", "split_digest", "feature_digest", "package_digest"}
    if not required_provenance.issubset(provenance):
        missing = sorted(required_provenance.difference(provenance))
        raise ValueError(f"Experiment provenance is missing: {missing}")
    fold_text = "na" if fold_number is None else str(int(fold_number))
    digest = configuration_digest({
        "stage_id": stage_id,
        "feature_pack": feature_pack,
        "configuration": configuration,
        "provenance": provenance,
    })
    return (
        f"{stage_id}__{model_name}__{sensitive_mode}__{target_mode}__"
        f"{evaluation_stage}__fold-{fold_text}__cfg-{digest}"
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, destination)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def atomic_write_joblib(value: Any, path: str | Path, compress: int = 3) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    joblib.dump(value, temporary, compress=compress)
    os.replace(temporary, destination)


def validate_cache_file(
    path: str | Path,
    expected_sha256: str | None = None,
    minimum_bytes: int = 1,
) -> tuple[bool, str]:
    """Validate file presence, size, and an optional content digest."""
    candidate = Path(path)
    if not candidate.is_file():
        return False, "missing"
    if candidate.stat().st_size < minimum_bytes:
        return False, "too_small"
    if expected_sha256 is not None and sha256_file(candidate) != expected_sha256:
        return False, "sha256_mismatch"
    return True, "valid"


def validate_cache_artifact(
    path: str | Path,
    *,
    artifact_type: str,
    expected_sha256: str,
    expected_metadata: dict[str, Any],
    required_columns: Iterable[str] | None = None,
    expected_rows: int | None = None,
    unique_column: str | None = None,
    finite_columns: Iterable[str] | None = None,
) -> tuple[bool, str, Any | None]:
    """Validate content, schema, completion state, and provenance before cache reuse."""
    valid, reason = validate_cache_file(path, expected_sha256=expected_sha256, minimum_bytes=2)
    if not valid:
        return False, reason, None
    candidate = Path(path)
    try:
        if artifact_type == "json":
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if value.get("status") not in {"complete", "success", "PASS"}:
                return False, "incomplete_status", value
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                return False, "missing_metadata", value
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    return False, f"metadata_mismatch:{key}", value
            return True, "valid", value
        if artifact_type == "csv":
            value = pd.read_csv(candidate)
            columns = set(required_columns or ())
            if not columns.issubset(value.columns):
                return False, "missing_required_columns", value
            for key, expected in expected_metadata.items():
                if key not in value.columns:
                    return False, f"missing_metadata_column:{key}", value
                observed = set(value[key].astype(str))
                if observed != {str(expected)}:
                    return False, f"metadata_mismatch:{key}", value
            if expected_rows is not None and len(value) != int(expected_rows):
                return False, "row_count_mismatch", value
            if unique_column is not None and (
                unique_column not in value.columns or not value[unique_column].is_unique
            ):
                return False, "unique_key_failure", value
            numeric = list(finite_columns or ())
            if numeric and not np.isfinite(value[numeric].to_numpy(dtype=float)).all():
                return False, "non_finite_values", value
            return True, "valid", value
        if artifact_type == "joblib":
            value = joblib.load(candidate)
            if not isinstance(value, dict) or value.get("status") not in {"complete", "success", "PASS"}:
                return False, "invalid_joblib_contract", value
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                return False, "missing_metadata", value
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    return False, f"metadata_mismatch:{key}", value
            return True, "valid", value
        return False, "unknown_artifact_type", None
    except Exception as exc:
        return False, f"read_failure:{type(exc).__name__}", None


def _require_stage4_frame(value: Any, name: str = "X") -> pd.DataFrame:
    """Return a DataFrame or raise a clear transformer contract error."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return value


def _stage4_category_text(series: pd.Series, missing_token: str) -> pd.Series:
    """Create stable category text without changing the input Series."""
    return series.astype("string").fillna(missing_token).astype(object)


class Stage4FixedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add frozen target-independent Stage 4B row-level features."""

    _feature_sources = {
        "estimated_tract_family_income_000s": (
            "hud_median_family_income", "tract_income_ratio"
        ),
        "applicant_vs_area_income_gap_000s": (
            "applicant_income_000s", "hud_median_family_income"
        ),
        "purpose_lien_status_group": ("loan_purpose_name", "lien_status_name"),
        "occupancy_lien_status_group": ("owner_occupancy_name", "lien_status_name"),
        "loan_type_lien_status_group": ("loan_type_name", "lien_status_name"),
        "state_lien_status_group": ("state_name", "lien_status_name"),
        "property_purpose_group": ("property_type_name", "loan_purpose_name"),
        "agency_lien_status_group": ("agency_name", "lien_status_name"),
    }

    def __init__(
        self,
        selected_features: Sequence[str] = STAGE4B_FIXED_FEATURES,
        missing_token: str = "<MISSING>",
    ) -> None:
        self.selected_features = selected_features
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Stage4FixedFeatureEngineer":
        frame = _require_stage4_frame(X)
        selected = tuple(self.selected_features)
        unknown = sorted(set(selected).difference(self._feature_sources))
        if unknown:
            raise ValueError(f"Unknown fixed Stage 4B features: {unknown}")
        required = sorted({source for name in selected for source in self._feature_sources[name]})
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Fixed feature sources are missing: {missing}")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.selected_features_ = selected
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_stage4_frame(X)
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Stage4FixedFeatureEngineer is not fitted.")
        missing = sorted(set(self.feature_names_in_).difference(frame.columns))
        if missing:
            raise ValueError(f"Transform input is missing fitted columns: {missing}")
        result = frame.copy()
        selected = set(self.selected_features_)
        if "estimated_tract_family_income_000s" in selected:
            area = pd.to_numeric(result["hud_median_family_income"], errors="coerce") / 1000.0
            ratio = pd.to_numeric(result["tract_income_ratio"], errors="coerce")
            values = area * ratio
            result["estimated_tract_family_income_000s"] = values.where(np.isfinite(values))
        if "applicant_vs_area_income_gap_000s" in selected:
            applicant = pd.to_numeric(result["applicant_income_000s"], errors="coerce")
            area = pd.to_numeric(result["hud_median_family_income"], errors="coerce") / 1000.0
            values = applicant - area
            result["applicant_vs_area_income_gap_000s"] = values.where(np.isfinite(values))
        combinations = {
            "purpose_lien_status_group": ("loan_purpose_name", "lien_status_name"),
            "occupancy_lien_status_group": ("owner_occupancy_name", "lien_status_name"),
            "loan_type_lien_status_group": ("loan_type_name", "lien_status_name"),
            "state_lien_status_group": ("state_name", "lien_status_name"),
            "property_purpose_group": ("property_type_name", "loan_purpose_name"),
            "agency_lien_status_group": ("agency_name", "lien_status_name"),
        }
        for output, (left, right) in combinations.items():
            if output in selected:
                left_text = _stage4_category_text(result[left], self.missing_token)
                right_text = _stage4_category_text(result[right], self.missing_token)
                result[output] = left_text + " | " + right_text
        if not result.index.equals(frame.index):
            raise AssertionError("Fixed feature engineering changed row order.")
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(base + [name for name in self.selected_features_ if name not in base], dtype=object)


class Stage4CategoricalSanitizer(BaseEstimator, TransformerMixin):
    """Convert selected category columns to stable strings with a missing token."""

    def __init__(self, columns: Sequence[str], missing_token: str = "<MISSING>") -> None:
        self.columns = columns
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Stage4CategoricalSanitizer":
        frame = _require_stage4_frame(X)
        columns = tuple(self.columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Categorical columns are missing: {missing}")
        self.columns_ = columns
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_stage4_frame(X)
        if not hasattr(self, "columns_"):
            raise RuntimeError("Stage4CategoricalSanitizer is not fitted.")
        result = frame.copy()
        for column in self.columns_:
            if column not in result.columns:
                raise ValueError(f"Transform input is missing category column: {column}")
            result[column] = _stage4_category_text(result[column], self.missing_token)
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)


class Stage4RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Learn frequent category values on fit rows and group all other values."""

    def __init__(
        self,
        columns: Sequence[str],
        min_count: int = 2,
        rare_token: str = "<RARE>",
        missing_token: str = "<MISSING>",
    ) -> None:
        self.columns = columns
        self.min_count = min_count
        self.rare_token = rare_token
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Stage4RareCategoryGrouper":
        frame = _require_stage4_frame(X)
        if int(self.min_count) < 1:
            raise ValueError("min_count must be at least 1.")
        columns = tuple(self.columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Rare-category columns are missing: {missing}")
        self.columns_ = columns
        self.kept_categories_ = {}
        for column in columns:
            text = _stage4_category_text(frame[column], self.missing_token)
            counts = text.value_counts(dropna=False)
            self.kept_categories_[column] = frozenset(counts[counts >= int(self.min_count)].index.tolist())
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_stage4_frame(X)
        if not hasattr(self, "kept_categories_"):
            raise RuntimeError("Stage4RareCategoryGrouper is not fitted.")
        result = frame.copy()
        for column in self.columns_:
            if column not in result.columns:
                raise ValueError(f"Transform input is missing rare-category column: {column}")
            text = _stage4_category_text(result[column], self.missing_token)
            result[column] = text.where(text.isin(self.kept_categories_[column]), self.rare_token)
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)


class Stage4FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Learn category frequencies on fit rows and add stable numeric columns."""

    def __init__(
        self,
        columns: Sequence[str],
        suffix: str = "__frequency",
        unseen_value: float = 0.0,
        drop_original: bool = False,
        missing_token: str = "<MISSING>",
    ) -> None:
        self.columns = columns
        self.suffix = suffix
        self.unseen_value = unseen_value
        self.drop_original = drop_original
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Stage4FrequencyEncoder":
        frame = _require_stage4_frame(X)
        columns = tuple(self.columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Frequency columns are missing: {missing}")
        self.columns_ = columns
        row_count = max(len(frame), 1)
        self.frequency_maps_ = {
            column: (_stage4_category_text(frame[column], self.missing_token).value_counts(dropna=False) / row_count).to_dict()
            for column in columns
        }
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_stage4_frame(X)
        if not hasattr(self, "frequency_maps_"):
            raise RuntimeError("Stage4FrequencyEncoder is not fitted.")
        result = frame.copy()
        for column in self.columns_:
            if column not in result.columns:
                raise ValueError(f"Transform input is missing frequency column: {column}")
            text = _stage4_category_text(result[column], self.missing_token)
            result[f"{column}{self.suffix}"] = (
                text.map(self.frequency_maps_[column]).fillna(float(self.unseen_value)).astype(float)
            )
        if bool(self.drop_original):
            result = result.drop(columns=list(self.columns_))
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        added = [f"{column}{self.suffix}" for column in self.columns_]
        if bool(self.drop_original):
            base = [name for name in base if name not in self.columns_]
        return np.asarray(base + [name for name in added if name not in base], dtype=object)


class Stage4ColumnSelector(BaseEstimator, TransformerMixin):
    """Select named DataFrame columns in a fixed order without mutation."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Stage4ColumnSelector":
        frame = _require_stage4_frame(X)
        columns = tuple(self.columns)
        if len(columns) != len(set(columns)):
            raise ValueError("Selected columns must be unique.")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Selected columns are missing: {missing}")
        self.columns_ = columns
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_stage4_frame(X)
        if not hasattr(self, "columns_"):
            raise RuntimeError("Stage4ColumnSelector is not fitted.")
        missing = [column for column in self.columns_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Transform input is missing selected columns: {missing}")
        return frame.loc[:, list(self.columns_)].copy()

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


def transform_target(y: Iterable[float], target_mode: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Target values must be finite.")
    if target_mode == "raw":
        return values.copy()
    if target_mode == "log1p":
        if (values < 0).any():
            raise ValueError("log1p target mode requires non-negative targets.")
        return np.log1p(values)
    raise ValueError(f"Unknown target mode: {target_mode}")


def inverse_target(values: Iterable[float], target_mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if target_mode == "raw":
        return array.copy()
    if target_mode == "log1p":
        return np.expm1(array)
    raise ValueError(f"Unknown target mode: {target_mode}")


def evaluate_regression_predictions(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, Any]:
    """Calculate the shared metrics on the original target scale."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred) or not len(true):
        raise ValueError("Targets and predictions must be non-empty aligned vectors.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Targets and predictions must be finite.")
    error = pred - true
    absolute = np.abs(error)
    squared = error**2
    mae = float(absolute.mean())
    mse = float(squared.mean())
    rmse = float(np.sqrt(mse))
    nonzero = true != 0
    mape = float(np.mean(absolute[nonzero] / np.abs(true[nonzero])) * 100) if nonzero.all() else None
    denominator = float(np.sum((true - true.mean()) ** 2))
    r_squared = float(1 - squared.sum() / denominator) if len(true) >= 2 and denominator > 0 else None
    rmsle = None
    if (true >= 0).all() and (pred >= 0).all():
        rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    clipped = None
    if (true >= 0).all():
        clipped = float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(true)) ** 2)))
    true_sum = float(np.abs(true).sum())
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape_percent": mape,
        "r_squared": r_squared,
        "rmsle": rmsle,
        "rmsle_clipped_zero": clipped,
        "median_absolute_error": float(np.median(absolute)),
        "wape_percent": float(absolute.sum() / true_sum * 100) if true_sum > 0 else None,
        "mean_signed_error": float(error.mean()),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "negative_prediction_rate": float(np.mean(pred < 0)),
        "mae_usd": mae * 1000,
        "rmse_usd": rmse * 1000,
        "original_scale": True,
    }


def evaluate_model_output(
    y_true_original: Iterable[float],
    model_output: Iterable[float],
    target_mode: str,
) -> dict[str, Any]:
    """Invert a transformed target prediction before metric calculation."""
    return evaluate_regression_predictions(y_true_original, inverse_target(model_output, target_mode))


def upsert_registry(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    allowed_stage_ids: Iterable[str] = (STAGE_ID,),
) -> pd.DataFrame:
    """Upsert allowed later-Stage rows while keeping the fixed Registry schema."""
    if list(existing.columns) != REGISTRY_COLUMNS:
        raise ValueError("The existing Registry schema does not match the fixed Stage 1 schema.")
    if "experiment_id" not in new_rows.columns:
        raise ValueError("New Registry rows require experiment_id.")
    prefixes = tuple(f"{stage_id}__" for stage_id in allowed_stage_ids)
    invalid_ids = [value for value in new_rows["experiment_id"].astype(str) if not value.startswith(prefixes)]
    if invalid_ids:
        raise ValueError(f"New experiment IDs must use one of these Stage prefixes: {prefixes}")
    aligned = new_rows.copy()
    extra = [column for column in aligned.columns if column not in existing.columns]
    if extra:
        raise ValueError(f"New Registry rows contain unknown columns: {extra}")
    for column in existing.columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    aligned = aligned[existing.columns]
    prior = existing.loc[~existing["experiment_id"].isin(aligned["experiment_id"])].copy()
    combined = pd.concat([prior, aligned], ignore_index=True)
    if combined["experiment_id"].duplicated().any():
        raise AssertionError("Registry experiment IDs are not unique after upsert.")
    return combined


def validate_row_ids(
    row_ids: Iterable[int],
    train_ids: Iterable[int],
    test_ids: Iterable[int],
    require_unique: bool = True,
) -> dict[str, Any]:
    ids = np.asarray(list(row_ids), dtype=np.int64)
    train_set = set(int(value) for value in train_ids)
    test_set = set(int(value) for value in test_ids)
    actual = set(int(value) for value in ids)
    result = {
        "rows": int(len(ids)),
        "unique_rows": int(len(actual)),
        "all_in_saved_train": actual.issubset(train_set),
        "test_overlap_rows": int(len(actual.intersection(test_set))),
    }
    result["valid"] = bool(
        result["all_in_saved_train"]
        and result["test_overlap_rows"] == 0
        and (not require_unique or result["rows"] == result["unique_rows"])
    )
    return result


def read_training_rows(
    csv_path: str | Path,
    train_row_ids: Iterable[int],
    usecols: Iterable[str],
) -> pd.DataFrame:
    """Read only saved Train rows and never materialize locked Test targets."""
    ids = np.sort(np.asarray(list(train_row_ids), dtype=np.int64))
    id_set = set(int(value) for value in ids)
    frame = pd.read_csv(
        csv_path,
        usecols=list(usecols),
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in id_set,
        low_memory=False,
    )
    if len(frame) != len(ids):
        raise AssertionError(f"Training-only load returned {len(frame)} rows, expected {len(ids)}.")
    frame.index = pd.Index(ids, name="row_id")
    return frame


def assign_target_bins(target: pd.Series, bin_edges: Sequence[float]) -> pd.Series:
    bins = pd.cut(
        pd.to_numeric(target, errors="raise"),
        bins=np.asarray(bin_edges, dtype=float),
        labels=False,
        include_lowest=True,
        right=True,
    )
    if bins.isna().any():
        raise AssertionError("Some training targets fall outside the saved target-bin edges.")
    return bins.astype("int64")


def _largest_remainder_counts(bin_counts: pd.Series, total: int) -> dict[int, int]:
    proportions = bin_counts / int(bin_counts.sum())
    raw = proportions * int(total)
    counts = np.floor(raw).astype(int)
    remaining = int(total - counts.sum())
    order = (raw - counts).sort_values(ascending=False, kind="mergesort").index.tolist()
    for bin_id in order[:remaining]:
        counts.loc[bin_id] += 1
    return {int(bin_id): int(value) for bin_id, value in counts.items()}


def create_stage4_samples(root: str | Path) -> dict[str, Any]:
    """Create three deterministic, disjoint, training-only sample manifests."""
    project = Path(root).resolve()
    split_dir = project / "artifacts/splits"
    output_dir = split_dir / "stage4"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = pd.read_csv(split_dir / "train_row_ids.csv", dtype={"row_id": "int64"})["row_id"]
    test_ids = pd.read_csv(split_dir / "test_row_ids.csv", dtype={"row_id": "int64"})["row_id"]
    split_config = json.loads((split_dir / "split_config.json").read_text(encoding="utf-8"))
    targets = read_training_rows(
        project / "data/regression_without_sensitive_features.csv",
        train_ids,
        [TARGET_COLUMN],
    )[TARGET_COLUMN]
    target_bins = assign_target_bins(targets, split_config["bin_edges"])

    # One seeded shuffle per target bin gives deterministic disjoint allocations.
    rng = np.random.default_rng(RANDOM_SEED)
    available: dict[int, list[int]] = {}
    for bin_id in sorted(target_bins.unique()):
        ids = target_bins.index[target_bins.eq(bin_id)].to_numpy(dtype=np.int64)
        rng.shuffle(ids)
        available[int(bin_id)] = ids.tolist()
    cursors = {bin_id: 0 for bin_id in available}
    full_bin_counts = target_bins.value_counts().sort_index()
    created: dict[str, pd.DataFrame] = {}

    for sample_name, roles in SAMPLE_SPECS.items():
        parts: list[pd.DataFrame] = []
        for role_name, role_size in roles.items():
            allocation = _largest_remainder_counts(full_bin_counts, role_size)
            role_parts: list[pd.DataFrame] = []
            for bin_id in sorted(allocation):
                count = allocation[bin_id]
                start = cursors[bin_id]
                stop = start + count
                selected = available[bin_id][start:stop]
                if len(selected) != count:
                    raise RuntimeError("The requested disjoint stratified samples do not fit the Train set.")
                cursors[bin_id] = stop
                role_parts.append(pd.DataFrame({
                    "row_id": np.asarray(selected, dtype=np.int64),
                    "sample_role": role_name,
                    "target_bin": int(bin_id),
                }))
            role_frame = pd.concat(role_parts, ignore_index=True)
            role_frame = role_frame.sort_values("row_id", kind="mergesort").reset_index(drop=True)
            parts.append(role_frame)
        sample = pd.concat(parts, ignore_index=True)
        filename = {
            "discovery": "stage4_discovery_sample.csv",
            "feature_confirmation": "stage4_feature_confirmation_sample.csv",
            "final_selection": "stage4_final_selection_sample.csv",
        }[sample_name]
        atomic_write_csv(sample, output_dir / filename)
        created[sample_name] = sample

    all_rows = pd.concat(
        [frame.assign(sample_name=name) for name, frame in created.items()],
        ignore_index=True,
    )
    row_validation = validate_row_ids(all_rows["row_id"], train_ids, test_ids)
    train_distribution = (full_bin_counts / full_bin_counts.sum()).sort_index()
    sample_checks: dict[str, Any] = {}
    for name, frame in created.items():
        sample_distribution = frame["target_bin"].value_counts(normalize=True).sort_index()
        difference = sample_distribution.reindex(train_distribution.index, fill_value=0) - train_distribution
        expected = SAMPLE_SPECS[name]
        role_differences = {}
        for role_name in ("train", "validation"):
            role_distribution = frame.loc[frame["sample_role"].eq(role_name), "target_bin"].value_counts(normalize=True).sort_index()
            role_difference = role_distribution.reindex(train_distribution.index, fill_value=0) - train_distribution
            role_differences[role_name] = float(role_difference.abs().max())
        sample_checks[name] = {
            "rows": int(len(frame)),
            "train_rows": int(frame["sample_role"].eq("train").sum()),
            "validation_rows": int(frame["sample_role"].eq("validation").sum()),
            "expected_rows": int(sum(expected.values())),
            "row_ids_unique": bool(frame["row_id"].is_unique),
            "maximum_target_bin_proportion_difference": float(difference.abs().max()),
            "role_maximum_target_bin_proportion_difference": role_differences,
            "sha256": sha256_file(output_dir / {
                "discovery": "stage4_discovery_sample.csv",
                "feature_confirmation": "stage4_feature_confirmation_sample.csv",
                "final_selection": "stage4_final_selection_sample.csv",
            }[name]),
        }
        sample_checks[name]["valid"] = bool(
            sample_checks[name]["rows"] == sample_checks[name]["expected_rows"]
            and sample_checks[name]["train_rows"] == expected["train"]
            and sample_checks[name]["validation_rows"] == expected["validation"]
            and sample_checks[name]["row_ids_unique"]
            and sample_checks[name]["maximum_target_bin_proportion_difference"] <= 0.001
            and max(role_differences.values()) <= 0.001
        )
    verification = {
        "stage": STAGE_ID,
        "random_seed": RANDOM_SEED,
        "sampling_method": "one seeded per-bin shuffle with largest-remainder proportional allocation",
        "target_bin_source": "saved split_config.json bin_edges",
        "source_train_rows": int(len(train_ids)),
        "locked_test_rows": int(len(test_ids)),
        "total_sample_rows": int(len(all_rows)),
        "all_sample_rows_unique": bool(all_rows["row_id"].is_unique),
        "all_rows_in_saved_train": row_validation["all_in_saved_train"],
        "test_overlap_rows": row_validation["test_overlap_rows"],
        "samples": sample_checks,
    }
    verification["status"] = "PASS" if (
        verification["all_sample_rows_unique"]
        and verification["all_rows_in_saved_train"]
        and verification["test_overlap_rows"] == 0
        and all(item["valid"] for item in sample_checks.values())
    ) else "FAIL"
    atomic_write_json(output_dir / "stage4_sample_verification.json", verification)
    return verification


def validate_existing_stage4_samples(root: str | Path) -> dict[str, Any]:
    """Validate and reuse the three saved Stage 4A sample manifests."""
    project = Path(root).resolve()
    split_dir = project / "artifacts/splits"
    output_dir = split_dir / "stage4"
    verification_path = output_dir / "stage4_sample_verification.json"
    if not verification_path.is_file():
        raise FileNotFoundError("The Stage 4A sample verification report is missing.")
    saved = json.loads(verification_path.read_text(encoding="utf-8"))
    train_ids = set(pd.read_csv(split_dir / "train_row_ids.csv", usecols=["row_id"])["row_id"].astype("int64"))
    test_ids = set(pd.read_csv(split_dir / "test_row_ids.csv", usecols=["row_id"])["row_id"].astype("int64"))
    filenames = {
        "discovery": "stage4_discovery_sample.csv",
        "feature_confirmation": "stage4_feature_confirmation_sample.csv",
        "final_selection": "stage4_final_selection_sample.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    checks: dict[str, Any] = {}
    for name, filename in filenames.items():
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 4A sample: {filename}")
        frame = pd.read_csv(path, dtype={"row_id": "int64"})
        frames[name] = frame
        expected = SAMPLE_SPECS[name]
        expected_hash = saved["samples"][name]["sha256"]
        checks[name] = {
            "rows": int(len(frame)),
            "train_rows": int(frame["sample_role"].eq("train").sum()),
            "validation_rows": int(frame["sample_role"].eq("validation").sum()),
            "row_ids_unique": bool(frame["row_id"].is_unique),
            "hash_matches": sha256_file(path) == expected_hash,
            "all_in_saved_train": bool(set(frame["row_id"]).issubset(train_ids)),
            "test_overlap_rows": int(len(set(frame["row_id"]).intersection(test_ids))),
        }
        checks[name]["valid"] = bool(
            checks[name]["rows"] == sum(expected.values())
            and checks[name]["train_rows"] == expected["train"]
            and checks[name]["validation_rows"] == expected["validation"]
            and checks[name]["row_ids_unique"]
            and checks[name]["hash_matches"]
            and checks[name]["all_in_saved_train"]
            and checks[name]["test_overlap_rows"] == 0
        )
    combined = pd.concat(
        [frame[["row_id"]].assign(sample_name=name) for name, frame in frames.items()],
        ignore_index=True,
    )
    valid = bool(
        saved.get("status") == "PASS"
        and saved.get("total_sample_rows") == len(combined)
        and combined["row_id"].is_unique
        and all(item["valid"] for item in checks.values())
    )
    if not valid:
        raise AssertionError({"saved_status": saved.get("status"), "checks": checks})
    result = dict(saved)
    result["cache_reused"] = True
    result["current_validation"] = checks
    return result


def ensure_stage4_directories(root: str | Path) -> list[str]:
    project = Path(root).resolve()
    relative_paths = [
        "artifacts/results/stage4",
        "artifacts/results/stage4/catboost",
        "artifacts/results/stage4/lightgbm",
        "artifacts/results/stage4/xgboost",
        "artifacts/models/catboost",
        "artifacts/models/lightgbm",
        "artifacts/models/xgboost",
        "artifacts/predictions/catboost",
        "artifacts/predictions/lightgbm",
        "artifacts/predictions/xgboost",
        "artifacts/features/stage4",
        "artifacts/figures/stage4",
        "artifacts/reports",
        "artifacts/manifests/stage4",
        "artifacts/checkpoints/stage4",
        "artifacts/backups",
        "artifacts/splits/stage4",
        "artifacts/environment/stage4_packages",
    ]
    for relative in relative_paths:
        (project / relative).mkdir(parents=True, exist_ok=True)
    return relative_paths


def collect_protected_paths(root: str | Path) -> list[Path]:
    """Collect immutable files that existed before Stage 4A."""
    project = Path(root).resolve()
    paths: set[Path] = set()
    explicit = [
        project / "data/regression_with_sensitive_features.csv",
        project / "data/regression_without_sensitive_features.csv",
        project / "REGRESSION_PART2_MODELING.ipynb",
        project / "REGRESSION_PART3_TREE_MODELS.ipynb",
        Path(r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb"),
    ]
    paths.update(path.resolve() for path in explicit if path.is_file())
    artifacts = project / "artifacts"
    if artifacts.is_dir():
        for path in artifacts.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(project).as_posix().lower()
            if "stage4" in relative or relative.startswith("artifacts/backups/"):
                continue
            paths.add(path.resolve())
    return sorted(paths, key=lambda value: str(value).lower())


def build_protected_manifest(root: str | Path, paths: Iterable[str | Path] | None = None) -> dict[str, Any]:
    project = Path(root).resolve()
    selected = collect_protected_paths(project) if paths is None else [Path(path).resolve() for path in paths]
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in selected:
        key = path.relative_to(project).as_posix() if path.is_relative_to(project) else str(path)
        if not path.is_file():
            hashes[key] = "MISSING"
            sizes[key] = -1
        else:
            hashes[key] = sha256_file(path)
            sizes[key] = int(path.stat().st_size)
    return {"created_at_utc": utc_now(), "file_count": len(hashes), "hashes": hashes, "sizes": sizes}


def recheck_protected_manifest(root: str | Path, before: dict[str, Any]) -> dict[str, Any]:
    project = Path(root).resolve()
    current: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for key, expected in before["hashes"].items():
        path = Path(key) if Path(key).is_absolute() else project / key
        actual = sha256_file(path) if path.is_file() else "MISSING"
        current[key] = actual
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return {
        "created_at_utc": utc_now(),
        "file_count": len(current),
        "hashes": current,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def safe_package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def activate_local_packages(root: str | Path) -> Path:
    """Add the project-local package target to the current Python process."""
    project = Path(root).resolve()
    local_package_dir = project / "artifacts/environment/stage4_packages"
    local_text = str(local_package_dir)
    if local_package_dir.is_dir() and local_text not in sys.path:
        sys.path.insert(0, local_text)
    return local_package_dir


def worker_environment(
    root: str | Path,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that exposes project-local packages to a clean worker."""
    environment = dict(os.environ if base_environment is None else base_environment)
    local_package_dir = activate_local_packages(root)
    existing = environment.get("PYTHONPATH", "")
    entries = [str(local_package_dir)] + ([existing] if existing else [])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["STAGE4_PROJECT_ROOT"] = str(Path(root).resolve())
    environment["STAGE4_LOCAL_PACKAGES"] = str(local_package_dir)
    return environment


def _gpu_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_available": False,
        "nvidia_smi_output": None,
        "cuda_available": False,
        "cuda_version_reported": None,
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            report["nvidia_smi_available"] = result.returncode == 0
            report["nvidia_smi_output"] = result.stdout.strip() or result.stderr.strip()
            if result.returncode == 0:
                full = subprocess.run(
                    [executable], capture_output=True, text=True, timeout=5, check=False
                )
                match = re.search(r"CUDA Version:\s*([0-9.]+)", full.stdout)
                report["cuda_available"] = True
                report["cuda_version_reported"] = match.group(1) if match else None
        except Exception as exc:
            report["nvidia_smi_output"] = f"{type(exc).__name__}: {exc}"
    return report


def _import_and_construct(module_name: str, local_package_dir: Path | None) -> dict[str, Any]:
    if local_package_dir and local_package_dir.is_dir():
        activate_local_packages(local_package_dir.parent.parent.parent)
    result: dict[str, Any] = {"import_ok": False, "construction_ok": None, "error": None}
    try:
        module = importlib.import_module(module_name)
        result["import_ok"] = True
        result["module_version"] = getattr(module, "__version__", "unknown")
        if module_name == "catboost":
            module.CatBoostRegressor(iterations=2, depth=2, random_seed=RANDOM_SEED, verbose=False)
            result["construction_ok"] = True
        elif module_name == "lightgbm":
            module.LGBMRegressor(n_estimators=2, random_state=RANDOM_SEED, verbosity=-1)
            result["construction_ok"] = True
        elif module_name == "xgboost":
            module.XGBRegressor(n_estimators=2, random_state=RANDOM_SEED, verbosity=0)
            result["construction_ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["import_ok"] and module_name != "shap":
            result["construction_ok"] = False
    return result


def environment_report(root: str | Path, installation_attempts: dict[str, Any] | None = None) -> dict[str, Any]:
    project = Path(root).resolve()
    local_package_dir = project / "artifacts/environment/stage4_packages"
    try:
        import psutil

        memory = psutil.virtual_memory()
        ram = {"total_bytes": int(memory.total), "available_bytes": int(memory.available)}
    except Exception as exc:
        ram = {"error": f"{type(exc).__name__}: {exc}"}
    disk = shutil.disk_usage(project)
    packages: dict[str, Any] = {}
    distributions = {
        "pandas": "pandas", "numpy": "numpy", "scikit_learn": "scikit-learn",
        "joblib": "joblib", "catboost": "catboost", "lightgbm": "lightgbm",
        "xgboost": "xgboost", "shap": "shap",
    }
    for key, distribution in distributions.items():
        packages[key] = {"global_version": safe_package_version(distribution)}
    for module_name in ("catboost", "lightgbm", "xgboost", "shap"):
        packages[module_name].update(_import_and_construct(module_name, local_package_dir))
    report = {
        "stage": STAGE_ID,
        "created_at_utc": utc_now(),
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": packages,
        "installation_attempts": installation_attempts or {},
        "resources": {
            "logical_cpu_count": os.cpu_count(),
            "ram": ram,
            "disk": {"total_bytes": int(disk.total), "free_bytes": int(disk.free)},
            "gpu": _gpu_report(),
        },
        "local_package_directory": str(local_package_dir.relative_to(project)),
    }
    report["audit_completed"] = True
    atomic_write_json(project / "artifacts/reports/stage4a_environment.json", report)
    atomic_write_json(project / "artifacts/manifests/stage4/stage4a_package_availability.json", {
        "created_at_utc": report["created_at_utc"],
        "packages": packages,
        "installation_attempts": report["installation_attempts"],
    })
    return report


def write_progress(path: str | Path, step: str, status: str, details: dict[str, Any] | None = None) -> None:
    atomic_write_json(path, {
        "stage": STAGE_ID,
        "updated_at_utc": utc_now(),
        "step": step,
        "status": status,
        "details": details or {},
    })


def write_heartbeat(path: str | Path, worker_id: str, state: str = "running") -> None:
    atomic_write_json(path, {
        "stage": STAGE_ID,
        "worker_id": worker_id,
        "process_id": os.getpid(),
        "state": state,
        "updated_at_utc": utc_now(),
        "monotonic_seconds": time.monotonic(),
    })


def terminate_process_tree(process: subprocess.Popen[Any]) -> dict[str, Any]:
    """Terminate a worker and its children with a Windows-safe fallback."""
    result: dict[str, Any] = {"pid": process.pid, "method": None, "return_code": None}
    if process.poll() is not None:
        result.update({"method": "already_exited", "return_code": process.returncode})
        return result
    try:
        import psutil

        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        result["descendant_pids"] = [child.pid for child in descendants]
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        _, alive = psutil.wait_procs(descendants + [parent], timeout=10)
        for item in alive:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
        result.update({"method": "psutil_recursive_tree", "alive_after_cleanup": [item.pid for item in alive]})
    except Exception as exc:
        result["psutil_error"] = f"{type(exc).__name__}: {exc}"
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            result.update({
                "method": "taskkill_tree_fallback",
                "return_code": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            })
        else:
            process.kill()
            result.update({"method": "process_kill_fallback"})
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        result["fallback_kill"] = True
    result["worker_return_code"] = process.returncode
    return result


def run_worker_process(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: str | Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a worker under a parent-enforced hard wall-clock timeout."""
    start = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
        env=worker_environment(cwd) if environment is None else environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=float(timeout_seconds))
        return {
            "status": "success" if process.returncode == 0 else "failed",
            "timed_out": False,
            "return_code": process.returncode,
            "wall_seconds": time.monotonic() - start,
            "stdout": stdout,
            "stderr": stderr,
            "cleanup": None,
        }
    except subprocess.TimeoutExpired:
        cleanup = terminate_process_tree(process)
        stdout, stderr = process.communicate(timeout=10)
        return {
            "status": "timed_out",
            "timed_out": True,
            "return_code": process.returncode,
            "wall_seconds": time.monotonic() - start,
            "stdout": stdout,
            "stderr": stderr,
            "cleanup": cleanup,
        }


def timeout_smoke_test(root: str | Path, timeout_seconds: float = 1.0) -> dict[str, Any]:
    project = Path(root).resolve()
    heartbeat = project / "artifacts/checkpoints/stage4/timeout_smoke_heartbeat.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--timeout-smoke-worker",
        str(heartbeat),
        "30",
    ]
    result = run_worker_process(command, timeout_seconds=timeout_seconds, cwd=project)
    result["heartbeat_created"] = heartbeat.is_file()
    heartbeat_payload = json.loads(heartbeat.read_text(encoding="utf-8")) if heartbeat.is_file() else {}
    child_pid = heartbeat_payload.get("child_process_id")
    child_alive = None
    if child_pid is not None:
        try:
            import psutil

            child_alive = psutil.pid_exists(int(child_pid))
        except Exception:
            child_alive = None
    result["worker_process_stopped"] = result["return_code"] is not None
    result["child_process_id"] = child_pid
    result["child_process_alive_after_cleanup"] = child_alive
    result["pass"] = bool(
        result["timed_out"]
        and result["heartbeat_created"]
        and result["worker_process_stopped"]
        and child_pid is not None
        and child_alive is False
    )
    atomic_write_json(project / "artifacts/reports/stage4a_timeout_smoke_test.json", result)
    return result


def clean_worker_package_smoke_test(root: str | Path) -> dict[str, Any]:
    """Prove that a clean child can import local boosting packages without fitting."""
    project = Path(root).resolve()
    script = (
        "import json,catboost,lightgbm,xgboost;"
        "print(json.dumps({'catboost':catboost.__version__,'lightgbm':lightgbm.__version__,"
        "'xgboost':xgboost.__version__},sort_keys=True))"
    )
    worker = run_worker_process([sys.executable, "-c", script], timeout_seconds=30, cwd=project)
    versions = None
    if worker["status"] == "success":
        try:
            versions = json.loads(worker["stdout"].strip().splitlines()[-1])
        except Exception:
            versions = None
    passed = bool(worker["status"] == "success" and versions and set(versions) == {"catboost", "lightgbm", "xgboost"})
    result = {
        "status": "PASS" if passed else "FAIL",
        "versions": versions,
        "worker_status": worker["status"],
        "return_code": worker["return_code"],
        "fit_called": False,
        "local_package_path_propagated": str(project / "artifacts/environment/stage4_packages") in worker_environment(project)["PYTHONPATH"],
        "stderr": worker["stderr"],
    }
    atomic_write_json(project / "artifacts/reports/stage4a_clean_worker_package_smoke_test.json", result)
    return result


def atomic_write_smoke_test(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    directory = project / "artifacts/checkpoints/stage4/atomic_smoke"
    csv_path = directory / "sample.csv"
    json_path = directory / "sample.json"
    joblib_path = directory / "sample.joblib"
    frame = pd.DataFrame({"value": [1, 2, 3]})
    payload = {"stage": STAGE_ID, "values": [1, 2, 3]}
    atomic_write_csv(frame, csv_path)
    atomic_write_json(json_path, payload)
    atomic_write_joblib(payload, joblib_path)
    passed = (
        pd.read_csv(csv_path).equals(frame)
        and json.loads(json_path.read_text(encoding="utf-8")) == payload
        and joblib.load(joblib_path) == payload
        and not list(directory.glob("*.tmp"))
    )
    result = {"status": "PASS" if passed else "FAIL", "paths": [str(p.relative_to(project)) for p in (csv_path, json_path, joblib_path)]}
    atomic_write_json(project / "artifacts/reports/stage4a_atomic_write_smoke_test.json", result)
    return result


def cache_validation_smoke_test(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    path = project / "artifacts/checkpoints/stage4/cache_validation_smoke.json"
    metadata = {
        "experiment_id": "stage4a__cache_contract_smoke",
        "configuration_digest": "configuration-v1",
        "data_digest": "data-v1",
        "split_digest": "split-v1",
        "feature_digest": "feature-v1",
        "package_digest": "packages-v1",
    }
    payload = {"stage": STAGE_ID, "status": "complete", "metadata": metadata, "payload": [1, 2, 3]}
    atomic_write_json(path, payload)
    digest = sha256_file(path)
    valid_ok, valid_reason, _ = validate_cache_artifact(
        path, artifact_type="json", expected_sha256=digest, expected_metadata=metadata
    )
    bad_metadata = dict(metadata, split_digest="wrong-split")
    mismatch_ok, mismatch_reason, _ = validate_cache_artifact(
        path, artifact_type="json", expected_sha256=digest, expected_metadata=bad_metadata
    )
    hash_ok, hash_reason, _ = validate_cache_artifact(
        path, artifact_type="json", expected_sha256="0" * 64, expected_metadata=metadata
    )
    csv_path = project / "artifacts/checkpoints/stage4/cache_validation_smoke.csv"
    csv_frame = pd.DataFrame({
        "row_id": [10, 11, 12],
        "prediction": [1.0, 2.0, 3.0],
        **{key: [value] * 3 for key, value in metadata.items()},
    })
    atomic_write_csv(csv_frame, csv_path)
    csv_digest = sha256_file(csv_path)
    csv_ok, csv_reason, _ = validate_cache_artifact(
        csv_path,
        artifact_type="csv",
        expected_sha256=csv_digest,
        expected_metadata=metadata,
        required_columns=["row_id", "prediction"],
        expected_rows=3,
        unique_column="row_id",
        finite_columns=["prediction"],
    )
    csv_bad_ok, csv_bad_reason, _ = validate_cache_artifact(
        csv_path,
        artifact_type="csv",
        expected_sha256=csv_digest,
        expected_metadata=dict(metadata, feature_digest="wrong-feature"),
        required_columns=["row_id", "prediction"],
        expected_rows=3,
        unique_column="row_id",
        finite_columns=["prediction"],
    )
    passed = (
        valid_ok and valid_reason == "valid"
        and not mismatch_ok and mismatch_reason == "metadata_mismatch:split_digest"
        and not hash_ok and hash_reason == "sha256_mismatch"
        and csv_ok and csv_reason == "valid"
        and not csv_bad_ok and csv_bad_reason == "metadata_mismatch:feature_digest"
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "complete_provenance_cache_accepted": valid_ok,
        "wrong_split_provenance_rejected": not mismatch_ok,
        "wrong_file_hash_rejected": not hash_ok,
        "csv_schema_rows_unique_finite_and_provenance_accepted": csv_ok,
        "csv_wrong_feature_provenance_rejected": not csv_bad_ok,
        "valid_reason": valid_reason,
        "wrong_split_reason": mismatch_reason,
        "wrong_hash_reason": hash_reason,
        "csv_valid_reason": csv_reason,
        "csv_wrong_feature_reason": csv_bad_reason,
        "path": str(path.relative_to(project)),
        "csv_path": str(csv_path.relative_to(project)),
    }
    atomic_write_json(project / "artifacts/reports/stage4a_cache_validation_smoke_test.json", result)
    return result


def registry_smoke_test(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    main_path = project / "artifacts/results/experiment_results.csv"
    existing = pd.read_csv(main_path)
    prior_digest = sha256_file(main_path)
    provenance = {
        "data_digest": "not-applicable-foundation",
        "split_digest": "not-applicable-foundation",
        "feature_digest": "not-applicable-foundation",
        "package_digest": "not-applicable-foundation",
    }
    experiment_id = deterministic_experiment_id(
        "adapter_smoke", "without_sensitive", "raw", "foundation_smoke", None,
        {"fit_performed": False}, "none",
        provenance=provenance,
    )
    stage4b_id = deterministic_experiment_id(
        "adapter_smoke", "without_sensitive", "raw", "foundation_smoke", None,
        {"fit_performed": False}, "none", provenance=provenance, stage_id="stage4b",
    )
    changed_provenance_id = deterministic_experiment_id(
        "adapter_smoke", "without_sensitive", "raw", "foundation_smoke", None,
        {"fit_performed": False}, "none",
        provenance=dict(provenance, split_digest="changed-split"),
    )
    row = pd.DataFrame([{
        "experiment_id": experiment_id,
        "timestamp_utc": "2000-01-01T00:00:00+00:00",
        "model_family": "infrastructure",
        "model_name": "adapter_smoke",
        "sensitive_mode": "without_sensitive",
        "feature_set": "none",
        "target_mode": "raw",
        "evaluation_stage": "foundation_smoke",
        "training_row_count": 0,
        "validation_row_count": 0,
        "test_row_count": 0,
        "parameter_json": canonical_json({"fit_performed": False}),
        "status": "success",
        "notes": "Temporary adapter smoke row. No model fit.",
    }])
    first = upsert_registry(existing, row)
    second = upsert_registry(first, row)
    smoke_path = project / "artifacts/checkpoints/stage4/stage4a_registry_smoke.csv"
    atomic_write_csv(second, smoke_path)
    passed = bool(
        len(first) == len(existing) + 1
        and len(second) == len(first)
        and second["experiment_id"].is_unique
        and sha256_file(main_path) == prior_digest
        and second.loc[second["experiment_id"].eq(experiment_id)].shape[0] == 1
        and stage4b_id.startswith("stage4b__")
        and changed_provenance_id != experiment_id
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "main_registry_rows": int(len(existing)),
        "smoke_registry_rows": int(len(second)),
        "main_registry_unchanged": sha256_file(main_path) == prior_digest,
        "idempotent_upsert": len(second) == len(first),
        "experiment_id": experiment_id,
        "stage_parameterized": stage4b_id.startswith("stage4b__"),
        "provenance_changes_id": changed_provenance_id != experiment_id,
        "smoke_path": str(smoke_path.relative_to(project)),
    }
    atomic_write_json(project / "artifacts/reports/stage4a_registry_smoke_test.json", result)
    return result


def metric_smoke_test(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    true = np.array([1.0, 2.0, 4.0, 8.0])
    raw = evaluate_model_output(true, true, "raw")
    log_mode = evaluate_model_output(true, np.log1p(true), "log1p")
    passed = raw["mae"] == 0.0 and log_mode["mae"] < 1e-12 and raw["original_scale"] and log_mode["original_scale"]
    result = {"status": "PASS" if passed else "FAIL", "raw": raw, "log1p": log_mode}
    atomic_write_json(project / "artifacts/reports/stage4a_metric_smoke_test.json", result)
    return result


def previous_stage_validation(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    report_names = ["prompt1_verification.json", "prompt2_verification.json", "stage3_verification.json"]
    statuses: dict[str, Any] = {}
    for name in report_names:
        path = project / "artifacts/reports" / name
        if not path.is_file():
            statuses[name] = {"exists": False, "status": None}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        statuses[name] = {"exists": True, "status": payload.get("status"), "sha256": sha256_file(path)}
    split_dir = project / "artifacts/splits"
    train = pd.read_csv(split_dir / "train_row_ids.csv", dtype={"row_id": "int64"})
    test = pd.read_csv(split_dir / "test_row_ids.csv", dtype={"row_id": "int64"})
    folds = pd.read_csv(split_dir / "cv_fold_assignments.csv", dtype={"row_id": "int64", "fold": "int64"})
    split_valid = bool(
        train["row_id"].is_unique
        and test["row_id"].is_unique
        and set(train["row_id"]).isdisjoint(set(test["row_id"]))
        and set(folds["row_id"]) == set(train["row_id"])
        and folds["row_id"].is_unique
        and set(folds["fold"]) == {0, 1, 2}
    )
    result = {
        "reports": statuses,
        "train_rows": int(len(train)),
        "locked_test_rows": int(len(test)),
        "fold_rows": int(len(folds)),
        "fold_values": sorted(int(value) for value in folds["fold"].unique()),
        "split_valid": split_valid,
        "test_targets_loaded": False,
        "test_predictions_created": False,
    }
    result["status"] = "PASS" if split_valid and all(item["status"] == "PASS" for item in statuses.values()) else "FAIL"
    atomic_write_json(project / "artifacts/reports/stage4a_previous_stage_validation.json", result)
    return result


def run_foundation_smoke_suite(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    progress_path = project / "artifacts/checkpoints/stage4/stage4a_progress.json"
    write_progress(progress_path, "smoke_suite", "running")
    result = {
        "metric": metric_smoke_test(project),
        "atomic_write": atomic_write_smoke_test(project),
        "cache_validation": cache_validation_smoke_test(project),
        "registry": registry_smoke_test(project),
        "clean_worker_packages": clean_worker_package_smoke_test(project),
        "timeout": timeout_smoke_test(project),
    }
    result["status"] = "PASS" if all(item.get("status", "PASS" if item.get("pass") else "FAIL") == "PASS" or item.get("pass") for item in result.values()) else "FAIL"
    write_progress(progress_path, "smoke_suite", result["status"].lower(), result)
    atomic_write_json(project / "artifacts/reports/stage4a_smoke_suite.json", result)
    return result


def logical_foundation_snapshot(root: str | Path) -> dict[str, Any]:
    """Return stable logical evidence for notebook idempotence."""
    project = Path(root).resolve()
    samples = json.loads((project / "artifacts/splits/stage4/stage4_sample_verification.json").read_text(encoding="utf-8"))
    environment = json.loads((project / "artifacts/reports/stage4a_environment.json").read_text(encoding="utf-8"))
    smoke = json.loads((project / "artifacts/reports/stage4a_smoke_suite.json").read_text(encoding="utf-8"))
    protected = json.loads((project / "artifacts/manifests/stage4/stage4a_protected_hashes_before.json").read_text(encoding="utf-8"))
    registry = pd.read_csv(project / "artifacts/results/experiment_results.csv")
    notebook_path = project / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb"
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    markdown_lines = markdown.splitlines()
    section_counts = {
        f"section_{section}": sum(line.startswith(f"## {section}. ") for line in markdown_lines)
        for section in range(19)
    }
    sample_counts = {
        name: {
            "rows": int(item["rows"]),
            "train_rows": int(item["train_rows"]),
            "validation_rows": int(item["validation_rows"]),
        }
        for name, item in samples["samples"].items()
    }
    substantive_roots = {
        "stage4_results": project / "artifacts/results/stage4",
        "stage4_features": project / "artifacts/features/stage4",
    }
    substantive_counts = {
        name: sum(1 for path in root_path.rglob("*") if path.is_file())
        for name, root_path in substantive_roots.items()
    }
    substantive_counts["boosting_models"] = sum(
        1 for name in ("catboost", "lightgbm", "xgboost")
        for path in (project / "artifacts/models" / name).rglob("*") if path.is_file()
    )
    substantive_counts["boosting_predictions"] = sum(
        1 for name in ("catboost", "lightgbm", "xgboost")
        for path in (project / "artifacts/predictions" / name).rglob("*") if path.is_file()
    )
    return {
        "sample_status": samples["status"],
        "sample_total_rows": samples["total_sample_rows"],
        "sample_counts": sample_counts,
        "sample_test_overlap_rows": int(samples["test_overlap_rows"]),
        "sample_rows_disjoint": bool(samples["all_sample_rows_unique"]),
        "sample_hashes": {name: item["sha256"] for name, item in samples["samples"].items()},
        "package_modules": {
            name: {
                "module_version": item.get("module_version") or item.get("global_version"),
                "import_ok": item.get("import_ok", item.get("global_version") is not None),
                "construction_ok": item.get("construction_ok"),
            }
            for name, item in environment["packages"].items()
        },
        "smoke_statuses": {
            "metric": smoke["metric"]["status"],
            "atomic_write": smoke["atomic_write"]["status"],
            "registry": smoke["registry"]["status"],
            "timeout_pass": smoke["timeout"]["pass"],
        },
        "protected_file_count": protected["file_count"],
        "main_registry_rows": int(len(registry)),
        "main_registry_unique_ids": int(registry["experiment_id"].nunique()),
        "main_registry_sha256": sha256_file(project / "artifacts/results/experiment_results.csv"),
        "notebook_section_counts": section_counts,
        "substantive_artifact_counts": substantive_counts,
        "boosting_model_files": sum(
            1 for name in ("catboost", "lightgbm", "xgboost")
            for path in (project / "artifacts/models" / name).glob("*") if path.is_file()
        ),
        "boosting_prediction_files": sum(
            1 for name in ("catboost", "lightgbm", "xgboost")
            for path in (project / "artifacts/predictions" / name).glob("*") if path.is_file()
        ),
    }


def record_notebook_success(
    root: str | Path,
    run_id: str,
    recovery_id: str | None = None,
    implementation_digest: str | None = None,
) -> dict[str, Any]:
    """Record a notebook run only when its final code cell is reached."""
    project = Path(root).resolve()
    path = project / "artifacts/reports/stage4a_notebook_executions.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"runs": []}
    recovery_id = recovery_id or os.environ.get("STAGE4A_RECOVERY_ID")
    implementation_digest = implementation_digest or os.environ.get("STAGE4A_IMPLEMENTATION_DIGEST")
    if recovery_id is not None and implementation_digest is None:
        raise ValueError("Recovery runs require an implementation digest.")
    snapshot = logical_foundation_snapshot(project)
    record = {
        "run_id": str(run_id),
        "completed_at_utc": utc_now(),
        "status": "success",
        "recovery_id": recovery_id,
        "implementation_digest": implementation_digest,
        "snapshot": snapshot,
        "snapshot_digest": configuration_digest(snapshot, length=64),
    }
    history["runs"] = [item for item in history.get("runs", []) if item.get("run_id") != str(run_id)]
    history["runs"].append(record)
    successful = [item for item in history["runs"] if item.get("status") == "success"]
    history["successful_run_count"] = len(successful)
    history["latest_snapshot_digest"] = record["snapshot_digest"]
    atomic_write_json(path, history)
    return history


def build_internal_verification(root: str | Path) -> dict[str, Any]:
    """Check foundation artifacts without claiming notebook or review completion."""
    project = Path(root).resolve()
    previous = json.loads((project / "artifacts/reports/stage4a_previous_stage_validation.json").read_text(encoding="utf-8"))
    samples = json.loads((project / "artifacts/splits/stage4/stage4_sample_verification.json").read_text(encoding="utf-8"))
    environment = json.loads((project / "artifacts/reports/stage4a_environment.json").read_text(encoding="utf-8"))
    smoke = json.loads((project / "artifacts/reports/stage4a_smoke_suite.json").read_text(encoding="utf-8"))
    before = json.loads((project / "artifacts/manifests/stage4/stage4a_protected_hashes_before.json").read_text(encoding="utf-8"))
    protected = recheck_protected_manifest(project, before)
    atomic_write_json(project / "artifacts/manifests/stage4/stage4a_protected_hashes_after.json", protected)
    checks = {
        "previous_stages_pass": previous["status"] == "PASS",
        "saved_split_and_folds_valid": previous["split_valid"],
        "test_targets_not_loaded": not previous["test_targets_loaded"],
        "test_predictions_not_created": not previous["test_predictions_created"],
        "protected_hashes_unchanged": protected["status"] == "PASS",
        "three_samples_valid": samples["status"] == "PASS" and len(samples["samples"]) == 3,
        "samples_disjoint": samples["all_sample_rows_unique"],
        "samples_have_zero_test_overlap": samples["test_overlap_rows"] == 0,
        "package_audit_complete": environment["audit_completed"],
        "clean_worker_package_import_pass": smoke["clean_worker_packages"]["status"] == "PASS",
        "metric_smoke_pass": smoke["metric"]["status"] == "PASS",
        "atomic_write_smoke_pass": smoke["atomic_write"]["status"] == "PASS",
        "cache_validation_smoke_pass": smoke["cache_validation"]["status"] == "PASS",
        "registry_smoke_pass": smoke["registry"]["status"] == "PASS",
        "timeout_smoke_pass": smoke["timeout"]["pass"],
        "main_registry_unchanged": smoke["registry"]["main_registry_unchanged"],
        "no_boosting_model_fit_artifact": logical_foundation_snapshot(project)["boosting_model_files"] == 0,
        "no_boosting_prediction_artifact": logical_foundation_snapshot(project)["boosting_prediction_files"] == 0,
    }
    result = {"stage": STAGE_ID, "created_at_utc": utc_now(), "checks": checks}
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    atomic_write_json(project / "artifacts/reports/stage4a_internal_verification.json", result)
    return result


def build_artifact_summary(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    destination = project / "artifacts/manifests/stage4/stage4a_artifact_summary.json"
    owned_roots = [
        project / "artifacts/results/stage4",
        project / "artifacts/features/stage4",
        project / "artifacts/figures/stage4",
        project / "artifacts/manifests/stage4",
        project / "artifacts/checkpoints/stage4",
        project / "artifacts/splits/stage4",
        project / "artifacts/environment/stage4_packages",
        project / "artifacts/models/catboost",
        project / "artifacts/models/lightgbm",
        project / "artifacts/models/xgboost",
        project / "artifacts/predictions/catboost",
        project / "artifacts/predictions/lightgbm",
        project / "artifacts/predictions/xgboost",
    ]
    files = {
        path.resolve() for owned in owned_roots if owned.exists()
        for path in owned.rglob("*") if path.is_file()
    }
    files.update(path.resolve() for path in (project / "artifacts/reports").glob("stage4a*") if path.is_file())
    files.update(path.resolve() for path in (project / "artifacts/backups").glob("REGRESSION_PART4_BOOSTING_FOUNDATION_run*.ipynb") if path.is_file())
    files.update(
        path.resolve() for path in (
            project / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb",
            project / "stage4_boosting_utils.py",
            project / "build_stage4a_notebook.py",
            project / "prepare_stage4a_recovery.py",
            project / "run_stage4a_notebook.py",
        ) if path.is_file()
    )
    files.discard(destination.resolve())
    files = sorted(files, key=lambda path: path.as_posix())
    result = {
        "stage": STAGE_ID,
        "created_at_utc": utc_now(),
        "artifact_count": len(files),
        "artifacts": [
            {"path": str(path.relative_to(project)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
        "real_boosting_models_trained": 0,
        "test_predictions_created": 0,
    }
    atomic_write_json(destination, result)
    return result


def audit_stage4a_notebook_outputs(notebook: Any) -> dict[str, Any]:
    """Audit execution counts, errors, and required visible result outputs."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        output for cell in code_cells for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    required_sections = {
        "package_audit": "## 5. Environment and Package Audit",
        "registry_smoke": "## 8. Shared Registry System",
        "atomic_write_smoke": "## 9. Atomic Artifact Writers",
        "timeout_smoke": "## 10. Worker and Timeout Design",
        "discovery_sample": "## 12. Discovery Sample",
        "confirmation_sample": "## 13. Feature Confirmation Sample",
        "final_selection_sample": "## 14. Final Selection Sample",
        "sample_verification": "## 15. Sample Verification",
        "verification_summary": "## 17. Stage 4A Verification",
        "completion_note": "## 18. Stage 4A Completion Note",
    }
    section_outputs: dict[str, bool] = {}
    for label, heading in required_sections.items():
        heading_index = next(
            (index for index, cell in enumerate(notebook.cells)
             if cell.cell_type == "markdown" and heading in cell.source),
            None,
        )
        output_present = False
        if heading_index is not None:
            for cell in notebook.cells[heading_index + 1:]:
                if cell.cell_type == "markdown":
                    break
                if cell.cell_type == "code":
                    output_present = bool(cell.get("outputs"))
                    break
        section_outputs[label] = output_present
    checks = {
        "notebook_opens": True,
        "all_code_cells_executed": all(cell.get("execution_count") is not None for cell in code_cells),
        "zero_error_outputs": len(error_outputs) == 0,
        "required_sections_have_outputs": all(section_outputs.values()),
    }
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "total_code_cells": len(code_cells),
        "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
        "cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
        "error_output_count": len(error_outputs),
        "key_section_outputs": section_outputs,
    }


def finalize_stage4a_verification(root: str | Path, notebook_path: str | Path) -> dict[str, Any]:
    """Create final PASS evidence after executions and independent review."""
    import nbformat

    project = Path(root).resolve()
    notebook_file = Path(notebook_path).resolve()
    notebook = nbformat.read(notebook_file, as_version=4)
    expected_headings = [
        "## 0. Stage Objective", "## 1. Imports and Configuration", "## 2. Project Discovery",
        "## 3. Previous-Stage Validation", "## 4. Protected File Manifest",
        "## 5. Environment and Package Audit", "## 6. Artifact Directory Design",
        "## 7. Shared Metric System", "## 8. Shared Registry System",
        "## 9. Atomic Artifact Writers", "## 10. Worker and Timeout Design",
        "## 11. Progress and Heartbeat System", "## 12. Discovery Sample",
        "## 13. Feature Confirmation Sample", "## 14. Final Selection Sample",
        "## 15. Sample Verification", "## 16. Stage 4A Artifact Summary",
        "## 17. Stage 4A Verification", "## 18. Stage 4A Completion Note",
    ]
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    heading_counts = {heading: markdown.count(heading) for heading in expected_headings}
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"
    ]
    history_path = project / "artifacts/reports/stage4a_notebook_executions.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else {"runs": []}
    implementation_digest = stage4a_implementation_digest(project, notebook_file)
    recovery_id = STAGE4A_RECOVERY_ID
    successful = [
        item for item in history.get("runs", [])
        if item.get("status") == "success"
        and item.get("recovery_id") == recovery_id
        and item.get("implementation_digest") == implementation_digest
    ]
    digests = [item.get("snapshot_digest") for item in successful[-2:]]
    output_audit_path = project / "artifacts/reports/stage4a_notebook_output_audit.json"
    output_audit = json.loads(output_audit_path.read_text(encoding="utf-8")) if output_audit_path.is_file() else {"status": "MISSING"}
    internal = build_internal_verification(project)
    before = json.loads((project / "artifacts/manifests/stage4/stage4a_protected_hashes_before.json").read_text(encoding="utf-8"))
    protected = recheck_protected_manifest(project, before)
    atomic_write_json(project / "artifacts/manifests/stage4/stage4a_protected_hashes_after.json", protected)
    reviewer_path = project / "artifacts/reports/stage4a_reviewer.md"
    reviewer_text = reviewer_path.read_text(encoding="utf-8") if reviewer_path.is_file() else ""
    state_text = (project / "TASK.md").read_text(encoding="utf-8")
    environment = json.loads((project / "artifacts/reports/stage4a_environment.json").read_text(encoding="utf-8"))
    protected_paths = set(before["hashes"])
    source_paths = {
        "data/regression_with_sensitive_features.csv",
        "data/regression_without_sensitive_features.csv",
    }
    previous_notebooks = {
        "REGRESSION_PART2_MODELING.ipynb",
        "REGRESSION_PART3_TREE_MODELS.ipynb",
        r"D:\SHARIF\TERM7\DATA\PROJECT\main\REGRESION_PART1.ipynb",
    }
    split_paths = {
        "artifacts/splits/train_row_ids.csv",
        "artifacts/splits/test_row_ids.csv",
        "artifacts/splits/cv_fold_assignments.csv",
        "artifacts/splits/split_config.json",
    }
    checks = {
        "stage1_pass_loaded": json.loads((project / "artifacts/reports/prompt1_verification.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "stage2_pass_loaded": json.loads((project / "artifacts/reports/prompt2_verification.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "stage3_pass_loaded": json.loads((project / "artifacts/reports/stage3_verification.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "protected_hashes_recorded_and_unchanged": protected["status"] == "PASS" and protected["file_count"] == before["file_count"],
        "source_data_unchanged": source_paths.issubset(protected_paths) and protected["status"] == "PASS",
        "previous_notebooks_unchanged": previous_notebooks.issubset(protected_paths) and protected["status"] == "PASS",
        "saved_split_and_folds_unchanged": split_paths.issubset(protected_paths) and protected["status"] == "PASS",
        "test_set_unused": internal["checks"]["test_targets_not_loaded"] and internal["checks"]["test_predictions_not_created"],
        "three_disjoint_sample_manifests_exist": internal["checks"]["three_samples_valid"] and internal["checks"]["samples_disjoint"],
        "no_test_row_in_samples": internal["checks"]["samples_have_zero_test_overlap"],
        "target_distribution_preserved": all(
            item["maximum_target_bin_proportion_difference"] <= 0.001
            for item in json.loads((project / "artifacts/splits/stage4/stage4_sample_verification.json").read_text(encoding="utf-8"))["samples"].values()
        ),
        "package_audit_completed": internal["checks"]["package_audit_complete"],
        "catboost_import_passes": bool(environment["packages"]["catboost"].get("import_ok")),
        "lightgbm_import_passes": bool(environment["packages"]["lightgbm"].get("import_ok")),
        "xgboost_import_passes": bool(environment["packages"]["xgboost"].get("import_ok")),
        "shap_status_recorded": "shap" in environment["packages"],
        "clean_worker_package_import_works": internal["checks"]["clean_worker_package_import_pass"],
        "utility_module_imports": True,
        "timeout_smoke_test_works": internal["checks"]["timeout_smoke_pass"],
        "atomic_write_smoke_test_works": internal["checks"]["atomic_write_smoke_pass"],
        "cache_validation_smoke_test_works": internal["checks"]["cache_validation_smoke_pass"],
        "registry_smoke_test_works": internal["checks"]["registry_smoke_pass"] and internal["checks"]["main_registry_unchanged"],
        "notebook_executes_twice": len(successful) >= 2,
        "notebook_outputs_saved": all(cell.get("execution_count") is not None for cell in code_cells) and not error_outputs,
        "notebook_output_audit_passes": output_audit.get("status") == "PASS",
        "no_duplicate_notebook_sections": all(count == 1 for count in heading_counts.values()),
        "notebook_runs_idempotent": len(digests) == 2 and digests[0] == digests[1],
        "independent_review_completed": reviewer_path.is_file() and "Overall result: PASS" in reviewer_text,
        "accepted_critical_and_major_findings_fixed": "Open critical findings: 0" in reviewer_text and "Open major findings: 0" in reviewer_text,
        "state_files_updated": "Next Step: Begin Stage 4B — Initial Boosting Feature Packs." in state_text,
        "no_real_boosting_fit": internal["checks"]["no_boosting_model_fit_artifact"],
        "no_test_prediction": internal["checks"]["no_boosting_prediction_artifact"],
    }
    result = {
        "stage": STAGE_ID,
        "official_name": STAGE_NAME,
        "created_at_utc": utc_now(),
        "checks": checks,
        "notebook": {
            "path": str(notebook_file.relative_to(project)),
            "cells": len(notebook.cells),
            "code_cells": len(code_cells),
            "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
            "error_outputs": len(error_outputs),
            "section_counts": heading_counts,
            "successful_runs": len(successful),
            "recovery_id": recovery_id,
            "implementation_digest": implementation_digest,
            "last_two_snapshot_digests": digests,
        },
        "protected_file_count": protected["file_count"],
        "protected_mismatches": protected["mismatches"],
        "test_predictions": 0,
        "real_boosting_fits": 0,
        "next_step": "Begin Stage 4B — Initial Boosting Feature Packs.",
    }
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    atomic_write_json(project / "artifacts/reports/stage4a_verification.json", result)
    atomic_write_json(project / "artifacts/reports/stage4a_idempotence_report.json", {
        "status": "PASS" if checks["notebook_runs_idempotent"] else "FAIL",
        "recovery_id": recovery_id,
        "implementation_digest": implementation_digest,
        "successful_recovery_runs": len(successful),
        "last_two_snapshot_digests": digests,
        "logical_results_match": len(digests) == 2 and digests[0] == digests[1],
        "sample_stability": len(successful) >= 2 and successful[-2]["snapshot"]["sample_hashes"] == successful[-1]["snapshot"]["sample_hashes"] and successful[-2]["snapshot"]["sample_counts"] == successful[-1]["snapshot"]["sample_counts"],
        "registry_stability": len(successful) >= 2 and successful[-2]["snapshot"]["main_registry_sha256"] == successful[-1]["snapshot"]["main_registry_sha256"] and successful[-2]["snapshot"]["main_registry_rows"] == successful[-1]["snapshot"]["main_registry_rows"],
        "section_stability": len(successful) >= 2 and successful[-2]["snapshot"]["notebook_section_counts"] == successful[-1]["snapshot"]["notebook_section_counts"],
        "artifact_count_stability": len(successful) >= 2 and successful[-2]["snapshot"]["substantive_artifact_counts"] == successful[-1]["snapshot"]["substantive_artifact_counts"],
        "package_version_stability": len(successful) >= 2 and successful[-2]["snapshot"]["package_modules"] == successful[-1]["snapshot"]["package_modules"],
        "protected_hashes_unchanged": protected["status"] == "PASS",
        "registry_ids_unique": pd.read_csv(project / "artifacts/results/experiment_results.csv")["experiment_id"].is_unique,
        "section_counts": heading_counts,
    })
    build_artifact_summary(project)
    return result


def _timeout_smoke_worker(heartbeat_path: str, seconds: float) -> int:
    child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({float(seconds)})"])
    try:
        deadline = time.monotonic() + float(seconds)
        while time.monotonic() < deadline:
            atomic_write_json(heartbeat_path, {
                "stage": STAGE_ID,
                "worker_id": "timeout_smoke_worker",
                "process_id": os.getpid(),
                "child_process_id": child.pid,
                "state": "running",
                "updated_at_utc": utc_now(),
                "monotonic_seconds": time.monotonic(),
            })
            time.sleep(0.2)
        return 0
    finally:
        if child.poll() is None:
            child.kill()


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-smoke-worker", nargs=2, metavar=("HEARTBEAT", "SECONDS"))
    arguments = parser.parse_args()
    if arguments.timeout_smoke_worker:
        return _timeout_smoke_worker(arguments.timeout_smoke_worker[0], float(arguments.timeout_smoke_worker[1]))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
