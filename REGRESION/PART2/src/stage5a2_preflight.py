"""Stage 5A2 pre-target preflight, protected baseline, backup, and freeze."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parent
STAGE_ID = "stage5a2"
SAMPLE = ROOT / "artifacts/splits/stage4/stage4_final_selection_sample.csv"
DISCOVERY = ROOT / "artifacts/splits/stage4/stage4_discovery_sample.csv"
TRAIN_IDS = ROOT / "artifacts/splits/train_row_ids.csv"
TEST_IDS = ROOT / "artifacts/splits/test_row_ids.csv"
SOURCE_WITHOUT = ROOT / "data/regression_without_sensitive_features.csv"
SOURCE_WITH = ROOT / "data/regression_with_sensitive_features.csv"
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"
BASELINE = ROOT / "artifacts/manifests/stage5/stage5a2_protected_hashes_before.json"
FREEZE = ROOT / "artifacts/reports/stage5a2_prevalidation_freeze.json"
PREFLIGHT = ROOT / "artifacts/reports/stage5a2_preflight.json"
RESOURCE = ROOT / "artifacts/reports/stage5a2_resource_recheck.json"
ENV_PYTHON = ROOT / "artifacts/environment/stage5_env/Scripts/python.exe"


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def digest_int64(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def normalized_cell_hash(cell: dict[str, Any]) -> str:
    payload = copy.deepcopy(cell)
    payload.pop("id", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_notebook_json() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def create_backup_and_notebook_prefix() -> tuple[Path, list[str]]:
    notebook = load_notebook_json()
    if len(notebook["cells"]) != 51:
        raise RuntimeError(f"Expected the finalized 51-cell Stage 5A1 Notebook, found {len(notebook['cells'])}")
    text = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])
    if "### 24. Stage 5A1 Completion Note" not in text:
        raise RuntimeError("Stage 5A1 section 24 is missing")
    if "### 25. Stage 5A2 Objective" in text:
        raise RuntimeError("Stage 5A2 Notebook cells already exist; resume audit is required")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / f"artifacts/backups/REGRESSION_PART5_DEEP_TABULAR_MODELS.stage5a2_pre_{timestamp}.ipynb"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK, backup)
    if sha256_file(backup) != sha256_file(NOTEBOOK):
        raise RuntimeError("Notebook backup hash mismatch")
    return backup, [normalized_cell_hash(cell) for cell in notebook["cells"]]


def _stage5a1_paths() -> set[Path]:
    paths: set[Path] = set()
    for pattern in (
        "artifacts/checkpoints/stage5/deep_core/screening/**/*",
        "artifacts/figures/stage5a1/**/*",
        "artifacts/models/deep/core_screening/**/*",
        "artifacts/predictions/stage5/deep_core/screening/**/*",
        "artifacts/preprocessing/stage5/deep_core/**/*",
        "artifacts/results/stage5/deep_core/screening/**/*",
        "artifacts/reports/stage5a1*",
        "artifacts/manifests/stage5/stage5a1*",
    ):
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return paths


def protected_paths() -> list[Path]:
    paths: set[Path] = set()
    stage5a1_baseline = ROOT / "artifacts/manifests/stage5/stage5a1_protected_hashes_before.json"
    prior = json.loads(stage5a1_baseline.read_text(encoding="utf-8"))
    for item in prior["files"]:
        path = Path(item["path"])
        if path.name != "experiment_results.csv":
            paths.add(path)
    paths.update(_stage5a1_paths())
    paths.update(
        {
            ROOT / "stage5_deep_models.py",
            ROOT / "stage5_deep_preprocessing.py",
            ROOT / "stage5_deep_worker.py",
        }
    )
    return sorted((path for path in paths if path.exists()), key=lambda path: str(path).lower())


def capture_baseline(prefix_hashes: list[str], backup: Path) -> dict[str, Any]:
    entries = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in protected_paths()
    ]
    registry = ROOT / "artifacts/results/experiment_results.csv"
    payload = {
        "stage_id": STAGE_ID,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "protected_file_count": len(entries),
        "files": entries,
        "registry_prefix": {
            "path": str(registry.resolve()),
            "size": registry.stat().st_size,
            "sha256": sha256_file(registry),
        },
        "stage5a1_notebook": {
            "path": str(NOTEBOOK.resolve()),
            "cell_count": 51,
            "cell_hashes": prefix_hashes,
            "file_sha256_before_append": sha256_file(NOTEBOOK),
            "backup_path": str(backup.relative_to(ROOT)),
            "backup_sha256": sha256_file(backup),
        },
        "status": "PASS",
    }
    atomic_json(payload, BASELINE)
    reloaded = json.loads(BASELINE.read_text(encoding="utf-8"))
    if reloaded != payload:
        raise RuntimeError("Protected baseline reload mismatch")
    return payload


def env_probe() -> dict[str, Any]:
    code = r'''
import json, joblib, torch, sklearn, pandas, numpy, psutil
import pytabkit, rtdl_revisiting_models
import stage5_deep_models, stage5_deep_preprocessing, stage5_deep_worker
bundle = joblib.load(r"artifacts/models/deep/core_screening/stage5a1__realmlp__raw__replacement1.joblib")
model = bundle["model"]
config = model.get_config()
probe = model.__class__(
    device="cpu", random_state=42, n_cv=1, n_refit=0, n_repeats=2,
    n_threads=4, verbosity=0, n_epochs=64, batch_size=256,
    predict_batch_size=1024, train_metric_name="mae", val_metric_name="mae",
    use_early_stopping=True,
)
probe_config = probe.get_config()
print(json.dumps({
    "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    "sklearn": sklearn.__version__, "pandas": pandas.__version__, "numpy": numpy.__version__,
    "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(False),
    "realmlp_class": model.__class__.__module__ + "." + model.__class__.__name__,
    "realmlp_frozen_config": config, "realmlp_refined_probe_config": probe_config,
    "imports_pass": True,
}, default=str))
'''
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(ROOT / "artifacts/environment/stage5_matplotlib")
    completed = subprocess.run(
        [str(ENV_PYTHON), "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Deep environment probe failed: {completed.stderr}")
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"Deep environment probe returned no JSON: {completed.stdout}")
    return json.loads(lines[-1])


def validate_handoff_and_sample() -> dict[str, Any]:
    gate_path = ROOT / "artifacts/reports/stage5a1_gate_verification.json"
    reviewer_path = ROOT / "artifacts/reports/stage5a1_reviewer_cycle3.json"
    validity_path = ROOT / "artifacts/reports/stage5a1_screening_validity.json"
    top_two_path = ROOT / "artifacts/results/stage5/deep_core/screening/stage5a1_top_two_families.json"
    schema_path = ROOT / "artifacts/preprocessing/stage5/deep_core/deep_core_v1_schema.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    reviewer = json.loads(reviewer_path.read_text(encoding="utf-8"))
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    top_two = json.loads(top_two_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    sample = pd.read_csv(SAMPLE, usecols=["row_id", "sample_role"])
    discovery = pd.read_csv(DISCOVERY, usecols=["row_id"])
    train_ids = pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    sample_ids = sample["row_id"].to_numpy(np.int64)
    train_rows = sample.loc[sample["sample_role"] == "train", "row_id"].to_numpy(np.int64)
    val_rows = sample.loc[sample["sample_role"] == "validation", "row_id"].to_numpy(np.int64)
    roles = sample["sample_role"].value_counts().to_dict()

    checks = {
        "stage5a1_gate_pass": gate.get("status") == "PASS",
        "stage5a1_reviewer_cycle3_pass": reviewer.get("status") == "PASS",
        "physical_fit_count_11": validity.get("physical_screening_fit_count") == 11,
        "valid_candidate_count_6": validity.get("final_valid_candidate_count") == 6,
        "continuing_families_exact": top_two.get("selected_families") == ["realmlp", "ft_transformer"],
        "continuing_candidates_exact": top_two.get("selected_candidate_ids") == [
            "stage5a1__realmlp__raw__replacement1", "stage5a1__ft_transformer__log1p"
        ],
        "schema_deep_core_v1": schema.get("schema_id") == "deep_core_v1",
        "sample_rows_125000": len(sample) == 125_000,
        "sample_train_rows_100000": roles.get("train") == 100_000,
        "sample_validation_rows_25000": roles.get("validation") == 25_000,
        "sample_roles_exact": set(roles) == {"train", "validation"},
        "sample_ids_unique": sample["row_id"].is_unique,
        "sample_in_saved_train": bool(np.isin(sample_ids, train_ids).all()),
        "sample_test_overlap_zero": len(np.intersect1d(sample_ids, test_ids)) == 0,
        "sample_discovery_overlap_zero": len(np.intersect1d(sample_ids, discovery["row_id"].to_numpy(np.int64))) == 0,
        "source_headers_match_schema": all(
            column in pd.read_csv(SOURCE_WITHOUT, nrows=0).columns
            for column in [schema["target"], *schema["feature_order"]]
        ),
        "no_final_selection_target_values_loaded": True,
        "no_test_feature_or_target_rows_loaded": True,
        "no_stage4l_test_metrics_loaded": True,
    }
    return {
        "checks": checks,
        "gate_path": str(gate_path.relative_to(ROOT)),
        "gate_sha256": sha256_file(gate_path),
        "reviewer_path": str(reviewer_path.relative_to(ROOT)),
        "reviewer_sha256": sha256_file(reviewer_path),
        "top_two_path": str(top_two_path.relative_to(ROOT)),
        "top_two_sha256": sha256_file(top_two_path),
        "schema_path": str(schema_path.relative_to(ROOT)),
        "schema_sha256": sha256_file(schema_path),
        "schema_digest": schema["feature_schema_digest"],
        "sample_sha256": sha256_file(SAMPLE),
        "sample_row_id_hash": digest_int64(sample_ids),
        "train_row_id_hash": digest_int64(train_rows),
        "validation_row_id_hash": digest_int64(val_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def candidate_plan(handoff: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    real_frozen = copy.deepcopy(environment["realmlp_frozen_config"])
    real_frozen.update({"n_cv": 1, "n_refit": 0, "n_repeats": 1, "n_epochs": 64,
                        "batch_size": 256, "predict_batch_size": 1024,
                        "use_early_stopping": True, "random_state": 42})
    real_refined = copy.deepcopy(real_frozen)
    real_refined["n_repeats"] = 2
    ft_architecture = {
        "implementation": "rtdl_revisiting_models.FTTransformer",
        "feature_tokenization": True,
        "token_dimension": 96,
        "n_blocks": 3,
        "attention_heads": 8,
        "attention_dropout": 0.15,
        "ffn_hidden_multiplier": 4.0 / 3.0,
        "ffn_dropout": 0.10,
        "residual_dropout": 0.0,
    }
    ft_training = {
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-5,
        "loss": "SmoothL1Loss(beta=0.5)", "batch_size": 256,
        "maximum_epochs": 24, "early_stopping_patience": 5,
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
        "gradient_clip_norm": 1.0, "checkpoint_metric": "original_scale_validation_mae",
    }
    ft_refined_training = copy.deepcopy(ft_training)
    ft_refined_training["weight_decay"] = 1e-4
    return {
        "freeze_timestamp": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "official_stage_name": "Stage 5A2 — Top-Two Deep Validation and Core Final Models",
        "stage5a1_gate_hash": handoff["gate_sha256"],
        "stage5a1_top_two_selection_hash": handoff["top_two_sha256"],
        "feature_schema_hash": handoff["schema_sha256"],
        "feature_schema_digest": handoff["schema_digest"],
        "final_selection_manifest_hash": handoff["sample_sha256"],
        "final_selection_row_id_hash": handoff["sample_row_id_hash"],
        "final_selection_train_row_id_hash": handoff["train_row_id_hash"],
        "final_selection_validation_row_id_hash": handoff["validation_row_id_hash"],
        "selected_families": ["realmlp", "ft_transformer"],
        "frozen_target_modes": {"realmlp": "raw", "ft_transformer": "log1p"},
        "frozen_preprocessing_versions": {
            "realmlp": "realmlp_train_only_vocab_v2_plus_official_pytabkit",
            "ft_transformer": "ft_transformer_v1_quantile_normal_train_only",
        },
        "feature_schema": "deep_core_v1",
        "seed": 42,
        "device": "cpu",
        "precision": "float32",
        "batch_policy": {"training": 256, "inference": 1024, "num_workers": 0},
        "regular_candidate_count": 4,
        "candidates": [
            {
                "candidate_id": "stage5a2__realmlp__frozen", "family": "realmlp",
                "candidate_type": "frozen", "target_mode": "raw",
                "architecture_and_training": real_frozen,
                "preprocessing": "Train-only RealMLP vocabulary, numeric imputation, official PyTabKit transforms",
                "refinement_reason": "Exact Stage 5A1 RealMLP raw winner.",
            },
            {
                "candidate_id": "stage5a2__realmlp__refined", "family": "realmlp",
                "candidate_type": "refined", "target_mode": "raw",
                "architecture_and_training": real_refined,
                "preprocessing": "Train-only RealMLP vocabulary, numeric imputation, official PyTabKit transforms",
                "refinement_reason": "Increase only official packed-ensemble repeats from 1 to 2; installed API support was verified. n_cv remains 1.",
            },
            {
                "candidate_id": "stage5a2__ft_transformer__frozen", "family": "ft_transformer",
                "candidate_type": "frozen", "target_mode": "log1p",
                "architecture": ft_architecture, "training": ft_training,
                "preprocessing": "Training-fit median, Quantile-normal transform, and Missing/Unknown/Rare categorical vocabulary",
                "refinement_reason": "Exact Stage 5A1 FT-Transformer log1p winner.",
            },
            {
                "candidate_id": "stage5a2__ft_transformer__refined", "family": "ft_transformer",
                "candidate_type": "refined", "target_mode": "log1p",
                "architecture": ft_architecture, "training": ft_refined_training,
                "preprocessing": "Training-fit median, Quantile-normal transform, and Missing/Unknown/Rare categorical vocabulary",
                "refinement_reason": "Stage 5A1 training loss fell after Validation MAE worsened; increase only AdamW weight decay from 1e-5 to 1e-4.",
            },
        ],
        "regular_fit_policy": {
            "early_stopping": True,
            "best_checkpoint": True,
            "checkpoint_metric": "original_scale_validation_mae",
            "technical_retry_maximum": 1,
            "heavy_fits_sequential": True,
        },
        "statement_final_selection_targets_not_loaded": True,
        "statement_test_and_stage4l_test_metrics_not_used": True,
        "status": "PASS",
    }


def main() -> None:
    if FREEZE.exists() or BASELINE.exists():
        raise RuntimeError("Stage 5A2 preflight artifacts already exist; validate and resume instead of overwriting")
    handoff = validate_handoff_and_sample()
    if handoff["status"] != "PASS":
        raise RuntimeError(f"Stage 5A2 handoff failed: {handoff['checks']}")
    environment = env_probe()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    resource = {
        "stage_id": STAGE_ID,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "total_ram_gib": memory.total / (1024**3),
        "available_ram_gib": memory.available / (1024**3),
        "free_disk_gib": disk.free / (1024**3),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "device": "cpu", "precision": "float32", "num_workers": 0,
        "environment": environment,
        "status": "PASS",
    }
    atomic_json(resource, RESOURCE)
    backup, prefix_hashes = create_backup_and_notebook_prefix()
    baseline = capture_baseline(prefix_hashes, backup)
    freeze = candidate_plan(handoff, environment)
    atomic_json(freeze, FREEZE)
    reloaded = json.loads(FREEZE.read_text(encoding="utf-8"))
    if reloaded != freeze:
        raise RuntimeError("Pre-validation freeze reload mismatch")
    report = {
        "stage_id": STAGE_ID,
        "handoff": handoff,
        "resource_report": str(RESOURCE.relative_to(ROOT)),
        "resource_report_sha256": sha256_file(RESOURCE),
        "notebook_backup": str(backup.relative_to(ROOT)),
        "protected_baseline": str(BASELINE.relative_to(ROOT)),
        "protected_baseline_sha256": sha256_file(BASELINE),
        "prevalidation_freeze": str(FREEZE.relative_to(ROOT)),
        "prevalidation_freeze_sha256": sha256_file(FREEZE),
        "final_selection_targets_loaded": False,
        "test_rows_loaded": False,
        "stage4l_test_metrics_loaded": False,
        "status": "PASS",
    }
    atomic_json(report, PREFLIGHT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
