"""Build the independent Stage 4A foundation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb"


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


cells = [
    md("""
# Stage 4A — Boosting Infrastructure and Experiment Foundation

This notebook builds safe shared tools for later CatBoost, LightGBM, and XGBoost work. It does not fit a real boosting model. The saved Test Set stays locked.
"""),
    md("""
## 0. Stage Objective

The goal is to prepare one safe and reusable foundation. This includes fixed training-only samples, shared metrics, Registry rules, atomic file writes, worker timeouts, progress files, and environment evidence.
"""),
    code("""
from datetime import datetime, timezone
run_id = datetime.now(timezone.utc).isoformat()
print("Stage 4A foundation only")
print("Real boosting fits allowed:", 0)
print("Test predictions allowed:", 0)
"""),
    md("""
## 1. Imports and Configuration

The random seed is 42. New internal identifiers use `stage4a`. Sample roles and target bins are metadata only. They are never model features.
"""),
    code("""
import json
import sys
from pathlib import Path
import pandas as pd
import stage4_boosting_utils as s4

SEED = s4.RANDOM_SEED
STAGE_ID = s4.STAGE_ID
print({"stage_id": STAGE_ID, "seed": SEED, "python": sys.version.split()[0]})
"""),
    md("""
## 2. Project Discovery

The project root is found from `AGENTS.md` and the saved Train row file. This avoids depending on one fixed working directory.
"""),
    code("""
ROOT = s4.discover_project_root()
print("Project root:", ROOT)
print("Source CSV files:", sorted(path.name for path in (ROOT / "data").glob("*.csv")))
"""),
    md("""
## 3. Previous-Stage Validation

Stage 1, Stage 2, and Stage 3 must report PASS. The saved Train, Test, and Fold files must be complete and separate. No replacement split is allowed.
"""),
    code("""
previous = json.loads((ROOT / "artifacts/reports/stage4a_previous_stage_validation.json").read_text(encoding="utf-8"))
assert previous["status"] == "PASS", previous
previous
"""),
    md("""
## 4. Protected File Manifest

The before-run manifest includes source data, all prior notebooks, saved splits, prior results, prior models, and prior predictions. Any changed hash is a critical failure.
"""),
    code("""
before_path = ROOT / "artifacts/manifests/stage4/stage4a_protected_hashes_before.json"
before = json.loads(before_path.read_text(encoding="utf-8"))
protected_check = s4.recheck_protected_manifest(ROOT, before)
s4.atomic_write_json(ROOT / "artifacts/manifests/stage4/stage4a_protected_hashes_after.json", protected_check)
assert protected_check["status"] == "PASS", protected_check["mismatches"]
print({"protected_files": protected_check["file_count"], "mismatches": len(protected_check["mismatches"])})
"""),
    md("""
## 5. Environment and Package Audit

The audit records Python, packages, CPU, RAM, disk, GPU, and CUDA. Missing authorized packages received at most one project-local installation attempt before notebook execution. This section imports and constructs small estimator objects only. It does not call `fit`.
"""),
    code("""
environment = json.loads((ROOT / "artifacts/reports/stage4a_environment.json").read_text(encoding="utf-8"))
worker_package_smoke = json.loads((ROOT / "artifacts/reports/stage4a_clean_worker_package_smoke_test.json").read_text(encoding="utf-8"))
assert worker_package_smoke["status"] == "PASS"
package_view = {
    name: {
        "version": item.get("module_version") or item.get("global_version"),
        "import_ok": item.get("import_ok", item.get("global_version") is not None),
        "construction_ok": item.get("construction_ok"),
    }
    for name, item in environment["packages"].items()
}
pd.DataFrame(package_view).T
"""),
    md("""
## 6. Artifact Directory Design

Each boosting library has separate result, model, and prediction folders. Shared Stage 4 reports, features, figures, manifests, and checkpoints also have separate folders. Old artifacts are not moved.
"""),
    code("""
stage4_directories = s4.ensure_stage4_directories(ROOT)
assert all((ROOT / path).is_dir() for path in stage4_directories)
pd.DataFrame({"stage4_directory": stage4_directories})
"""),
    md("""
## 7. Shared Metric System

The shared adapter supports raw and `log1p` targets. Predictions are changed back to the original target scale before metrics are calculated.
"""),
    code("""
metric_smoke = json.loads((ROOT / "artifacts/reports/stage4a_metric_smoke_test.json").read_text(encoding="utf-8"))
assert metric_smoke["status"] == "PASS"
pd.DataFrame([metric_smoke["raw"], metric_smoke["log1p"]], index=["raw", "log1p"])
"""),
    md("""
## 8. Shared Registry System

Experiment IDs are deterministic and start with `stage4a`. Upsert uses the fixed 31-column schema. The smoke test uses a scratch copy, so all 215 prior Registry rows stay unchanged.
"""),
    code("""
registry_smoke = json.loads((ROOT / "artifacts/reports/stage4a_registry_smoke_test.json").read_text(encoding="utf-8"))
assert registry_smoke["status"] == "PASS"
registry_smoke
"""),
    md("""
## 9. Atomic Artifact Writers

CSV, JSON, and joblib files are written to a temporary file first. The complete temporary file then replaces the destination.
"""),
    code("""
atomic_smoke = json.loads((ROOT / "artifacts/reports/stage4a_atomic_write_smoke_test.json").read_text(encoding="utf-8"))
assert atomic_smoke["status"] == "PASS"
cache_smoke = json.loads((ROOT / "artifacts/reports/stage4a_cache_validation_smoke_test.json").read_text(encoding="utf-8"))
assert cache_smoke["status"] == "PASS"
{"atomic_write": atomic_smoke, "cache_validation": cache_smoke}
"""),
    md("""
## 10. Worker and Timeout Design

A parent process starts each later heavy worker. If time expires, the parent stops the worker and its child process tree. A check after `fit` is not used as the timeout.
"""),
    code("""
timeout_smoke = json.loads((ROOT / "artifacts/reports/stage4a_timeout_smoke_test.json").read_text(encoding="utf-8"))
assert timeout_smoke["pass"], timeout_smoke
timeout_smoke
"""),
    md("""
## 11. Progress and Heartbeat System

Progress files describe the current Stage step. Heartbeat files show that a worker is alive. Both use atomic JSON writes.
"""),
    code("""
smoke_suite = json.loads((ROOT / "artifacts/reports/stage4a_smoke_suite.json").read_text(encoding="utf-8"))
assert smoke_suite["status"] == "PASS"
heartbeat_path = ROOT / "artifacts/checkpoints/stage4/stage4a_notebook_heartbeat.json"
s4.write_heartbeat(heartbeat_path, "stage4a_notebook", state="foundation_ready")
heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
print({"smoke_suite": smoke_suite["status"], "heartbeat_state": heartbeat["state"]})
"""),
    md("""
## 12. Discovery Sample

The discovery sample has 50,000 training rows and 15,000 validation rows. It is for first models, first feature importance, and first error analysis in Stage 4B.
"""),
    code("""
sample_verification = s4.validate_existing_stage4_samples(ROOT)
assert sample_verification["status"] == "PASS"
discovery = pd.read_csv(ROOT / "artifacts/splits/stage4/stage4_discovery_sample.csv")
discovery.groupby(["sample_role", "target_bin"]).size().unstack(fill_value=0)
"""),
    md("""
## 13. Feature Confirmation Sample

The feature confirmation sample has 80,000 training rows and 20,000 validation rows. It checks that a later feature idea also helps on different rows.
"""),
    code("""
confirmation = pd.read_csv(ROOT / "artifacts/splits/stage4/stage4_feature_confirmation_sample.csv")
assert len(confirmation) == 100_000 and confirmation["row_id"].is_unique
confirmation.groupby(["sample_role", "target_bin"]).size().unstack(fill_value=0)
"""),
    md("""
## 14. Final Selection Sample

The final selection sample has 100,000 training rows and 25,000 validation rows. It is reserved for limited final tuning and the later sensitive comparison.
"""),
    code("""
final_selection = pd.read_csv(ROOT / "artifacts/splits/stage4/stage4_final_selection_sample.csv")
assert len(final_selection) == 125_000 and final_selection["row_id"].is_unique
final_selection.groupby(["sample_role", "target_bin"]).size().unstack(fill_value=0)
"""),
    md("""
## 15. Sample Verification

All three samples use saved Train rows only. They are disjoint, have zero Test overlap, and closely follow the saved training target-bin distribution.
"""),
    code("""
sample_verification = json.loads((ROOT / "artifacts/splits/stage4/stage4_sample_verification.json").read_text(encoding="utf-8"))
assert sample_verification["status"] == "PASS"
pd.DataFrame(sample_verification["samples"]).T
"""),
    md("""
## 16. Stage 4A Artifact Summary

The summary lists Stage 4A foundation artifacts. Model and prediction folders stay empty because no real boosting experiment is allowed here.
"""),
    code("""
artifact_summary = s4.build_artifact_summary(ROOT)
assert artifact_summary["real_boosting_models_trained"] == 0
assert artifact_summary["test_predictions_created"] == 0
print({"artifact_count": artifact_summary["artifact_count"], "real_boosting_models": 0, "test_predictions": 0})
"""),
    md("""
## 17. Stage 4A Verification

Internal verification checks prior PASS evidence, protected hashes, Test isolation, sample safety, package evidence, and all utility smoke tests. Final external verification also checks two saved notebook runs and the independent review.
"""),
    code("""
internal_verification = s4.build_internal_verification(ROOT)
assert internal_verification["status"] == "PASS", internal_verification
pd.Series(internal_verification["checks"], name="passed")
"""),
    md("""
## 18. Stage 4A Completion Note

Stage 4A is complete after final external verification. Three training-only samples were created and validated. Test data was not used. CatBoost, LightGBM, and XGBoost are available. Shared utilities are ready. No real boosting experiment was trained. The next Stage is Stage 4B.
"""),
    code("""
execution_history = s4.record_notebook_success(ROOT, run_id)
print("Stage 4A notebook run completed successfully.")
print("Successful clean runs recorded:", execution_history["successful_run_count"])
print("Next step after final external verification: Begin Stage 4B — Initial Boosting Feature Packs.")
"""),
]

notebook = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "stage": "stage4a",
        "official_stage_name": "Stage 4A — Boosting Infrastructure and Experiment Foundation",
    },
)

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"stage4a-cell-{index}", "exec")

nbformat.write(notebook, OUTPUT)
print(f"Built {OUTPUT.name}: {len(notebook.cells)} cells, {sum(c.cell_type == 'code' for c in notebook.cells)} code cells")
