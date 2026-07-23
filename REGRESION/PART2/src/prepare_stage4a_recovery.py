"""Apply the narrow Stage 4A recovery edits while preserving unknown cells."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb"

REPLACEMENTS = {
    "## 3. Previous-Stage Validation": '''previous = json.loads((ROOT / "artifacts/reports/stage4a_previous_stage_validation.json").read_text(encoding="utf-8"))
assert previous["status"] == "PASS", previous
previous''',
    "## 5. Environment and Package Audit": '''environment = json.loads((ROOT / "artifacts/reports/stage4a_environment.json").read_text(encoding="utf-8"))
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
pd.DataFrame(package_view).T''',
    "## 7. Shared Metric System": '''metric_smoke = json.loads((ROOT / "artifacts/reports/stage4a_metric_smoke_test.json").read_text(encoding="utf-8"))
assert metric_smoke["status"] == "PASS"
pd.DataFrame([metric_smoke["raw"], metric_smoke["log1p"]], index=["raw", "log1p"])''',
    "## 8. Shared Registry System": '''registry_smoke = json.loads((ROOT / "artifacts/reports/stage4a_registry_smoke_test.json").read_text(encoding="utf-8"))
assert registry_smoke["status"] == "PASS"
registry_smoke''',
    "## 9. Atomic Artifact Writers": '''atomic_smoke = json.loads((ROOT / "artifacts/reports/stage4a_atomic_write_smoke_test.json").read_text(encoding="utf-8"))
assert atomic_smoke["status"] == "PASS"
cache_smoke = json.loads((ROOT / "artifacts/reports/stage4a_cache_validation_smoke_test.json").read_text(encoding="utf-8"))
assert cache_smoke["status"] == "PASS"
{"atomic_write": atomic_smoke, "cache_validation": cache_smoke}''',
    "## 10. Worker and Timeout Design": '''timeout_smoke = json.loads((ROOT / "artifacts/reports/stage4a_timeout_smoke_test.json").read_text(encoding="utf-8"))
assert timeout_smoke["pass"], timeout_smoke
timeout_smoke''',
    "## 11. Progress and Heartbeat System": '''smoke_suite = json.loads((ROOT / "artifacts/reports/stage4a_smoke_suite.json").read_text(encoding="utf-8"))
assert smoke_suite["status"] == "PASS"
heartbeat_path = ROOT / "artifacts/checkpoints/stage4/stage4a_notebook_heartbeat.json"
s4.write_heartbeat(heartbeat_path, "stage4a_notebook", state="foundation_ready")
heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
print({"smoke_suite": smoke_suite["status"], "heartbeat_state": heartbeat["state"]})''',
    "## 12. Discovery Sample": '''sample_verification = s4.validate_existing_stage4_samples(ROOT)
assert sample_verification["status"] == "PASS"
discovery = pd.read_csv(ROOT / "artifacts/splits/stage4/stage4_discovery_sample.csv")
discovery.groupby(["sample_role", "target_bin"]).size().unstack(fill_value=0)''',
}


def following_code_cell(notebook, heading_index: int):
    for cell in notebook.cells[heading_index + 1:]:
        if cell.cell_type == "markdown":
            break
        if cell.cell_type == "code":
            return cell
    raise RuntimeError(f"No code cell follows heading cell {heading_index}.")


notebook = nbformat.read(NOTEBOOK, as_version=4)
for heading, source in REPLACEMENTS.items():
    matches = [
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and heading in cell.source
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {heading!r} heading, found {len(matches)}.")
    following_code_cell(notebook, matches[0]).source = source

completion_heading = "## 18. Stage 4A Completion Note"
completion_matches = [
    cell for cell in notebook.cells
    if cell.cell_type == "markdown" and completion_heading in cell.source
]
if len(completion_matches) != 1:
    raise RuntimeError("Expected one Stage 4A completion note.")
completion_matches[0].source = '''## 18. Stage 4A Completion Note

Stage 4A is complete after final external verification. Three training-only samples were created and validated. Test data was not used. CatBoost, LightGBM, and XGBoost are available. Shared utilities are ready. No real boosting experiment was trained. The next Stage is Stage 4B.'''

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"stage4a-recovery-cell-{index}", "exec")

temporary = NOTEBOOK.with_suffix(NOTEBOOK.suffix + f".{os.getpid()}.tmp")
nbformat.write(notebook, temporary)
os.replace(temporary, NOTEBOOK)
print(f"Prepared {NOTEBOOK.name}: {len(notebook.cells)} cells preserved.")
