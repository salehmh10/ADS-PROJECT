"""Perform external checks on the saved executed Prompt 1 notebook."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import nbformat


root = Path(__file__).resolve().parent
notebook_path = root / "REGRESSION_PART2_MODELING.ipynb"
verification_path = root / "artifacts" / "reports" / "prompt1_verification.json"
reviewer_path = root / "artifacts" / "reports" / "prompt1_reviewer.md"

notebook = nbformat.read(notebook_path, as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
headings = {
    line.strip()
    for cell in notebook.cells
    if cell.cell_type == "markdown"
    for line in cell.source.splitlines()
    if line.startswith("## ")
}
required_headings = {f"## {number}. {title}" for number, title in [
    (0, "Project Objective"), (1, "Imports and Configuration"), (2, "Project and File Discovery"),
    (3, "Source Data Protection"), (4, "Data Loading"), (5, "Data Contract Validation"),
    (6, "Target and Feature Review"), (7, "Sensitive Feature Comparison"),
    (8, "Leakage and Suspicious-Column Checks"), (9, "Shared Train-Test Split"),
    (10, "Shared Cross-Validation Folds"), (11, "Regression Evaluation Metrics"),
    (12, "Metric Unit Tests"), (13, "Experiment Result Registry"),
    (14, "Saved Artifacts"), (15, "Verification Summary"),
]}

report = json.loads(verification_path.read_text(encoding="utf-8"))
before = json.loads((root / "artifacts" / "data_contract" / "source_hashes_before.json").read_text(encoding="utf-8"))
after = json.loads((root / "artifacts" / "data_contract" / "source_hashes_after.json").read_text(encoding="utf-8"))

def current_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

state = {name: (root / name).read_text(encoding="utf-8") for name in ["AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md"]}
external_checks = {
    "notebook_execution": bool(code_cells) and all(cell.execution_count is not None for cell in code_cells),
    "notebook_error_outputs_zero": len(error_outputs) == 0,
    "all_required_notebook_sections": required_headings.issubset(headings),
    "internal_verification_passed": report.get("status") == "INTERNAL_PASS_PENDING_EXTERNAL_EXECUTION" and all(report["checks"].values()),
    "source_fingerprints_match_current_files": all(
        before[name]["sha256"] == after[name]["sha256"] == current_hash(after[name]["resolved_path"])
        for name in before
    ),
    "reviewer_report_and_adjudication_exist": reviewer_path.is_file() and "Disposition: **Accepted.**" in reviewer_path.read_text(encoding="utf-8"),
    "final_state_files_current": "Prompt 1 is fully complete" in state["TASK.md"] and "Final full execution" in state["LOG.md"],
    "verification_report_exists": verification_path.is_file(),
}
report["external_execution_verification"] = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "checks": external_checks,
    "executed_code_cell_count": len(code_cells),
    "error_output_count": len(error_outputs),
}
report["checks"]["notebook_execution"] = external_checks["notebook_execution"] and external_checks["notebook_error_outputs_zero"]
report["checks"]["required_markdown_files_updated"] = external_checks["final_state_files_current"]
report["status"] = "PASS" if all(report["checks"].values()) and all(external_checks.values()) else "FAIL"
verification_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"status": report["status"], "external_checks": external_checks}, indent=2))
if report["status"] != "PASS":
    raise SystemExit(1)
