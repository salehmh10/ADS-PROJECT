"""Update and cache-execute Stage 5A2 governance reporting cells only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb"
REPORT = ROOT / "artifacts/reports/stage5a2_governance_notebook_execution.json"
PREFLIGHT = ROOT / "artifacts/reports/stage5a2_governance_notebook_guard_preflight.json"
ADJUDICATION = ROOT / "artifacts/reports/stage5a2_governance_adjudication.json"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def atomic_notebook(notebook: Any) -> None:
    temporary = NOTEBOOK.with_suffix(NOTEBOOK.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    nbformat.read(temporary, as_version=4)
    os.replace(temporary, NOTEBOOK)


def static_fit_calls(notebook: Any) -> list[dict[str, Any]]:
    found = []
    blocked = {"fit", "fit_transform", "partial_fit"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"notebook-cell-{index}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in blocked:
                found.append({"cell_index": index, "line": node.lineno, "call": name})
    return found


def update_notebook() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if len(notebook.cells) != 99 or sum(cell.cell_type == "code" for cell in notebook.cells) != 49:
        raise RuntimeError("Unexpected Notebook structure before governance update")

    notebook.cells[51].source = """### 25. Stage 5A2 Objective and Governance

Stage 5A2 is a post-Test extension. It uses saved Train-only scientific evidence. A loader parsed complete source chunks before applying the Train-only row mask, so requested Test-row Feature and target values were temporarily in memory. This violated the literal no-Test-loading rule. The mask was applied before every learned operation. Zero Test rows entered preprocessing fitting, model fitting, selection, Validation metrics, or Validation predictions. No Stage 5 Test prediction was generated. Human adjudication accepts this as a limited procedural exception without demonstrated statistical leakage. Both final bundles remain accepted without refit. Future training must use the stricter parser-boundary Train-only loader."""
    notebook.cells[52].source = """import ast, json, os, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
ROOT=Path.cwd()
MODE=os.environ.get('STAGE5A2_PRESENTATION_MODE','governance_cache_only')
_document=ast.parse((ROOT/'REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb').read_text(encoding='utf-8')) if False else None
_blocked={'fit','fit_transform','partial_fit'}
_notebook_json=json.loads((ROOT/'REGRESSION_PART5_DEEP_TABULAR_MODELS.ipynb').read_text(encoding='utf-8'))
STATIC_FIT_CALLS=[]
for _index,_cell in enumerate(_notebook_json['cells']):
    if _cell.get('cell_type')!='code':
        continue
    _tree=ast.parse(''.join(_cell.get('source',[])),filename=f'cell-{_index}')
    for _node in ast.walk(_tree):
        if isinstance(_node,ast.Call):
            _name=_node.func.attr if isinstance(_node.func,ast.Attribute) else (_node.func.id if isinstance(_node.func,ast.Name) else None)
            if _name in _blocked:
                STATIC_FIT_CALLS.append({'cell':_index,'line':_node.lineno,'call':_name})
assert STATIC_FIT_CALLS==[], STATIC_FIT_CALLS
RUNTIME_FIT_CALLS=[]
def _stage5a2_no_fit_profile(frame,event,arg):
    if event=='call' and frame.f_code.co_name in _blocked:
        RUNTIME_FIT_CALLS.append({'function':frame.f_code.co_name,'file':frame.f_code.co_filename})
        raise RuntimeError('A fit call is prohibited in the Stage 5A2 cache-only Notebook run')
    return _stage5a2_no_fit_profile
sys.setprofile(_stage5a2_no_fit_profile)
g=json.loads((ROOT/'artifacts/reports/stage5a2_governance_adjudication.json').read_text(encoding='utf-8'))
print({'mode':MODE,'stage':'stage5a2','static_fit_calls':len(STATIC_FIT_CALLS),'literal_zero_test_loading':g['classifications']['literal_zero_test_loading'],'procedural_compliance':g['classifications']['procedural_compliance']})"""

    notebook.cells[93].source = """### 46. Required Figures and Governance Registry

The required scientific figures remain unchanged. The Registry contains one idempotent governance-adjudication row after the protected 323-row prefix."""
    notebook.cells[94].source = """f=json.loads((ROOT/'artifacts/reports/stage5a2_figure_manifest.json').read_text(encoding='utf-8'))
r=json.loads((ROOT/'artifacts/reports/stage5a2_governance_registry_update.json').read_text(encoding='utf-8'))
print({'figures':len(f['figures']),'figure_status':f['status'],'registry_status':r['status'],'registry_rows':r['registry_row_count'],'governance_rows':r['governance_row_count'],'prior_prefix_preserved':r['prior_323_row_byte_prefix_preserved']})
display(Image(filename=str(ROOT/'artifacts/figures/stage5a2/stage5a_summary_dashboard.png')))"""

    notebook.cells[95].source = """### 47. Governance Incident and Human Adjudication

The shared loader read full requested-column source chunks before it applied the saved Train-row mask. It temporarily materialized Feature and target values for 99,948 Test rows. This is a literal governance failure. The Train-only mask was applied before numerical medians, missing-value rules, categorical vocabularies, target transformation, internal RealMLP preprocessing, model fitting, selection, Validation metrics, and Validation predictions. Zero Test rows entered those learned or decision operations. No Stage 4L Test metric was used and no Stage 5 Test prediction was generated. The human adjudication accepts the incident as a limited procedural exception without demonstrated statistical leakage. Both final bundles remain valid and no refit is required."""
    notebook.cells[96].source = """g=json.loads((ROOT/'artifacts/reports/stage5a2_governance_adjudication.json').read_text(encoding='utf-8'))
s=json.loads((ROOT/'artifacts/reports/stage5a2_zero_test_loading_claim_supersession.json').read_text(encoding='utf-8'))
m=json.loads((ROOT/'artifacts/reports/stage5a2_learned_membership_audit.json').read_text(encoding='utf-8'))
u=json.loads((ROOT/'artifacts/reports/stage5a2_future_safe_loader_smoke.json').read_text(encoding='utf-8'))
print({'adjudication_id':g['adjudication_id'],'literal_zero_test_loading':g['classifications']['literal_zero_test_loading'],'procedural_compliance':g['classifications']['procedural_compliance'],'statistical_test_leakage':g['classifications']['statistical_test_leakage'],'transient_test_rows':g['test_row_count_transiently_materialized'],'preprocessing_fit_test_rows':g['preprocessing_fit_test_row_count'],'model_fit_test_rows':g['model_fit_test_row_count'],'model_selection_test_rows':g['model_selection_test_row_count'],'test_metrics_used':g['test_metric_use_count'],'stage5_test_predictions':g['stage5_test_prediction_count'],'bundle_validity':g['classifications']['bundle_validity'],'refit_required':g['classifications']['refit_required'],'superseded_literal_claims':len(s['superseded_claims']),'membership_status':m['status'],'future_loader_smoke':u['status']})"""

    notebook.cells[97].source = """### 48. Final Verification and Completion

Stage 5A2 and Stage 5A pass with a documented governance exception only if every scientific and artifact check passes, the literal failure stays visible, the Reviewer accepts the adjudication, no fit call occurs, protected files remain unchanged, and Stage 5B remains unstarted."""
    notebook.cells[98].source = """sys.setprofile(None)
assert RUNTIME_FIT_CALLS==[], RUNTIME_FIT_CALLS
final_path=ROOT/'artifacts/reports/stage5a_verification.json'
d=json.loads(final_path.read_text(encoding='utf-8'))
print({'stage5a_status':d['status'],'literal_zero_test_loading':'FAIL' if not d['literal_zero_test_loading'] else 'PASS','procedural_exception':d['procedural_exception'],'reviewer_status':d['reviewer_result'],'notebook_static_fit_calls':len(STATIC_FIT_CALLS),'notebook_runtime_fit_calls':len(RUNTIME_FIT_CALLS),'stage5b_started':d['stage5b_started'],'next_step':d['next_step']})"""

    calls = static_fit_calls(notebook)
    if calls:
        raise RuntimeError(f"Static no-fit guard failed after Notebook update: {calls}")
    atomic_notebook(notebook)
    print(json.dumps({"status": "UPDATED", "cells": 99, "code_cells": 49, "static_fit_calls": 0}))


def snapshot() -> dict[str, str]:
    paths = [
        ROOT / "artifacts/models/deep/core_final/components/stage5a2__realmlp__full_train__without_sensitive__direct_no_refit_recovery2_model.joblib",
        ROOT / "artifacts/models/deep/core_final/stage5a2__realmlp__full_train__without_sensitive__direct_no_refit_recovery2.joblib",
        ROOT / "artifacts/models/deep/core_final/components/stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30__technical_retry1_model.joblib",
        ROOT / "artifacts/models/deep/core_final/stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30__technical_retry1.joblib",
        ADJUDICATION,
        REGISTRY,
    ]
    paths.extend(sorted((ROOT / "artifacts/predictions/stage5").rglob("*.csv")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def execute_notebook() -> None:
    verification = json.loads((ROOT / "artifacts/reports/stage5a_verification.json").read_text(encoding="utf-8"))
    reviewer = json.loads((ROOT / "artifacts/reports/stage5a_governance_reviewer.json").read_text(encoding="utf-8"))
    if verification["status"] != "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION" or reviewer["status"] != "PASS":
        raise RuntimeError("Final verification and independent review must pass before Notebook execution")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    calls = static_fit_calls(notebook)
    if calls:
        raise RuntimeError(f"External static no-fit guard failed: {calls}")
    before = snapshot()
    registry_before = pd.read_csv(REGISTRY)
    preflight = {
        "adjudication_id": "stage5a2_governance_adjudication_1", "recorded_at": now(),
        "status": "PASS", "execution_mode": "governance_cache_only",
        "static_fit_calls": calls, "model_bundle_prediction_governance_registry_hashes": before,
        "registry_rows": int(len(registry_before)),
        "registry_unique_ids": int(registry_before["experiment_id"].nunique()),
    }
    atomic_json(preflight, PREFLIGHT)
    started = time.perf_counter()
    old_mode = os.environ.get("STAGE5A2_PRESENTATION_MODE")
    os.environ["STAGE5A2_PRESENTATION_MODE"] = "governance_cache_only"
    try:
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False)
        executed = client.execute(cwd=str(ROOT))
    finally:
        if old_mode is None:
            os.environ.pop("STAGE5A2_PRESENTATION_MODE", None)
        else:
            os.environ["STAGE5A2_PRESENTATION_MODE"] = old_mode
    elapsed = time.perf_counter() - started
    atomic_notebook(executed)
    after = snapshot()
    registry_after = pd.read_csv(REGISTRY)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    final_text = "\n".join(
        output.get("text", "") for output in code_cells[-1].get("outputs", []) if output.get("output_type") == "stream"
    )
    checks = {
        "total_cells_99": len(executed.cells) == 99,
        "code_cells_49": len(code_cells) == 49,
        "all_code_cells_have_outputs": all(bool(cell.get("outputs")) for cell in code_cells),
        "zero_error_outputs": len(errors) == 0,
        "zero_static_fit_calls": len(calls) == 0,
        "zero_runtime_fit_calls_displayed": "'notebook_runtime_fit_calls': 0" in final_text,
        "pass_with_exception_displayed": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION" in final_text,
        "literal_failure_displayed": "'literal_zero_test_loading': 'FAIL'" in final_text,
        "protected_runtime_hashes_unchanged": before == after,
        "registry_row_count_unchanged": len(registry_after) == len(registry_before) == 324,
        "registry_ids_unique": registry_after["experiment_id"].nunique() == len(registry_after),
        "governance_row_exactly_one": int((registry_after["experiment_id"] == "stage5a2_governance_adjudication_1").sum()) == 1,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "adjudication_id": "stage5a2_governance_adjudication_1",
        "attempt": 1, "maximum_attempts": 2, "recorded_at": now(), "status": status,
        "elapsed_seconds": elapsed, "checks": checks,
        "new_model_fits": 0, "new_preprocessing_fits": 0,
        "new_prediction_generations": 0, "model_bundle_prediction_changes": [],
        "before_hashes": before, "after_hashes": after,
        "notebook_sha256": sha256(NOTEBOOK),
    }
    atomic_json(report, REPORT)
    if status != "PASS":
        raise RuntimeError(f"Governance cache-only Notebook execution failed: {checks}")
    print(json.dumps({"status": status, "attempt": 1, "elapsed_seconds": elapsed, "notebook_sha256": report["notebook_sha256"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["update", "execute"])
    action = parser.parse_args().action
    if action == "update":
        update_notebook()
    else:
        execute_notebook()


if __name__ == "__main__":
    main()

