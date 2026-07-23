"""Append or refresh only owned Stage 4B cells in the foundation notebook."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


NOTEBOOK = "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb"
STAGE4A_BACKUP = "artifacts/backups/REGRESSION_PART4_BOOSTING_FOUNDATION_recovery_run2_20260714_135751.ipynb"
STAGE4A_CELL_COUNT = 39
TAG = "stage4b_owned"


SECTIONS = (
    (
        19,
        "Stage 4B Objective",
        """Stage 4B creates safe initial Feature Packs for CatBoost, LightGBM, and XGBoost. It uses saved Train rows only. It does not compare models or tune settings.""",
        """stage4b_summary = b.build_stage4b_artifacts(ROOT)
assert stage4b_summary["status"] == "PASS"
stage4b_start = json.loads((ROOT / "artifacts/reports/stage4b_start_validation.json").read_text(encoding="utf-8"))
display(pd.DataFrame([stage4b_start["checks"]]).T.rename(columns={0: "passed"}))
display(stage4b_summary)""",
    ),
    (
        20,
        "Existing Feature Review",
        """The review uses the source inventory, Stage 1 features, Stage 3 proposals, selected packs, importance, and leakage reports. Strong prior signals include applicant income, lien status, state, area income, and tract income ratio.""",
        """feature_audit = pd.read_csv(ROOT / "artifacts/features/stage4/feature_audit.csv")
stage3_importance = pd.read_csv(ROOT / "artifacts/features/tree/importance/stage3_feature_importance_summary.csv")
display(feature_audit[["feature_name", "source_or_engineered", "cardinality", "missing_rate", "decision"]])
display(stage3_importance.loc[stage3_importance["sensitive_mode"].eq("without_sensitive")].head(15))""",
    ),
    (
        21,
        "Leakage and Redundancy Audit",
        """The target, row ID, target bins, Fold IDs, sensitive interactions, exact duplicates, and post-outcome information are excluded. Lender and geography fields may be proxy signals, so later results must stay associative.""",
        """leakage_text = (ROOT / "artifacts/reports/stage4b_leakage_review.md").read_text(encoding="utf-8")
assert "Status: PASS" in leakage_text
display(Markdown(leakage_text))""",
    ),
    (
        22,
        "Initial Feature Proposals",
        """Twelve fixed features were reviewed. Eight were selected. Four were rejected because they repeat Stage 3 work or duplicate existing information.""",
        """initial_feature_proposals = pd.read_csv(ROOT / "artifacts/features/stage4/initial_feature_proposals.csv")
assert len(initial_feature_proposals) <= 12
display(initial_feature_proposals)""",
    ),
    (
        23,
        "Common Base Pack",
        """`boosting_base_v1` keeps reviewed original and useful Stage 1 fields. It removes the target, IDs used for splitting, sensitive fields, and exact tree duplicates.""",
        """boosting_packs = json.loads((ROOT / "artifacts/features/stage4/boosting_feature_packs.json").read_text(encoding="utf-8"))
base_pack = boosting_packs["packs"]["boosting_base_v1"]
assert "loan_amount_000s" not in base_pack["raw"]
display(base_pack)""",
    ),
    (
        24,
        "Initial Engineered Pack",
        """`boosting_engineered_v1` adds two numeric fields and six small category combinations. These are fixed row-level calculations and use no learned statistics.""",
        """engineered_pack = boosting_packs["packs"]["boosting_engineered_v1"]
assert len(engineered_pack["fixed_features"]) == 8
display(engineered_pack)""",
    ),
    (
        25,
        "CatBoost Native Pack",
        """`catboost_native_v1` keeps lender, metro area, county, tract, state, and other categories as native CatBoost text fields. Rare grouping is learned only inside the Pipeline.""",
        """catboost_schema = json.loads((ROOT / "artifacts/features/stage4/catboost_feature_schema.json").read_text(encoding="utf-8"))
assert catboost_schema["categorical_strategy"].startswith("native CatBoost")
display(catboost_schema)""",
    ),
    (
        26,
        "LightGBM Pack",
        """`lightgbm_encoded_v1` uses fold-fit frequency values for high-cardinality fields and fold-fit ordinal values for the remaining categories. Both steps stay inside the Pipeline.""",
        """lightgbm_schema = json.loads((ROOT / "artifacts/features/stage4/lightgbm_feature_schema.json").read_text(encoding="utf-8"))
assert "pipeline" in lightgbm_schema["categorical_strategy"]
display(lightgbm_schema)""",
    ),
    (
        27,
        "XGBoost Pack",
        """`xgboost_sparse_v1` uses sparse one-hot output for controlled categories. High-cardinality fields use fold-fit frequency values, so the matrix does not become uncontrolled and dense.""",
        """xgboost_schema = json.loads((ROOT / "artifacts/features/stage4/xgboost_feature_schema.json").read_text(encoding="utf-8"))
assert xgboost_schema["sparse_output"] is True
display(xgboost_schema)""",
    ),
    (
        28,
        "Serializable Feature Transformers",
        """Five named transformers accept pandas DataFrames, keep row order, avoid source mutation, handle unseen categories, and reload in a clean process.""",
        """transformer_roundtrips = pd.read_csv(ROOT / "artifacts/features/stage4/transformer_roundtrip_results.csv")
assert transformer_roundtrips["status"].eq("PASS").all()
display(transformer_roundtrips)""",
    ),
    (
        29,
        "Compatibility Smoke Tests",
        """Each available package fits only five trees or iterations on 4,000 saved Train rows and predicts 1,000 other saved Train rows. These tests check compatibility only. They are not model screening.""",
        """smoke_tests = json.loads((ROOT / "artifacts/reports/stage4b_smoke_tests.json").read_text(encoding="utf-8"))
assert smoke_tests["status"] == "PASS"
smoke_table = pd.DataFrame(smoke_tests["results"]).T.reset_index(drop=True)
display(smoke_table[["model", "fit_rows", "validation_rows", "fit_seconds", "finite_predictions", "feature_names_stable", "serialization_reload_match", "representation_sparse", "status"]])""",
    ),
    (
        30,
        "Stage 4B Artifact Summary",
        """The Stage 4B files record the audit, proposals, five Feature Packs, three schemas, transformer checks, leakage review, and bounded smoke evidence.""",
        """artifact_summary = json.loads((ROOT / "artifacts/manifests/stage4/stage4b_artifact_summary.json").read_text(encoding="utf-8"))
display(artifact_summary)""",
    ),
    (
        31,
        "Stage 4B Verification",
        """Internal verification checks leakage, pack design, serialization, smoke limits, Test exclusion, artifact completeness, and protected hashes. Final completion also needs two matching notebook runs and independent review.""",
        """stage4b_internal = b.build_internal_verification(ROOT)
assert stage4b_internal["status"] == "PASS"
display(pd.DataFrame([stage4b_internal["checks"]]).T.rename(columns={0: "passed"}))""",
    ),
    (
        32,
        "Stage 4B Completion Note",
        """The initial Feature Packs are ready for the first CatBoost experiment after external execution and review checks pass. No real boosting comparison or tuning was performed.""",
        """completion_note = {
    "stage": "Stage 4B — Initial Boosting Feature Packs",
    "implementation_status": "PASS",
    "feature_packs": list(boosting_packs["packs"]),
    "selected_fixed_features": int(initial_feature_proposals["selected"].sum()),
    "test_rows_used": 0,
    "real_screening_performed": False,
    "next_step": "Begin Stage 4C — Initial CatBoost Model and Importance Analysis.",
}
display(completion_note)""",
    ),
)


def _owned_metadata(section: int) -> dict:
    return {"tags": [TAG], "stage4b_section": section, "stage4b_version": "v1"}


def desired_cells() -> list:
    cells = []
    for section, title, explanation, code in SECTIONS:
        markdown = new_markdown_cell(
            source=f"## {section}. {title}\n\n{explanation}",
            metadata=_owned_metadata(section),
        )
        markdown["id"] = f"stage4b-{section}-markdown"
        code_cell = new_code_cell(source=code, metadata=_owned_metadata(section))
        code_cell["id"] = f"stage4b-{section}-code"
        cells.extend([markdown, code_cell])
    return cells


def prepare(root: str | Path = ".") -> Path:
    project = Path(root).resolve()
    notebook_path = project / NOTEBOOK
    backup_path = project / STAGE4A_BACKUP
    notebook = nbformat.read(notebook_path, as_version=4)
    backup = nbformat.read(backup_path, as_version=4)
    if len(notebook.cells) < STAGE4A_CELL_COUNT or notebook.cells[:STAGE4A_CELL_COUNT] != backup.cells[:STAGE4A_CELL_COUNT]:
        raise AssertionError("The finalized Stage 4A notebook prefix changed.")
    desired = desired_cells()
    desired_by_id = {cell["id"]: cell for cell in desired}
    present = [cell for cell in notebook.cells if TAG in cell.get("metadata", {}).get("tags", [])]
    present_ids = [cell.get("id") for cell in present]
    unowned_markdown = "\n".join(
        cell.source for cell in notebook.cells
        if cell.cell_type == "markdown" and TAG not in cell.get("metadata", {}).get("tags", [])
    )
    if not present:
        if any(f"## {number}." in unowned_markdown for number in range(19, 33)):
            raise AssertionError("An unowned Stage 4B heading already exists.")
        notebook.cells.extend(desired)
    else:
        if len(present_ids) != len(set(present_ids)) or set(present_ids) != set(desired_by_id):
            raise AssertionError("The owned Stage 4B block is partial, duplicated, or malformed.")
        for index, cell in enumerate(notebook.cells):
            cell_id = cell.get("id")
            if cell_id in desired_by_id:
                replacement = desired_by_id[cell_id]
                if cell.cell_type != replacement.cell_type:
                    raise AssertionError(f"Owned cell type changed: {cell_id}")
                cell.source = replacement.source
                cell.metadata = replacement.metadata
    notebook.metadata["stage4b"] = {
        "version": "stage4b_initial_boosting_packs_v1_20260714",
        "owned_tag": TAG,
        "section_range": [19, 32],
        "stage4a_prefix_cells": STAGE4A_CELL_COUNT,
    }
    temporary = notebook_path.with_suffix(".stage4b.tmp.ipynb")
    nbformat.write(notebook, temporary)
    os.replace(temporary, notebook_path)
    return notebook_path


if __name__ == "__main__":
    print(prepare())
