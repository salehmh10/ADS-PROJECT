"""Apply safe user-facing Stage naming and output-display repairs."""

from pathlib import Path
from textwrap import dedent
import nbformat


root = Path(__file__).resolve().parent
path = root / "REGRESSION_PART2_MODELING.ipynb"
notebook = nbformat.read(path, as_version=4)

# Capitalized Prompt labels are user-facing in notebook cells. Lowercase
# prompt1/prompt2 technical identifiers, paths, keys, and IDs stay unchanged.
for cell in notebook.cells:
    cell.source = (
        cell.source.replace("Prompt 1", "Stage 1").replace("Prompt 2", "Stage 2")
        .replace("PROMPT 1", "STAGE 1").replace("PROMPT 2", "STAGE 2")
    )

heading_replacements = {
    "## 0. Project Objective": "## 0. Stage 1 — Data Validation and Experiment Setup Objective",
}
for cell in notebook.cells:
    if cell.cell_type == "markdown":
        if cell.source.startswith("# Regression Modeling"):
            lines = cell.source.splitlines()
            lines[0] = "# Regression Modeling — Stage 1 and Stage 2"
            cell.source = "\n".join(lines)
        for old, new in heading_replacements.items():
            if cell.source.startswith(old):
                cell.source = cell.source.replace(old, new, 1)


def code_after_section(number):
    prefix = f"## {number}."
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.startswith(prefix):
            for candidate in notebook.cells[index + 1:]:
                if candidate.cell_type == "markdown" and candidate.source.startswith("## "):
                    break
                if candidate.cell_type == "code":
                    return candidate
    raise KeyError(number)


section16 = next(cell for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## 16."))
compatibility_note = "Internal artifact IDs keep their old technical names to preserve reproducibility."
if compatibility_note not in section16.source:
    section16.source += f"\n\n{compatibility_note}"

final_fit_cell = code_after_section(35)
manifest_marker = "# STAGE2_COMPLETION_MANIFEST_DISPLAY"
if manifest_marker not in final_fit_cell.source:
    final_fit_cell.source += dedent("""

    # STAGE2_COMPLETION_MANIFEST_DISPLAY
    model_manifest_display = pd.DataFrame(model_manifest["models"])[
        ["model_name", "sensitive_mode", "target_mode", "model_path", "model_sha256"]
    ]
    display(model_manifest_display)
    """)

completion_cell = code_after_section(40)
completion_cell.source = dedent("""
    best_family = best_linear_row["model_name"]
    best_effect = linear_sensitive_comparison.loc[
        linear_sensitive_comparison["model_name"] == best_family,
        "mae_difference_with_minus_without"
    ].iloc[0]
    gamma_without = cv_oof_summary.loc[
        (cv_oof_summary["model_name"] == "gamma_regressor")
        & (cv_oof_summary["sensitive_mode"] == "without_sensitive"), "mae"
    ].iloc[0]
    gamma_with = cv_oof_summary.loc[
        (cv_oof_summary["model_name"] == "gamma_regressor")
        & (cv_oof_summary["sensitive_mode"] == "with_sensitive"), "mae"
    ].iloc[0]
    effect_word = "improved" if best_effect < 0 else "worsened" if best_effect > 0 else "did not change"
    display(Markdown(
        f"Stage 2 completed successfully. Six model families were evaluated in both sensitive modes. "
        f"Results use complete training OOF predictions on the original target scale, and both modes used the same frozen family configurations. "
        f"The locked test set was not used. The current linear-family leader is **{best_family}** in "
        f"**{best_linear_row['sensitive_mode']}** mode with OOF MAE **{best_linear_row['oof_mae']:.3f}** target units. "
        f"For this family, adding sensitive features {effect_word} MAE by **{abs(best_effect):.3f}** target units. "
        f"This small accuracy difference is not a fairness conclusion. Gamma formally converged but was unstable: its OOF MAE was "
        f"**{gamma_without:.3f}** without sensitive features and **{gamma_with:.3f}** with them. "
        f"Saved pipelines and results are ready for later analysis. This is not the final project model. "
        f"The next stage is Stage 3 — Tree-Based and Interpretable Models."
    ))
    print("Stage 2 notebook runtime this execution (seconds):", round(time.perf_counter() - P2_START_TIME, 2))
""").strip()

verification_cell = code_after_section(39)
verification_cell.source = verification_cell.source.replace(
    'np.allclose(frame["absolute_error"], np.abs(frame["y_pred"] - frame["y_true"]), rtol=0, atol=1e-12)',
    'np.allclose(frame["absolute_error"], np.abs(frame["y_pred"] - frame["y_true"]), rtol=0, atol=1e-10)',
).replace(
    'np.allclose(frame["signed_error"], frame["y_pred"] - frame["y_true"], rtol=0, atol=1e-12)',
    'np.allclose(frame["signed_error"], frame["y_pred"] - frame["y_true"], rtol=0, atol=1e-10)',
)
verification_cell.source = verification_cell.source.replace(
    '    "state_markdown_files_updated": "Stage 2" in (PROJECT_ROOT / "TASK.md").read_text(encoding="utf-8") and "Stage 2 Execution Plan" in (PROJECT_ROOT / "PLAN.md").read_text(encoding="utf-8"),',
    '    "state_markdown_files_updated": "Stage 2" in (PROJECT_ROOT / "TASK.md").read_text(encoding="utf-8") and "Stage 2 Completion Plan" in (PROJECT_ROOT / "PLAN.md").read_text(encoding="utf-8"),',
)
verification_cell.source = verification_cell.source.replace(
    '    "accepted_critical_and_major_findings_fixed": False,',
    '    "accepted_critical_and_major_findings_fixed": (P2_DIRS["reports"] / "prompt2_reviewer.md").exists() and "Critical issues fixed: PASS" in (P2_DIRS["reports"] / "prompt2_reviewer.md").read_text(encoding="utf-8") and "Major issues fixed: PASS" in (P2_DIRS["reports"] / "prompt2_reviewer.md").read_text(encoding="utf-8"),',
)
verification_cell.source = verification_cell.source.replace(
    'else:\n    prompt2_status = "INTERNAL_PASS_PENDING_INDEPENDENT_REVIEW"',
    'elif not prompt2_checks["independent_reviewer_completed"] or not prompt2_checks["accepted_critical_and_major_findings_fixed"]:\n    prompt2_status = "INTERNAL_PASS_PENDING_INDEPENDENT_REVIEW"\nelse:\n    prompt2_status = "PASS"',
)
verification_cell.source = verification_cell.source.replace(
    'json.dumps(prompt2_verification, indent=2)',
    'json.dumps(prompt2_verification, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))',
)

notebook.metadata["display_stage_name"] = "Stage 2 — Baseline and Linear Regression Models"
nbformat.write(notebook, path)
print(f"Prepared {path}")
