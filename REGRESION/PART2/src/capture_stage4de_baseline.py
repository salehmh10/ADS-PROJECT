"""Capture immutable inputs before Stage 4D–E implementation."""

from __future__ import annotations

import json
from pathlib import Path

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/manifests/stage4/catboost/stage4de_protected_hashes_before.json"
NOTEBOOK_BACKUP = ROOT / "artifacts/backups/REGRESSION_PART4_CATBOOST_before_stage4de_20260714_184951.ipynb"


def capture() -> dict:
    if OUTPUT.is_file():
        return json.loads(OUTPUT.read_text(encoding="utf-8"))
    prior = c4.read_json(ROOT / "artifacts/manifests/stage4/catboost/stage4c_protected_hashes_before.json")
    candidates: set[Path] = set()
    for name in prior["hashes"]:
        item = Path(name)
        candidates.add(item if item.is_absolute() else ROOT / item)
    candidates.add(NOTEBOOK_BACKUP)
    for name in (
        "stage4_catboost_utils.py", "stage4_catboost_worker.py", "run_stage4c.py",
        "build_stage4c_notebook.py", "run_stage4c_notebook.py", "run_stage4c_recovery.py",
        "finalize_stage4c.py",
    ):
        candidates.add(ROOT / name)
    for pattern in (
        "artifacts/results/stage4/catboost/initial/**/*",
        "artifacts/predictions/catboost/initial/**/*",
        "artifacts/models/catboost/preliminary/**/*",
        "artifacts/models/catboost/initial_candidates/**/*",
        "artifacts/features/stage4/catboost/*",
        "artifacts/figures/stage4/catboost/*",
        "artifacts/checkpoints/stage4/catboost/*",
        "artifacts/manifests/stage4/catboost/stage4c*",
        "artifacts/manifests/stage4/catboost/catboost*",
        "artifacts/reports/stage4c*",
    ):
        candidates.update(path for path in ROOT.glob(pattern) if path.is_file())
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 4D–E protected inputs are missing: {missing[:10]}")
    hashes = {}
    sizes = {}
    for path in sorted(candidates, key=lambda value: str(value).lower()):
        try:
            name = str(path.resolve().relative_to(ROOT))
        except ValueError:
            name = str(path.resolve())
        hashes[name] = s4.sha256_file(path)
        sizes[name] = path.stat().st_size
    result = {
        "stage": "stage4de",
        "created_at_utc": s4.utc_now(),
        "file_count": len(hashes),
        "hashes": hashes,
        "sizes": sizes,
        "stage4c_notebook_backup": str(NOTEBOOK_BACKUP.relative_to(ROOT)),
        "stage4c_notebook_backup_sha256": s4.sha256_file(NOTEBOOK_BACKUP),
        "status": "PASS",
    }
    s4.atomic_write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
