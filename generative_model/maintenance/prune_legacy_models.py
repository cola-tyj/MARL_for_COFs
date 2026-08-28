"""Prune superseded SemlaFlow, MiDi and UAE-3D experiment artifacts.

Dry-run is the default.  Pass ``--apply`` only after reviewing the printed
inventory.  The cleanup preserves frozen JSON reports and the minimum runnable
entry points documented in ``generative_model/models/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "generative_model/maintenance/legacy_cleanup_manifest_20260828.json"

DELETE_TREES = (
    "generative_model/checkpoints/midi",
    "generative_model/checkpoints/semlaflow",
    "generative_model/external/midi/data",
    "generative_model/runs/midi_overfit1",
    "generative_model/runs/midi_overfit32",
    "generative_model/runs/midi_overfit4",
    "generative_model/runs/midi_zero_shot_geom_approx_5seed",
    "generative_model/runs/midi_zero_shot_geom_approx_stratified100",
    "generative_model/runs/semlaflow_official",
    "generative_model/runs/semlaflow_overfit1",
    "generative_model/runs/semlaflow_overfit32",
    "generative_model/runs/semlaflow_overfit4",
    "generative_model/runs/uae3d_engineering",
    "generative_model/runs/uae3d_explicit_engineering",
    "generative_model/runs/uae3d_overfit1",
    "generative_model/configs/midi",
)

UAE_OVERFIT4_DELETE_DIRS = (
    "bond_refine_w10_coord10_lr1em5_0256",
    "explicit_bond_head_0064_lr1em4",
    "hard_pair_v3_engineering_step1_cpu",
    "hard_pair_v3_engineering_step1_cpu_repeat",
    "hard_pair_v3_pcgrad_0032",
    "hard_pair_v3_pcgrad_engineering_step1_cpu",
    "hard_pair_v3_pcgrad_engineering_step1_cpu_repeat",
    "hierarchical_v4_balanced_0016",
    "hierarchical_v4_balanced_engineering_step1_cpu",
    "hierarchical_v4_balanced_engineering_step1_cpu_repeat",
    "hierarchical_v4_engineering_step1_cpu",
    "hierarchical_v4_engineering_step1_cpu_repeat",
)

KEEP_MODEL_SCRIPTS = {
    "midi_bridge.py",
    "midi_cof_runtime.py",
    "semlaflow_bridge.py",
    "uae3d_bond_calibration.py",
    "uae3d_bond_capacity.py",
    "uae3d_bridge.py",
    "uae3d_gate_a.py",
    "uae3d_hierarchical_bonds.py",
    "uae3d_objectives.py",
    "uae3d_reconstruction.py",
}

KEEP_SMOKE_SCRIPTS = {
    "audit_uae3d_bond_capacity.py",
    "audit_uae3d_c23_reconstruction.py",
    "audit_uae3d_gate_a.py",
    "build_uae3d_c23_readonly_protocol.py",
    "build_uae3d_gate_a_protocol.py",
    "build_uae3d_split.py",
    "finalize_uae3d_c23_readonly_audit.py",
    "run_midi_forward.py",
    "run_semlaflow_forward.py",
    "run_uae3d_forward.py",
    "train_uae3d_reconstruction.py",
}

KEEP_TESTS = {
    "test_midi_bridge.py",
    "test_uae3d_bond_capacity.py",
    "test_uae3d_bridge.py",
    "test_uae3d_gate_a.py",
}


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise RuntimeError(f"拒绝处理项目根目录外或根目录本身: {resolved}")
    return resolved


def _size(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(item.lstat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_named_scripts(directory: Path, keep: set[str]) -> Iterable[Path]:
    for path in sorted(directory.glob("*.py")):
        lower = path.name.lower()
        if any(token in lower for token in ("midi", "semlaflow", "uae3d")):
            if path.name not in keep:
                yield path


def candidates() -> list[Path]:
    paths: list[Path] = [ROOT / relative for relative in DELETE_TREES]
    uae_root = ROOT / "generative_model/runs/uae3d_overfit4"
    paths.extend(uae_root / name for name in UAE_OVERFIT4_DELETE_DIRS)

    final_run = uae_root / "from_tier1_w10_lr3em5_0512"
    paths.append(final_run / "last.pt")
    checkpoint_dir = final_run / "checkpoints"
    paths.extend(
        path for path in sorted(checkpoint_dir.glob("step-*.pt"))
        if path.name != "step-003008.pt"
    )

    paths.extend(_legacy_named_scripts(ROOT / "generative_model/models", KEEP_MODEL_SCRIPTS))
    paths.extend(_legacy_named_scripts(ROOT / "generative_model/smoke", KEEP_SMOKE_SCRIPTS))
    paths.extend(_legacy_named_scripts(ROOT / "generative_model/tests", KEEP_TESTS))
    paths.append(ROOT / "generative_model/models/UAE3D_PLAN.md")

    statistics = ROOT / "generative_model/smoke/statistics"
    paths.extend(sorted(statistics.glob("midi*.json")))

    paths.extend(sorted((ROOT / "generative_model").rglob("__pycache__")))
    paths.extend(sorted((ROOT / "generative_model").rglob("*.py[co]")))

    unique: dict[str, Path] = {}
    for path in paths:
        resolved = _inside_root(path)
        if resolved.exists() or resolved.is_symlink():
            unique[str(resolved)] = resolved
    # When a parent directory is already selected, do not separately list children.
    ordered = sorted(unique.values(), key=lambda item: (len(item.parts), str(item)))
    selected: list[Path] = []
    for path in ordered:
        if not any(parent == path or parent in path.parents for parent in selected):
            selected.append(path)
    return selected


def _remove(path: Path) -> None:
    _inside_root(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="execute the reviewed deletion")
    args = parser.parse_args()
    selected = candidates()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "kind": "directory" if path.is_dir() else "file",
            "bytes": _size(path),
        }
        for path in selected
    ]
    reclaimable = sum(record["bytes"] for record in records)
    payload = {
        "schema_version": "legacy-model-cleanup-v1",
        "status": "DRY_RUN" if not args.apply else "APPLIED",
        "policy": {
            "models": ["SemlaFlow", "MiDi", "UAE-3D"],
            "frozen_json_reports_preserved": True,
            "canonical_data_preserved": True,
            "our_etflow_preserved": True,
            "midi_and_semlaflow_checkpoints_preserved": False,
            "uae3d_step_3008_checkpoint_preserved": True,
        },
        "candidate_count": len(records),
        "reclaimable_bytes": reclaimable,
        "targets": records,
        "retained_entrypoints": {
            "models": sorted(KEEP_MODEL_SCRIPTS),
            "smoke": sorted(KEEP_SMOKE_SCRIPTS),
            "tests": sorted(KEEP_TESTS),
        },
        "retained_uae_checkpoint": {
            "path": (
                "generative_model/runs/uae3d_overfit4/from_tier1_w10_lr3em5_0512/"
                "checkpoints/step-003008.pt"
            ),
            "sha256": _sha256(
                ROOT
                / "generative_model/runs/uae3d_overfit4/from_tier1_w10_lr3em5_0512/"
                "checkpoints/step-003008.pt"
            ),
        },
        "cleanup_script_sha256": _sha256(Path(__file__)),
    }
    if args.apply:
        for path in selected:
            _remove(path)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
