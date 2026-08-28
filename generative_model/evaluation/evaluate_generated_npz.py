"""将模型导出的 canonical compact NPZ 送入统一 P1 评测。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import EvaluationConfig, evaluate, write_report


REQUIRED_ARRAYS = {
    "atomic_numbers",
    "formal_charges",
    "positions",
    "bond_index",
    "bond_types",
    "atom_offsets",
    "bond_offsets",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        missing = sorted(REQUIRED_ARRAYS - set(arrays.files))
        if missing:
            raise ValueError(f"generated NPZ 缺少 arrays: {missing}")
        atom_offsets = arrays["atom_offsets"]
        bond_offsets = arrays["bond_offsets"]
        if atom_offsets.ndim != 1 or bond_offsets.ndim != 1:
            raise ValueError("atom_offsets/bond_offsets 必须为一维")
        if len(atom_offsets) != len(bond_offsets) or len(atom_offsets) < 2:
            raise ValueError("atom_offsets/bond_offsets 长度不一致或没有分子")
        if atom_offsets[0] != 0 or bond_offsets[0] != 0:
            raise ValueError("offsets 必须从 0 开始")
        if np.any(np.diff(atom_offsets) <= 0) or np.any(np.diff(bond_offsets) < 0):
            raise ValueError("atom offsets 必须严格递增，bond offsets 必须单调")
        if atom_offsets[-1] != len(arrays["atomic_numbers"]):
            raise ValueError("atom_offsets 终点与 atomic_numbers 长度不符")
        if atom_offsets[-1] != len(arrays["formal_charges"]) or atom_offsets[-1] != len(arrays["positions"]):
            raise ValueError("atom arrays 长度不一致")
        if arrays["bond_index"].shape != (2, int(bond_offsets[-1])):
            raise ValueError("bond_index shape 与 bond_offsets 不一致")
        if len(arrays["bond_types"]) != bond_offsets[-1]:
            raise ValueError("bond_types 长度与 bond_offsets 不一致")

        samples = []
        for index in range(len(atom_offsets) - 1):
            atom_start, atom_end = map(int, atom_offsets[index : index + 2])
            bond_start, bond_end = map(int, bond_offsets[index : index + 2])
            samples.append(
                {
                    "molecule_id": f"generated_{index:06d}",
                    "atomic_numbers": arrays["atomic_numbers"][atom_start:atom_end].copy(),
                    "formal_charges": arrays["formal_charges"][atom_start:atom_end].copy(),
                    "positions": arrays["positions"][atom_start:atom_end].copy(),
                    "bond_index": arrays["bond_index"][:, bond_start:bond_end].copy(),
                    "bond_types": arrays["bond_types"][bond_start:bond_end].copy(),
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sampling-report", type=Path)
    parser.add_argument("--skip-mmff", action="store_true")
    parser.add_argument("--skip-symmetry", action="store_true")
    args = parser.parse_args()

    input_hash = _sha256_file(args.input)
    provenance: dict[str, Any] = {
        "source_type": "model_generated",
        "source_fingerprint": input_hash,
        "coordinate_unit": "angstrom",
        "model_name": args.model_name,
        "checkpoint_sha256": args.checkpoint_sha256,
    }
    if args.sampling_report is not None:
        provenance["sampling_report_sha256"] = _sha256_file(args.sampling_report)

    report = evaluate(
        load_samples(args.input),
        provenance=provenance,
        report_type="model_generation",
        config=EvaluationConfig(
            run_mmff=not args.skip_mmff,
            run_symmetry=not args.skip_symmetry,
        ),
    )
    write_report(report, args.output)
    print(
        f"{report['summary']['evaluated_molecules']}/{report['summary']['input_molecules']} evaluated; "
        f"status={report['summary']['status']}; fingerprint={report['report_fingerprint']}"
    )


if __name__ == "__main__":
    main()
