"""为冻结的 canonical dataset 生成 IID/Core-OOD 参考评测报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from generative_model.data import COFSymmetryDataset

from .chemistry import build_reference
from .evaluator import EvaluationConfig, evaluate, write_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = REPOSITORY_ROOT / "generative_model/data/processed/v2"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "generative_model/evaluation/reports/v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        # 外部包不写入机器相关的绝对路径，只保留稳定目录名。
        return resolved.name


def _source_fingerprint(
    dataset_fingerprint: str,
    split_file_sha256: str,
    split_scheme: str,
    split: str,
) -> str:
    value = {
        "dataset_fingerprint": dataset_fingerprint,
        "split_file_sha256": split_file_sha256,
        "split_scheme": split_scheme,
        "split": split,
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_reference_report(
    package_dir: str | Path,
    *,
    split_scheme: str,
    split: str = "test",
) -> dict[str, Any]:
    package_path = Path(package_dir)
    dataset = COFSymmetryDataset(package_path, split=split, split_scheme=split_scheme)
    split_path = package_path / f"split_{split_scheme}.json"
    split_hash = _sha256_file(split_path)
    dataset_fingerprint = str(dataset.manifest["dataset_fingerprint"])
    config = EvaluationConfig()
    split_data = json.loads(split_path.read_text(encoding="utf-8"))["indices"]
    train_indices = list(map(int, split_data["train"]))
    reference = build_reference(
        dataset.metadata.iloc[train_indices]["SMILES"].astype(str),
        radius=config.fingerprint_radius,
        bits=config.fingerprint_bits,
        include_chirality=config.fingerprint_include_chirality,
        identity={
            "dataset_fingerprint": dataset_fingerprint,
            "split_file_sha256": split_hash,
            "split_scheme": split_scheme,
            "split": "train",
        },
    )
    provenance = {
        "source_type": "canonical_dataset",
        "source_fingerprint": _source_fingerprint(
            dataset_fingerprint, split_hash, split_scheme, split
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "package_schema_version": str(dataset.manifest["schema_version"]),
        "package_path": _relative_path(package_path),
        "coordinate_unit": str(dataset.manifest["coordinate"]["unit"]),
        "split_scheme": split_scheme,
        "split": split,
        "split_file_sha256": split_hash,
    }
    samples = (dataset[index] for index in range(len(dataset)))
    return evaluate(
        samples,
        provenance=provenance,
        report_type="reference_dataset",
        config=config,
        reference=reference,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--split-schemes",
        nargs="+",
        choices=("iid", "core_ood"),
        default=("iid", "core_ood"),
    )
    args = parser.parse_args()

    for split_scheme in args.split_schemes:
        report = build_reference_report(
            args.package,
            split_scheme=split_scheme,
            split=args.split,
        )
        output_path = args.output_dir / split_scheme / f"reference_{args.split}.json"
        write_report(report, output_path)
        print(
            f"{split_scheme}/{args.split}: {report['summary']['status']} "
            f"({report['summary']['evaluated_molecules']} molecules) -> {output_path}"
        )


if __name__ == "__main__":
    main()
