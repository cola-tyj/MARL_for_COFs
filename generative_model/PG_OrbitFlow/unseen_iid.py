"""Strict helpers for frozen, read-only unseen IID panels."""

from __future__ import annotations

from .data import PGOrbitFlowDataset


def samples_for_split_records(protocol: dict, *, split: str):
    """Resolve frozen records from one canonical split without fallback."""

    dataset = PGOrbitFlowDataset(
        protocol["package_dir"],
        split=split,
        split_scheme="iid",
        point_groups=("C2", "C3"),
    )
    positions = {}
    for position, local_index in enumerate(dataset.local_indices):
        raw = dataset.canonical[int(local_index)]
        positions[int(raw["package_index"])] = position
    records = protocol["panel"]["records"]
    requested = [int(record["package_index"]) for record in records]
    if len(requested) != len(set(requested)):
        raise ValueError("unseen panel contains duplicate package indices")
    if any(index not in positions for index in requested):
        raise ValueError(f"unseen panel contains a non-{split} package index")
    samples = [dataset[positions[index]] for index in requested]
    for record, sample in zip(records, samples, strict=True):
        if (
            sample.molecule_id != record["molecule_id"]
            or sample.target_pg != record["target_pg"]
            or len(sample.atom_types) != int(record["num_atoms"])
        ):
            raise RuntimeError("unseen panel identity changed")
    return samples
