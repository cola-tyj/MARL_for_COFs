"""Freeze the C2/C3-only read-only UAE reconstruction audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE_A_PROTOCOL = ROOT / "generative_model/smoke/reports/uae3d_gate_a_protocol_v1.json"
GATE_A_REPORT = ROOT / "generative_model/smoke/reports/uae3d_gate_a_v1.json"
METADATA = ROOT / "generative_model/data/processed/v2/metadata.csv"
MANIFEST = ROOT / "generative_model/data/processed/v2/manifest.json"
OUTPUT = ROOT / "generative_model/smoke/reports/uae3d_c23_readonly_protocol_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _size_bin(atom_count: int) -> str:
    if atom_count <= 16:
        return "small_le_16"
    if atom_count <= 40:
        return "medium_17_40"
    return "large_gt_40"


def main() -> None:
    gate_protocol = json.loads(GATE_A_PROTOCOL.read_text(encoding="utf-8"))
    gate_report = json.loads(GATE_A_REPORT.read_text(encoding="utf-8"))
    if gate_report["status"] != "FAIL_GATE_A_STOP_UAE_BRANCH":
        raise RuntimeError("C23 audit 必须保持 UAE Gate A 已关闭状态")
    validation = list(map(int, gate_protocol["panels"]["iid_validation_transfer"]["package_indices"]))
    metadata = {int(row["Package_Index"]): row for row in csv.DictReader(METADATA.open())}
    c23 = [index for index in validation if metadata[index]["Target_PG"] in {"C2", "C3"}]
    counts = {pg: sum(metadata[index]["Target_PG"] == pg for index in c23) for pg in ("C2", "C3")}
    if len(c23) != 30 or counts != {"C2": 24, "C3": 6}:
        raise RuntimeError(f"冻结 C23 panel 组成变化: {counts}, total={len(c23)}")
    panel = [{
        "package_index": index,
        "molecule_id": metadata[index]["Molecule_ID"],
        "target_pg": metadata[index]["Target_PG"],
        "atom_count": int(metadata[index]["Num_Atoms"]),
        "size_bin": _size_bin(int(metadata[index]["Num_Atoms"])),
        "split_iid": metadata[index]["Split_IID"],
    } for index in c23]
    if any(item["split_iid"] != "val" for item in panel):
        raise RuntimeError("C23 panel 含非 IID-validation 分子")
    source = gate_protocol["source"]
    checkpoint = Path(source["checkpoint"])
    protocol = {
        "schema_version": "uae3d-c23-readonly-protocol-v1",
        "status": "FROZEN_BEFORE_UAE_C23_READONLY_AUDIT",
        "purpose": (
            "Measure frozen step-3008 UAE reconstruction and reconstructed-coordinate "
            "symmetry on the existing IID-validation C2/C3 subset while separating size effects."
        ),
        "source": {
            **source,
            "checkpoint_sha256": _sha256(checkpoint),
            "weights_updated": False,
            "decode": "official encoder z_mean and official six-class argmax decoder",
        },
        "panel": {
            "role": "read-only IID-validation diagnostic; never used for threshold or weight fitting",
            "molecule_count": len(panel),
            "target_pg_counts": counts,
            "records": panel,
        },
        "metrics": [
            "atom/bond/categorical exact", "sanitize/connected", "coordinate RMSD",
            "minimum pair distance", "actual PG exact/compatible", "joint molecule success",
            "per-PG and per-size-bin summaries",
        ],
        "joint_success_definition": {
            "atom_exact": True, "bond_exact": True, "sanitize_valid": True,
            "connected": True, "coordinate_rmsd_angstrom_max": 0.15,
            "minimum_pair_distance_angstrom_min": 0.6, "pg_compatible": True,
        },
        "descriptive_evidence_thresholds": {
            "strong_joint_success_molecules_min": 24,
            "partial_joint_success_molecules_min": 8,
            "strong_atom_exact_molecules_min": 30,
            "strong_bond_exact_molecules_min": 24,
            "strong_sanitize_connected_molecules_min": 24,
            "strong_pg_compatible_molecules_min": 27,
        },
        "decision_scope": {
            "uae_branch_reopened": False,
            "gate_b_reopened": False,
            "dataset_level_finetuning_authorized": False,
            "target_pg_conditioning_claimed": False,
            "test_used": False,
            "core_ood_used": False,
            "possible_use_if_strong": "coordinate representation initialization evidence only",
        },
        "identity": {
            "dataset_fingerprint": json.loads(MANIFEST.read_text())["dataset_fingerprint"],
            "manifest_sha256": _sha256(MANIFEST),
            "metadata_sha256": _sha256(METADATA),
            "gate_a_protocol_sha256": _sha256(GATE_A_PROTOCOL),
            "gate_a_report_sha256": _sha256(GATE_A_REPORT),
            "symmetry_analyzer": json.loads(MANIFEST.read_text())["symmetry_protocol"],
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT}")
    print(f"sha256={_sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
