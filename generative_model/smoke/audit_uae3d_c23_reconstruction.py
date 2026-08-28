"""Export frozen UAE step-3008 reconstruction metrics and coordinates for C2/C3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.models.uae3d_bridge import (
    DEFAULT_SOURCE,
    OFFICIAL_GEOM_POSITION_STD,
    assert_uae3d_official_source,
    canonical_to_uae3d_data,
    uae3d_model_args,
)
from generative_model.models.uae3d_reconstruction import evaluate_mean_reconstruction
from generative_model.smoke.train_uae3d_reconstruction import _state_fingerprint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "generative_model/smoke/reports/uae3d_c23_readonly_protocol_v1.json"
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/uae3d_c23_readonly/reconstruction_v1.json"
DEFAULT_COORDS = ROOT / "generative_model/runs/uae3d_c23_readonly/reconstructed_coordinates_v1.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(protocol: dict) -> str:
    payload = dict(protocol); payload.pop("protocol_fingerprint", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run(args) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "uae3d-c23-readonly-protocol-v1" or protocol.get("status") != "FROZEN_BEFORE_UAE_C23_READONLY_AUDIT":
        raise RuntimeError("UAE C23 protocol schema/status 错误")
    if protocol.get("protocol_fingerprint") != _fingerprint(protocol):
        raise RuntimeError("UAE C23 protocol fingerprint 不匹配")
    if _sha256(args.package_dir / "manifest.json") != protocol["identity"]["manifest_sha256"]:
        raise RuntimeError("UAE C23 manifest SHA 不匹配")
    checkpoint_path = Path(protocol["source"]["checkpoint"])
    if _sha256(checkpoint_path) != protocol["source"]["checkpoint_sha256"]:
        raise RuntimeError("UAE C23 checkpoint SHA 不匹配")
    source_commit = assert_uae3d_official_source(args.source)
    source_text = str(args.source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import torch
    import torch_geometric
    from rdkit import rdBase
    from torch_geometric.data import Batch
    from model.autoencoder.unified_autoencoder import UnifiedAutoEncoder

    torch.manual_seed(20260821)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    dataset = COFSymmetryDataset(args.package_dir)
    if dataset.manifest["dataset_fingerprint"] != protocol["identity"]["dataset_fingerprint"]:
        raise RuntimeError("UAE C23 dataset fingerprint 不匹配")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "uae3d-reconstruction-checkpoint-v1" or int(checkpoint.get("step", -1)) != 3008:
        raise RuntimeError("UAE C23 checkpoint schema/step 错误")
    model = UnifiedAutoEncoder(uae3d_model_args("official"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    state_before = _state_fingerprint(model.state_dict())
    records = []
    predicted_parts = []
    target_parts = []
    atom_type_parts = []
    offsets = [0]
    for panel_record in protocol["panel"]["records"]:
        index = int(panel_record["package_index"])
        sample = dataset[index]
        data = canonical_to_uae3d_data(
            sample, source_root=args.source, position_std=OFFICIAL_GEOM_POSITION_STD,
            verify_source=False,
        )
        metrics = evaluate_mean_reconstruction(model, [data], device)
        record = {**metrics["records"][0], **panel_record}
        records.append(record)
        batch = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            z_mean, _ = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
            _, _, coordinates = model.decode(z_mean, batch=batch.batch)
        predicted = coordinates.detach().cpu().double().numpy() * OFFICIAL_GEOM_POSITION_STD
        target = batch.pos.detach().cpu().double().numpy() * OFFICIAL_GEOM_POSITION_STD
        predicted_parts.append(predicted)
        target_parts.append(target)
        atom_type_parts.append(np.asarray(sample["atom_types"], dtype=np.int16))
        offsets.append(offsets[-1] + len(predicted))
        print(f"package_index={index} pg={panel_record['target_pg']} atoms={len(predicted)} atom_exact={record['atom_exact']} bond_exact={record['bond_exact']} rmsd={record['coordinate_rmsd_angstrom']:.6f}A", flush=True)
    state_after = _state_fingerprint(model.state_dict())
    if state_before != state_after:
        raise RuntimeError("UAE C23 read-only audit 修改了 model state")
    args.coords.parent.mkdir(parents=True, exist_ok=True)
    save_deterministic_npz(args.coords, {
        "package_indices": np.asarray([r["package_index"] for r in records], dtype=np.int64),
        "atom_offsets": np.asarray(offsets, dtype=np.int64),
        "atom_types": np.concatenate(atom_type_parts),
        "predicted_positions": np.concatenate(predicted_parts),
        "target_positions": np.concatenate(target_parts),
    })
    report = {
        "schema_version": "uae3d-c23-reconstruction-intermediate-v1",
        "status": "PASS_UAE_C23_READONLY_RECONSTRUCTION_EXECUTION",
        "protocol_sha256": _sha256(args.protocol),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_state_fingerprint": state_before,
        "model_state_unchanged": True,
        "weights_updated": False,
        "records": records,
        "coordinate_artifact": {"path": str(args.coords.resolve()), "sha256": _sha256(args.coords)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "rdkit": rdBase.rdkitVersion, "device": str(device)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "records": len(report["records"]), "output": str(args.output), "coordinates": report["coordinate_artifact"]}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
