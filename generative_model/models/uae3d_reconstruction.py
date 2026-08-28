"""UAE-3D ``z_mean`` 确定性重构指标与严格 RDKit 检查。"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from generative_model.data.model_adapters import UAE_GEOM_VOCAB
from generative_model.models.uae3d_bridge import OFFICIAL_GEOM_POSITION_STD


RECONSTRUCTION_METRICS_VERSION = "uae3d-mean-reconstruction-v1.1"


def full_pair_targets(edge_attr):
    """复现官方 none/bond/self-loop 六分类 target，拒绝非法 one-hot。"""

    import torch

    if edge_attr.ndim != 2 or edge_attr.shape[1] != 5:
        raise ValueError("UAE-3D edge_attr 必须为 [sum(N²),5]")
    if not torch.all((edge_attr == 0) | (edge_attr == 1)):
        raise ValueError("UAE-3D edge_attr 含非 0/1 值")
    if not torch.all(edge_attr.sum(dim=-1) <= 1):
        raise ValueError("一个 full pair 不能同时属于多个类别")
    return torch.cat(
        [(edge_attr == 0).all(dim=-1, keepdim=True).float(), edge_attr], dim=-1
    ).argmax(dim=-1)


def _strict_rdkit_status(atom_classes: np.ndarray, bond_classes: np.ndarray) -> dict[str, Any]:
    """按官方非对角键语义构建分子；不对键、价态或元素做静默修复。"""

    from rdkit import Chem

    atom_classes = np.asarray(atom_classes, dtype=np.int64)
    bond_classes = np.asarray(bond_classes, dtype=np.int64)
    atom_count = len(atom_classes)
    if bond_classes.shape != (atom_count, atom_count):
        return {"sanitize_valid": False, "connected": False, "status": "BAD_SHAPE"}
    if np.any(atom_classes < 0) or np.any(atom_classes >= len(UAE_GEOM_VOCAB)):
        return {"sanitize_valid": False, "connected": False, "status": "BAD_ATOM_CLASS"}
    if not np.array_equal(bond_classes, bond_classes.T):
        return {"sanitize_valid": False, "connected": False, "status": "ASYMMETRIC_BONDS"}
    off_diagonal = ~np.eye(atom_count, dtype=bool)
    if np.any((bond_classes[off_diagonal] < 0) | (bond_classes[off_diagonal] > 4)):
        return {"sanitize_valid": False, "connected": False, "status": "SELF_LOOP_OFF_DIAGONAL"}

    editable = Chem.RWMol()
    try:
        for class_index in atom_classes:
            atom = Chem.Atom(UAE_GEOM_VOCAB[int(class_index)])
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            editable.AddAtom(atom)
        rdkit_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }
        for right in range(atom_count):
            for left in range(right):
                kind = int(bond_classes[left, right])
                if kind == 0:
                    continue
                editable.AddBond(left, right, rdkit_types[kind])
                if kind == 4:
                    bond = editable.GetBondBetweenAtoms(left, right)
                    if bond is None:
                        raise RuntimeError("aromatic bond 添加失败")
                    bond.SetIsAromatic(True)
                    editable.GetAtomWithIdx(left).SetIsAromatic(True)
                    editable.GetAtomWithIdx(right).SetIsAromatic(True)
        molecule = editable.GetMol()
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return {
                "sanitize_valid": False,
                "connected": False,
                "status": str(flag),
            }
        return {
            "sanitize_valid": True,
            "connected": len(Chem.GetMolFrags(molecule)) == 1,
            "status": "VALID",
        }
    except (RuntimeError, ValueError, KeyError) as error:
        return {
            "sanitize_valid": False,
            "connected": False,
            "status": f"BUILD_ERROR:{type(error).__name__}",
        }


def _bond_representation_status(bond_classes: np.ndarray) -> dict[str, Any]:
    """独立审计 full-pair 对称性和 self-loop sentinel，不冒充化学 sanitize。"""

    bond_classes = np.asarray(bond_classes, dtype=np.int64)
    if bond_classes.ndim != 2 or bond_classes.shape[0] != bond_classes.shape[1]:
        return {
            "bond_symmetric": False,
            "off_diagonal_no_self_loop": False,
            "self_loop_correct": 0,
            "self_loop_total": 0,
            "self_loop_exact": False,
            "bond_representation_valid": False,
            "bond_representation_status": "BAD_SHAPE",
        }
    atom_count = bond_classes.shape[0]
    symmetric = bool(np.array_equal(bond_classes, bond_classes.T))
    diagonal = np.diag(bond_classes)
    self_loop_correct = int(np.count_nonzero(diagonal == 5))
    off_diagonal = ~np.eye(atom_count, dtype=bool)
    off_diagonal_no_self_loop = not bool(np.any(bond_classes[off_diagonal] == 5))
    if not symmetric:
        status = "ASYMMETRIC_BONDS"
    elif not off_diagonal_no_self_loop:
        status = "SELF_LOOP_OFF_DIAGONAL"
    elif self_loop_correct != atom_count:
        status = "BAD_SELF_LOOP"
    else:
        status = "VALID"
    return {
        "bond_symmetric": symmetric,
        "off_diagonal_no_self_loop": off_diagonal_no_self_loop,
        "self_loop_correct": self_loop_correct,
        "self_loop_total": atom_count,
        "self_loop_exact": self_loop_correct == atom_count,
        "bond_representation_valid": status == "VALID",
        "bond_representation_status": status,
    }


def evaluate_mean_reconstruction(
    model,
    data_list: Sequence[Any],
    device,
    *,
    position_std: float = OFFICIAL_GEOM_POSITION_STD,
    bond_prediction_transform: Callable[[Any, Sequence[int]], Any] | None = None,
    bond_decode_mode: str | None = None,
) -> dict[str, Any]:
    """使用 encoder 的 ``z_mean``，在固定输入上执行无采样重构审计。"""

    import torch
    from torch_geometric.data import Batch

    if not data_list:
        raise ValueError("重构审计不能使用空数据集")
    batch = Batch.from_data_list(list(data_list)).to(device)
    prior_mode = model.training
    model.eval()
    with torch.no_grad():
        z_mean, z_log_var = model.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.pos
        )
        atom_logits, bond_logits, coordinates = model.decode(z_mean, batch=batch.batch)
    tensors = (z_mean, z_log_var, atom_logits, bond_logits, coordinates)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError("UAE-3D 确定性重构输出含 NaN/Inf")

    atom_counts = [int(data.x.shape[0]) for data in data_list]
    atom_targets = batch.x[:, :len(UAE_GEOM_VOCAB)].argmax(dim=-1)
    bond_targets = full_pair_targets(batch.edge_attr)
    atom_predictions = atom_logits.argmax(dim=-1)
    if bond_prediction_transform is None:
        if bond_decode_mode is not None:
            raise ValueError("bond_decode_mode 只能与 bond_prediction_transform 同时使用")
        bond_predictions = bond_logits.argmax(dim=-1)
        decode_mode = "encoder_z_mean_no_sampling"
    else:
        if not bond_decode_mode:
            raise ValueError("自定义 bond prediction 必须显式给出 bond_decode_mode")
        bond_predictions = bond_prediction_transform(bond_logits, atom_counts)
        decode_mode = f"encoder_z_mean_no_sampling+{bond_decode_mode}"
    if bond_predictions.shape != bond_targets.shape:
        raise RuntimeError("自定义 bond prediction shape 与 full-pair target 不一致")
    if bond_predictions.dtype != torch.long:
        raise RuntimeError("自定义 bond prediction 必须返回 torch.long class index")
    if not bool(((bond_predictions >= 0) & (bond_predictions <= 5)).all()):
        raise RuntimeError("自定义 bond prediction 含非法 class index")
    predicted_positions = coordinates * float(position_std)
    target_positions = batch.pos * float(position_std)

    records: list[dict[str, Any]] = []
    atom_begin = pair_begin = 0
    for item, data in enumerate(data_list):
        atom_count = int(data.x.shape[0])
        atom_end = atom_begin + atom_count
        pair_end = pair_begin + atom_count * atom_count
        target_atoms = atom_targets[atom_begin:atom_end]
        predicted_atoms = atom_predictions[atom_begin:atom_end]
        target_bonds = bond_targets[pair_begin:pair_end].reshape(atom_count, atom_count)
        predicted_bonds = bond_predictions[pair_begin:pair_end].reshape(atom_count, atom_count)
        atom_matches = predicted_atoms == target_atoms
        bond_matches = predicted_bonds == target_bonds
        off_diagonal = ~torch.eye(atom_count, dtype=torch.bool, device=device)
        present = off_diagonal & (target_bonds >= 1) & (target_bonds <= 4)
        none = off_diagonal & (target_bonds == 0)
        delta = predicted_positions[atom_begin:atom_end] - target_positions[atom_begin:atom_end]
        coordinate_rmsd = torch.sqrt(delta.square().sum(dim=-1).mean())
        if atom_count > 1:
            pred_dist = torch.pdist(predicted_positions[atom_begin:atom_end])
            target_dist = torch.pdist(target_positions[atom_begin:atom_end])
            pair_distance_mae = (pred_dist - target_dist).abs().mean()
        else:
            pair_distance_mae = torch.zeros((), device=device)
        chemistry = _strict_rdkit_status(
            predicted_atoms.detach().cpu().numpy(),
            predicted_bonds.detach().cpu().numpy(),
        )
        representation = _bond_representation_status(
            predicted_bonds.detach().cpu().numpy()
        )
        atom_exact = bool(atom_matches.all())
        bond_exact = bool(bond_matches.all())
        records.append({
            "package_index": int(data.idx),
            "molecule_id": str(data.molecule_id),
            "atoms": atom_count,
            "atom_correct": int(atom_matches.sum()),
            "atom_exact": atom_exact,
            "bond_pair_correct": int(bond_matches.sum()),
            "bond_pair_total": atom_count * atom_count,
            "bond_exact": bond_exact,
            "off_diagonal_correct": int(bond_matches[off_diagonal].sum()),
            "off_diagonal_total": int(off_diagonal.sum()),
            "present_bond_correct": int(bond_matches[present].sum()),
            "present_bond_total": int(present.sum()),
            "none_bond_correct": int(bond_matches[none].sum()),
            "none_bond_total": int(none.sum()),
            "categorical_exact": atom_exact and bond_exact,
            "coordinate_rmsd_angstrom": float(coordinate_rmsd.cpu()),
            "pair_distance_mae_angstrom": float(pair_distance_mae.cpu()),
            **representation,
            **chemistry,
        })
        atom_begin, pair_begin = atom_end, pair_end
    if atom_begin != len(atom_targets) or pair_begin != len(bond_targets):
        raise RuntimeError("UAE-3D batch offsets 与输出长度不一致")
    model.train(prior_mode)

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 1.0

    sums = {
        key: sum(int(record[key]) for record in records)
        for key in (
            "atom_correct", "atoms", "bond_pair_correct", "bond_pair_total",
            "off_diagonal_correct", "off_diagonal_total", "present_bond_correct",
            "present_bond_total", "none_bond_correct", "none_bond_total",
        )
    }
    coordinate_values = [record["coordinate_rmsd_angstrom"] for record in records]
    pair_values = [record["pair_distance_mae_angstrom"] for record in records]
    return {
        "schema_version": RECONSTRUCTION_METRICS_VERSION,
        "decode_mode": decode_mode,
        "molecule_count": len(records),
        "atom_accuracy": ratio(sums["atom_correct"], sums["atoms"]),
        "atom_exact_molecules": sum(record["atom_exact"] for record in records),
        "bond_full_pair_accuracy": ratio(sums["bond_pair_correct"], sums["bond_pair_total"]),
        "bond_exact_molecules": sum(record["bond_exact"] for record in records),
        "bond_representation_valid_molecules": sum(
            record["bond_representation_valid"] for record in records
        ),
        "self_loop_accuracy": ratio(
            sum(record["self_loop_correct"] for record in records),
            sum(record["self_loop_total"] for record in records),
        ),
        "self_loop_exact_molecules": sum(record["self_loop_exact"] for record in records),
        "off_diagonal_accuracy": ratio(sums["off_diagonal_correct"], sums["off_diagonal_total"]),
        "present_bond_accuracy": ratio(sums["present_bond_correct"], sums["present_bond_total"]),
        "none_bond_accuracy": ratio(sums["none_bond_correct"], sums["none_bond_total"]),
        "categorical_exact_molecules": sum(record["categorical_exact"] for record in records),
        "sanitize_valid_molecules": sum(record["sanitize_valid"] for record in records),
        "connected_molecules": sum(record["connected"] for record in records),
        "coordinate_rmsd_mean_angstrom": float(np.mean(coordinate_values)),
        "coordinate_rmsd_max_angstrom": float(np.max(coordinate_values)),
        "pair_distance_mae_mean_angstrom": float(np.mean(pair_values)),
        "records": records,
    }


def reconstruction_gate(metrics: dict[str, Any], coordinate_rmsd_limit: float = 0.05) -> dict[str, Any]:
    """冻结 U2 门槛；只接受整分子 exact、有效连通及坐标阈值同时通过。"""

    molecule_count = int(metrics["molecule_count"])
    checks = {
        "atom_exact_all": int(metrics["atom_exact_molecules"]) == molecule_count,
        "bond_exact_all": int(metrics["bond_exact_molecules"]) == molecule_count,
        "sanitize_valid_all": int(metrics["sanitize_valid_molecules"]) == molecule_count,
        "connected_all": int(metrics["connected_molecules"]) == molecule_count,
        "coordinate_rmsd_max": (
            float(metrics["coordinate_rmsd_max_angstrom"]) <= coordinate_rmsd_limit
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "coordinate_rmsd_limit_angstrom": coordinate_rmsd_limit,
    }
