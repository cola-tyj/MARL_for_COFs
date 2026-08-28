"""与 UAE-3D 固定 ``z_mean`` 重构 Gate 对齐的全原子辅助目标。"""

from __future__ import annotations

from typing import Any

from generative_model.models.uae3d_reconstruction import full_pair_targets
from generative_model.models.uae3d_hierarchical_bonds import hierarchical_bond_losses


EXACT_RECONSTRUCTION_OBJECTIVE_VERSION = (
    "uae3d-exact-reconstruction-aux-v2-molecule-balanced"
)
HARD_PAIR_OBJECTIVE_VERSION = (
    "uae3d-exact-reconstruction-aux-v3-bond-head-hard-margin"
)
HIERARCHICAL_BOND_OBJECTIVE_VERSION = (
    "uae3d-exact-reconstruction-aux-v4-constrained-hierarchical-bonds-strata-balanced"
)


def stratified_bond_cross_entropy(
    bond_logits, edge_attr, *, atom_counts: list[int] | None = None
) -> dict[str, Any]:
    """先按分子、再按 strata 等权，避免大分子及 N² 多数类支配。"""

    import torch
    import torch.nn.functional as functional

    if bond_logits.ndim != 2 or bond_logits.shape[1] != 6:
        raise ValueError("UAE-3D bond_logits 必须为 [sum(N²),6]")
    if len(bond_logits) != len(edge_attr):
        raise ValueError("bond_logits 与 edge_attr pair 数不一致")
    if not bool(torch.isfinite(bond_logits).all()):
        raise ValueError("bond_logits 含 NaN/Inf")
    targets = full_pair_targets(edge_attr)
    if atom_counts is None:
        atom_counts = [int(round(len(targets) ** 0.5))]
    if not atom_counts or any(count <= 0 for count in atom_counts):
        raise ValueError("atom_counts 必须为非空正整数列表")
    pair_counts = [count * count for count in atom_counts]
    if sum(pair_counts) != len(targets):
        raise ValueError("atom_counts 的 N² 总数与 full-pair 数不一致")

    per_molecule: list[dict[str, Any]] = []
    begin = 0
    for molecule_index, pair_count in enumerate(pair_counts):
        end = begin + pair_count
        local_targets = targets[begin:end]
        local_logits = bond_logits[begin:end]
        masks = {
            "none": local_targets == 0,
            "present": (local_targets >= 1) & (local_targets <= 4),
            "self": local_targets == 5,
        }
        empty = [name for name, mask in masks.items() if not bool(mask.any())]
        if empty:
            raise ValueError(
                f"molecule {molecule_index} balanced bond strata 为空: {empty}"
            )
        per_molecule.append({
            name: functional.cross_entropy(local_logits[mask], local_targets[mask])
            for name, mask in masks.items()
        })
        begin = end
    losses = {
        name: torch.stack([local[name] for local in per_molecule]).mean()
        for name in ("none", "present", "self")
    }
    total = torch.stack(list(losses.values())).mean()
    global_masks = {
        "none": targets == 0,
        "present": (targets >= 1) & (targets <= 4),
        "self": targets == 5,
    }
    return {
        "balanced_bond_loss": total,
        "bond_none_stratum_loss": losses["none"],
        "bond_present_stratum_loss": losses["present"],
        "bond_self_stratum_loss": losses["self"],
        "bond_none_pairs": int(global_masks["none"].sum()),
        "bond_present_pairs": int(global_masks["present"].sum()),
        "bond_self_pairs": int(global_masks["self"].sum()),
        "aux_molecule_count": len(atom_counts),
    }


def molecule_balanced_coordinate_mse(coordinates, targets, atom_counts: list[int]):
    """每个分子内按坐标元素求 MSE，再对分子等权。"""

    import torch
    import torch.nn.functional as functional

    if coordinates.shape != targets.shape or coordinates.ndim != 2:
        raise ValueError("coordinate tensors 必须为相同二维 shape")
    if not atom_counts or any(count <= 0 for count in atom_counts):
        raise ValueError("atom_counts 必须为非空正整数列表")
    if sum(atom_counts) != len(coordinates):
        raise ValueError("atom_counts 总数与 coordinate 原子数不一致")
    losses = []
    begin = 0
    for count in atom_counts:
        end = begin + count
        losses.append(functional.mse_loss(coordinates[begin:end], targets[begin:end]))
        begin = end
    return torch.stack(losses).mean()


def hard_pair_bond_losses(
    bond_logits,
    edge_attr,
    *,
    atom_counts: list[int],
    margin_threshold: float = 0.5,
    symmetry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """对每分子/strata 的错误或低 margin unique pairs 等权计算 v3 loss。"""

    import torch
    import torch.nn.functional as functional

    if bond_logits.ndim != 2 or bond_logits.shape[1] != 6:
        raise ValueError("UAE-3D bond_logits 必须为 [sum(N²),6]")
    if len(bond_logits) != len(edge_attr):
        raise ValueError("bond_logits 与 edge_attr pair 数不一致")
    if not bool(torch.isfinite(bond_logits).all()):
        raise ValueError("bond_logits 含 NaN/Inf")
    if not atom_counts or any(count <= 0 for count in atom_counts):
        raise ValueError("atom_counts 必须为非空正整数列表")
    if sum(count * count for count in atom_counts) != len(bond_logits):
        raise ValueError("atom_counts 的 N² 总数与 full-pair 数不一致")
    if margin_threshold <= 0 or symmetry_tolerance < 0:
        raise ValueError("margin_threshold 必须为正且 symmetry_tolerance 不得为负")

    targets = full_pair_targets(edge_attr)
    per_molecule: list[dict[str, Any]] = []
    begin = 0
    global_counts = {name: 0 for name in ("none", "present", "self")}
    for molecule_index, atom_count in enumerate(atom_counts):
        end = begin + atom_count * atom_count
        local_logits = bond_logits[begin:end].reshape(atom_count, atom_count, 6)
        local_targets = targets[begin:end].reshape(atom_count, atom_count)
        if not bool(torch.equal(local_targets, local_targets.T)):
            raise RuntimeError(f"molecule {molecule_index} bond target 非对称")
        symmetry_error = (local_logits - local_logits.transpose(0, 1)).abs().max()
        if float(symmetry_error.detach().cpu()) > symmetry_tolerance:
            raise RuntimeError(
                f"molecule {molecule_index} bond logits 非对称: "
                f"{float(symmetry_error.detach().cpu())}"
            )
        diagonal = torch.eye(atom_count, dtype=torch.bool, device=bond_logits.device)
        upper = torch.triu(
            torch.ones(
                (atom_count, atom_count), dtype=torch.bool, device=bond_logits.device
            ),
            diagonal=1,
        )
        masks = {
            "none": upper & (local_targets == 0),
            "present": upper & (local_targets >= 1) & (local_targets <= 4),
            "self": diagonal & (local_targets == 5),
        }
        empty = [name for name, mask in masks.items() if not bool(mask.any())]
        if empty:
            raise ValueError(f"molecule {molecule_index} hard-pair strata 为空: {empty}")
        local: dict[str, Any] = {}
        for name, mask in masks.items():
            logits = local_logits[mask]
            target = local_targets[mask]
            target_logits = logits.gather(1, target[:, None]).squeeze(1)
            competing = logits.masked_fill(
                functional.one_hot(target, num_classes=6).bool(), -torch.inf
            ).max(dim=1).values
            margin = target_logits - competing
            wrong = logits.argmax(dim=-1) != target
            hard = wrong | (margin < margin_threshold)
            count = int(hard.sum())
            global_counts[name] += count
            if count:
                local[name] = {
                    "ce": functional.cross_entropy(logits[hard], target[hard]),
                    "margin": functional.relu(margin_threshold - margin[hard]).mean(),
                    "count": count,
                }
            else:
                # 保留与 bond head 的计算图，使全 exact 时 backward 仍合法且梯度为零。
                zero = logits.sum() * 0.0
                local[name] = {"ce": zero, "margin": zero, "count": 0}
        per_molecule.append(local)
        begin = end
    if begin != len(targets):
        raise RuntimeError("hard-pair offsets 未消费全部 full pairs")

    ce_by_stratum = {
        name: torch.stack([local[name]["ce"] for local in per_molecule]).mean()
        for name in ("none", "present", "self")
    }
    margin_by_stratum = {
        name: torch.stack([local[name]["margin"] for local in per_molecule]).mean()
        for name in ("none", "present", "self")
    }
    return {
        "hard_pair_ce_loss": torch.stack(list(ce_by_stratum.values())).mean(),
        "hard_pair_margin_loss": torch.stack(list(margin_by_stratum.values())).mean(),
        "hard_pair_none_ce_loss": ce_by_stratum["none"],
        "hard_pair_present_ce_loss": ce_by_stratum["present"],
        "hard_pair_self_ce_loss": ce_by_stratum["self"],
        "hard_pair_none_margin_loss": margin_by_stratum["none"],
        "hard_pair_present_margin_loss": margin_by_stratum["present"],
        "hard_pair_self_margin_loss": margin_by_stratum["self"],
        "hard_pair_count": sum(global_counts.values()),
        "hard_pair_none_count": global_counts["none"],
        "hard_pair_present_count": global_counts["present"],
        "hard_pair_self_count": global_counts["self"],
        "hard_pair_margin_threshold": float(margin_threshold),
    }


def deterministic_mean_auxiliaries(
    model,
    batch,
    *,
    hard_pair_margin: float | None = None,
    hierarchical_present_score: str | None = None,
) -> dict[str, Any]:
    """关闭 dropout，用 encoder mean 解码并计算与固定评估同口径的辅助 loss。"""

    import torch

    prior_mode = model.training
    model.eval()
    try:
        z_mean, _ = model.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.pos
        )
        _, bond_logits, coordinates = model.decode(z_mean, batch=batch.batch)
    finally:
        model.train(prior_mode)
    if not bool(torch.isfinite(coordinates).all()):
        raise RuntimeError("deterministic z_mean coordinates 含 NaN/Inf")
    atom_counts = [int(value) for value in (batch.ptr[1:] - batch.ptr[:-1]).tolist()]
    bond = stratified_bond_cross_entropy(
        bond_logits, batch.edge_attr, atom_counts=atom_counts
    )
    coordinate_loss = molecule_balanced_coordinate_mse(
        coordinates, batch.pos, atom_counts
    )
    if hard_pair_margin is None:
        zero = bond_logits.sum() * 0.0
        hard = {
            "hard_pair_ce_loss": zero,
            "hard_pair_margin_loss": zero,
            "hard_pair_none_ce_loss": zero,
            "hard_pair_present_ce_loss": zero,
            "hard_pair_self_ce_loss": zero,
            "hard_pair_none_margin_loss": zero,
            "hard_pair_present_margin_loss": zero,
            "hard_pair_self_margin_loss": zero,
            "hard_pair_count": 0,
            "hard_pair_none_count": 0,
            "hard_pair_present_count": 0,
            "hard_pair_self_count": 0,
            "hard_pair_margin_threshold": None,
        }
    else:
        hard = hard_pair_bond_losses(
            bond_logits,
            batch.edge_attr,
            atom_counts=atom_counts,
            margin_threshold=hard_pair_margin,
        )
    if hierarchical_present_score is None:
        zero = bond_logits.sum() * 0.0
        hierarchical = {
            "hierarchical_bond_loss": zero,
            "bond_existence_loss": zero,
            "bond_existence_none_loss": zero,
            "bond_existence_present_loss": zero,
            "conditional_bond_type_loss": zero,
            "bond_existence_correct": 0,
            "bond_existence_total": 0,
            "conditional_bond_type_correct": 0,
            "conditional_bond_type_total": 0,
        }
    else:
        hierarchical = hierarchical_bond_losses(
            bond_logits,
            batch.edge_attr,
            atom_counts=atom_counts,
            present_score_reduction=hierarchical_present_score,
        )
    return {
        **bond,
        **hard,
        **hierarchical,
        "mean_coordinate_loss": coordinate_loss,
    }
