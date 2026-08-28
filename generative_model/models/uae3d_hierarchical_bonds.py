"""UAE-3D 的受约束 hierarchical full-pair bond 解码与损失。

canonical 标签仍保持官方六类 ``none/四种键/self``。本模块不改 checkpoint：
对角 self 由矩阵位置固定，off-diagonal 仅使用旧 head 的前五类 logits；训练目标拆为
``none/present`` existence 和仅在真实 present pair 上计算的四分类 bond type。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from generative_model.models.uae3d_reconstruction import full_pair_targets


HIERARCHICAL_BOND_SCHEMA_VERSION = (
    "uae3d-hierarchical-bonds-v1.2-global-existence-threshold"
)
DECODE_REDUCTIONS = ("max", "logsumexp")


def _validate_inputs(
    bond_logits,
    atom_counts: Sequence[int],
    *,
    symmetry_tolerance: float,
) -> list[int]:
    import torch

    if bond_logits.ndim != 2 or bond_logits.shape[1] != 6:
        raise ValueError("UAE-3D bond_logits 必须为 [sum(N²),6]")
    if not bool(torch.isfinite(bond_logits).all()):
        raise ValueError("bond_logits 含 NaN/Inf")
    counts = [int(value) for value in atom_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("atom_counts 必须为非空正整数")
    if sum(value * value for value in counts) != len(bond_logits):
        raise ValueError("atom_counts 的 N² 总数与 full-pair 数不一致")
    if symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance 不得为负")

    begin = 0
    for molecule_index, atom_count in enumerate(counts):
        end = begin + atom_count * atom_count
        local = bond_logits[begin:end].reshape(atom_count, atom_count, 6)
        symmetry_error = float(
            (local - local.transpose(0, 1)).abs().max().detach().cpu()
        )
        if symmetry_error > symmetry_tolerance:
            raise RuntimeError(
                f"molecule {molecule_index} bond logits 非对称: {symmetry_error}"
            )
        begin = end
    return counts


def _present_score(type_logits, reduction: str):
    """将四个条件键型 logits 聚合为一个 present score。"""

    import torch

    if reduction == "max":
        return type_logits.max(dim=-1).values
    if reduction == "logsumexp":
        return torch.logsumexp(type_logits, dim=-1)
    raise ValueError(f"未知 present score reduction: {reduction}")


def constrained_hierarchical_predictions(
    bond_logits,
    atom_counts: Sequence[int],
    *,
    present_score_reduction: str = "max",
    existence_threshold: float = 0.0,
    symmetry_tolerance: float = 1e-6,
):
    """输出六类 full-pair index，但以结构规则消除非法 self 竞争。

    ``max`` 与“在旧 logits 的 0..4 类中 argmax”严格等价，适合 checkpoint 零拷贝
    初始化；``logsumexp`` 将四种真实键的概率质量聚合成 present，供分层概率审计。
    """

    import torch

    if present_score_reduction not in DECODE_REDUCTIONS:
        raise ValueError(
            f"present_score_reduction 必须属于 {DECODE_REDUCTIONS}"
        )
    if not math.isfinite(existence_threshold):
        raise ValueError("existence_threshold 必须为有限数")
    counts = _validate_inputs(
        bond_logits, atom_counts, symmetry_tolerance=symmetry_tolerance
    )
    predictions = torch.empty(
        len(bond_logits), dtype=torch.long, device=bond_logits.device
    )
    begin = 0
    for atom_count in counts:
        end = begin + atom_count * atom_count
        local_logits = bond_logits[begin:end].reshape(atom_count, atom_count, 6)
        type_logits = local_logits[..., 1:5]
        existence_score = (
            _present_score(type_logits, present_score_reduction)
            - local_logits[..., 0]
        )
        # 严格大于保持 threshold=0 时与 argmax 并列选 none 的语义一致。
        present = existence_score > existence_threshold
        local_predictions = torch.where(
            present,
            type_logits.argmax(dim=-1) + 1,
            torch.zeros_like(present, dtype=torch.long),
        )
        diagonal = torch.arange(atom_count, device=bond_logits.device)
        local_predictions[diagonal, diagonal] = 5
        if not bool(torch.equal(local_predictions, local_predictions.T)):
            raise RuntimeError("受约束 hierarchical bond prediction 非对称")
        predictions[begin:end] = local_predictions.reshape(-1)
        begin = end
    return predictions


def hierarchical_bond_losses(
    bond_logits,
    edge_attr,
    *,
    atom_counts: Sequence[int],
    present_score_reduction: str = "logsumexp",
    symmetry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """按分子等权计算 off-diagonal existence 与 conditional type loss。

    只使用上三角 unique pairs，避免对无向键重复计权；diagonal self 是硬约束，不进入
    学习目标。严格要求每个分子同时含 none 和 present pair，不进行静默 fallback。
    """

    import torch
    import torch.nn.functional as functional

    if present_score_reduction not in DECODE_REDUCTIONS:
        raise ValueError(
            f"present_score_reduction 必须属于 {DECODE_REDUCTIONS}"
        )
    counts = _validate_inputs(
        bond_logits, atom_counts, symmetry_tolerance=symmetry_tolerance
    )
    if len(edge_attr) != len(bond_logits):
        raise ValueError("bond_logits 与 edge_attr pair 数不一致")
    targets = full_pair_targets(edge_attr)
    existence_losses = []
    existence_none_losses = []
    existence_present_losses = []
    type_losses = []
    existence_correct = existence_total = 0
    type_correct = type_total = 0
    begin = 0
    for molecule_index, atom_count in enumerate(counts):
        end = begin + atom_count * atom_count
        local_logits = bond_logits[begin:end].reshape(atom_count, atom_count, 6)
        local_targets = targets[begin:end].reshape(atom_count, atom_count)
        if not bool(torch.equal(local_targets, local_targets.T)):
            raise RuntimeError(f"molecule {molecule_index} bond target 非对称")
        diagonal = torch.diag(local_targets)
        if not bool((diagonal == 5).all()):
            raise RuntimeError(f"molecule {molecule_index} diagonal target 不是 self")
        upper = torch.triu(
            torch.ones(
                (atom_count, atom_count),
                dtype=torch.bool,
                device=bond_logits.device,
            ),
            diagonal=1,
        )
        upper_targets = local_targets[upper]
        if not bool(((upper_targets >= 0) & (upper_targets <= 4)).all()):
            raise RuntimeError(f"molecule {molecule_index} off-diagonal target 非法")
        present_targets = (upper_targets > 0).long()
        if not bool((present_targets == 0).any()) or not bool((present_targets == 1).any()):
            raise ValueError(
                f"molecule {molecule_index} existence strata 必须同时含 none/present"
            )
        upper_logits = local_logits[upper]
        type_logits = upper_logits[:, 1:5]
        existence_logits = torch.stack(
            (
                upper_logits[:, 0],
                _present_score(type_logits, present_score_reduction),
            ),
            dim=-1,
        )
        none_mask = present_targets == 0
        present_mask = present_targets == 1
        none_loss = functional.cross_entropy(
            existence_logits[none_mask], present_targets[none_mask]
        )
        present_loss = functional.cross_entropy(
            existence_logits[present_mask], present_targets[present_mask]
        )
        existence_none_losses.append(none_loss)
        existence_present_losses.append(present_loss)
        existence_losses.append((none_loss + present_loss) / 2)
        local_type_targets = upper_targets[present_mask] - 1
        type_losses.append(
            functional.cross_entropy(type_logits[present_mask], local_type_targets)
        )
        existence_correct += int(
            (existence_logits.argmax(dim=-1) == present_targets).sum()
        )
        existence_total += int(present_targets.numel())
        type_correct += int(
            (type_logits[present_mask].argmax(dim=-1) == local_type_targets).sum()
        )
        type_total += int(local_type_targets.numel())
        begin = end

    existence_loss = torch.stack(existence_losses).mean()
    existence_none_loss = torch.stack(existence_none_losses).mean()
    existence_present_loss = torch.stack(existence_present_losses).mean()
    type_loss = torch.stack(type_losses).mean()
    return {
        "schema_version": HIERARCHICAL_BOND_SCHEMA_VERSION,
        "present_score_reduction": present_score_reduction,
        "hierarchical_bond_loss": (existence_loss + type_loss) / 2,
        "bond_existence_loss": existence_loss,
        "bond_existence_none_loss": existence_none_loss,
        "bond_existence_present_loss": existence_present_loss,
        "conditional_bond_type_loss": type_loss,
        "bond_existence_correct": existence_correct,
        "bond_existence_total": existence_total,
        "conditional_bond_type_correct": type_correct,
        "conditional_bond_type_total": type_total,
        "diagonal_self_is_structural": True,
        "legacy_checkpoint_parameter_mapping": {
            "none_logit": 0,
            "conditional_type_logits": [1, 2, 3, 4],
            "legacy_self_logit": 5,
            "legacy_self_logit_usage": "excluded; diagonal class 5 is fixed structurally",
        },
    }


def legacy_parameter_mapping() -> dict[str, Any]:
    """返回与旧六分类 head 的确定性、无参数拷贝映射。"""

    return {
        "schema_version": HIERARCHICAL_BOND_SCHEMA_VERSION,
        "source_class_order": ["none", "single", "double", "triple", "aromatic", "self"],
        "off_diagonal_class_order": ["none", "single", "double", "triple", "aromatic"],
        "existence_none_source": "legacy_logit[0]",
        "existence_present_source": "reduce(legacy_logits[1:5])",
        "conditional_type_source": "legacy_logits[1:5]",
        "diagonal_source": "structural_self_class_5",
        "new_parameters": 0,
        "checkpoint_state_dict_change": False,
        "max_reduction_equivalence": "argmax(legacy_logits[0:5]) off diagonal",
        "logsumexp_reduction_semantics": "aggregate legacy probability mass of four present classes",
        "existence_score": "reduce(legacy_logits[1:5])-legacy_logit[0]",
        "existence_decision": "present iff existence_score > global_threshold",
        "zero_threshold_tie_break": "none",
        "present_class_count": 4,
        "logsumexp_uniform_offset": math.log(4.0),
    }
