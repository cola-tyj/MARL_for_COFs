"""显式、可审计的生成图化学约束 decoder；不用于修改 canonical ground truth。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


DECODER_VERSION = "valence_greedy_v2"
PRUNE_DECODER_VERSION = "valence_prune_v2"

# 与锁定 SemlaFlow ``Metrics.ALLOWED_VALENCIES`` 的上界一致。这里使用 2 倍键级，
# 从而可精确表示 aromatic=1.5；约束只阻止过价，不声称保证 RDKit sanitize。
_ALLOWED_VALENCES: dict[str, int | list[int] | dict[int, int | list[int]]] = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}
_BOND_ORDER_X2 = (0, 2, 4, 6, 3)
_SPECIAL_TOKENS = {"<PAD>", "<MASK>"}


@dataclass(frozen=True)
class DecodeAudit:
    molecule_count: int
    raw_argmax_edges: int
    decoded_edges: int
    candidates_positive_gain: int
    rejected_overvalence: int
    rejected_pair_already_assigned: int
    aromatic_bridge_edges_replaced: int
    aromatic_bridge_edges_removed: int
    atom_types_changed: int
    formal_charges_changed: int
    initial_overvalent_atoms: int = 0
    downgraded_bonds: int = 0
    removed_bonds: int = 0
    bridge_removals: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


def _max_valence_x2(symbol: str, charge: int) -> int:
    if symbol in _SPECIAL_TOKENS:
        raise ValueError(f"生成结果含特殊原子 token: {symbol}")
    if symbol not in _ALLOWED_VALENCES:
        raise ValueError(f"没有定义价态约束的元素: {symbol}")
    allowed = _ALLOWED_VALENCES[symbol]
    if isinstance(allowed, dict):
        if charge not in allowed:
            raise ValueError(f"不支持的元素/电荷价态组合: {symbol}{charge:+d}")
        allowed = allowed[charge]
    maximum = max(allowed) if isinstance(allowed, list) else allowed
    # SemlaFlow stability metric 先把 aromatic 记作 1.5，再对总价态取 long/floor。
    # 因而 max+0.5 仍属于 max 价态；偶数整数键级不受这个额外半单位影响。
    return int(maximum) * 2 + 1


def _aromatic_bridges(bond_types: np.ndarray) -> list[tuple[int, int]]:
    """返回完整分子图中的 aromatic 桥；这种边不可能属于任何分子环。"""

    atom_count = bond_types.shape[0]
    adjacency = [[] for _ in range(atom_count)]
    for begin in range(atom_count):
        for end in range(begin + 1, atom_count):
            if int(bond_types[begin, end]) != 0:
                adjacency[begin].append(end)
                adjacency[end].append(begin)

    discovery = [-1] * atom_count
    low = [-1] * atom_count
    parent = [-1] * atom_count
    bridges: list[tuple[int, int]] = []
    counter = 0

    def visit(node: int) -> None:
        nonlocal counter
        discovery[node] = low[node] = counter
        counter += 1
        for neighbour in adjacency[node]:
            if discovery[neighbour] < 0:
                parent[neighbour] = node
                visit(neighbour)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    begin, end = min(node, neighbour), max(node, neighbour)
                    if int(bond_types[begin, end]) == 4:
                        bridges.append((begin, end))
            elif neighbour != parent[node]:
                low[node] = min(low[node], discovery[neighbour])

    for node in range(atom_count):
        if discovery[node] < 0:
            visit(node)
    return sorted(bridges)


def _is_bridge(bond_types: np.ndarray, begin: int, end: int) -> bool:
    if int(bond_types[begin, end]) == 0:
        return False
    visited = {begin}
    stack = [begin]
    while stack:
        node = stack.pop()
        for neighbour in np.flatnonzero(bond_types[node] != 0).tolist():
            if (node == begin and neighbour == end) or (node == end and neighbour == begin):
                continue
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    return end not in visited


def _decode_one(
    atom_probs: np.ndarray,
    bond_probs: np.ndarray,
    charge_probs: np.ndarray,
    symbols: Sequence[str],
    charge_values: Sequence[int],
    *,
    minimum_log_gain: float,
) -> tuple[np.ndarray, dict[str, int]]:
    atom_indices = atom_probs.argmax(axis=-1)
    charge_indices = charge_probs.argmax(axis=-1)
    atom_symbols = [str(symbols[int(index)]) for index in atom_indices]
    formal_charges = [int(charge_values[int(index)]) for index in charge_indices]
    capacities = np.asarray(
        [_max_valence_x2(symbol, charge) for symbol, charge in zip(atom_symbols, formal_charges)],
        dtype=np.int64,
    )
    used = np.zeros_like(capacities)
    atom_count = len(atom_symbols)
    symmetric_probs = (bond_probs + bond_probs.transpose(1, 0, 2)) / 2.0
    raw_types = symmetric_probs.argmax(axis=-1)
    raw_edges = int(np.count_nonzero(np.triu(raw_types, k=1)))

    eps = np.finfo(np.float64).tiny
    candidates: list[tuple[float, int, int, int]] = []
    for begin in range(atom_count):
        for end in range(begin + 1, atom_count):
            none_log = math.log(max(float(symmetric_probs[begin, end, 0]), eps))
            for bond_type in range(1, 5):
                gain = math.log(max(float(symmetric_probs[begin, end, bond_type]), eps)) - none_log
                if gain > minimum_log_gain:
                    candidates.append((gain, begin, end, bond_type))
    # 完全相同的 gain 以原子索引和 bond type 固定打破平局，保证可复现。
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    decoded = np.zeros((atom_count, atom_count), dtype=np.int64)
    rejected_overvalence = 0
    rejected_assigned = 0
    for _, begin, end, bond_type in candidates:
        if decoded[begin, end] != 0:
            rejected_assigned += 1
            continue
        order = _BOND_ORDER_X2[bond_type]
        if used[begin] + order > capacities[begin] or used[end] + order > capacities[end]:
            rejected_overvalence += 1
            continue
        decoded[begin, end] = decoded[end, begin] = bond_type
        used[begin] += order
        used[end] += order

    replaced = removed = 0
    # 芳香桥显式降为该 pair 最可能且仍满足价态的非芳香键；没有正 gain 候选则移除。
    for begin, end in _aromatic_bridges(decoded):
        used[begin] -= _BOND_ORDER_X2[4]
        used[end] -= _BOND_ORDER_X2[4]
        none_log = math.log(max(float(symmetric_probs[begin, end, 0]), eps))
        alternatives = sorted(
            (
                (
                    math.log(max(float(symmetric_probs[begin, end, bond_type]), eps)) - none_log,
                    bond_type,
                )
                for bond_type in range(1, 4)
            ),
            reverse=True,
        )
        replacement = 0
        for gain, bond_type in alternatives:
            order = _BOND_ORDER_X2[bond_type]
            if gain <= minimum_log_gain:
                continue
            if used[begin] + order <= capacities[begin] and used[end] + order <= capacities[end]:
                replacement = bond_type
                used[begin] += order
                used[end] += order
                break
        decoded[begin, end] = decoded[end, begin] = replacement
        if replacement:
            replaced += 1
        else:
            removed += 1

    if np.any(used > capacities):
        raise RuntimeError("decoder 内部错误：输出仍存在过价原子")
    np.fill_diagonal(decoded, 0)
    return decoded, {
        "raw_argmax_edges": raw_edges,
        "decoded_edges": int(np.count_nonzero(np.triu(decoded, k=1))),
        "candidates_positive_gain": len(candidates),
        "rejected_overvalence": rejected_overvalence,
        "rejected_pair_already_assigned": rejected_assigned,
        "aromatic_bridge_edges_replaced": replaced,
        "aromatic_bridge_edges_removed": removed,
    }


def valence_greedy_decode(
    output: dict[str, Any],
    *,
    symbols: Sequence[str],
    charge_values: Sequence[int],
    minimum_log_gain: float = 0.0,
) -> tuple[dict[str, Any], DecodeAudit]:
    """以概率 gain 贪心构图；返回新 tensors，绝不原地修改模型原始输出。"""

    import torch
    import torch.nn.functional as functional

    required = {"coords", "atomics", "bonds", "charges", "mask"}
    missing = sorted(required - set(output))
    if missing:
        raise KeyError(f"生成 output 缺字段: {missing}")
    atomics = output["atomics"]
    bonds = output["bonds"]
    charges = output["charges"]
    mask = output["mask"].bool()
    if atomics.ndim != 3 or charges.ndim != 3 or bonds.ndim != 4 or mask.ndim != 2:
        raise ValueError("生成 output tensor 维度不符合 SemlaFlow 契约")
    if bonds.shape[-1] != 5:
        raise ValueError("bond vocabulary 必须为 none/single/double/triple/aromatic 共 5 类")
    if len(symbols) != atomics.shape[-1] or len(charge_values) != charges.shape[-1]:
        raise ValueError("atom/charge vocabulary 与概率 tensor 不一致")

    decoded_bonds = torch.zeros_like(bonds)
    totals = {
        "raw_argmax_edges": 0,
        "decoded_edges": 0,
        "candidates_positive_gain": 0,
        "rejected_overvalence": 0,
        "rejected_pair_already_assigned": 0,
        "aromatic_bridge_edges_replaced": 0,
        "aromatic_bridge_edges_removed": 0,
    }
    for batch_index in range(mask.shape[0]):
        atom_count = int(mask[batch_index].sum().item())
        if atom_count <= 0:
            raise ValueError("生成 batch 含空分子")
        decoded, audit = _decode_one(
            atomics[batch_index, :atom_count].detach().cpu().numpy(),
            bonds[batch_index, :atom_count, :atom_count].detach().cpu().numpy(),
            charges[batch_index, :atom_count].detach().cpu().numpy(),
            symbols,
            charge_values,
            minimum_log_gain=minimum_log_gain,
        )
        one_hot = functional.one_hot(
            torch.as_tensor(decoded, dtype=torch.long, device=bonds.device),
            num_classes=5,
        ).to(dtype=bonds.dtype)
        decoded_bonds[batch_index, :atom_count, :atom_count] = one_hot
        for key, value in audit.items():
            totals[key] += int(value)

    decoded_output = {key: value.clone() for key, value in output.items()}
    decoded_output["bonds"] = decoded_bonds
    return decoded_output, DecodeAudit(
        molecule_count=int(mask.shape[0]),
        atom_types_changed=0,
        formal_charges_changed=0,
        **totals,
    )


def valence_prune_decode(
    output: dict[str, Any],
    *,
    symbols: Sequence[str],
    charge_values: Sequence[int],
    bridge_removal_penalty: float = 20.0,
) -> tuple[dict[str, Any], DecodeAudit]:
    """从 raw argmax 图做最小价态修剪，优先降级并避免删除图桥。"""

    import torch
    import torch.nn.functional as functional

    required = {"coords", "atomics", "bonds", "charges", "mask"}
    missing = sorted(required - set(output))
    if missing:
        raise KeyError(f"生成 output 缺字段: {missing}")
    atomics = output["atomics"]
    bonds = output["bonds"]
    charges = output["charges"]
    mask = output["mask"].bool()
    if atomics.ndim != 3 or charges.ndim != 3 or bonds.ndim != 4 or mask.ndim != 2:
        raise ValueError("生成 output tensor 维度不符合 SemlaFlow 契约")
    if bonds.shape[-1] != 5:
        raise ValueError("bond vocabulary 必须为 none/single/double/triple/aromatic 共 5 类")
    if len(symbols) != atomics.shape[-1] or len(charge_values) != charges.shape[-1]:
        raise ValueError("atom/charge vocabulary 与概率 tensor 不一致")
    if bridge_removal_penalty < 0:
        raise ValueError("bridge_removal_penalty 不能为负")

    decoded_bonds = torch.zeros_like(bonds)
    totals = {
        "raw_argmax_edges": 0,
        "decoded_edges": 0,
        "candidates_positive_gain": 0,
        "rejected_overvalence": 0,
        "rejected_pair_already_assigned": 0,
        "aromatic_bridge_edges_replaced": 0,
        "aromatic_bridge_edges_removed": 0,
        "initial_overvalent_atoms": 0,
        "downgraded_bonds": 0,
        "removed_bonds": 0,
        "bridge_removals": 0,
    }
    eps = np.finfo(np.float64).tiny
    for batch_index in range(mask.shape[0]):
        atom_count = int(mask[batch_index].sum().item())
        if atom_count <= 0:
            raise ValueError("生成 batch 含空分子")
        atom_probs = atomics[batch_index, :atom_count].detach().cpu().numpy()
        bond_probs = bonds[batch_index, :atom_count, :atom_count].detach().cpu().numpy()
        charge_probs = charges[batch_index, :atom_count].detach().cpu().numpy()
        atom_symbols = [str(symbols[int(index)]) for index in atom_probs.argmax(axis=-1)]
        formal_charges = [int(charge_values[int(index)]) for index in charge_probs.argmax(axis=-1)]
        capacities = np.asarray(
            [_max_valence_x2(symbol, charge) for symbol, charge in zip(atom_symbols, formal_charges)],
            dtype=np.int64,
        )
        symmetric_probs = (bond_probs + bond_probs.transpose(1, 0, 2)) / 2.0
        decoded = symmetric_probs.argmax(axis=-1).astype(np.int64)
        upper = np.triu(decoded, k=1)
        decoded = upper + upper.T
        used = np.zeros(atom_count, dtype=np.int64)
        for begin in range(atom_count):
            for end in range(begin + 1, atom_count):
                order = _BOND_ORDER_X2[int(decoded[begin, end])]
                used[begin] += order
                used[end] += order
        totals["raw_argmax_edges"] += int(np.count_nonzero(np.triu(decoded, k=1)))
        totals["initial_overvalent_atoms"] += int(np.count_nonzero(used > capacities))

        while np.any(used > capacities):
            actions: list[tuple[float, float, int, int, int, bool]] = []
            for begin in range(atom_count):
                for end in range(begin + 1, atom_count):
                    current = int(decoded[begin, end])
                    if current == 0 or not (used[begin] > capacities[begin] or used[end] > capacities[end]):
                        continue
                    current_order = _BOND_ORDER_X2[current]
                    current_log = math.log(max(float(symmetric_probs[begin, end, current]), eps))
                    for alternative in range(5):
                        alternative_order = _BOND_ORDER_X2[alternative]
                        if alternative_order >= current_order:
                            continue
                        alternative_log = math.log(
                            max(float(symmetric_probs[begin, end, alternative]), eps)
                        )
                        cost = current_log - alternative_log
                        removes_bridge = alternative == 0 and _is_bridge(decoded, begin, end)
                        score = cost + (bridge_removal_penalty if removes_bridge else 0.0)
                        actions.append(
                            (score, cost, begin, end, alternative, removes_bridge)
                        )
            if not actions:
                raise RuntimeError("无法通过键降级消除过价原子")
            _, _, begin, end, alternative, removes_bridge = min(actions)
            current = int(decoded[begin, end])
            reduction = _BOND_ORDER_X2[current] - _BOND_ORDER_X2[alternative]
            decoded[begin, end] = decoded[end, begin] = alternative
            used[begin] -= reduction
            used[end] -= reduction
            if alternative == 0:
                totals["removed_bonds"] += 1
                totals["bridge_removals"] += int(removes_bridge)
            else:
                totals["downgraded_bonds"] += 1

        # 非环 aromatic 边显式改为 single；键级下降，因此不会重新造成过价或断图。
        for begin, end in _aromatic_bridges(decoded):
            decoded[begin, end] = decoded[end, begin] = 1
            used[begin] -= _BOND_ORDER_X2[4] - _BOND_ORDER_X2[1]
            used[end] -= _BOND_ORDER_X2[4] - _BOND_ORDER_X2[1]
            totals["aromatic_bridge_edges_replaced"] += 1
            totals["downgraded_bonds"] += 1

        if np.any(used > capacities):
            raise RuntimeError("prune decoder 内部错误：输出仍存在过价原子")
        np.fill_diagonal(decoded, 0)
        totals["decoded_edges"] += int(np.count_nonzero(np.triu(decoded, k=1)))
        decoded_bonds[batch_index, :atom_count, :atom_count] = functional.one_hot(
            torch.as_tensor(decoded, dtype=torch.long, device=bonds.device),
            num_classes=5,
        ).to(dtype=bonds.dtype)

    decoded_output = {key: value.clone() for key, value in output.items()}
    decoded_output["bonds"] = decoded_bonds
    return decoded_output, DecodeAudit(
        molecule_count=int(mask.shape[0]),
        atom_types_changed=0,
        formal_charges_changed=0,
        **totals,
    )
