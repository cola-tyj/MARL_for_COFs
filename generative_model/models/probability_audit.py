"""模型无关的 categorical probability 审计工具。"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


PROBABILITY_AUDIT_VERSION = "categorical_probability_v1"


def _validate(probabilities: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("probabilities 必须为 [items, classes]，targets 必须为 [items]")
    if probabilities.shape[0] != targets.shape[0] or probabilities.shape[1] < 2:
        raise ValueError("probabilities/targets 形状不兼容")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("概率包含非有限值或负值")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("categorical probability 每行之和必须为 1")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("target class 越界")
    return probabilities, targets


def categorical_item_records(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    """返回逐 item 的预测、目标概率、排名、margin 与 NLL。"""

    probabilities, targets = _validate(probabilities, targets)
    item_indices = np.arange(targets.size)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    target_probabilities = probabilities[item_indices, targets]
    predicted_probabilities = probabilities[item_indices, predictions]
    # stable sort 使完全相同概率时以 class index 固定打破平局。
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = (order == targets[:, None]).argmax(axis=1).astype(np.int64) + 1
    return {
        "prediction": predictions,
        "target_probability": target_probabilities,
        "predicted_probability": predicted_probabilities,
        "target_rank": ranks,
        "margin": predicted_probabilities - target_probabilities,
        "negative_log_probability": -np.log(
            np.clip(target_probabilities, np.finfo(np.float64).tiny, 1.0)
        ),
    }


def _float_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def summarize_categorical(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    probabilities, targets = _validate(probabilities, targets)
    records = categorical_item_records(probabilities, targets)
    predictions = records["prediction"]
    ranks = records["target_rank"]
    class_count = probabilities.shape[1]
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    rank_counts = Counter(int(rank) for rank in ranks.tolist())
    return {
        "items": int(targets.size),
        "classes": int(class_count),
        "top1_accuracy": float(np.mean(predictions == targets)) if targets.size else None,
        "target_top2_rate": float(np.mean(ranks <= 2)) if targets.size else None,
        "target_rank_counts": {
            str(rank): int(rank_counts.get(rank, 0)) for rank in range(1, class_count + 1)
        },
        "target_probability": _float_summary(records["target_probability"]),
        "wrong_item_margin": _float_summary(records["margin"][predictions != targets]),
        "negative_log_probability": _float_summary(records["negative_log_probability"]),
        "confusion_target_rows_prediction_columns": confusion.tolist(),
    }


def pair_upper_triangle(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对 [N,N,C] 无向 pair 概率作对称平均并取严格上三角。"""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] != probabilities.shape[1]:
        raise ValueError("pair probabilities 必须为 [N,N,C]")
    atom_count = probabilities.shape[0]
    symmetric = (probabilities + probabilities.transpose(1, 0, 2)) / 2.0
    begin, end = np.triu_indices(atom_count, k=1)
    return symmetric[begin, end], np.stack((begin, end), axis=1).astype(np.int64)


def target_rank_route_decision(
    wrong_present_ranks: np.ndarray,
    *,
    wrong_present_target_probabilities: np.ndarray | None = None,
    wrong_present_margins: np.ndarray | None = None,
    minimum_items: int = 5,
    top2_threshold: float = 0.8,
    target_probability_threshold: float = 0.05,
    median_margin_threshold: float = 0.5,
) -> dict[str, Any]:
    """综合目标排名和置信度，区分解码边界问题与训练侧饱和错误。"""

    ranks = np.asarray(wrong_present_ranks, dtype=np.int64)
    target_probabilities = (
        None if wrong_present_target_probabilities is None
        else np.asarray(wrong_present_target_probabilities, dtype=np.float64)
    )
    margins = (
        None if wrong_present_margins is None
        else np.asarray(wrong_present_margins, dtype=np.float64)
    )
    if target_probabilities is not None and target_probabilities.shape != ranks.shape:
        raise ValueError("wrong target probabilities 与 ranks 形状不一致")
    if margins is not None and margins.shape != ranks.shape:
        raise ValueError("wrong margins 与 ranks 形状不一致")
    top2_rate = float(np.mean(ranks <= 2)) if ranks.size else None
    plausible_rate = (
        None if target_probabilities is None or not ranks.size
        else float(np.mean(target_probabilities >= target_probability_threshold))
    )
    median_margin = (
        None if margins is None or not ranks.size else float(np.median(margins))
    )
    if ranks.size < minimum_items:
        route = "insufficient_evidence"
    elif (
        top2_rate is not None
        and top2_rate >= top2_threshold
        and (plausible_rate is None or plausible_rate >= top2_threshold)
        and (median_margin is None or median_margin <= median_margin_threshold)
    ):
        route = "structured_decoder_first"
    else:
        route = "training_objective_first"
    return {
        "rule": (
            f"at least {minimum_items} wrong present bonds; target top-2 rate >= "
            f"{top2_threshold:.2f}; target probability >= {target_probability_threshold:.2f} "
            f"rate >= {top2_threshold:.2f}; median argmax-target margin <= "
            f"{median_margin_threshold:.2f}"
        ),
        "wrong_present_bond_items": int(ranks.size),
        "target_top2_rate": top2_rate,
        "target_probability_at_least_threshold_rate": plausible_rate,
        "median_argmax_target_margin": median_margin,
        "recommended_route": route,
    }


def _assignment_objective(
    assignment: np.ndarray,
    atom_probabilities: np.ndarray,
    charge_probabilities: np.ndarray,
    symmetric_bond_probabilities: np.ndarray,
    target_atoms: np.ndarray,
    target_charges: np.ndarray,
    target_adjacency: np.ndarray,
) -> float:
    """target node ``assignment[predicted node]`` 的联合 categorical NLL。"""

    atom_count = len(assignment)
    predicted = np.arange(atom_count)
    eps = np.finfo(np.float64).tiny
    objective = -np.log(np.clip(
        atom_probabilities[predicted, target_atoms[assignment]], eps, 1.0
    )).sum()
    objective -= np.log(np.clip(
        charge_probabilities[predicted, target_charges[assignment]], eps, 1.0
    )).sum()
    begin, end = np.triu_indices(atom_count, k=1)
    mapped_types = target_adjacency[assignment[begin], assignment[end]]
    objective -= np.log(np.clip(
        symmetric_bond_probabilities[begin, end, mapped_types], eps, 1.0
    )).sum()
    return float(objective)


def align_target_nodes(
    atom_probabilities: np.ndarray,
    charge_probabilities: np.ndarray,
    bond_probabilities: np.ndarray,
    predicted_coordinates: np.ndarray,
    target_atoms: np.ndarray,
    target_charges: np.ndarray,
    target_adjacency: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    seed: int,
    random_starts: int = 4,
    max_swap_passes: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """以多起点 Hungarian + pair-swap 最小化联合 NLL，消除节点置换任意性。

    坐标只用于构造一个旋转/平移/置换不变的 distance-profile 初值；最终选择和
    swap 优化均由 atom/charge/bond categorical NLL 决定。因此它是明确标注的
    oracle target alignment，不是生成时可用的 decoder。
    """

    from scipy.optimize import linear_sum_assignment

    atom_probabilities, target_atoms = _validate(atom_probabilities, target_atoms)
    charge_probabilities, target_charges = _validate(charge_probabilities, target_charges)
    bond_probabilities = np.asarray(bond_probabilities, dtype=np.float64)
    predicted_coordinates = np.asarray(predicted_coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target_coordinates, dtype=np.float64)
    target_adjacency = np.asarray(target_adjacency, dtype=np.int64)
    atom_count = atom_probabilities.shape[0]
    expected_pair_shape = (atom_count, atom_count, target_adjacency.max(initial=0) + 1)
    if bond_probabilities.shape[:2] != (atom_count, atom_count):
        raise ValueError("bond probability 原子维度不匹配")
    if target_adjacency.shape != (atom_count, atom_count):
        raise ValueError("target adjacency 形状不匹配")
    if predicted_coordinates.shape != (atom_count, 3) or target_coordinates.shape != (atom_count, 3):
        raise ValueError("坐标必须为 [N,3]")
    if target_adjacency.max(initial=0) >= bond_probabilities.shape[-1]:
        raise ValueError(f"target bond class 超过预测类别数: {expected_pair_shape}")
    symmetric_bonds = (bond_probabilities + bond_probabilities.transpose(1, 0, 2)) / 2.0
    eps = np.finfo(np.float64).tiny

    node_cost = -np.log(np.clip(atom_probabilities[:, target_atoms], eps, 1.0))
    node_cost -= np.log(np.clip(charge_probabilities[:, target_charges], eps, 1.0))
    pred_distances = np.linalg.norm(
        predicted_coordinates[:, None, :] - predicted_coordinates[None, :, :], axis=-1
    )
    target_distances = np.linalg.norm(
        target_coordinates[:, None, :] - target_coordinates[None, :, :], axis=-1
    )
    pred_profiles = np.sort(pred_distances, axis=1)
    target_profiles = np.sort(target_distances, axis=1)
    geometry_cost = np.linalg.norm(
        pred_profiles[:, None, :] - target_profiles[None, :, :], axis=-1
    )

    def hungarian(cost: np.ndarray) -> np.ndarray:
        rows, columns = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(atom_count)):
            raise RuntimeError("Hungarian assignment 未覆盖全部预测节点")
        return columns.astype(np.int64)

    node_scale = max(float(np.std(node_cost)), 1e-12)
    geometry_scale = max(float(np.std(geometry_cost)), 1e-12)
    starts: list[tuple[str, np.ndarray]] = [
        ("identity", np.arange(atom_count, dtype=np.int64)),
        ("node_hungarian", hungarian(node_cost)),
        ("geometry_hungarian", hungarian(geometry_cost)),
        ("combined_hungarian", hungarian(node_cost / node_scale + geometry_cost / geometry_scale)),
    ]
    generator = np.random.default_rng(seed)
    starts.extend(
        (f"random_{index}", generator.permutation(atom_count).astype(np.int64))
        for index in range(random_starts)
    )

    candidates: list[tuple[float, str, np.ndarray, int]] = []
    for name, start in starts:
        assignment = start.copy()
        objective = _assignment_objective(
            assignment, atom_probabilities, charge_probabilities, symmetric_bonds,
            target_atoms, target_charges, target_adjacency,
        )
        accepted_swaps = 0
        for _ in range(max_swap_passes):
            best: tuple[float, int, int] | None = None
            for begin in range(atom_count):
                for end in range(begin + 1, atom_count):
                    proposal = assignment.copy()
                    proposal[begin], proposal[end] = proposal[end], proposal[begin]
                    proposal_objective = _assignment_objective(
                        proposal, atom_probabilities, charge_probabilities, symmetric_bonds,
                        target_atoms, target_charges, target_adjacency,
                    )
                    improvement = objective - proposal_objective
                    if improvement > 1e-10 and (best is None or improvement > best[0] + 1e-12):
                        best = (improvement, begin, end)
            if best is None:
                break
            _, begin, end = best
            assignment[begin], assignment[end] = assignment[end], assignment[begin]
            objective = _assignment_objective(
                assignment, atom_probabilities, charge_probabilities, symmetric_bonds,
                target_atoms, target_charges, target_adjacency,
            )
            accepted_swaps += 1
        candidates.append((objective, name, assignment, accepted_swaps))

    candidates.sort(key=lambda item: (item[0], item[1], item[2].tolist()))
    objective, start_name, assignment, accepted_swaps = candidates[0]
    if sorted(assignment.tolist()) != list(range(atom_count)):
        raise RuntimeError("target alignment 不是合法 permutation")
    identity_objective = _assignment_objective(
        np.arange(atom_count), atom_probabilities, charge_probabilities, symmetric_bonds,
        target_atoms, target_charges, target_adjacency,
    )
    return assignment, {
        "method": "multi-start Hungarian initialization plus deterministic best pair swaps",
        "oracle_uses_target": True,
        "starts": len(starts),
        "selected_start": start_name,
        "accepted_swaps": accepted_swaps,
        "identity_joint_nll": identity_objective,
        "aligned_joint_nll": objective,
        "joint_nll_improvement": identity_objective - objective,
    }
