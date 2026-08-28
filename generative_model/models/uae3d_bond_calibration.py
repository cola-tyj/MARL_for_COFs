"""UAE-3D hierarchical bond existence 的确定性全局阈值校准。

校准只使用 unique off-diagonal pairs，一个 checkpoint 只产生一个全局
threshold。本模块不读取 split，由调用方保证 test/Core-OOD 不参与拟合。
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from generative_model.models.uae3d_reconstruction import full_pair_targets


BOND_CALIBRATION_SCHEMA_VERSION = "uae3d-global-bond-threshold-calibration-v1"
CALIBRATION_OBJECTIVE = (
    "maximize macro molecule-balanced existence accuracy; then maximize minimum "
    "molecule-balanced existence accuracy; then minimize total existence errors; "
    "then choose threshold closest to zero; then choose the smaller threshold"
)


def _little_endian(array: np.ndarray, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False))


def _fingerprint_arrays(table: dict[str, Any]) -> str:
    digest = hashlib.sha256(BOND_CALIBRATION_SCHEMA_VERSION.encode("ascii"))
    fields = (
        ("scores", "<f8"),
        ("targets", "<i2"),
        ("type_predictions", "<i2"),
        ("molecule_offsets", "<i8"),
        ("package_indices", "<i8"),
    )
    for name, dtype in fields:
        array = _little_endian(table[name], dtype)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def calibration_table_from_logits(
    bond_logits,
    edge_attr,
    atom_counts: Sequence[int],
    package_indices: Sequence[int],
    *,
    symmetry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Extract ``max(type)-none`` scores on unique pairs with strict validation."""

    import torch

    if bond_logits.ndim != 2 or tuple(bond_logits.shape[1:]) != (6,):
        raise ValueError("bond_logits 必须为 [sum(N²),6]")
    if not bool(torch.isfinite(bond_logits).all()):
        raise ValueError("bond_logits 含 NaN/Inf")
    counts = [int(value) for value in atom_counts]
    indices = [int(value) for value in package_indices]
    if not counts or len(counts) != len(indices) or any(value <= 1 for value in counts):
        raise ValueError("atom_counts/package_indices 数量不一致或非法")
    if len(set(indices)) != len(indices):
        raise ValueError("package_indices 必须唯一")
    if sum(value * value for value in counts) != len(bond_logits):
        raise ValueError("atom_counts 与 full-pair logits 长度不一致")
    if len(edge_attr) != len(bond_logits):
        raise ValueError("edge_attr 与 bond_logits 长度不一致")
    if symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance 不得为负")

    targets = full_pair_targets(edge_attr)
    score_parts = []
    target_parts = []
    type_parts = []
    offsets = [0]
    begin = 0
    for molecule_index, atom_count in enumerate(counts):
        end = begin + atom_count * atom_count
        local_logits = bond_logits[begin:end].reshape(atom_count, atom_count, 6)
        local_targets = targets[begin:end].reshape(atom_count, atom_count)
        logit_error = float(
            (local_logits - local_logits.transpose(0, 1)).abs().max().detach().cpu()
        )
        if logit_error > symmetry_tolerance:
            raise RuntimeError(
                f"molecule {molecule_index} bond logits 非对称: {logit_error}"
            )
        if not bool(torch.equal(local_targets, local_targets.T)):
            raise RuntimeError(f"molecule {molecule_index} bond target 非对称")
        if not bool((torch.diag(local_targets) == 5).all()):
            raise RuntimeError(f"molecule {molecule_index} diagonal target 不是 self")
        upper = torch.triu(
            torch.ones(
                (atom_count, atom_count), dtype=torch.bool, device=bond_logits.device
            ),
            diagonal=1,
        )
        upper_targets = local_targets[upper]
        if not bool(((upper_targets >= 0) & (upper_targets <= 4)).all()):
            raise RuntimeError(f"molecule {molecule_index} off-diagonal target 非法")
        if not bool((upper_targets == 0).any()) or not bool((upper_targets > 0).any()):
            raise ValueError(
                f"molecule {molecule_index} 必须同时包含 none/present pair"
            )
        upper_logits = local_logits[upper]
        type_logits = upper_logits[:, 1:5]
        scores = type_logits.max(dim=-1).values - upper_logits[:, 0]
        type_predictions = type_logits.argmax(dim=-1) + 1
        score_parts.append(scores.detach().to("cpu", torch.float64).numpy())
        target_parts.append(upper_targets.detach().to("cpu", torch.int16).numpy())
        type_parts.append(type_predictions.detach().to("cpu", torch.int16).numpy())
        offsets.append(offsets[-1] + int(upper_targets.numel()))
        begin = end

    table = {
        "schema_version": BOND_CALIBRATION_SCHEMA_VERSION,
        "score_definition": "max(logits[1:5])-logits[0]",
        "decision_rule": "present iff score > threshold; ties are none",
        "scores": np.concatenate(score_parts).astype(np.float64, copy=False),
        "targets": np.concatenate(target_parts).astype(np.int16, copy=False),
        "type_predictions": np.concatenate(type_parts).astype(np.int16, copy=False),
        "molecule_offsets": np.asarray(offsets, dtype=np.int64),
        "package_indices": np.asarray(indices, dtype=np.int64),
    }
    table["fingerprint"] = _fingerprint_arrays(table)
    return table


def calibration_table_from_arrays(
    scores: np.ndarray,
    targets: np.ndarray,
    type_predictions: np.ndarray,
    molecule_offsets: np.ndarray,
    package_indices: np.ndarray,
    *,
    score_definition: str,
) -> dict[str, Any]:
    """Build a strict table for an auxiliary frozen-feature probe."""

    table = {
        "schema_version": BOND_CALIBRATION_SCHEMA_VERSION,
        "score_definition": str(score_definition),
        "decision_rule": "present iff score > threshold; ties are none",
        "scores": np.asarray(scores, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.int16),
        "type_predictions": np.asarray(type_predictions, dtype=np.int16),
        "molecule_offsets": np.asarray(molecule_offsets, dtype=np.int64),
        "package_indices": np.asarray(package_indices, dtype=np.int64),
    }
    # Reuse the full validator before exposing a trusted fingerprint.
    table["fingerprint"] = _fingerprint_arrays(table)
    _table_arrays(table)
    return table


def merge_calibration_tables(tables: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge independently extracted batches without changing molecule order."""

    if not tables:
        raise ValueError("不能合并空 calibration tables")
    for table in tables:
        if table.get("schema_version") != BOND_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("calibration table schema 不一致")
        if table.get("fingerprint") != _fingerprint_arrays(table):
            raise RuntimeError("calibration table fingerprint 不匹配")
    scores = np.concatenate([np.asarray(table["scores"]) for table in tables])
    targets = np.concatenate([np.asarray(table["targets"]) for table in tables])
    type_predictions = np.concatenate([
        np.asarray(table["type_predictions"]) for table in tables
    ])
    package_indices = np.concatenate([
        np.asarray(table["package_indices"]) for table in tables
    ])
    offsets = [0]
    for table in tables:
        sizes = np.diff(np.asarray(table["molecule_offsets"], dtype=np.int64))
        offsets.extend((offsets[-1] + np.cumsum(sizes)).tolist())
    merged = {
        "schema_version": BOND_CALIBRATION_SCHEMA_VERSION,
        "score_definition": "max(logits[1:5])-logits[0]",
        "decision_rule": "present iff score > threshold; ties are none",
        "scores": scores.astype(np.float64, copy=False),
        "targets": targets.astype(np.int16, copy=False),
        "type_predictions": type_predictions.astype(np.int16, copy=False),
        "molecule_offsets": np.asarray(offsets, dtype=np.int64),
        "package_indices": package_indices.astype(np.int64, copy=False),
    }
    if len(set(map(int, package_indices))) != len(package_indices):
        raise ValueError("合并后 package_indices 不唯一")
    merged["fingerprint"] = _fingerprint_arrays(merged)
    return merged


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Enumerate every distinct decision partition plus the raw zero threshold."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores 必须为非空有限一维数组")
    unique = np.unique(values)
    below_minimum = np.nextafter(unique[0], -np.inf)
    candidates = np.unique(np.concatenate((np.asarray([below_minimum, 0.0]), unique)))
    if not np.isfinite(candidates).all():
        raise RuntimeError("threshold candidate 含非有限值")
    return candidates.astype(np.float64, copy=False)


def _table_arrays(table: dict[str, Any]):
    if table.get("schema_version") != BOND_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("calibration table schema 错误")
    if table.get("fingerprint") != _fingerprint_arrays(table):
        raise RuntimeError("calibration table fingerprint 不匹配")
    scores = np.asarray(table["scores"], dtype=np.float64)
    targets = np.asarray(table["targets"], dtype=np.int16)
    type_predictions = np.asarray(table["type_predictions"], dtype=np.int16)
    offsets = np.asarray(table["molecule_offsets"], dtype=np.int64)
    indices = np.asarray(table["package_indices"], dtype=np.int64)
    if scores.ndim != 1 or targets.shape != scores.shape or type_predictions.shape != scores.shape:
        raise ValueError("calibration arrays shape 不一致")
    if len(offsets) != len(indices) + 1 or offsets[0] != 0 or offsets[-1] != len(scores):
        raise ValueError("molecule_offsets 非法")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("molecule_offsets 必须严格递增")
    if np.any((targets < 0) | (targets > 4)):
        raise ValueError("targets 超出 0..4")
    if np.any((type_predictions < 1) | (type_predictions > 4)):
        raise ValueError("type_predictions 超出 1..4")
    return scores, targets, type_predictions, offsets, indices


def evaluate_threshold(table: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Evaluate one global threshold; no molecule-specific branch is available."""

    if not math.isfinite(threshold):
        raise ValueError("threshold 必须为有限数")
    scores, targets, type_predictions, offsets, indices = _table_arrays(table)
    present_prediction = scores > float(threshold)
    predictions = np.where(present_prediction, type_predictions, 0).astype(np.int16)
    present_target = targets > 0
    none_target = ~present_target
    records = []
    macro_balanced = []
    existence_errors = 0
    exact_molecules = 0
    for molecule_index, package_index in enumerate(indices):
        begin, end = map(int, offsets[molecule_index : molecule_index + 2])
        local_present_target = present_target[begin:end]
        local_none_target = none_target[begin:end]
        local_present_prediction = present_prediction[begin:end]
        local_predictions = predictions[begin:end]
        local_targets = targets[begin:end]
        none_correct = int((~local_present_prediction[local_none_target]).sum())
        present_correct = int(local_present_prediction[local_present_target].sum())
        none_total = int(local_none_target.sum())
        present_total = int(local_present_target.sum())
        if none_total == 0 or present_total == 0:
            raise RuntimeError("calibration molecule 缺少 none/present stratum")
        balanced_fraction = (
            Fraction(none_correct, none_total)
            + Fraction(present_correct, present_total)
        ) / 2
        macro_balanced.append(balanced_fraction)
        local_existence_errors = (
            none_total - none_correct + present_total - present_correct
        )
        existence_errors += local_existence_errors
        exact_type_correct = int((local_predictions == local_targets).sum())
        molecule_exact = exact_type_correct == end - begin
        exact_molecules += int(molecule_exact)
        records.append({
            "package_index": int(package_index),
            "pair_count_unique": end - begin,
            "none_correct": none_correct,
            "none_total": none_total,
            "present_existence_correct": present_correct,
            "present_total": present_total,
            "balanced_existence_accuracy": float(balanced_fraction),
            "existence_error_count": local_existence_errors,
            "exact_type_correct": exact_type_correct,
            "wrong_unique_pair_count": end - begin - exact_type_correct,
            "bond_exact": molecule_exact,
        })

    none_correct = int((~present_prediction[none_target]).sum())
    present_existence_correct = int(present_prediction[present_target].sum())
    present_exact_type_correct = int((predictions[present_target] == targets[present_target]).sum())
    exact_type_correct = int((predictions == targets).sum())
    macro_fraction = sum(macro_balanced, Fraction(0, 1)) / len(macro_balanced)
    minimum_fraction = min(macro_balanced)
    return {
        "threshold": float(threshold),
        "molecule_count": len(indices),
        "pair_count_unique": len(scores),
        "macro_molecule_balanced_existence_accuracy": float(macro_fraction),
        "minimum_molecule_balanced_existence_accuracy": float(minimum_fraction),
        "existence_accuracy": float(1.0 - existence_errors / len(scores)),
        "existence_error_count": existence_errors,
        "none_accuracy": float(none_correct / int(none_target.sum())),
        "present_existence_recall": float(
            present_existence_correct / int(present_target.sum())
        ),
        "present_exact_type_accuracy": float(
            present_exact_type_correct / int(present_target.sum())
        ),
        "full_unique_pair_exact_type_accuracy": float(exact_type_correct / len(scores)),
        "wrong_unique_pair_count": len(scores) - exact_type_correct,
        "bond_exact_molecules": exact_molecules,
        "records": records,
        "_objective_exact": {
            "macro": macro_fraction,
            "minimum": minimum_fraction,
        },
    }


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "_objective_exact"}


def fit_global_threshold(table: dict[str, Any]) -> dict[str, Any]:
    """Fit a deterministic single threshold using the frozen lexicographic objective."""

    scores, targets, _, offsets, indices = _table_arrays(table)
    candidates = candidate_thresholds(scores)
    molecule_ids = np.repeat(np.arange(len(indices), dtype=np.int64), np.diff(offsets))
    none_target = targets == 0
    present_target = ~none_target
    none_totals = np.bincount(
        molecule_ids[none_target], minlength=len(indices)
    ).astype(np.int64, copy=False)
    present_totals = np.bincount(
        molecule_ids[present_target], minlength=len(indices)
    ).astype(np.int64, copy=False)
    if np.any(none_totals == 0) or np.any(present_totals == 0):
        raise RuntimeError("calibration molecule 缺少 none/present stratum")

    # threshold 低于最小 score 时所有 pair 均预测 present。threshold 递增时，
    # score <= threshold 的 pair 各自只从 present 切换为 none 一次。
    none_correct = np.zeros(len(indices), dtype=np.int64)
    present_correct = present_totals.copy()
    existence_errors = int(none_totals.sum())
    order = np.argsort(scores, kind="mergesort")
    cursor = 0
    best_threshold = None
    best_key = None
    for threshold in candidates:
        while cursor < len(order) and scores[order[cursor]] <= threshold:
            pair_index = int(order[cursor])
            molecule_index = int(molecule_ids[pair_index])
            if none_target[pair_index]:
                none_correct[molecule_index] += 1
                existence_errors -= 1
            else:
                present_correct[molecule_index] -= 1
                existence_errors += 1
            cursor += 1
        balanced = (
            none_correct.astype(np.float64) / none_totals
            + present_correct.astype(np.float64) / present_totals
        ) / 2.0
        macro = float(balanced.mean())
        minimum = float(balanced.min())
        key = (
            macro,
            minimum,
            -existence_errors,
            -abs(float(threshold)),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    if best_threshold is None:
        raise RuntimeError("未能选出 calibration threshold")
    best_metrics = evaluate_threshold(table, best_threshold)
    candidate_bytes = _little_endian(candidates, "<f8").tobytes()
    return {
        "schema_version": BOND_CALIBRATION_SCHEMA_VERSION,
        "fit_table_fingerprint": table["fingerprint"],
        "score_definition": str(table["score_definition"]),
        "decision_rule": "present iff score > threshold; ties are none",
        "objective": CALIBRATION_OBJECTIVE,
        "candidate_protocol": (
            "sorted unique fit scores + nextafter(min_score,-inf) + zero"
        ),
        "candidate_count": len(candidates),
        "candidate_fingerprint": hashlib.sha256(candidate_bytes).hexdigest(),
        "selected_threshold": float(best_metrics["threshold"]),
        "raw_zero": _public_metrics(evaluate_threshold(table, 0.0)),
        "selected": _public_metrics(best_metrics),
    }


def public_threshold_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Strip internal exact-comparison values before JSON serialization."""

    return _public_metrics(metrics)
