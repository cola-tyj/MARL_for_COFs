"""Deterministic auxiliary probes for frozen UAE-3D bond representations.

These probes diagnose representation/capacity only. They neither modify UAE-3D
weights nor constitute a tier-training gate.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from generative_model.models.uae3d_bond_calibration import (
    calibration_table_from_arrays,
    evaluate_threshold,
    fit_global_threshold,
    public_threshold_metrics,
)


CAPACITY_SCHEMA_VERSION = "uae3d-frozen-bond-capacity-v1"
PROBE_L2 = 1e-2


def _as_little_endian(array: np.ndarray, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False))


def feature_table_fingerprint(table: dict[str, Any]) -> str:
    digest = hashlib.sha256(CAPACITY_SCHEMA_VERSION.encode("ascii"))
    fields = (
        ("pair_features", "<f4"),
        ("hidden_features", "<f4"),
        ("legacy_logits", "<f4"),
        ("targets", "<i2"),
        ("molecule_offsets", "<i8"),
        ("package_indices", "<i8"),
    )
    for name, dtype in fields:
        value = _as_little_endian(table[name], dtype)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_feature_table(table: dict[str, Any]) -> None:
    if table.get("schema_version") != CAPACITY_SCHEMA_VERSION:
        raise ValueError("capacity feature table schema 错误")
    if table.get("fingerprint") != feature_table_fingerprint(table):
        raise RuntimeError("capacity feature table fingerprint 不匹配")
    pair = np.asarray(table["pair_features"])
    hidden = np.asarray(table["hidden_features"])
    logits = np.asarray(table["legacy_logits"])
    targets = np.asarray(table["targets"])
    offsets = np.asarray(table["molecule_offsets"])
    indices = np.asarray(table["package_indices"])
    if pair.ndim != 2 or pair.shape[1] != 64:
        raise ValueError("pair_features 必须为 [P,64]")
    if hidden.shape != pair.shape or logits.shape != (len(pair), 6):
        raise ValueError("hidden/logits shape 与 pair_features 不一致")
    if targets.shape != (len(pair),):
        raise ValueError("targets shape 错误")
    if len(offsets) != len(indices) + 1 or offsets[0] != 0 or offsets[-1] != len(pair):
        raise ValueError("molecule_offsets 非法")
    if np.any(np.diff(offsets) <= 0) or len(set(map(int, indices))) != len(indices):
        raise ValueError("molecule offsets/indices 非法")
    if np.any((targets < 0) | (targets > 4)):
        raise ValueError("off-diagonal target 超出 0..4")
    if not all(np.isfinite(value).all() for value in (pair, hidden, logits)):
        raise ValueError("capacity features 含 NaN/Inf")


def make_feature_table(
    pair_features: np.ndarray,
    hidden_features: np.ndarray,
    legacy_logits: np.ndarray,
    targets: np.ndarray,
    molecule_offsets: np.ndarray,
    package_indices: np.ndarray,
) -> dict[str, Any]:
    table = {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "pair_features": np.asarray(pair_features, dtype=np.float32),
        "hidden_features": np.asarray(hidden_features, dtype=np.float32),
        "legacy_logits": np.asarray(legacy_logits, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.int16),
        "molecule_offsets": np.asarray(molecule_offsets, dtype=np.int64),
        "package_indices": np.asarray(package_indices, dtype=np.int64),
    }
    table["fingerprint"] = feature_table_fingerprint(table)
    validate_feature_table(table)
    return table


def merge_feature_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    if not tables:
        raise ValueError("不能合并空 capacity feature tables")
    for table in tables:
        validate_feature_table(table)
    offsets = [0]
    for table in tables:
        sizes = np.diff(np.asarray(table["molecule_offsets"], dtype=np.int64))
        offsets.extend((offsets[-1] + np.cumsum(sizes)).tolist())
    return make_feature_table(
        np.concatenate([table["pair_features"] for table in tables]),
        np.concatenate([table["hidden_features"] for table in tables]),
        np.concatenate([table["legacy_logits"] for table in tables]),
        np.concatenate([table["targets"] for table in tables]),
        np.asarray(offsets, dtype=np.int64),
        np.concatenate([table["package_indices"] for table in tables]),
    )


def balanced_fit_rows(table: dict[str, Any]) -> np.ndarray:
    """Select all present pairs and an equal deterministic none sample per molecule."""

    validate_feature_table(table)
    targets = np.asarray(table["targets"])
    offsets = np.asarray(table["molecule_offsets"])
    selected: list[np.ndarray] = []
    for molecule_index in range(len(offsets) - 1):
        begin, end = map(int, offsets[molecule_index : molecule_index + 2])
        local = targets[begin:end]
        present = np.flatnonzero(local > 0)
        none = np.flatnonzero(local == 0)
        if not len(present) or len(none) < len(present):
            raise RuntimeError("probe molecule 缺少足够的 none/present pair")
        if len(present) == 1:
            none_positions = np.asarray([len(none) // 2], dtype=np.int64)
        else:
            none_positions = (
                np.arange(len(present), dtype=np.int64) * (len(none) - 1)
            ) // (len(present) - 1)
        chosen_none = none[none_positions]
        local_selected = np.sort(np.concatenate((present, chosen_none))) + begin
        selected.append(local_selected)
    result = np.concatenate(selected).astype(np.int64, copy=False)
    labels = targets[result] > 0
    if int(labels.sum()) * 2 != len(labels):
        raise RuntimeError("balanced row selection 未形成 1:1 existence 样本")
    return result


def _standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    constant = scale <= np.finfo(np.float64).eps
    scale[constant] = 1.0
    return mean, scale, int(constant.sum())


def _transform(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = (np.asarray(features, dtype=np.float64) - mean) / scale
    if not np.isfinite(values).all():
        raise RuntimeError("standardized probe features 含 NaN/Inf")
    return values


def fit_binary_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = PROBE_L2,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fit deterministic float64 Newton/IRLS logistic regression."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("binary probe features/labels shape 非法")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("binary probe 必须同时包含 0/1")
    if l2 <= 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("binary probe optimizer 参数非法")
    mean, scale, constant_count = _standardize_fit(x)
    standardized = _transform(x, mean, scale)
    design = np.column_stack((np.ones(len(x)), standardized))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.ones(design.shape[1], dtype=np.float64)
    regularizer[0] = 0.0
    converged = False
    final_step = None
    for iteration in range(1, max_iterations + 1):
        logits = np.clip(design @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        gradient = design.T @ (probabilities - y) + l2 * regularizer * weights
        hessian = (design.T * curvature) @ design
        hessian.flat[:: len(weights) + 1] += l2 * regularizer
        step = np.linalg.solve(hessian, gradient)
        if not np.isfinite(step).all():
            raise RuntimeError("binary probe Newton step 含 NaN/Inf")
        weights -= step
        final_step = float(np.max(np.abs(step)))
        if final_step <= tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"binary probe 未在 {max_iterations} 次迭代内收敛; step={final_step}"
        )
    return {
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "l2": float(l2),
        "iterations": iteration,
        "converged": converged,
        "max_abs_final_step": final_step,
        "constant_feature_count": constant_count,
    }


def binary_scores(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = _transform(features, model["mean"], model["scale"])
    return model["weights"][0] + standardized @ model["weights"][1:]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact tie-aware binary ROC-AUC without an external sklearn dependency."""

    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if values.shape != target.shape or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ROC-AUC input 非法")
    positives = int(target.sum())
    negatives = len(target) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC 必须同时包含正负样本")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_target = target[order]
    favorable = 0.0
    negatives_before = 0
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and sorted_values[end] == sorted_values[begin]:
            end += 1
        group_positive = int(sorted_target[begin:end].sum())
        group_negative = end - begin - group_positive
        favorable += group_positive * (negatives_before + 0.5 * group_negative)
        negatives_before += group_negative
        begin = end
    return float(favorable / (positives * negatives))


def fit_type_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float = PROBE_L2,
) -> dict[str, Any]:
    """Fit fixed-L2 class-balanced ridge scores on target-present pairs."""

    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or labels.shape != (len(x),) or not len(x):
        raise ValueError("type probe features/targets shape 非法")
    if np.any((labels < 1) | (labels > 4)):
        raise ValueError("type probe targets 必须为 1..4")
    counts = np.bincount(labels, minlength=5)[1:]
    if np.any(counts == 0):
        raise RuntimeError(f"type probe train 缺少 bond class: {np.flatnonzero(counts == 0) + 1}")
    mean, scale, constant_count = _standardize_fit(x)
    design = np.column_stack((np.ones(len(x)), _transform(x, mean, scale)))
    sample_weight = np.asarray(
        [len(labels) / (4.0 * counts[label - 1]) for label in labels],
        dtype=np.float64,
    )
    one_hot = np.eye(4, dtype=np.float64)[labels - 1]
    weighted_design = design * sample_weight[:, None]
    normal = design.T @ weighted_design
    regularizer = np.ones(design.shape[1], dtype=np.float64)
    regularizer[0] = 0.0
    normal.flat[:: design.shape[1] + 1] += l2 * regularizer
    coefficients = np.linalg.solve(normal, design.T @ (sample_weight[:, None] * one_hot))
    if not np.isfinite(coefficients).all():
        raise RuntimeError("type ridge coefficients 含 NaN/Inf")
    return {
        "coefficients": coefficients,
        "mean": mean,
        "scale": scale,
        "l2": float(l2),
        "class_counts": counts,
        "constant_feature_count": constant_count,
    }


def type_predictions(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    design = np.column_stack((
        np.ones(len(features)),
        _transform(features, model["mean"], model["scale"]),
    ))
    return (np.argmax(design @ model["coefficients"], axis=1) + 1).astype(np.int16)


def type_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    if predicted.shape != target.shape or np.any((target < 1) | (target > 4)):
        raise ValueError("type metrics input 非法")
    per_class = {}
    accuracies = []
    for class_index, name in enumerate(("single", "double", "triple", "aromatic"), 1):
        mask = target == class_index
        total = int(mask.sum())
        correct = int((predicted[mask] == class_index).sum())
        accuracy = float(correct / total) if total else None
        if accuracy is not None:
            accuracies.append(accuracy)
        per_class[name] = {"class_index": class_index, "correct": correct, "total": total, "accuracy": accuracy}
    return {
        "micro_accuracy": float((predicted == target).mean()),
        "macro_supported_class_accuracy": float(np.mean(accuracies)),
        "per_class": per_class,
    }


def _existence_public(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "threshold", "molecule_count", "pair_count_unique",
        "macro_molecule_balanced_existence_accuracy",
        "minimum_molecule_balanced_existence_accuracy", "existence_accuracy",
        "existence_error_count", "none_accuracy", "present_existence_recall",
    )
    return {key: metrics[key] for key in keys}


def audit_feature_probe(
    fit_table: dict[str, Any],
    transfer_table: dict[str, Any],
    feature_name: str,
) -> dict[str, Any]:
    """Fit one frozen-feature probe and transfer it unchanged to IID validation."""

    validate_feature_table(fit_table)
    validate_feature_table(transfer_table)
    if feature_name not in {"pair_features", "hidden_features"}:
        raise ValueError("unknown capacity feature")
    fit_features = np.asarray(fit_table[feature_name])
    transfer_features = np.asarray(transfer_table[feature_name])
    fit_targets = np.asarray(fit_table["targets"])
    transfer_targets = np.asarray(transfer_table["targets"])
    rows = balanced_fit_rows(fit_table)
    binary_model = fit_binary_logistic(fit_features[rows], fit_targets[rows] > 0)
    fit_scores = binary_scores(binary_model, fit_features)
    transfer_scores = binary_scores(binary_model, transfer_features)
    fit_binary_targets = (fit_targets > 0).astype(np.int16)
    transfer_binary_targets = (transfer_targets > 0).astype(np.int16)
    fit_calibration_table = calibration_table_from_arrays(
        fit_scores, fit_binary_targets, np.ones(len(fit_scores), dtype=np.int16),
        fit_table["molecule_offsets"], fit_table["package_indices"],
        score_definition=f"auxiliary logistic score on frozen {feature_name}",
    )
    transfer_calibration_table = calibration_table_from_arrays(
        transfer_scores, transfer_binary_targets,
        np.ones(len(transfer_scores), dtype=np.int16),
        transfer_table["molecule_offsets"], transfer_table["package_indices"],
        score_definition=f"auxiliary logistic score on frozen {feature_name}",
    )
    calibration = fit_global_threshold(fit_calibration_table)
    threshold = float(calibration["selected_threshold"])
    transfer_raw = evaluate_threshold(transfer_calibration_table, 0.0)
    transfer_calibrated = evaluate_threshold(transfer_calibration_table, threshold)

    fit_present = fit_targets > 0
    transfer_present = transfer_targets > 0
    type_model = fit_type_ridge(fit_features[fit_present], fit_targets[fit_present])
    fit_type_predictions = type_predictions(type_model, fit_features[fit_present])
    transfer_type_predictions = type_predictions(
        type_model, transfer_features[transfer_present]
    )
    calibrated_public = _existence_public(transfer_calibrated)
    auc = roc_auc(transfer_scores, transfer_present)
    if (
        auc >= 0.85
        and calibrated_public["macro_molecule_balanced_existence_accuracy"] >= 0.75
        and calibrated_public["none_accuracy"] >= 0.70
        and calibrated_public["present_existence_recall"] >= 0.70
    ):
        signal = "FEATURE_SEPARABLE"
    elif auc >= 0.70:
        signal = "WEAK_LINEAR_SIGNAL"
    else:
        signal = "NOT_LINEARLY_SEPARABLE"
    return {
        "feature_name": feature_name,
        "probe_scope": "diagnostic_only_not_a_tier_gate",
        "fit_balanced_pair_count": len(rows),
        "fit_present_pair_count": int(fit_present.sum()),
        "transfer_present_pair_count": int(transfer_present.sum()),
        "binary_probe": {
            "algorithm": "float64 Newton/IRLS logistic regression",
            "l2": binary_model["l2"],
            "iterations": binary_model["iterations"],
            "converged": binary_model["converged"],
            "constant_feature_count": binary_model["constant_feature_count"],
            "fit_roc_auc_all_pairs": roc_auc(fit_scores, fit_present),
            "transfer_roc_auc_all_pairs": auc,
            "fit_global_threshold": threshold,
            "fit_calibration": {
                "raw_zero": _existence_public(calibration["raw_zero"]),
                "selected": _existence_public(calibration["selected"]),
            },
            "transfer_raw_zero": _existence_public(transfer_raw),
            "transfer_calibrated": calibrated_public,
        },
        "conditional_type_probe": {
            "algorithm": "class-balanced float64 ridge least squares",
            "l2": type_model["l2"],
            "constant_feature_count": type_model["constant_feature_count"],
            "fit": type_metrics(fit_type_predictions, fit_targets[fit_present]),
            "transfer": type_metrics(
                transfer_type_predictions, transfer_targets[transfer_present]
            ),
        },
        "linear_signal_classification": signal,
    }


def audit_legacy_head(
    fit_table: dict[str, Any], transfer_table: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen official 6-class head under the same split protocol."""

    validate_feature_table(fit_table)
    validate_feature_table(transfer_table)
    fit_logits = np.asarray(fit_table["legacy_logits"], dtype=np.float64)
    transfer_logits = np.asarray(transfer_table["legacy_logits"], dtype=np.float64)
    fit_targets = np.asarray(fit_table["targets"])
    transfer_targets = np.asarray(transfer_table["targets"])
    fit_scores = fit_logits[:, 1:5].max(axis=1) - fit_logits[:, 0]
    transfer_scores = transfer_logits[:, 1:5].max(axis=1) - transfer_logits[:, 0]
    fit_type = fit_logits[:, 1:5].argmax(axis=1).astype(np.int16) + 1
    transfer_type = transfer_logits[:, 1:5].argmax(axis=1).astype(np.int16) + 1
    fit_binary = (fit_targets > 0).astype(np.int16)
    transfer_binary = (transfer_targets > 0).astype(np.int16)
    fit_calibration = calibration_table_from_arrays(
        fit_scores, fit_binary, np.ones(len(fit_scores), dtype=np.int16),
        fit_table["molecule_offsets"], fit_table["package_indices"],
        score_definition="official max(type)-none existence score",
    )
    transfer_calibration = calibration_table_from_arrays(
        transfer_scores, transfer_binary, np.ones(len(transfer_scores), dtype=np.int16),
        transfer_table["molecule_offsets"], transfer_table["package_indices"],
        score_definition="official max(type)-none existence score",
    )
    fitted = fit_global_threshold(fit_calibration)
    threshold = float(fitted["selected_threshold"])
    transfer_raw = evaluate_threshold(transfer_calibration, 0.0)
    transfer_calibrated = evaluate_threshold(transfer_calibration, threshold)
    fit_present = fit_targets > 0
    transfer_present = transfer_targets > 0
    return {
        "head": "official 64->64 GELU -> 6-class bond head",
        "probe_scope": "frozen_baseline_not_a_tier_gate",
        "fit_roc_auc_all_pairs": roc_auc(fit_scores, fit_present),
        "transfer_roc_auc_all_pairs": roc_auc(transfer_scores, transfer_present),
        "fit_global_threshold": threshold,
        "fit_calibration": {
            "raw_zero": _existence_public(fitted["raw_zero"]),
            "selected": _existence_public(fitted["selected"]),
        },
        "transfer_raw_zero": _existence_public(transfer_raw),
        "transfer_calibrated": _existence_public(transfer_calibrated),
        "conditional_type": {
            "fit": type_metrics(fit_type[fit_present], fit_targets[fit_present]),
            "transfer": type_metrics(
                transfer_type[transfer_present], transfer_targets[transfer_present]
            ),
        },
    }
