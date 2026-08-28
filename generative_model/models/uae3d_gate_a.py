"""Deterministic read-only probes for the final UAE-3D Gate A.

Gate A asks whether the frozen step-3008 post-GELU pair representation contains
enough bond-existence information for a small nonlinear readout.  The probe is
diagnostic only: it never updates or replaces UAE-3D parameters.
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
from generative_model.models.uae3d_bond_capacity import (
    PROBE_L2,
    balanced_fit_rows,
    binary_scores,
    fit_binary_logistic,
    roc_auc,
    validate_feature_table,
)


GATE_A_SCHEMA_VERSION = "uae3d-gate-a-v1"


def _array_fingerprint(parts: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256(GATE_A_SCHEMA_VERSION.encode("ascii"))
    for name, value in parts:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_random_fourier_map(
    features: np.ndarray,
    *,
    output_dim: int,
    seed: int,
) -> dict[str, Any]:
    """Fit only deterministic normalization; Fourier weights stay random/frozen."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or output_dim <= 0:
        raise ValueError("Gate A random Fourier features shape 非法")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    constant = scale <= np.finfo(np.float64).eps
    scale[constant] = 1.0
    rng = np.random.default_rng(seed)
    weights = rng.normal(
        0.0,
        1.0 / np.sqrt(values.shape[1]),
        size=(values.shape[1], output_dim),
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=output_dim)
    model = {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "phase": phase,
        "output_dim": int(output_dim),
        "seed": int(seed),
        "constant_feature_count": int(constant.sum()),
    }
    model["fingerprint"] = _array_fingerprint([
        ("mean", mean),
        ("scale", scale),
        ("weights", weights),
        ("phase", phase),
    ])
    return model


def random_fourier_features(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    values = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    projected = np.sqrt(2.0 / model["output_dim"]) * np.cos(
        values @ model["weights"] + model["phase"]
    )
    result = np.concatenate((values, projected), axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("Gate A nonlinear features 含 NaN/Inf")
    return result


def _calibration_table(
    table: dict[str, Any], scores: np.ndarray, *, score_definition: str
) -> dict[str, Any]:
    targets = (np.asarray(table["targets"]) > 0).astype(np.int16)
    return calibration_table_from_arrays(
        scores,
        targets,
        np.ones(len(scores), dtype=np.int16),
        table["molecule_offsets"],
        table["package_indices"],
        score_definition=score_definition,
    )


def _public_existence(metrics: dict[str, Any]) -> dict[str, Any]:
    public = public_threshold_metrics(metrics)
    records = public.pop("records")
    public["existence_exact_molecules"] = int(
        sum(record["existence_error_count"] == 0 for record in records)
    )
    public["records"] = [
        {
            "package_index": record["package_index"],
            "pair_count_unique": record["pair_count_unique"],
            "none_correct": record["none_correct"],
            "none_total": record["none_total"],
            "present_existence_correct": record["present_existence_correct"],
            "present_total": record["present_total"],
            "balanced_existence_accuracy": record["balanced_existence_accuracy"],
            "existence_error_count": record["existence_error_count"],
        }
        for record in records
    ]
    for key in (
        "present_exact_type_accuracy",
        "full_unique_pair_exact_type_accuracy",
        "wrong_unique_pair_count",
        "bond_exact_molecules",
    ):
        public.pop(key, None)
    return public


def fit_existence_probe(
    fit_table: dict[str, Any],
    evaluate_table: dict[str, Any],
    *,
    nonlinear: bool,
    random_feature_dim: int = 64,
    random_feature_seed: int = 20260821,
    l2: float = PROBE_L2,
) -> dict[str, Any]:
    """Fit on a frozen feature table and apply one unchanged global threshold."""

    validate_feature_table(fit_table)
    validate_feature_table(evaluate_table)
    fit_raw = np.asarray(fit_table["hidden_features"], dtype=np.float64)
    evaluate_raw = np.asarray(evaluate_table["hidden_features"], dtype=np.float64)
    fit_targets = np.asarray(fit_table["targets"]) > 0
    evaluate_targets = np.asarray(evaluate_table["targets"]) > 0
    fit_rows = balanced_fit_rows(fit_table)

    mapping = None
    if nonlinear:
        mapping = fit_random_fourier_map(
            fit_raw[fit_rows], output_dim=random_feature_dim, seed=random_feature_seed
        )
        fit_features = random_fourier_features(mapping, fit_raw)
        evaluate_features = random_fourier_features(mapping, evaluate_raw)
        name = "frozen random-Fourier nonlinear map + logistic readout"
    else:
        fit_features = fit_raw
        evaluate_features = evaluate_raw
        name = "linear logistic readout"

    model = fit_binary_logistic(
        fit_features[fit_rows], fit_targets[fit_rows], l2=l2
    )
    fit_scores = binary_scores(model, fit_features)
    evaluate_scores = binary_scores(model, evaluate_features)
    fit_calibration = _calibration_table(
        fit_table, fit_scores, score_definition=f"Gate A {name}"
    )
    evaluate_calibration = _calibration_table(
        evaluate_table, evaluate_scores, score_definition=f"Gate A {name}"
    )
    fitted = fit_global_threshold(fit_calibration)
    threshold = float(fitted["selected_threshold"])
    evaluated = evaluate_threshold(evaluate_calibration, threshold)
    model_parts = [
        ("logistic_weights", model["weights"]),
        ("logistic_mean", model["mean"]),
        ("logistic_scale", model["scale"]),
    ]
    if mapping is not None:
        model_parts.extend([
            ("rff_mean", mapping["mean"]),
            ("rff_scale", mapping["scale"]),
            ("rff_weights", mapping["weights"]),
            ("rff_phase", mapping["phase"]),
        ])
    return {
        "probe": name,
        "uae3d_weights_updated": False,
        "fit_balanced_pair_count": int(len(fit_rows)),
        "fit_roc_auc_all_pairs": roc_auc(fit_scores, fit_targets),
        "evaluation_roc_auc_all_pairs": roc_auc(evaluate_scores, evaluate_targets),
        "selected_global_threshold_on_fit": threshold,
        "fit_selected": _public_existence(fitted["selected"]),
        "evaluation_at_unchanged_threshold": _public_existence(evaluated),
        "optimizer": {
            "algorithm": "deterministic float64 Newton/IRLS logistic regression",
            "l2": float(l2),
            "iterations": int(model["iterations"]),
            "converged": bool(model["converged"]),
        },
        "nonlinear_map": None if mapping is None else {
            "kind": "fixed random Fourier features concatenated with standardized input",
            "input_dim": int(fit_raw.shape[1]),
            "random_feature_dim": int(random_feature_dim),
            "output_dim": int(fit_features.shape[1]),
            "seed": int(random_feature_seed),
            "frequency_std": float(1.0 / np.sqrt(fit_raw.shape[1])),
            "fingerprint": mapping["fingerprint"],
        },
        "probe_fingerprint": _array_fingerprint(model_parts),
    }


def gate_a_checks(result: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, bool]:
    tier4 = result["tier4_oracle"]["nonlinear"]["evaluation_at_unchanged_threshold"]
    transfer = result["iid_transfer"]["nonlinear"]["evaluation_at_unchanged_threshold"]
    linear_macro = result["iid_transfer"]["linear"]["evaluation_at_unchanged_threshold"][
        "macro_molecule_balanced_existence_accuracy"
    ]
    nonlinear_macro = transfer["macro_molecule_balanced_existence_accuracy"]
    return {
        "source_model_fingerprint_unchanged": bool(result["source_model_unchanged"]),
        "all_probe_outputs_finite": bool(result["all_probe_outputs_finite"]),
        "tier4_existence_exact_molecules_min": tier4["existence_exact_molecules"]
        >= thresholds["tier4_existence_exact_molecules_min"],
        "iid_transfer_roc_auc_min": result["iid_transfer"]["nonlinear"][
            "evaluation_roc_auc_all_pairs"
        ] >= thresholds["iid_transfer_roc_auc_min"],
        "iid_transfer_macro_balanced_min": nonlinear_macro
        >= thresholds["iid_transfer_macro_balanced_min"],
        "iid_transfer_minimum_balanced_min": transfer[
            "minimum_molecule_balanced_existence_accuracy"
        ] >= thresholds["iid_transfer_minimum_balanced_min"],
        "iid_transfer_none_accuracy_min": transfer["none_accuracy"]
        >= thresholds["iid_transfer_none_accuracy_min"],
        "iid_transfer_present_recall_min": transfer["present_existence_recall"]
        >= thresholds["iid_transfer_present_recall_min"],
        "nonlinear_macro_gain_over_linear_min": nonlinear_macro - linear_macro
        >= thresholds["nonlinear_macro_gain_over_linear_min"],
    }
