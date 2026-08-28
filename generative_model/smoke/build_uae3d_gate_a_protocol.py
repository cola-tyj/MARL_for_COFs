"""Freeze the final low-cost UAE-3D Gate A protocol before observing results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_SPLIT = ROOT / "generative_model/smoke/splits/uae3d_bond_capacity_v1.json"
UAE_SPLIT = ROOT / "generative_model/smoke/splits/uae3d_v1.json"
SOURCE_CHECKPOINT = (
    ROOT / "generative_model/runs/uae3d_overfit4/"
    "from_tier1_w10_lr3em5_0512/checkpoints/step-003008.pt"
)
OUTPUT = ROOT / "generative_model/smoke/reports/uae3d_gate_a_protocol_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    capacity = json.loads(CAPACITY_SPLIT.read_text(encoding="utf-8"))
    uae = json.loads(UAE_SPLIT.read_text(encoding="utf-8"))
    protocol = {
        "schema_version": "uae3d-gate-a-protocol-v1",
        "status": "FROZEN_BEFORE_GATE_A_EXECUTION",
        "purpose": (
            "Final read-only diagnosis of whether frozen step-3008 post-GELU pair "
            "features support a small transferable nonlinear bond-existence readout."
        ),
        "source": {
            "checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
            "checkpoint_step": 3008,
        },
        "panels": {
            "iid_probe_fit": {
                "role": "fit probe and one global threshold",
                "molecule_count": len(capacity["fit"]["package_indices"]),
                "package_indices": capacity["fit"]["package_indices"],
            },
            "iid_validation_transfer": {
                "role": "apply fitted probe and threshold unchanged",
                "molecule_count": len(capacity["transfer"]["package_indices"]),
                "package_indices": capacity["transfer"]["package_indices"],
            },
            "tier4_oracle": {
                "role": "same-panel memorization/capacity upper bound only",
                "molecule_count": len(uae["tier_indices"]["4"]),
                "package_indices": uae["tier_indices"]["4"],
            },
        },
        "probe": {
            "feature": "64-d decoder post-GELU hidden pair feature",
            "fit_sampling": (
                "per molecule all present unique pairs plus an equal deterministic "
                "evenly-spaced sample of none pairs"
            ),
            "baseline": "float64 L2 logistic linear readout",
            "nonlinear": {
                "kind": (
                    "64 fixed random Fourier features concatenated with the 64 "
                    "standardized inputs; only an L2 logistic readout is fitted"
                ),
                "random_feature_dim": 64,
                "seed": 20260821,
                "frequency_std": 0.125,
            },
            "optimizer": "deterministic float64 Newton/IRLS",
            "l2": 0.01,
            "threshold": (
                "one deterministic global threshold selected on the fit panel and "
                "applied unchanged to the evaluation panel"
            ),
        },
        "gate_thresholds": {
            "tier4_existence_exact_molecules_min": 4,
            "iid_transfer_roc_auc_min": 0.90,
            "iid_transfer_macro_balanced_min": 0.82,
            "iid_transfer_minimum_balanced_min": 0.65,
            "iid_transfer_none_accuracy_min": 0.75,
            "iid_transfer_present_recall_min": 0.75,
            "nonlinear_macro_gain_over_linear_min": 0.01,
        },
        "decision": {
            "pass_if": "all predeclared checks are true",
            "pass_status": "PASS_GATE_A_OPTIONAL_GATE_B_SUPPORTED",
            "fail_status": "FAIL_GATE_A_STOP_UAE_BRANCH",
            "gate_b_is_mainline_prerequisite": False,
            "graph_plus_target_pg_to_3d_starts_after_gate_a_regardless": True,
        },
        "leakage_and_mutation_policy": {
            "test_used": False,
            "core_ood_used": False,
            "uae3d_weights_updated": False,
            "tier4_oracle_used_for_transfer_threshold": False,
            "iid_validation_used_for_probe_or_threshold_fit": False,
        },
        "identity": {
            "dataset_fingerprint": capacity["dataset_fingerprint"],
            "package_manifest_sha256": capacity["package_manifest_sha256"],
            "capacity_split": str(CAPACITY_SPLIT.resolve()),
            "capacity_split_sha256": sha256_file(CAPACITY_SPLIT),
            "capacity_split_fingerprint": capacity["split_fingerprint"],
            "uae_split": str(UAE_SPLIT.resolve()),
            "uae_split_sha256": sha256_file(UAE_SPLIT),
            "uae_split_fingerprint": uae["split_fingerprint"],
        },
    }
    protocol["protocol_fingerprint"] = payload_fingerprint(protocol)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {OUTPUT}")
    print(f"sha256={sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
