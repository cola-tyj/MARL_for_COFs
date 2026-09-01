"""Read-only checkpoint/time/integrator diagnosis for a failed overfit tier.

This command never trains, changes a checkpoint, projects generated coordinates,
or changes the frozen gate. It answers three narrow questions:

1. Did an earlier checkpoint perform better than the final checkpoint?
2. At which flow times is endpoint recovery weakest?
3. Is raw geometry sensitive to the numerical ODE step count?
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import PGOrbitFlow
from .overfit import (
    OVERFIT_SCHEMA_VERSION,
    _dump,
    _fixed_time_evaluation,
    _load_protocol,
    _panel_samples,
    _raw_evaluation,
    _sha256,
)


DIAGNOSIS_SCHEMA_VERSION = "pg-orbitflow-overfit-readonly-diagnosis-v1"


def _model(protocol: dict, checkpoint_path: Path, device: str):
    import torch

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("schema_version") != OVERFIT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema mismatch: {checkpoint_path}")
    model = PGOrbitFlow(
        **{
            key: protocol["model"][key]
            for key in ("hidden_dim", "layers", "radial_dim", "radial_max")
        }
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tier", choices=("4", "16", "32"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ode-steps", type=int, nargs="+", default=(25, 50, 100))
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    protocol_path = args.protocol.resolve()
    run_dir = args.run_dir.resolve()
    protocol, tier_config = _load_protocol(protocol_path, args.tier)
    protocol_sha = _sha256(protocol_path)
    samples = _panel_samples(protocol, tier_config)
    checkpoints = sorted(run_dir.glob("step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no step checkpoints in {run_dir}")

    checkpoint_results = []
    for checkpoint_path in checkpoints:
        model, state = _model(protocol, checkpoint_path, args.device)
        if state.get("protocol_sha256") != protocol_sha or state.get("tier") != args.tier:
            raise ValueError(f"checkpoint protocol/tier mismatch: {checkpoint_path}")
        fixed = _fixed_time_evaluation(model, samples, protocol, tier_config, args.device)
        by_time = {
            time: values["means"] for time, values in fixed["by_flow_time"].items()
        }
        checkpoint_results.append(
            {
                "step": int(state["step"]),
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "fixed_time_means": fixed["means"],
                "by_flow_time": by_time,
            }
        )

    final_checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("-")[-1]))
    final_model, final_state = _model(protocol, final_checkpoint, args.device)
    integrator_results = {}
    for ode_steps in args.ode_steps:
        if ode_steps <= 0:
            raise ValueError("ODE steps must be positive")
        setting = dict(tier_config)
        setting["evaluation"] = dict(tier_config["evaluation"])
        setting["evaluation"]["raw_ode_steps"] = int(ode_steps)
        raw, _ = _raw_evaluation(final_model, samples, protocol, setting, args.device)
        integrator_results[str(ode_steps)] = {
            "means": raw["means"],
            "collision_free_fraction": raw["collision_free_fraction"],
            "max_operation_atom_error_angstrom": raw[
                "max_operation_atom_error_angstrom"
            ],
            "by_point_group": raw["by_point_group"],
        }

    best_fixed = min(
        checkpoint_results,
        key=lambda result: result["fixed_time_means"]["endpoint_rmsd_angstrom"],
    )
    final_result = next(
        result for result in checkpoint_results if result["step"] == int(final_state["step"])
    )
    endpoint_by_time = final_result["by_flow_time"]
    weakest_time = max(
        endpoint_by_time,
        key=lambda time: endpoint_by_time[time]["endpoint_rmsd_angstrom"],
    )
    report = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "status": "PASS_READONLY_DIAGNOSIS_COMPLETE",
        "scientific_gate_changed": False,
        "training_performed": False,
        "posthoc_hard_projection_used": False,
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "run_dir": str(run_dir),
        "tier": int(args.tier),
        "checkpoint_results": checkpoint_results,
        "final_checkpoint_step": int(final_state["step"]),
        "integrator_step_sensitivity": integrator_results,
        "diagnostic_summary": {
            "best_fixed_endpoint_checkpoint_step": int(best_fixed["step"]),
            "best_fixed_endpoint_rmsd_angstrom": best_fixed["fixed_time_means"][
                "endpoint_rmsd_angstrom"
            ],
            "final_fixed_endpoint_rmsd_angstrom": final_result["fixed_time_means"][
                "endpoint_rmsd_angstrom"
            ],
            "weakest_final_checkpoint_flow_time": float(weakest_time),
            "weakest_final_checkpoint_endpoint_rmsd_angstrom": endpoint_by_time[
                weakest_time
            ]["endpoint_rmsd_angstrom"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _dump(args.output.resolve(), report)
    print(f"PASS_READONLY_DIAGNOSIS_COMPLETE report={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
