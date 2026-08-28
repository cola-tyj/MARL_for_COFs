"""Paired raw Euler ODE sampling for official and PG-conditioned ET-Flow."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "etflow-pg-paired-raw-ode-v1"


def _device_graph(graph: Any, device: str) -> dict[str, Any]:
    import torch

    atom_count = len(graph.atomic_numbers)
    result = {
        "z": graph.atomic_numbers.to(device),
        "bond_index": graph.edge_index.to(device),
        "edge_attr": None if getattr(graph, "edge_attr", None) is None else graph.edge_attr.to(device),
        "node_attr": graph.node_attr.to(device),
        "batch": torch.zeros(atom_count, dtype=torch.long, device=device),
    }
    for name in ("chiral_index", "chiral_nbr_index", "chiral_tag"):
        value = getattr(graph, name, None)
        result[name] = None if value is None else value.to(device)
    return result


def sample_shared_prior(base_flow: Any, graph: Any, *, seed: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Sample one centered official prior deterministically."""

    import torch
    from etflow.models.utils import center_of_mass

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    fields = _device_graph(graph, device)
    prior = base_flow.sample_base_dist(
        (len(fields["z"]), 3), fields["bond_index"], fields["batch"]
    )
    prior = center_of_mass(prior, batch=fields["batch"])
    if not bool(torch.isfinite(prior).all()):
        raise RuntimeError("ET-Flow prior 含 NaN/Inf")
    return prior, fields


def integrate_raw_ode(
    base_flow: Any,
    model: Any,
    fields: dict[str, Any],
    initial_positions: Any,
    *,
    n_timesteps: int,
    symmetry_action: dict[str, Any] | None,
) -> Any:
    """Reproduce official Euler ODE; optional action invokes the A2 wrapper."""

    import torch
    from etflow.models.utils import unsqueeze_like

    if n_timesteps <= 0:
        raise ValueError("n_timesteps 必须为正")
    if symmetry_action is None and model is not base_flow:
        raise ValueError("action ablation 必须直接使用 official base_flow")
    positions = initial_positions.clone()
    schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=positions.device)
    with torch.no_grad():
        for step in range(n_timesteps):
            time = schedule[step].repeat(len(positions))
            time = unsqueeze_like(time, positions)
            delta = base_flow._compute_delta_t(schedule, t=step)
            common = {
                "z": fields["z"], "t": time, "pos": positions,
                "bond_index": fields["bond_index"], "edge_attr": fields["edge_attr"],
                "node_attr": fields["node_attr"], "batch": fields["batch"],
            }
            if symmetry_action is None:
                velocity = base_flow(**common)
            else:
                velocity = model(**common, symmetry_actions=[symmetry_action])
            positions = positions + delta * velocity
        if base_flow.parity_switch == "post_hoc":
            positions = base_flow.switch_parity_of_pos(
                positions,
                fields["chiral_index"], fields["chiral_nbr_index"],
                fields["chiral_tag"], fields["batch"],
            )
    if not bool(torch.isfinite(positions).all()):
        raise RuntimeError("raw ODE coordinates 含 NaN/Inf")
    return positions


def paired_raw_ode_sample(
    base_flow: Any,
    conditioned_model: Any,
    prepared: dict[str, Any],
    *,
    n_timesteps: int,
    seed: int,
    device: str,
    alternate_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample base/correct and optional alternate branches from identical prior."""

    prior, fields = sample_shared_prior(
        base_flow, prepared["graph"], seed=seed, device=device
    )
    base = integrate_raw_ode(
        base_flow, base_flow, fields, prior,
        n_timesteps=n_timesteps, symmetry_action=None,
    )
    correct = integrate_raw_ode(
        base_flow, conditioned_model, fields, prior,
        n_timesteps=n_timesteps, symmetry_action=prepared["symmetry_action"],
    )
    alternate = None
    if alternate_action is not None:
        alternate = integrate_raw_ode(
            base_flow, conditioned_model, fields, prior,
            n_timesteps=n_timesteps, symmetry_action=alternate_action,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "n_timesteps": int(n_timesteps),
        "base_positions": base,
        "correct_positions": correct,
        "alternate_positions": alternate,
        "shared_prior_exact": True,
        "hard_projection_used": False,
    }
