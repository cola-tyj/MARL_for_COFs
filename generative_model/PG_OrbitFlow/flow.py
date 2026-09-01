"""Flow-matching example preparation and raw ODE sampling."""

from __future__ import annotations

import numpy as np

from .data import OrbitFlowSample, graph_laplacian, make_symmetric_noise
from .group import operation_error_numpy


FLOW_SCHEMA_VERSION = "pg-orbitflow-linear-conditional-flow-v1"


def cyclic_axis(operation_matrices: np.ndarray) -> np.ndarray:
    """Extract the unoriented principal axis of a Cn rotation action."""

    matrices = np.asarray(operation_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or len(matrices) < 2:
        raise ValueError("cyclic axis requires non-trivial [G,3,3] operations")
    if np.any(np.linalg.det(matrices) < 1.0 - 1e-5):
        raise ValueError("Cn phase alignment supports proper rotations only")
    reynolds = np.mean(matrices, axis=0)
    symmetric = 0.5 * (reynolds + reynolds.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis /= np.linalg.norm(axis)
    # The axis is a line.  Fix a deterministic sign only for reproducible
    # Rodrigues rotations; the optimized final coordinates are sign invariant.
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0:
        axis = -axis
    projector = np.outer(axis, axis)
    if float(np.max(np.abs(reynolds - projector))) > 2e-4:
        raise ValueError("operations do not define one cyclic-axis projector")
    return axis


def align_cyclic_target_phase(
    target: np.ndarray,
    prior: np.ndarray,
    operation_matrices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """SO(2)-align target to prior around the Cn axis without changing action.

    The optimized rotation commutes with every Cn operation, so both endpoints
    remain in exactly the same group-invariant coordinate subspace.
    """

    target_values = np.asarray(target, dtype=np.float64)
    prior_values = np.asarray(prior, dtype=np.float64)
    if target_values.shape != prior_values.shape or target_values.ndim != 2 or target_values.shape[1] != 3:
        raise ValueError("target/prior must have equal [N,3] shape")
    if not np.isfinite(target_values).all() or not np.isfinite(prior_values).all():
        raise ValueError("target/prior contains NaN/Inf")
    target_values = target_values - target_values.mean(axis=0, keepdims=True)
    prior_values = prior_values - prior_values.mean(axis=0, keepdims=True)
    axis = cyclic_axis(operation_matrices)
    target_parallel = np.outer(target_values @ axis, axis)
    target_perpendicular = target_values - target_parallel
    prior_perpendicular = prior_values - np.outer(prior_values @ axis, axis)
    cosine_score = float(np.sum(prior_perpendicular * target_perpendicular))
    sine_basis = np.cross(axis[None, :], target_perpendicular)
    sine_score = float(np.sum(prior_perpendicular * sine_basis))
    angle = float(np.arctan2(sine_score, cosine_score))
    aligned = (
        target_parallel
        + np.cos(angle) * target_perpendicular
        + np.sin(angle) * sine_basis
    )
    aligned -= aligned.mean(axis=0, keepdims=True)
    # Explicitly prove that the phase rotation belongs to the centralizer.
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = np.eye(3) * np.cos(angle) + (1.0 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
    commutator = max(
        float(np.max(np.abs(rotation @ matrix - matrix @ rotation)))
        for matrix in np.asarray(operation_matrices, dtype=np.float64)
    )
    if commutator > 2e-5:
        raise RuntimeError("phase alignment rotation does not commute with group action")
    return aligned.astype(np.float32), angle


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def centralizer_rotations(
    operation_matrices: np.ndarray,
    *,
    target_pg: str,
    phase_count: int = 24,
) -> np.ndarray:
    """Deterministic Haar quadrature for the SO(3) centralizer of C2/C3.

    C3 has the connected SO(2) component around the principal axis.  C2 has
    an additional component consisting of pi rotations about axes normal to
    the principal axis; both components receive equal quadrature mass.
    """

    if target_pg not in {"C2", "C3"}:
        raise ValueError("centralizer averaging currently supports C2/C3 only")
    if phase_count <= 0:
        raise ValueError("phase_count must be positive")
    matrices = np.asarray(operation_matrices, dtype=np.float64)
    axis = cyclic_axis(matrices)
    phases = 2.0 * np.pi * np.arange(phase_count, dtype=np.float64) / phase_count
    connected = [_axis_rotation(axis, float(angle)) for angle in phases]
    rotations = list(connected)
    if target_pg == "C2":
        basis = np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ axis)))]
        perpendicular = np.cross(axis, basis)
        perpendicular /= np.linalg.norm(perpendicular)
        half_turn = 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
        rotations.extend(half_turn @ rotation for rotation in connected)
    result = np.asarray(rotations, dtype=np.float64)
    for rotation in result:
        if float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) > 1e-10:
            raise RuntimeError("centralizer quadrature contains a non-orthogonal matrix")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-10:
            raise RuntimeError("centralizer quadrature must contain proper rotations")
        commutator = max(
            float(np.max(np.abs(rotation @ matrix - matrix @ rotation)))
            for matrix in matrices
        )
        if commutator > 2e-5:
            raise RuntimeError("quadrature rotation does not centralize the point group")
    return result.astype(np.float32)


def centralizer_averaged_transport(
    sample: OrbitFlowSample,
    *,
    prior: np.ndarray,
    target: np.ndarray,
    time_value: float,
    seed: int,
    phase_count: int = 24,
    harmonic_alpha: float = 1.0,
) -> dict:
    """Construct a harmonic-posterior averaged C2/C3 transport label."""

    if not 0.0 < time_value < 1.0:
        raise ValueError("centralizer averaged transport requires 0<t<1")
    prior = np.asarray(prior, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prior.shape != target.shape or prior.ndim != 2 or prior.shape[1] != 3:
        raise ValueError("prior/target must have equal [N,3] shape")
    rotations = centralizer_rotations(
        sample.operation_matrices,
        target_pg=sample.target_pg,
        phase_count=phase_count,
    )
    candidates = np.asarray([target @ rotation.T for rotation in rotations])
    for candidate in candidates:
        error = operation_error_numpy(
            candidate, sample.operation_matrices, sample.permutation_index
        )
        if error["max_atom_error_angstrom"] > 1e-5:
            raise RuntimeError("centralizer target candidate breaks the group action")
    selected_index = int(np.random.default_rng(int(seed) + 2).integers(len(candidates)))
    selected = candidates[selected_index]
    positions_t = (1.0 - time_value) * prior + time_value * selected
    residual = positions_t[None, :, :] - time_value * candidates
    laplacian = graph_laplacian(sample, alpha=harmonic_alpha)
    energies = np.einsum("kni,nm,kmi->k", residual, laplacian, residual)
    log_weights = -0.5 * energies / ((1.0 - time_value) ** 2)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weight_sum = float(weights.sum())
    if not np.isfinite(weights).all() or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise RuntimeError("centralizer posterior weights are invalid")
    weights /= weight_sum
    averaged_target = np.einsum("k,kni->ni", weights, candidates)
    averaged_target -= averaged_target.mean(axis=0, keepdims=True)
    error = operation_error_numpy(
        averaged_target, sample.operation_matrices, sample.permutation_index
    )
    if error["max_atom_error_angstrom"] > 1e-5:
        raise RuntimeError("centralizer averaged target breaks the group action")
    velocity = (averaged_target - positions_t) / (1.0 - time_value)
    return {
        "positions_t": positions_t.astype(np.float32),
        "target_velocity": velocity.astype(np.float32),
        "transport_target_positions": averaged_target.astype(np.float32),
        "geometry_target_positions": selected.astype(np.float32),
        "selected_candidate_index": selected_index,
        "posterior_weights": weights.astype(np.float64),
    }


def tensorize_sample(sample: OrbitFlowSample, *, device: str, coordinate_scale_angstrom: float):
    import torch

    if coordinate_scale_angstrom <= 0:
        raise ValueError("coordinate_scale_angstrom must be positive")
    kwargs = {"device": device}
    return {
        "atom_types": torch.as_tensor(sample.atom_types, dtype=torch.long, **kwargs),
        "atomic_numbers": torch.as_tensor(sample.atomic_numbers, dtype=torch.long, **kwargs),
        "formal_charges": torch.as_tensor(sample.formal_charges, dtype=torch.long, **kwargs),
        "radical_electrons": torch.as_tensor(sample.radical_electrons, dtype=torch.long, **kwargs),
        "bond_index": torch.as_tensor(sample.bond_index, dtype=torch.long, **kwargs),
        "target_positions": torch.as_tensor(
            sample.symmetric_target_angstrom / coordinate_scale_angstrom,
            dtype=torch.float32,
            **kwargs,
        ),
        "operation_matrices": torch.as_tensor(sample.operation_matrices, dtype=torch.float32, **kwargs),
        "permutation_index": torch.as_tensor(sample.permutation_index, dtype=torch.long, **kwargs),
        "orbit_id": torch.as_tensor(sample.orbit_id, dtype=torch.long, **kwargs),
        "group_features": torch.as_tensor(sample.group_features, dtype=torch.float32, **kwargs),
        "atom_group_features": torch.as_tensor(sample.atom_group_features, dtype=torch.float32, **kwargs),
        "invariant_bond_order": torch.as_tensor(sample.invariant_bond_order, dtype=torch.float32, **kwargs),
        "target_pg_index": sample.target_pg_index,
    }


def make_training_example(
    sample: OrbitFlowSample,
    *,
    seed: int,
    device: str,
    coordinate_scale_angstrom: float,
    phase_align_target: bool = False,
    prior_type: str = "projected_gaussian",
    harmonic_alpha: float = 1.0,
    transport_type: str | None = None,
    centralizer_phase_count: int = 24,
    time_value: float | None = None,
):
    import torch

    tensors = tensorize_sample(
        sample, device=device, coordinate_scale_angstrom=coordinate_scale_angstrom
    )
    noise = make_symmetric_noise(
        sample,
        seed=seed,
        coordinate_scale_angstrom=coordinate_scale_angstrom,
        prior_type=prior_type,
        harmonic_alpha=harmonic_alpha,
    )
    generator = np.random.default_rng(int(seed) + 1)
    if time_value is None:
        time_value = float(generator.uniform(0.02, 0.98))
    else:
        time_value = float(time_value)
        if not 0.0 < time_value < 1.0:
            raise ValueError("time_value must satisfy 0<t<1")
    x0 = torch.as_tensor(noise, dtype=torch.float32, device=device)
    if transport_type is None:
        transport_type = "cn_phase_aligned" if phase_align_target else "independent"
    if transport_type not in {
        "independent",
        "cn_phase_aligned",
        "centralizer_averaged",
    }:
        raise ValueError(f"unsupported transport_type: {transport_type}")
    if phase_align_target and transport_type != "cn_phase_aligned":
        raise ValueError("phase_align_target conflicts with transport_type")
    canonical_target = tensors["target_positions"]
    if transport_type == "centralizer_averaged":
        if prior_type != "graph_harmonic":
            raise ValueError("centralizer_averaged transport requires graph_harmonic prior")
        transport = centralizer_averaged_transport(
            sample,
            prior=noise,
            target=sample.symmetric_target_angstrom / coordinate_scale_angstrom,
            time_value=time_value,
            seed=seed,
            phase_count=centralizer_phase_count,
            harmonic_alpha=harmonic_alpha,
        )
        tensors.update(
            {
                key: torch.as_tensor(value, dtype=torch.float32, device=device)
                for key, value in transport.items()
                if key
                in {
                    "positions_t",
                    "target_velocity",
                    "transport_target_positions",
                    "geometry_target_positions",
                }
            }
        )
        tensors["selected_candidate_index"] = transport["selected_candidate_index"]
        tensors["posterior_weights"] = transport["posterior_weights"]
        tensors["phase_alignment_angle_radians"] = 0.0
    elif transport_type == "cn_phase_aligned":
        aligned, phase_angle = align_cyclic_target_phase(
            sample.symmetric_target_angstrom / coordinate_scale_angstrom,
            noise,
            sample.operation_matrices,
        )
        x1 = torch.as_tensor(aligned, dtype=torch.float32, device=device)
        tensors["phase_alignment_angle_radians"] = phase_angle
        tensors["transport_target_positions"] = x1
        tensors["geometry_target_positions"] = x1
    else:
        x1 = canonical_target
        tensors["phase_alignment_angle_radians"] = 0.0
        tensors["transport_target_positions"] = x1
        tensors["geometry_target_positions"] = x1
    time = torch.as_tensor(time_value, dtype=torch.float32, device=device)
    tensors["time"] = time
    if transport_type != "centralizer_averaged":
        tensors["positions_t"] = (1.0 - time) * x0 + time * tensors[
            "transport_target_positions"
        ]
        tensors["target_velocity"] = tensors["transport_target_positions"] - x0
    # Keep the historical key as an alias for callers that only inspect the
    # transport endpoint. Geometry losses use the explicit separate key.
    tensors["target_positions"] = tensors["transport_target_positions"]
    return tensors


def _model_kwargs(tensors: dict) -> dict:
    return {
        key: tensors[key]
        for key in (
            "atom_types",
            "formal_charges",
            "radical_electrons",
            "target_pg_index",
            "group_features",
            "atom_group_features",
            "invariant_bond_order",
        )
    }


def sample_raw_ode(
    model,
    sample: OrbitFlowSample,
    *,
    seed: int,
    steps: int,
    device: str,
    coordinate_scale_angstrom: float,
    method: str = "heun",
    prior_type: str = "projected_gaussian",
    harmonic_alpha: float = 1.0,
):
    """Generate raw coordinates; no Reynolds/hard projection is applied."""

    import torch

    if steps <= 0 or method not in {"euler", "heun"}:
        raise ValueError("invalid ODE integration settings")
    tensors = tensorize_sample(
        sample, device=device, coordinate_scale_angstrom=coordinate_scale_angstrom
    )
    noise = make_symmetric_noise(
        sample,
        seed=seed,
        coordinate_scale_angstrom=coordinate_scale_angstrom,
        prior_type=prior_type,
        harmonic_alpha=harmonic_alpha,
    )
    positions = torch.as_tensor(noise, dtype=torch.float32, device=device)
    dt = 1.0 / steps
    kwargs = _model_kwargs(tensors)
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            time = torch.as_tensor(step / steps, dtype=positions.dtype, device=device)
            first = model(positions=positions, time=time, **kwargs)
            if method == "euler":
                positions = positions + dt * first
            else:
                proposed = positions + dt * first
                second = model(
                    positions=proposed,
                    time=torch.as_tensor((step + 1) / steps, dtype=positions.dtype, device=device),
                    **kwargs,
                )
                positions = positions + 0.5 * dt * (first + second)
            positions = positions - positions.mean(dim=0, keepdim=True)
    return positions.detach().cpu().numpy() * coordinate_scale_angstrom
