"""Graph-only local exchangeable-rotor contracts and circular set losses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


LOCAL_ROTOR_SCHEMA_VERSION = "pg-orbitflow-local-rotor-set-v1"


@dataclass(frozen=True)
class LocalRotorSetContract:
    """Unordered torsion slots caused by equal-element terminal substituents."""

    groups: tuple[tuple[int, ...], ...]
    terminal_atom_groups: tuple[tuple[int, ...], ...]
    coupled_group_components: tuple[tuple[int, ...], ...]
    coupled_component_slot_maps: tuple[tuple[tuple[int, ...], ...], ...]
    representative_tuples: np.ndarray
    orbit_to_group: np.ndarray


def _torsion_representatives(contract) -> np.ndarray:
    identifiers = np.asarray(contract.torsion_orbit_id, dtype=np.int64)
    count = int(np.max(identifiers)) + 1
    return np.asarray(
        [
            contract.torsion_index[:, np.flatnonzero(identifiers == current)[0]]
            for current in range(count)
        ],
        dtype=np.int64,
    )


def _bond_lookup(sample) -> dict[tuple[int, int], int]:
    result = {}
    for edge, kind in zip(sample.bond_index.T, sample.bond_types, strict=True):
        left, right = sorted(int(value) for value in edge)
        if (left, right) in result:
            raise ValueError("duplicate canonical bond in local-rotor contract")
        result[(left, right)] = int(kind)
    return result


def _operation_coupled_components(sample, terminal_groups):
    """Connect terminal sets related by an exact canonical group action.

    Each slot map sends the canonical (first-group) slot position to the local
    slot position.  It is graph/action-only and is later used to conjugate one
    component-level Hungarian permutation into every local view.
    """

    count = len(terminal_groups)
    parent = list(range(count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    edges = {index: [] for index in range(count)}
    for left, left_atoms in enumerate(terminal_groups):
        for right, right_atoms in enumerate(terminal_groups):
            if left != right and left_atoms == right_atoms:
                identity = tuple(range(len(left_atoms)))
                edges[left].append((right, identity))
                union(left, right)
    for permutation in np.asarray(sample.permutation_index, dtype=np.int64):
        for left, left_atoms in enumerate(terminal_groups):
            mapped = tuple(int(permutation[atom]) for atom in left_atoms)
            for right, right_atoms in enumerate(terminal_groups):
                if len(mapped) != len(right_atoms) or set(mapped) != set(right_atoms):
                    continue
                slot_map = tuple(right_atoms.index(atom) for atom in mapped)
                edges[left].append((right, slot_map))
                union(left, right)

    members = {}
    for index in range(count):
        members.setdefault(find(index), []).append(index)
    components = []
    component_maps = []
    for values in sorted(members.values(), key=lambda item: tuple(item)):
        component = tuple(sorted(values))
        root = component[0]
        size = len(terminal_groups[root])
        frames = {root: tuple(range(size))}
        queue = [root]
        while queue:
            current = queue.pop(0)
            for target, edge_map in sorted(edges[current]):
                if target not in component:
                    continue
                composed = tuple(edge_map[position] for position in frames[current])
                if target not in frames:
                    frames[target] = composed
                    queue.append(target)
        if set(frames) != set(component):
            raise RuntimeError("operation-coupled terminal component is disconnected")
        components.append(component)
        component_maps.append(tuple(frames[index] for index in component))
    return tuple(components), tuple(component_maps)


def build_local_rotor_set_contract(sample, contract) -> LocalRotorSetContract:
    """Identify unordered terminal slots using graph/type information only.

    A family shares three atoms of a directed torsion path and varies only the
    degree-one endpoint.  The varying atoms must have identical element, charge,
    radical and bond type.  No coordinates, torsion targets or atom-index token
    are used to create a family.
    """

    representatives = _torsion_representatives(contract)
    atom_types = np.asarray(sample.atom_types, dtype=np.int64)
    charges = np.asarray(sample.formal_charges, dtype=np.int64)
    radicals = np.asarray(sample.radical_electrons, dtype=np.int64)
    degree = np.zeros(len(atom_types), dtype=np.int64)
    for left, right in np.asarray(sample.bond_index, dtype=np.int64).T:
        degree[int(left)] += 1
        degree[int(right)] += 1
    lookup = _bond_lookup(sample)
    families: dict[tuple, list[tuple[int, int]]] = {}
    for orbit, values in enumerate(representatives):
        for oriented in (values, values[::-1]):
            fixed_outer, center_outer, center_terminal, terminal = (
                int(value) for value in oriented
            )
            if degree[terminal] != 1:
                continue
            edge = tuple(sorted((center_terminal, terminal)))
            key = (
                fixed_outer,
                center_outer,
                center_terminal,
                int(atom_types[terminal]),
                int(charges[terminal]),
                int(radicals[terminal]),
                lookup[edge],
            )
            families.setdefault(key, []).append((orbit, terminal))
    group_records = {}
    for values in families.values():
        by_orbit = {}
        for orbit, terminal in values:
            if orbit in by_orbit and by_orbit[orbit] != terminal:
                raise ValueError("torsion orbit has inconsistent terminal slot")
            by_orbit[orbit] = terminal
        ordered = sorted(by_orbit.items(), key=lambda item: item[1])
        if len(ordered) > 1:
            group = tuple(item[0] for item in ordered)
            terminals = tuple(item[1] for item in ordered)
            if group in group_records and group_records[group] != terminals:
                raise ValueError("local rotor group has inconsistent terminal atoms")
            group_records[group] = terminals
    ordered_records = sorted(group_records.items())
    groups = [item[0] for item in ordered_records]
    terminal_groups = [item[1] for item in ordered_records]
    membership: dict[int, int] = {}
    for group_id, group in enumerate(groups):
        if len(group) not in {2, 3}:
            raise ValueError("local rotor set currently supports exactly 2 or 3 slots")
        for orbit in group:
            if orbit in membership:
                raise ValueError("overlapping local rotor sets require a higher-order contract")
            membership[orbit] = group_id
    orbit_to_group = np.full(len(representatives), -1, dtype=np.int64)
    for orbit, group_id in membership.items():
        orbit_to_group[orbit] = group_id
    components, component_slot_maps = _operation_coupled_components(
        sample, tuple(terminal_groups)
    )
    return LocalRotorSetContract(
        groups=tuple(groups),
        terminal_atom_groups=tuple(terminal_groups),
        coupled_group_components=components,
        coupled_component_slot_maps=component_slot_maps,
        representative_tuples=representatives,
        orbit_to_group=orbit_to_group,
    )


def optimal_circular_assignment(prediction, target, groups):
    """Return target rows aligned to prediction under each unordered slot set."""

    import torch

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("circular prediction/target must both be [T,2]")
    aligned = target.clone()
    seen: set[int] = set()
    for group in groups:
        indices = tuple(int(value) for value in group)
        if any(index in seen for index in indices):
            raise ValueError("overlapping circular assignment groups")
        seen.update(indices)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=prediction.device)
        pred = prediction[index_tensor]
        candidates = []
        candidate_targets = []
        for order in permutations(range(len(indices))):
            current = target[index_tensor[list(order)]]
            candidates.append(torch.mean(1.0 - torch.sum(pred * current, dim=-1)))
            candidate_targets.append(current)
        choice = int(torch.argmin(torch.stack(candidates)).detach().cpu())
        aligned[index_tensor] = candidate_targets[choice]
    return aligned


def permutation_invariant_circular_loss(prediction, target, groups):
    """Orbit-balanced circular loss after optimal local slot assignment."""

    aligned = optimal_circular_assignment(prediction, target, groups)
    return (1.0 - (prediction * aligned).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def permutation_invariant_circular_errors_degrees(
    prediction: np.ndarray, target: np.ndarray, groups
) -> np.ndarray:
    """NumPy audit metric with the same assignment semantics as the loss."""

    import torch

    predicted = torch.as_tensor(prediction, dtype=torch.float64)
    targets = torch.as_tensor(target, dtype=torch.float64)
    aligned = optimal_circular_assignment(predicted, targets, groups).numpy()
    cosine = np.sum(prediction * aligned, axis=-1).clip(-1.0, 1.0)
    sine = prediction[:, 0] * aligned[:, 1] - prediction[:, 1] * aligned[:, 0]
    return np.abs(np.rad2deg(np.arctan2(sine, cosine)))


def optimal_coupled_circular_assignment(prediction, target, contract):
    """Align operation-linked torsion views with one conjugated permutation."""

    import torch

    aligned = target.clone()
    slot_maps = getattr(contract, "coupled_component_slot_maps", None)
    if slot_maps is None:
        slot_maps = tuple(
            tuple(tuple(range(len(contract.groups[index]))) for index in component)
            for component in contract.coupled_group_components
        )
    for component, frames in zip(
        contract.coupled_group_components, slot_maps, strict=True
    ):
        groups = [contract.groups[index] for index in component]
        size = len(groups[0])
        if any(len(group) != size for group in groups):
            raise ValueError("coupled local rotor views have inconsistent slot counts")
        losses = []
        candidates = []
        for order in permutations(range(size)):
            current_loss = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
            current_targets = []
            for group, frame in zip(groups, frames, strict=True):
                indices = torch.as_tensor(group, dtype=torch.long, device=prediction.device)
                inverse = [frame.index(index) for index in range(size)]
                local_order = tuple(
                    frame[order[inverse[index]]] for index in range(size)
                )
                reordered = target[indices[list(local_order)]]
                current_loss = current_loss + torch.mean(
                    1.0 - torch.sum(prediction[indices] * reordered, dim=-1)
                )
                current_targets.append((indices, reordered))
            losses.append(current_loss / len(groups)); candidates.append(current_targets)
        choice = int(torch.argmin(torch.stack(losses)).detach().cpu())
        for indices, reordered in candidates[choice]:
            aligned[indices] = reordered
    return aligned


def coupled_permutation_invariant_circular_loss(prediction, target, contract):
    import torch

    aligned = optimal_coupled_circular_assignment(prediction, target, contract)
    grouped = sorted({orbit for group in contract.groups for orbit in group})
    indices = torch.as_tensor(grouped, dtype=torch.long, device=prediction.device)
    return (1.0 - (prediction[indices] * aligned[indices]).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def coupled_circular_errors_degrees(prediction: np.ndarray, target: np.ndarray, contract) -> np.ndarray:
    import torch

    predicted = torch.as_tensor(prediction, dtype=torch.float64)
    targets = torch.as_tensor(target, dtype=torch.float64)
    aligned = optimal_coupled_circular_assignment(predicted, targets, contract).numpy()
    cosine = np.sum(prediction * aligned, axis=-1).clip(-1.0, 1.0)
    sine = prediction[:, 0] * aligned[:, 1] - prediction[:, 1] * aligned[:, 0]
    return np.abs(np.rad2deg(np.arctan2(sine, cosine)))
