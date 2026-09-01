"""Graph-constrained atom-label gauge matching for terminal sibling atoms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

from .metrics import kabsch_rmsd


@dataclass(frozen=True)
class TerminalSiblingContract:
    groups: tuple[tuple[int, ...], ...]


def build_terminal_sibling_contract(sample) -> TerminalSiblingContract:
    """Build disjoint same-parent terminal groups without using coordinates."""

    atom_types = np.asarray(sample.atom_types, dtype=np.int64)
    charges = np.asarray(sample.formal_charges, dtype=np.int64)
    radicals = np.asarray(sample.radical_electrons, dtype=np.int64)
    neighbors = [[] for _ in atom_types]
    bond_types = {}
    for edge, kind in zip(sample.bond_index.T, sample.bond_types, strict=True):
        left, right = (int(value) for value in edge)
        neighbors[left].append(right); neighbors[right].append(left)
        bond_types[tuple(sorted((left, right)))] = int(kind)
    groups = []
    for parent, adjacent in enumerate(neighbors):
        families = {}
        for atom in adjacent:
            if len(neighbors[atom]) != 1:
                continue
            key = (
                int(atom_types[atom]), int(charges[atom]), int(radicals[atom]),
                bond_types[tuple(sorted((parent, atom)))],
            )
            families.setdefault(key, []).append(atom)
        groups.extend(tuple(sorted(values)) for values in families.values() if len(values) > 1)
    groups = tuple(sorted(groups))
    flat = [atom for group in groups for atom in group]
    if len(flat) != len(set(flat)):
        raise RuntimeError("terminal sibling groups overlap")
    return TerminalSiblingContract(groups=groups)


def best_terminal_sibling_relabeling(sample, coordinates: np.ndarray) -> dict:
    """Find the best graph-allowed atom relabeling for reference-only evaluation."""

    candidate = np.asarray(coordinates, dtype=np.float64)
    target = np.asarray(sample.symmetric_target_angstrom, dtype=np.float64)
    if candidate.shape != target.shape:
        raise ValueError("terminal sibling matching coordinate shape mismatch")
    contract = build_terminal_sibling_contract(sample)
    choices = [tuple(permutations(group)) for group in contract.groups]
    best = None
    iterator = product(*choices) if choices else [()]
    evaluated = 0
    for orders in iterator:
        permutation = np.arange(len(candidate), dtype=np.int64)
        for group, order in zip(contract.groups, orders, strict=True):
            permutation[np.asarray(group, dtype=np.int64)] = np.asarray(order, dtype=np.int64)
        relabelled = candidate[permutation]
        rmsd = kabsch_rmsd(target, relabelled)
        evaluated += 1
        key = (rmsd, tuple(permutation.tolist()))
        if best is None or key < best[0]:
            best = (key, permutation, relabelled)
    if best is None:
        raise RuntimeError("terminal sibling matching evaluated no assignment")
    return {
        "coordinates": best[2],
        "permutation": best[1],
        "kabsch_rmsd_angstrom": float(best[0][0]),
        "assignment_count": evaluated,
        "terminal_sibling_groups": [list(group) for group in contract.groups],
        "nonidentity_assignment": bool(np.any(best[1] != np.arange(len(candidate)))),
    }
