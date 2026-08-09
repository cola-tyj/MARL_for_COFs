#!/usr/bin/env python3
"""Rebuild config.py from scratch using dataset + hierarchical assembly"""
import os, sys
from collections import OrderedDict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ── Helper: reverse-engineer core SMILES from full molecule ──
def extract_core_template(full_smi, arm_smi_template):
    """
    Given a full molecule (core + arm) and the arm template (*-FG),
    reverse the substitution to get back the core template (*-core-*).

    For simple arms like *Br, we replace Br atom with * in the molecule.
    """
    mol = Chem.MolFromSmiles(full_smi)
    arm_mol = Chem.MolFromSmiles(arm_smi_template)
    if mol is None or arm_mol is None:
        return None

    # Find the arm substructure in the full molecule
    # The arm is *Br, so we search for Br atoms (or the arm's heavy atom)
    arm_heavy = None
    for a in arm_mol.GetAtoms():
        if a.GetAtomicNum() > 1 and a.GetAtomicNum() != 0:  # not H, not dummy
            arm_heavy = a.GetSymbol()
            break
    if arm_heavy is None:
        return None

    # Find all atoms of this type in the full molecule
    pat = Chem.MolFromSmarts(arm_heavy)
    if pat is None:
        return None

    matches = mol.GetSubstructMatches(pat)
    if not matches:
        return None

    # Replace each matched atom with * and return the modified SMILES
    rw = Chem.RWMol(mol)
    # Get indices of atoms to replace
    to_replace = sorted([m[0] for m in matches], reverse=True)
    for idx in to_replace:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(0)  # Set to dummy atom
        atom.SetIsAromatic(False)

    try:
        Chem.SanitizeMol(rw)
        smi = Chem.MolToSmiles(rw)
        return smi
    except:
        # Fallback: just do string replacement of the terminal group
        return full_smi.replace(arm_heavy, '*')

# ── Load dataset ──
aug = pd.read_csv('cof_symmetry_pipeline/output/augmented_dataset.csv')

# Build core templates by reverse-engineering
core_templates = OrderedDict()

for core_name in sorted(aug['Core'].unique()):
    sub = aug[aug['Core'] == core_name]

    # Find simplest arm (direct attachment, minimal atoms)
    direct_arms = sub[sub['Arm'].str.contains('direct')]
    if len(direct_arms) == 0:
        direct_arms = sub

    best = direct_arms.loc[direct_arms['N_Atoms'].idxmin()]
    full_smi = best['SMILES']
    arm_name = best['Arm']
    target_pg = best['Target_PG']
    sym_fam = best['Symmetry_Family']

    # Get arm template from ARM_LIBRARY (we need to reconstruct this too)
    # For now, estimate core SMILES via SMARTS
    core_smi = extract_core_template(full_smi, f'*{arm_name.split("_")[0]}')
    if core_smi is None or '*' not in core_smi:
        # Fallback: 对于biphenyl-type, hardcode from known patterns
        # 实际会从对话记录中重建
        continue

    n_stars = core_smi.count('*')

    core_templates[core_name] = {
        'smiles': core_smi,
        'description': f'{core_name} 核心',
        'target_pg': target_pg,
        'n_arms': n_stars,
        'symmetry_family': sym_fam,
    }

print(f'Reverse-engineered cores: {len(core_templates)}')
for k, v in list(core_templates.items())[:5]:
    print(f'  {k}: arms={v["n_arms"]} pg={v["target_pg"]} smi={v["smiles"][:60]}')

# ── Generate hierarchical cores ──

def _cleanup_bridges_hier(mol):
    """Local copy of bridge cleanup to avoid circular imports"""
    bridges = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetDegree() == 2:
            nbrs = atom.GetNeighbors()
            bridges.append((atom.GetIdx(), nbrs[0].GetIdx(), nbrs[1].GetIdx()))
    if not bridges:
        return mol
    bridge_indices = sorted([b[0] for b in bridges])
    mol_rw = Chem.RWMol(mol)
    for bridge_idx, _, _ in sorted(bridges, reverse=True):
        mol_rw.RemoveAtom(bridge_idx)
    for _, n1, n2 in bridges:
        adj_n1 = n1 - sum(1 for bi in bridge_indices if bi < n1)
        adj_n2 = n2 - sum(1 for bi in bridge_indices if bi < n2)
        try:
            mol_rw.AddBond(adj_n1, adj_n2, Chem.BondType.SINGLE)
        except:
            pass
    return mol_rw.GetMol()

# For now, skip hierarchical generation and just write the basic config
# This is getting too complex without the original SMILES
print("\nWriting minimal config.py from extracted data...")

# Build the output
output = []
output.append('"""COF Symmetry Pipeline Configuration (rebuilt)"""\n')
output.append('from dataclasses import dataclass, field\n')
output.append('from typing import Dict, List, Optional\n\n')

# CORE_TEMPLATES
output.append('CORE_TEMPLATES: Dict[str, dict] = {\n')
for name, info in core_templates.items():
    smi = info['smiles']
    if len(smi) > 80:
        # Split long SMILES
        smi_parts = [smi[i:i+80] for i in range(0, len(smi), 80)]
        smi_str = '(\n            "' + '"\n            "'.join(smi_parts) + '"\n        )'
    else:
        smi_str = f'"{smi}"'

    output.append(f'    "{name}": {{\n')
    output.append(f'        "smiles": {smi_str},\n')
    output.append(f'        "description": "{info["description"]}",\n')
    output.append(f'        "target_pg": "{info["target_pg"]}",\n')
    output.append(f'        "n_arms": {info["n_arms"]},\n')
    output.append(f'        "symmetry_family": "{info["symmetry_family"]}",\n')
    output.append(f'    }},\n')
output.append('}\n\n')

# Since we can't fully rebuild the ARM_LIBRARY and config classes from extracted data,
# write a placeholder and note the issue
output.append('# ============================================================\n')
output.append('# WARNING: This file was rebuilt from dataset extraction.\n')
output.append('# ARM_LIBRARY and config classes need manual restoration.\n')
output.append('# Run: python cof_symmetry_pipeline/rebuild_config.py\n')
output.append('# ============================================================\n')

with open('cof_symmetry_pipeline/config_rebuilt.py', 'w') as f:
    f.writelines(output)

print(f"Written config_rebuilt.py with {len(core_templates)} cores")
print("Full rebuild requires manual restoration of ARM_LIBRARY and config classes.")
