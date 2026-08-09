#!/usr/bin/env python3
"""从 pyCOFBuilder 导入 connectors 和 cores 到 config.py"""
import json, os, re, sys
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cof_symmetry_pipeline'))
from config import ARM_LIBRARY

# ============================================================
# Part 1: Connectors
# ============================================================
conector_dir = '/home/tianyajun/MARL_for_COFs/pycofbuilder/data/conector'
existing_arm_smis = set()
for k, v in ARM_LIBRARY.items():
    try:
        mol = Chem.MolFromSmiles(v['smiles'])
        if mol:
            existing_arm_smis.add(Chem.MolToSmiles(mol, canonical=True))
    except:
        pass

new_arms = []
for f in sorted(os.listdir(conector_dir)):
    if not f.endswith('.cjson'):
        continue
    with open(os.path.join(conector_dir, f)) as fh:
        data = json.load(fh)
    code = data.get('properties', {}).get('code', '')
    name = data.get('name', '')
    xsmiles = data.get('properties', {}).get('xsmiles', '')
    if not xsmiles:
        continue
    smi = xsmiles.replace('[*]', '*')
    if smi == '*CH':
        continue
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except:
        continue
    if canonical in existing_arm_smis:
        continue
    new_arms.append({
        'key': f"PB_{code}" if code else f"PB_{f.replace('.cjson','')}",
        'smiles': canonical,
        'description': f'[pycofbuilder] {name}',
        'functional_group': code.lower() if code else 'unknown',
    })
    existing_arm_smis.add(canonical)

# ============================================================
# Part 2: Cores
# ============================================================
core_dir = '/home/tianyajun/MARL_for_COFs/pycofbuilder/data/core'

# 现有 CORE_TEMPLATES SMILES
with open('/home/tianyajun/MARL_for_COFs/cof_symmetry_pipeline/config.py') as f:
    config_text = f.read()
split_pt = config_text.find('ARM_LIBRARY')
all_names = re.findall(r'"([a-zA-Z0-9_]+)":\s*\{', config_text[:split_pt])
all_names = [n for n in all_names if n != 'CORE_TEMPLATES']
orig_names = [n for n in all_names if '__' not in n]

existing_core_smis = set()
for name in orig_names:
    m = re.search(rf'"{name}":\s*\{{\s*"smiles":\s*"([^"]+)"', config_text)
    if not m:
        continue
    try:
        mol = Chem.MolFromSmiles(m.group(1))
        if mol:
            existing_core_smis.add(Chem.MolToSmiles(mol, canonical=True))
    except:
        pass

sym_map = {
    'L2': ('C2', 'C2'), 'T3': ('C3', 'C3'), 'D4': ('C4', 'C4'),
    'S4': ('S4', 'C4'), 'R4': ('C4', 'C4'), 'H6': ('D6h', 'H6'),
}

new_cores = []
seen_canonical = set()

for sym in ['L2', 'T3', 'D4', 'S4', 'R4', 'H6']:
    d = os.path.join(core_dir, sym)
    for f in sorted(os.listdir(d)):
        if not f.endswith('.cjson'):
            continue
        if any(f.startswith(p) for p in ['SMILE', 'SD', 'SDIF']):
            continue
        with open(os.path.join(d, f)) as fh:
            data = json.load(fh)
        smi = data.get('properties', {}).get('smiles', '')
        if not smi:
            continue

        smi_clean = smi.replace('[Q]', '*')
        smi_clean = re.sub(r'\[R\d*\]', '[H]', smi_clean)
        try:
            mol = Chem.MolFromSmiles(smi_clean)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
        except:
            continue

        if canonical in existing_core_smis or canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        n_stars = canonical.count('*')
        # 过滤无效模板
        if n_stars == 0 or n_heavy == 0:
            continue
        # 目标臂数应匹配
        expected_arms = {'L2': 2, 'T3': 3, 'D4': 4, 'S4': 4, 'R4': 4, 'H6': 6}
        if n_stars != expected_arms.get(sym, n_stars):
            continue

        target_pg, sym_family = sym_map.get(sym, ('C1', 'C1'))
        file_key = f.replace('.cjson', '')
        key = f"pb_{file_key}_{sym}"

        # Format SMILES
        if len(canonical) <= 80:
            smi_str = f'"{canonical}"'
        else:
            smi_str = '(\n            "' + '"\n            "'.join(
                canonical[i:i+80] for i in range(0, len(canonical), 80)
            ) + '"\n        )'

        new_cores.append({
            'key': key, 'smiles_str': smi_str,
            'description': f'[pycofbuilder] {data.get("name", file_key)} ({sym})',
            'target_pg': target_pg, 'n_arms': n_stars, 'symmetry_family': sym_family,
        })

# ============================================================
# Output
# ============================================================
out_lines = []

out_lines.append(f"# === pyCOFBuilder connectors: {len(new_arms)} new ===\n")
for a in new_arms:
    out_lines.append(f'    "{a["key"]}": {{')
    out_lines.append(f'        "smiles": "{a["smiles"]}",')
    out_lines.append(f'        "description": "{a["description"]}",')
    out_lines.append(f'        "functional_group": "{a["functional_group"]}",')
    out_lines.append(f'    }},\n')

out_lines.append(f"# === pyCOFBuilder cores: {len(new_cores)} new ===\n")
for c in new_cores:
    out_lines.append(f'    "{c["key"]}": {{')
    out_lines.append(f'        "smiles": {c["smiles_str"]},')
    out_lines.append(f'        "description": "{c["description"]}",')
    out_lines.append(f'        "target_pg": "{c["target_pg"]}",')
    out_lines.append(f'        "n_arms": {c["n_arms"]},')
    out_lines.append(f'        "symmetry_family": "{c["symmetry_family"]}",')
    out_lines.append(f'    }},\n')

out_path = '/home/tianyajun/MARL_for_COFs/tools/new_entries.txt'
with open(out_path, 'w') as f:
    f.write('\n'.join(out_lines))

print(f"Arms: {len(new_arms)} new")
for a in new_arms:
    print(f"  {a['key']}")
print(f"\nCores: {len(new_cores)} new")
for c in new_cores:
    print(f"  {c['key']:50s} pg={c['target_pg']} arms={c['n_arms']}")
print(f"\nSaved to {out_path}")
