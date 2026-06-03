from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from ccjson import get_xmiles, get_labels

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def _add_R_group(mol, R, n):
    #R = 'H'
    path = '/home/tianyajun/MARL_for_COFs/data/all/'
    file_name = R + '.cjson'
    #xsmiles = "[*]Br"
    xsmiles = get_xmiles(path, file_name)
    bond_type = Chem.rdchem.BondType.SINGLE
    begin_atom_idx = n
    R = Chem.MolFromSmiles(xsmiles)
    for i, atom in enumerate(R.GetAtoms()):
        if atom.GetSymbol() == "*":
            # 删除R上连接'*'原子的键
            for neighbor_atom in atom.GetNeighbors():
                R = Chem.RWMol(R)
                end_atom_idx = neighbor_atom.GetIdx()
                R.RemoveBond(i,end_atom_idx)
            break
    combined = Chem.CombineMols(mol, R)
    rw_combined = Chem.RWMol(combined)
    end_atom_idx += mol.GetNumAtoms()
    rw_combined.AddBond(begin_atom_idx, end_atom_idx, bond_type)
    # 删除R上的'*'原子
    rw_combined.RemoveAtom(i+mol.GetNumAtoms())
    mol = rw_combined.GetMol()
    return mol
    
def add_R_groups(str):
    path = '/home/tianyajun/MARL_for_COFs/data/all/'
    file_name = str
    xsmiles = get_xmiles(path, file_name)
    labels = get_labels(path, file_name)
    mol = Chem.MolFromSmiles(xsmiles)
    #mol = Chem.RemoveHs(mol)
    # 所有'*'原子的序号
    idx = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "*":
            idx.append(i)

    # 把Q接上I，作为标记
    j = len(labels) - 1
    for i in reversed(idx):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == "*":
            if labels[j] == 'Q':
                # 删除连接'*'原子的键
                for neighbor_atom in atom.GetNeighbors():
                    rw_mol = Chem.RWMol(mol)
                    rw_mol.RemoveBond(i,neighbor_atom.GetIdx())
                    mol = rw_mol.GetMol()
                # 添加官能团
                mol = _add_R_group(mol, 'I', neighbor_atom.GetIdx())
                # 删除'*'原子
                rw_mol = Chem.RWMol(mol)
                rw_mol.RemoveAtom(i)
                mol = rw_mol.GetMol()
                j -= 1
            else:
                j -= 1
    return mol

def _add_substructure(mol_1, mol_2):
    # 所有 mol_1 中'I'原子的序号
    idx = []
    for i, atom in enumerate(mol_1.GetAtoms()):
        if atom.GetSymbol() == "I":
            idx.append(i)
    # 删除 mol_2 其中一个连接'I'原子的键
    for i, atom in enumerate(mol_2.GetAtoms()):
        if atom.GetSymbol() == "I":
            for neighbor_atom_2 in atom.GetNeighbors():
                rw_mol = Chem.RWMol(mol_2)
                rw_mol.RemoveBond(i,neighbor_atom_2.GetIdx())
                mol_2 = rw_mol.GetMol()
            break
    _idx = i   # mol_2 中要删掉的'I'原子

    for i in reversed(idx):
        atom = mol_1.GetAtomWithIdx(i)
        # 删除 mol_1 中连接'I'原子的键
        for neighbor_atom_1 in atom.GetNeighbors():
            rw_mol = Chem.RWMol(mol_1)
            rw_mol.RemoveBond(i,neighbor_atom_1.GetIdx())
            mol_1 = rw_mol.GetMol()

        # 连接mol_1和mol_2
        bond_type = Chem.rdchem.BondType.SINGLE
        combined = Chem.CombineMols(mol_1, mol_2)
        rw_combined = Chem.RWMol(combined)
        begin_atom_idx = neighbor_atom_1.GetIdx()
        end_atom_idx = neighbor_atom_2.GetIdx() + mol_1.GetNumAtoms()
        rw_combined.AddBond(begin_atom_idx, end_atom_idx, bond_type)
        rw_combined.RemoveAtom(_idx+mol_1.GetNumAtoms())
        rw_combined.RemoveAtom(i)
        mol_1 = rw_combined.GetMol()

    return mol_1

def get_substructures(str):
    str = str.split('_')[0]
    parts = str.split('-')[1:]   # 不考虑'CLS'
    substructures = []
    for part in parts:
        mol = add_R_groups(part)
        substructures.append(mol)
    return substructures

def combined(str):
    RDLogger.DisableLog('rdApp.*')
    substructures = get_substructures(str)
    combined = substructures[0]
    for substructure in substructures[1:]:
        combined = _add_substructure(combined,substructure)
    combined = Chem.AddHs(combined)
    #combined = Chem.RemoveHs(combined)
    #image = Draw.MolToImage(combined)
    #image.save("./imgs/"+str+".png")
    return combined

'''strs = 'CLS-204-202_CH3_Br_Br'
#get_substructures(strs)
#mol = combined(strs)
mols = get_substructures(strs)
mol = mols[0]
mol = combined(strs)
image = Draw.MolToImage(mol)
image.save(strs+".png")
smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)  # 从mol转换成smiles
print(smiles)
temp = smiles.replace('I', '[Q]')
print(temp)
i = 1
smiles = ''
for x in temp:
    if x == '*':
        smiles = smiles + '[R' + str(i) + ']'
        i += 1
    else:
        smiles += x
print(smiles)'''
#xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles)