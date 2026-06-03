from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdmolfiles import MolToSmiles, MolToXYZFile, MolFromXYZFile, MolFromSmiles
import os

# 把I原子换成Q，且键长缩短为0.2倍
def change_Q(mol, name, cid):
    # 打开xyz文件
    with open(name, 'r') as file:
        lines = file.readlines()

    # 找到所有与I原子相连的原子和键
    i_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'I']
    connected_atoms_and_bonds = []
    for i_atom in i_atoms:
        for neighbor_atom in i_atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(i_atom.GetIdx(), neighbor_atom.GetIdx())
            connected_atoms_and_bonds.append((i_atom, neighbor_atom, bond))

    # 修改坐标
    for i, (i_atom, neighbor_atom, bond) in enumerate(connected_atoms_and_bonds):
        # 得到H和neighbor的编号及坐标
        i_atom_idx = i_atom.GetIdx()
        neighbor_atom_idx = neighbor_atom.GetIdx()
        i_atom_coords = mol.GetConformer(cid).GetAtomPosition(i_atom_idx)
        neighbor_atom_coords = mol.GetConformer(cid).GetAtomPosition(neighbor_atom_idx)
        # I的键长缩短为0.2倍
        x = 0.8 * neighbor_atom_coords.x + 0.2 * i_atom_coords.x
        y = 0.8 * neighbor_atom_coords.y + 0.2 * i_atom_coords.y
        z = 0.8 * neighbor_atom_coords.z + 0.2 * i_atom_coords.z
        # 因为xyz文件前两行不是坐标，所以 +2
        lines[i_atom_idx+2] = 'Q ' + "{:.6f}".format(x) + ' ' + "{:.6f}".format(y) + ' ' + "{:.6f}".format(z) + '\n'

    # 
    with open(name, 'w') as file:
        file.writelines(lines)

def change_R(mol, name, cid):
    # 打开xyz文件
    with open(name, 'r') as file:
        lines = file.readlines()

    # 找到所有与*原子相连的原子和键
    i_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    connected_atoms_and_bonds = []
    for i_atom in i_atoms:
        for neighbor_atom in i_atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(i_atom.GetIdx(), neighbor_atom.GetIdx())
            connected_atoms_and_bonds.append((i_atom, neighbor_atom, bond))

    # 修改坐标
    for i, (i_atom, neighbor_atom, bond) in enumerate(connected_atoms_and_bonds):
        # 得到H和neighbor的编号及坐标
        i_atom_idx = i_atom.GetIdx()
        neighbor_atom_idx = neighbor_atom.GetIdx()
        i_atom_coords = mol.GetConformer(cid).GetAtomPosition(i_atom_idx)
        neighbor_atom_coords = mol.GetConformer(cid).GetAtomPosition(neighbor_atom_idx)
        # *的键长缩短为0.6倍
        x = 0.4 * neighbor_atom_coords.x + 0.6 * i_atom_coords.x
        y = 0.4 * neighbor_atom_coords.y + 0.6 * i_atom_coords.y
        z = 0.4 * neighbor_atom_coords.z + 0.6 * i_atom_coords.z
        # 因为xyz文件前两行不是坐标，所以 +2
        if i<20:
            lines[i_atom_idx+2] = 'R' + str(i+1) + ' ' + "{:.6f}".format(x) + ' ' + "{:.6f}".format(y) + ' ' + "{:.6f}".format(z) + '\n'
        else:
            lines[i_atom_idx+2] = 'H ' + "{:.6f}".format(x) + ' ' + "{:.6f}".format(y) + ' ' + "{:.6f}".format(z) + '\n'
    # 
    with open(name, 'w') as file:
        file.writelines(lines)

def change_H(mol, name, cid):
    # 打开xyz文件
    with open(name, 'r') as file:
        lines = file.readlines()

    # 找到所有与H原子相连的原子和键
    i_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    connected_atoms_and_bonds = []
    for i_atom in i_atoms:
        for neighbor_atom in i_atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(i_atom.GetIdx(), neighbor_atom.GetIdx())
            connected_atoms_and_bonds.append((i_atom, neighbor_atom, bond))

    # 修改坐标
    for i, (i_atom, neighbor_atom, bond) in enumerate(connected_atoms_and_bonds):
        # 得到H和neighbor的编号及坐标
        i_atom_idx = i_atom.GetIdx()
        neighbor_atom_idx = neighbor_atom.GetIdx()
        i_atom_coords = mol.GetConformer(cid).GetAtomPosition(i_atom_idx)
        neighbor_atom_coords = mol.GetConformer(cid).GetAtomPosition(neighbor_atom_idx)
        # *的键长缩短为0.6倍
        x = 0.4 * neighbor_atom_coords.x + 0.6 * i_atom_coords.x
        y = 0.4 * neighbor_atom_coords.y + 0.6 * i_atom_coords.y
        z = 0.4 * neighbor_atom_coords.z + 0.6 * i_atom_coords.z
        # 因为xyz文件前两行不是坐标，所以 +2
        lines[i_atom_idx+2] = 'H ' + "{:.6f}".format(x) + ' ' + "{:.6f}".format(y) + ' ' + "{:.6f}".format(z) + '\n'
    # 
    with open(name, 'w') as file:
        file.writelines(lines)

# 把双键的立体类型更改为STEREOANY，生成二维xyz坐标
def _2Dxyz(mol, name) :
    xyz_file_path = './xyzs/' + name + '.xyz'
    #mol = Chem.MolFromSmiles(smiles) #将 SMILES 表示转换为分子对象的代码
    Chem.Kekulize(mol)
    Chem.FindPotentialStereoBonds(mol)

    for bond in mol.GetBonds():
        if bond.GetBondType()==Chem.BondType.DOUBLE:
            if bond.GetStereo() == Chem.BondStereo.STEREONONE or bond.GetStereo() == Chem.BondStereo.STEREOZ or bond.GetStereo() == Chem.BondStereo.STEREOE:
                Chem.Bond.SetStereo(bond, Chem.BondStereo.STEREOANY) 
    
    # 计算分子的二维坐标
    AllChem.Compute2DCoords(mol)

    # 获取分子的原子数和二维坐标
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    coords_2d = conf.GetPositions()

    # 将二维坐标当作三维坐标保存为 XYZ 格式文件
    with open(xyz_file_path, 'w') as f:
        f.write(str(num_atoms) + '\n')
        f.write('\n')
        for atom_idx in range(num_atoms):
            atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            x, y, _ = coords_2d[atom_idx]  # 使用二维坐标，将 Z 坐标设为 0
            x = '%.8f' % float(x)
            y = '%.8f' % float(y)
            f.write(f"{atom_symbol} {x} {y} 0\n")
    change_Q(mol, xyz_file_path, -1)
    change_R(mol, xyz_file_path, -1)
    change_H(mol, xyz_file_path, -1)

'''smiles = 'C=C(N/N=C/I)C%13=CC(C(N/N=C/I)=C)=NC(C(N/N=C/I)=C)=C%13'
idx = 1
smiles2Dxyz_ANY(smiles, idx) '''