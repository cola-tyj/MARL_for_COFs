from combine_substructures import combined
from xyz import _2Dxyz
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from pycofbuilder.tools import smiles_to_xsmiles
from pycofbuilder.cjson import ChemJSON
from pycofbuilder.framework import Framework

#str = 'CLS-TPTA_F_EPO_OCOCH3-PYRN_SO2H_CN_NO_OProp'
#str = 'CLS-BENZ3_CH3-BENZ2_CH3_COOH'
#mol = combined(str)
def f1():  # 画出分子->smiles->xyz和正规smiles
    str = 'C(I)1=C(I)C(I)=C(I)C(I)=C1I'
    mol = Chem.MolFromSmiles(str)
    mol = AllChem.AddHs(mol)
    image = Draw.MolToImage(mol)
    image.save("./imgs/"+'str'+".png")
    _2Dxyz(mol, 'test')

    # 将分子对象转换为SMILES表示法，设置参数确保输出格式
    output_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)
    smiles = output_smiles.replace('*', '[R]')
    #smiles = output_smiles.replace('[H]', '[Q]')
    # 输出结果
    print(smiles)

def f2():  # 需要手动替换smiles里的Q和R -> 图片
    smiles = '[Q]C1=C([Q])C([Q])=C([Q])C([Q])=C1[Q]'
    #smiles = '[Q]C1=C([R2])C([R1])=C([Q])C([R2])=C1[R1]'
    xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles)
    print(xsmiles, xsmiles_label, composition, _labels)
    scaffold = Chem.MolFromSmiles(xsmiles + ' ' + xsmiles_label)
    image = Draw.MolToImage(scaffold)
    image.save("./imgs/"+smiles+".png")

def f3():  # smiles -> cjson
    smiles = '[Q]C1=C([Q])C([Q])=C([Q])C([Q])=C1[Q]'
    xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles)
    print(xsmiles, xsmiles_label, composition)

    new_BB = ChemJSON()

    new_BB.from_xyz('/home/liuhaoyu/code/rnd_1/xyzs', 'test.xyz')

    new_BB.name = 'BENZ6'

    new_BB.properties = {
        "smiles": smiles,
        "code": new_BB.name,
        "xsmiles": xsmiles,
        "xsmiles_label": xsmiles_label,
    }

    new_BB.write_cjson('/home/liuhaoyu/code/rnd_1/pycofbuilder/data/core/H6', '666.cjson')

def f4():  # 生成cif
    cof = Framework('H6_666_Cl-L2_217_Cl-HXL_A-AA')
    cof.save(fmt='cif', supercell = [1, 1, 1], save_dir = '/home/liuhaoyu/code/rnd_1/imgs')

# f3()
# f4()
f1()