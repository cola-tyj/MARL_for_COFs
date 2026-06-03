from tools import smiles_to_xsmiles
from cjson import ChemJSON
from framework import Framework

smiles_string = '[Q]C1=C(C)C(Br)=C(C2=C(C)C(Br)=C(C3=C(Br)C(C4=C(Br)C(C)=C(C5=C(Br)C(C)=C([Q])C(Br)=C5C)C(Br)=C4C)=C(Br)C(C4=C(Br)C(C)=C(C5=C(Br)C(C)=C([Q])C(Br)=C5C)C(Br)=C4C)=C3Br)C(C)=C2Br)C(C)=C1Br'
xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles_string)
#print(xsmiles, xsmiles_label, composition)

new_BB = ChemJSON()

new_BB.from_xyz('/home/liuhaoyu/code/mappo/cofs', 'test.xyz')

new_BB.name = 'C3'

new_BB.properties = {
    "smiles": smiles_string,
    "code": new_BB.name,
    "xsmiles": xsmiles,
    "xsmiles_label": xsmiles_label,
}

new_BB.write_cjson('/home/liuhaoyu/code/mappo/pycofbuilder/data/core/T3', 'test.cjson')
cof = Framework('T3_test_CHO-L2_BENZ_NH2_OH-HCB_A-AA')

cof.save(fmt='cif', supercell = [1, 1, 2], save_dir = '/home/liuhaoyu/code/mappo/cofs')