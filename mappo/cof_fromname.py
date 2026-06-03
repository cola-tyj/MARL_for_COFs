import csv
import re
import os
import shutil
from rdkit.DataStructs import ConvertToNumpyArray
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from combine_substructures import combined
from pycofbuilder.cjson import ChemJSON
from pycofbuilder.framework import Framework

def get_data(input):
    # 得到拓扑
    top = input.split('-')[-1:]
    input = input.split('-')[0:-1]
    # 得到对称类型
    SymmetricType = [part.replace('_CLS', '') for part in input if 'CLS' in part]
    input = '-'.join(input)[:-1]
    # 得到info
    parts = input.split('-')
    cls_indices = [i for i, part in enumerate(parts) if 'CLS' in part]
    info = ['CLS-'+'-'.join(parts[cls_indices[0] + 1:cls_indices[1]]), 'CLS-'+'-'.join(parts[cls_indices[1] + 1:])]
    # 得到new_info
    new_info = [i.split('_')[-1] for i in info]
    return SymmetricType,new_info,info

def get_mol(input):
    try:
        mol = combined(input)  # 从-连接的串转换成mol
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)  # 从mol转换成smiles
        return mol
    except:
        return Chem.Mol()  # 一个空的分子对象

def make_cof(SymmetricType,new_info,info,cofname):
    cof_dir = '/home/liuhaoyu/code/rnd_1/cofs'

    # 得到mol格式
    mols = []
    for i in range(len(info)):
        mol = get_mol(info[i])  # 从-连接的串转换成mol
    mols.append(mol)
    # 创建一个topology字典
    topology = {('T3', 'L2'): 'HCB_A',
                ('S4', 'S4'): 'SQL',
                ('S4', 'L2'): 'SQL_A',
                ('H6', 'T3'): 'KGD',
                ('H6', 'L2'): 'HXL_A'
                }
    reaction = [('NH2','CHO'),('CHO','NH2'),
                ('NHOH','CHO'),('CHO','NHOH'),
                ('CH2CN','CHO'),('CHO','CH2CN'),
                ('COOH','NH2'),('NH2','COOH'),
                ('OHc','NH2'),('NH2','OHc'),
                ('Cl','Cl')]
    i = 0
    j = 1
    top = topology[(SymmetricType[i],SymmetricType[j])]
    parts = info[i].split('_')[1:-1]
    R1 = '_'.join(parts)
    parts = info[j].split('_')[1:-1]
    R2 = '_'.join(parts)
    if SymmetricType[i]=='S4':
        if SymmetricType[j]=='S4':
            x = SymmetricType[i]+'_'+re.split('[-_]',info[i])[1]+'_'+new_info[i]+'_'+R1+'-'+SymmetricType[j]+'_'+re.split('[-_]',info[j])[1]+'_'+new_info[j]+'_'+R2+'-'+top+'-AA'
        else:
            x = SymmetricType[i]+'_'+re.split('[-_]',info[i])[1]+'_'+new_info[i]+'_'+R1+'-'+SymmetricType[j]+'_agent'+str(j)+'_'+new_info[j]+'_'+R2+'-'+top+'-AA'
    elif '310' in new_info[i]:
        x = SymmetricType[i]+'_'+re.split('[-_]',info[i])[1]+'_'+new_info[i]+'_'+R1+'-'+SymmetricType[j]+'_agent'+str(j)+'_'+new_info[j]+'_'+R2+'-'+top+'-AA'
    else:
        x = SymmetricType[i]+'_agent'+str(i)+'_'+new_info[i]+'_'+R1+'-'+SymmetricType[j]+'_agent'+str(j)+'_'+new_info[j]+'_'+R2+'-'+top+'-AA'
    try:
        print(x)
        cof = Framework(x)
        cof.save(fmt='cif', supercell = [1, 1, 1], save_dir = cof_dir)
        move_cof(cofname)
    except:
        print('不能生成COF')

def move_cof(cofname):
    cof_dir = '/home/liuhaoyu/code/rnd_1/cofs'
    
    # 保存的cof的cif文件名
    for file in os.listdir(cof_dir):
        # 检查文件扩展名是否为'.cif'
        if file.endswith('.cif'):
            x = file
            # 找到文件后就可以停止搜索
            break
    
    # 确保目标文件夹存在
    if not os.path.exists(os.path.join(cof_dir, 'cif')):
        os.makedirs(os.path.join(cof_dir, 'cif'))
    
    # 定义要移动和重命名的文件的名称
    original_file_name = x  # 原始文件名
    new_file_name = cofname + '.cif'

    # 构造源文件和目标文件的完整路径
    source_file = os.path.join(cof_dir, original_file_name)
    destination_file = os.path.join(cof_dir, 'cif', new_file_name)
    # 移动文件
    shutil.move(source_file, destination_file)

def main():
    max_count=500
    file_path = '/home/liuhaoyu/code/rnd_1/mappo/data_train/rand2.5.csv'
    inputs = []
    count = 0
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if count >= max_count:
                break  # 如果已经读取了足够的行数，就停止读取
            else:
                inputs.append(row[0])
                count += 1
    
    for input in inputs:
        SymmetricType,new_info,info = get_data(input)
        make_cof(SymmetricType,new_info,info,input)

if __name__ == "__main__":
    main()