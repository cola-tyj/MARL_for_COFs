# 随机生成cofs

import os
import re
import random
import json
import csv
import time
from cof_predictor.main import doPredict
from pycofbuilder.cjson import ChemJSON
from pycofbuilder.framework import Framework

cof_dir = '/home/liuhaoyu/code/rnd_1/cofs/KGD'
conector = ['CH2CN', 'CHO', 'Cl', 'COOH', 'NH2', 'NHOH', 'OHc']
func_groups = ['Br', 'CH3', 'CHO', 'CHS', 'Cl', 'CN', 'COOH', 'EPO', 'F','H','I', 'NH2', 'NO', 'NO2', 'OCOCH3', 'OH', 'OMe', 'SH', 'SO2H']
L2 = ['201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217']
T3 = ['301','302','303','304','305','306','307','308','309','310','311']
S4 = ['PTCA', 'PORP']
H6 = ['BENZ6', 'HECO'] 
MM = ['L2','T3','S4','H6'] 
cofs = []
inputPath = '/home/liuhaoyu/code/mappo_2/sql/input'
midPath = '/home/liuhaoyu/code/rnd_1/sql/mid'

topology = {('T3', 'L2'): 'HCB_A',
            #('T3', 'T3'): 'HCB',
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

def fff(x):
    new_BB = ChemJSON()
    new_BB.from_cjson('/home/liuhaoyu/code/rnd_1/data/all', x+'.cjson')
    input_string = new_BB.properties['xsmiles_label']
    matches = re.findall(r'R(\d+)', input_string)
    if matches:
        max_R_value = max(int(match) for match in matches)
    else:
        max_R_value = 0
    return max_R_value

def handleSingleCif(inputPath, outputPath, fileName):
    """
    去除单个CIF的重复单元

    Args:
        inputPath (str): 输入的文件夹路径,下一级直接包含CIF
        outputPath (str): 输出的文件夹路径,下一级直接包含CIF
        fileName (str): CIF带后缀的文件名,例如C301.cif
    """
    cifName = fileName.split(".")[0]
    os.system(f'ase convert {os.path.join(inputPath, fileName)} {cifName}.poscar')
    os.system(f'phonopy -c {cifName}.poscar --symmetry --tolerance=1e-2')
    os.system(f'ase convert BPOSCAR {os.path.join(outputPath, fileName)}')

def randchoice(a):
    if a == 'L2':
        return random.choice(L2)
    if a == 'T3':
        return random.choice(T3)
    if a == 'S4':
        return random.choice(S4)
    if a == 'H6':
        return random.choice(H6)

def predictor():
    cof_dir = '/home/liuhaoyu/code/rnd_1/cofs'
    top_path = os.path.join(cof_dir, 'topology.json')
    topology_dict = {}
    # 遍历指定目录下的所有文件
    for filename in os.listdir(os.path.join(cof_dir, 'cif')):
        # 检查文件是否以'.cif'结尾
        if filename.endswith('.cif'):
            print(filename)
            cifname = filename[:-4]
            top = cifname.split('-')[-2]
            print(top)
            if top == 'HCB' or top == 'HCB_A':
                new_value = "hcb"
            elif top == 'SQL' or top == 'SQL_A':
                new_value = "sql"
            elif top == 'KGD':
                new_value = "kgd-a"
            elif top == 'HXL_A':
                new_value = "hxl"
            topology_dict[cifname] = new_value

    # 将字典写入JSON文件
    with open(top_path, 'w', encoding='utf-8') as json_file:
        json.dump(topology_dict, json_file, indent=4, ensure_ascii=False)

    resultFilePath = doPredict(os.path.join(cof_dir, 'cif'), logPath=cof_dir, ts=str(int(time.time())))
    print(resultFilePath)
    return resultFilePath

def main():
    for i in range(150000):
        M1 = random.choice(MM)
        M2 = random.choice(MM)
        if (M1,M2) not in topology:
            continue
        C1 = randchoice(M1)
        C2 = randchoice(M2)
        r1 = random.choice(conector)
        r2 = random.choice(conector)
        if (r1,r2) in reaction:
            l1 = fff(C1)
            l2 = fff(C2)
            if l1 < 2:
                x = M1 + '_CLS-' + C1 + '-' + random.choice(L2) + '_' + random.choice(func_groups) + '_' + r1
            else:
                x = M1 + '_CLS-' + C1 + '_' + random.choice(func_groups) + '_' + random.choice(func_groups) + '_' + r1

            cof = x

            if l2 < 2:
                x = M2 + '_CLS-' + C1 + '-' + random.choice(L2) + '_' + random.choice(func_groups) + '_' + r2
            else:
                x = M2 + '_CLS-' + C1 + '_' + random.choice(func_groups) + '_' + random.choice(func_groups) + '_' + r2

            cof = cof + '-' + x + '_-' + topology[(M1,M2)]
            cofs.append(cof)

    # 指定要写入的CSV文件名
    filename = "/home/liuhaoyu/code/rnd_1/cofs.csv"

    # 使用 'write' 模式打开文件
    with open(filename, 'w', newline='') as csvfile:
        # 创建一个csv写入器
        writer = csv.writer(csvfile)
        
        # 循环遍历列表，将每个字符串写入CSV文件的一行
        for co in cofs:
            # 写入单个字符串，作为一个列表的单个元素
            writer.writerow([co])

def main1():
    list = []
    M1 = 'H6'
    M2 = 'T3'
    r1 = 'NH2'
    r2 = 'CHO'
    C1 = 'HEXB'
    for C2 in T3:
        '''for f in func_groups:
            x = M1 + '_' + C1 + '_' + r1 + '_' + f + '-' + M2 + '_' + C2 + '_' + r2 + '-HCB_A-AA'
            cof = Framework(x)
            cof.save(fmt='cif', supercell = [1, 1, 1], save_dir = cof_dir)
            list.append(x)

            x = M1 + '_' + C1 + '_' + r1 + '-' + M2 + '_' + C2 + '_' + r2 + '_' + f + '-HCB_A-AA'
            #cof = Framework(x)
            #cof.save(fmt='cif', supercell = [1, 1, 1], save_dir = cof_dir)
            list.append(x)'''
        x = M1 + '_' + C1 + '_' + r1 + '-' + M2 + '_' + C2 + '_' + r2 + '-KGD-AA'
        cof = Framework(x)
        cof.save(fmt='cif', supercell = [1, 1, 1], save_dir = cof_dir)
        list.append(x)
    print(len(list))

if __name__ == "__main__":
    predictor()
