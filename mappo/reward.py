import torch 
import torch.nn
import torch.nn.functional as F
import pandas as pd
import shutil
import csv
import sys
import os
import re
import json
import time
from rdkit import Chem
from xyz import _2Dxyz
from combine_substructures import combined
from cof_predictor.main import doPredict
from pycofbuilder.cjson import ChemJSON
from pycofbuilder.framework import Framework
from pycofbuilder.tools import smiles_to_xsmiles

class NN(torch.nn.Module):
    def __init__(self,in_dim,out_dim,n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid
        
        self.fc1 = torch.nn.Linear(in_dim,n_hid,'linear')
        self.fc2 = torch.nn.Linear(n_hid,n_hid,'linear')
        self.fc3 = torch.nn.Linear(n_hid,out_dim,'linear')
        #self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        #y = self.softmax(y)
        return y

class RND:
    def __init__(self,in_dim,out_dim,n_hid):
        self.in_dim = in_dim
        self.target = NN(in_dim,out_dim,n_hid)
        self.target.load_state_dict(torch.load('/home/liuhaoyu/code/rnd_1/mappo/model/decoder.pth'))
        self.model = NN(in_dim,out_dim,n_hid)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        
    def get_reward(self,input):
        #input = input.view(-1,self.in_dim)
        y_true = self.target(input).detach().mul(0.1)
        y_pred = self.model(input)
        a = y_pred - y_true
        rewards = torch.pow(y_pred - y_true,2).sum(dim=1)
        return rewards
    
    def update(self,replay_buffer):
        batch = replay_buffer.get_training_data()  # get training data
        S0 = batch['obs_n'].clone().detach()
        Ri = self.get_reward(S0)
        Ri.sum().backward()
        self.optimizer.step()
        
def get_mol(input):
    try:
        mol = combined(input)  # 从-连接的串转换成mol
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)  # 从mol转换成smiles
        return mol
    except:
        return Chem.Mol()  # 一个空的分子对象

def middle_reward(inputs):
    rewards = []
    for input in inputs:
        mol = get_mol(input)
        if mol.GetNumAtoms() == 0:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards

def final_reward(SymmetricType,new_info,info,episode_num):
    mols = []
    rewards = []
    # 转成mol格式
    for i in range(len(info)):
        mol = get_mol(info[i])  # 从-连接的串转换成mol
        mols.append(mol)
        if mol.GetNumAtoms() == 0:
            rewards.append(-1.0)
        else:
            try:
                _2Dxyz(mol, str(i))  # 生成xyz文件
            except:
                rewards.append(-1.0)
                continue
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)  # 从mol转换成smiles
            temp = smiles.replace('I', '[Q]')
            r = 1
            smiles = ''
            for x in temp:
                if x == '*':
                    smiles = smiles + '[R' + str(r) + ']'
                    r += 1
                else:
                    smiles += x
            xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles)
            new_BB = ChemJSON()
            print(info[i])
            new_BB.from_xyz('/home/liuhaoyu/code/rnd_1/xyzs', str(i)+'.xyz')
            new_BB.name = str(i)
            new_BB.properties = {
                "smiles": smiles,
                "code": new_BB.name,
                "xsmiles": xsmiles,
                "xsmiles_label": xsmiles_label,
            }
            new_BB.write_cjson('/home/liuhaoyu/code/rnd_1/pycofbuilder/data/core/'+SymmetricType[i], 'agent'+str(i)+'.cjson')
            rewards.append(0.0)

    # 生成cof
    make_cof(SymmetricType,new_info,info,mols,episode_num)
    # 预测器
    # predictor
    print('episode_num:',episode_num)
    return rewards

def make_cof(SymmetricType,new_info,info,mols,episode_num):
    cof_dir = '/home/liuhaoyu/code/rnd_1/cofs'
    # 创建一个topology字典
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
    print(new_info)
    #cof = Framework('T3_test_CHO-L2_BENZ_NH2_OH-HCB_A-AA')
    #cof.save(fmt='cif', supercell = [1, 1, 2], save_dir = '/home/liuhaoyu/code/rnd_1/cofs')
    for i in range(len(new_info)):
        for j in range(len(new_info)):
            if i == j or mols[i].GetNumAtoms() == 0 or mols[j].GetNumAtoms() == 0:
                continue
            else:
                if (SymmetricType[i],SymmetricType[j]) in topology and (new_info[i],new_info[j]) in reaction:  # 保证拓扑和反映类型的合理性
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
                        move_cof(SymmetricType,new_info,info,episode_num,i,j,top)
                    except:
                        print('不能生成COF')
                else:
                    #print('Not a Cof')
                    pass

def move_cof(SymmetricType,new_info,info,episode_num,i,j,top):
    cof_dir = '/home/liuhaoyu/code/rnd_1/cofs'
    cif_dir = '/home/liuhaoyu/code/rnd_1/cifs'
    top_path = os.path.join(cof_dir, 'topology.json')
    
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
    new_file_name = SymmetricType[i]+'_'+info[i]+'-'+SymmetricType[j]+'_'+info[j]+'-'+top+'.cif'

    with open('/home/liuhaoyu/code/rnd_1/number.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入一行数据
        writer.writerow([episode_num, i, j, new_file_name.replace(".cif", "")])

    # 构造源文件和目标文件的完整路径
    source_file = os.path.join(cof_dir, original_file_name)
    destination_file = os.path.join(cof_dir, 'cif', new_file_name)
    # 移动文件
    shutil.move(source_file, destination_file)

def predictor():
    cof_dir = '/home/liuhaoyu/code/rnd_1/cofs'
    top_path = os.path.join(cof_dir, 'topology.json')
    topology_dict = {}
    # 遍历指定目录下的所有文件
    for filename in os.listdir(os.path.join(cof_dir, 'cif')):
        # 检查文件是否以'.cif'结尾
        if filename.endswith('.cif'):
            cifname = filename[:-4]
            top = cifname.split('-')[-1]
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

def rm_dir(directory_path = '/home/liuhaoyu/code/rnd_1/cofs'):
    if os.path.exists(directory_path):
        # 使用shutil.rmtree递归删除目录及其内容
        shutil.rmtree(directory_path)
        print(f"目录 {directory_path} 已被清空。")
        os.makedirs(directory_path)
    else:
        print(f"目录 {directory_path} 不存在。")

def average_pred(FilePath):
    outputFilePath = '/home/liuhaoyu/code/rnd_1/ave_result.csv'
    x = pd.read_csv(FilePath)
    # 计算'pred'列的平均值
    average_pred = x['pred'].mean()
    average_df = pd.DataFrame([average_pred], columns=['Average of pred'])
    # 读取目标文件，如果文件不存在，将会创建一个空的DataFrame
    try:
        df_output = pd.read_csv(outputFilePath)
    except FileNotFoundError:
        df_output = pd.DataFrame()
    # 将平均值追加到目标文件的DataFrame中
    df_output = pd.concat([df_output, average_df], ignore_index=True)
    # 将新的DataFrame写入到目标文件
    df_output.to_csv(outputFilePath, index=False)

def max_pred(FilePath):
    outputFilePath = '/home/liuhaoyu/code/rnd_1/max_result.csv'
    x = pd.read_csv(FilePath)
    # 计算'pred'列的最大值
    max_pred = x['pred'].max()
    max_df = pd.DataFrame([max_pred], columns=['Maximum of pred'])
    # 读取目标文件，如果文件不存在，将会创建一个空的DataFrame
    try:
        df_output = pd.read_csv(outputFilePath)
    except FileNotFoundError:
        df_output = pd.DataFrame()
    # 将最大值追加到目标文件的DataFrame中
    df_output = pd.concat([df_output, max_df], ignore_index=True)
    # 将新的DataFrame写入到目标文件
    df_output.to_csv(outputFilePath, index=False)

