import gym
import os
from gym import spaces
import numpy as np
import re
import torch
import random
import shutil
from pycofbuilder.cjson import ChemJSON
from reward import middle_reward, final_reward
from transformer import Embedding_Layer
from transformer import TransformerEncoder

class CustomEnvironment(gym.Env):
    def __init__(self, env_name, args, discrete=True):
        super(CustomEnvironment, self).__init__()
        self.n = 6  # The number of agents
        # 设置文件夹路径
        folder_path = '/home/liuhaoyu/code/rnd_1/data'
        # 创建词汇表，设置mask
        self.vocab, self.linker_mask, self.core_mask, self.conector_mask, self.func_mask = self.get_data(folder_path)
        # 定义动作空间和状态空间
        self.action_space = 53
        self.observation_space = 128
        self.pad = 32  # 设置补齐长度
        # 定义Embedding层
        self.Embedding_Layer = Embedding_Layer(self.observation_space, self.pad)
        self.Transformer_model = TransformerEncoder(dim=self.observation_space)
        
        # 加载 Embedding_Layer 参数
        self.Embedding_Layer.load_state_dict(torch.load('/home/liuhaoyu/code/rnd_1/mappo/model/embedding_parameters.pth'))
        self.Transformer_model.load_state_dict(torch.load('/home/liuhaoyu/code/rnd_1/mappo/model/transformer_encoder.pth'))
        # 初始化环境的一些参数
        self.max_step = args.episode_limit
        self.now_step = [0 for i in range(self.n)]
        self.pre_step = [0 for i in range(self.n)]
        self.SymmetricType = ['' for i in range(self.n)]

    def reset(self):
        # 重置环境的状态
        self.now_step = [0 for i in range(self.n)]
        self.pre_step = [0 for i in range(self.n)]
        self.action_mask = [self.core_mask for i in range(self.n)]
        self.SymmetricType = ['' for i in range(self.n)]
        self.info = ['CLS'] * self.n # 生成初始状态
        observations = [[self.vocab[i]] for i in self.info] 
        # 计算需要补齐的数量
        padding_size = self.pad - len(observations[0])
        # 补齐
        for i in range(self.n):
            observations[i] = np.pad(observations[i], (0, padding_size), constant_values=0)
        # 嵌入 N x 32 -> N x 128
        observations = self.Embedding_observation(observations)
        return observations, self.info, self.action_mask

    def step(self, action, episode_num):
        self.new_info = [self.find_key_by_value(a) for a in action]
        for i in range(self.n):
            if self.new_info[i] == 'stop':
                self.now_step[i] = self.max_step - 2

        # 找到对称类型及拓扑网络
        if self.now_step[0] == 0:
            self.Symmetric_Type()
        # 新的状态
        for i in range(self.n):
            if self.new_info[i] == 'stop':
                continue
            elif self.now_step[i] >= self.max_step:
                continue
            elif self.now_step[i] == (self.max_step - 1):
                self.info[i] = self.info[i] + '_' + self.new_info[i]  # 最后一步用下划线加上connector
            elif self.new_info[i] in self.func_groups:
                if random.random() < 0.9:
                    self.new_info[i] = 'H'
                self.info[i] = self.info[i] + '_' + self.new_info[i]  # 用下划线加上官能团
            else:
                self.info[i] = self.info[i] + '-' + self.new_info[i]  # 用短横线加上子结构
        # 更新step并设置 action_mask
        for i in range(self.n):
            self.now_step[i] += 1
            self.pre_step[i] += 1
            num_R = self.max_R(self.new_info[i])
            if self.pre_step[i] == 1:
                self.pre_step[i] = self.pre_step[i] + num_R
            else:
                self.pre_step[i] = self.pre_step[i] + num_R * 2
            if self.now_step[i] >= self.max_step-1:
                self.action_mask[i] = self.conector_mask  # 下一步添加connector
            elif self.pre_step[i] < (self.max_step-1):
                self.action_mask[i] = self.linker_mask  # 下一步添加子结构(linker)
            else:
                self.action_mask[i] = self.func_mask  # 下一步添加官能团
        done = [step >= self.max_step for step in self.now_step]
        # 更新状态，映射到索引
        new_states = []
        for i in range(self.n):
            # 分解输入字符串为部分
            input_parts = re.split(r'[-_]', self.info[i])
            # 将输入字符串的每个部分映射到索引
            input_indices = [self.vocab[part] for part in input_parts]
            # 计算需要补齐的数量
            padding_size = self.pad - len(input_indices)
            # 补齐
            new_state = np.pad(input_indices, (0, padding_size), constant_values=0)
            new_states.append(new_state)
        # 嵌入 N x 32 -> N x 128
        observations = self.Embedding_observation(new_states)
        
        # 设置奖励
        if all(done):
            rewards = final_reward(self.SymmetricType, self.new_info, self.info, episode_num)
            print('final_reward:',rewards)
        else:
            rewards = middle_reward(self.info)
        print(self.info)
        print(self.now_step)
        print(self.pre_step)
        return observations, rewards, done, self.info, self.action_mask

    def max_R(self,x):
        try:
            new_BB = ChemJSON()
            new_BB.from_cjson('/home/liuhaoyu/code/rnd_1/data/all', x+'.cjson')
            # 输入字符串
            input_string = new_BB.properties['xsmiles_label']
            # 使用正则表达式找到所有的R后面的数字
            matches = re.findall(r'R(\d+)', input_string)
            # 如果找到了匹配项，将它们转换为整数并找到最大值
            if matches:
                max_R_value = max(int(match) for match in matches)
            else:
                max_R_value = 0
        except:
            max_R_value = 0
        return max_R_value
    
    def Symmetric_Type(self):
        for i in range(self.n):
            if self.new_info[i] in self.C2:
                self.SymmetricType[i] = 'L2'
            elif self.new_info[i] in self.C3:
                self.SymmetricType[i] = 'T3'
            elif self.new_info[i] in self.C4:
                self.SymmetricType[i] = 'S4'
            elif self.new_info[i] in self.C6:
                self.SymmetricType[i] = 'H6'
            else:
                self.SymmetricType[i] = 'Unknown'
        
    def render(self, mode='human', close=False):
        # 在终端中显示环境的当前状态，可以根据需要进行自定义
        print("Current State:")
        print(self.state)

    def find_key_by_value(self, target_value):
        # 根据值找键
        for key, value in self.vocab.items():
            if value == target_value:
                return key
        # 如果找不到匹配的键，返回 None 或者其他你认为合适的值
        return None
    
    def Embedding_observation(self,observations):
        with torch.no_grad():
            # 转换成 tensor
            observations = torch.tensor(np.array(observations), dtype=torch.long)
            # 嵌入层  N x 32 -> N x 32 x 128
            observations = self.Embedding_Layer(observations).float()
            # Transformer Encoder 嵌入层  N x 32 x 128 -> N x 128
            observations = self.Transformer_model(observations).float()
        return observations

    def get_data(self, path):
        # 设置文件夹路径
        folder_path = path

        # 获取文件夹中的所有文件
        conector = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'conector')) if f.endswith('.cjson')]
        self.func_groups = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'func_groups')) if f.endswith('.cjson')]
        self.C2 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C2')) if f.endswith('.cjson')]
        self.C3 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C3')) if f.endswith('.cjson')]
        self.C4 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C4')) if f.endswith('.cjson')]
        self.C6 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C6')) if f.endswith('.cjson')]
        self.core = self.C2 + self.C3 + self.C4 + self.C6

        file_names = ['PAD','CLS'] + [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'all')) if f.endswith('.cjson')] + ['stop']

        # 创建词汇表
        vocab = {file_name: idx for idx, file_name in enumerate(file_names)} 
        #vocab['stop'] = -1
        print(vocab)

        # linker_mask
        linker_mask = [0] * len(file_names)
        for element in self.C2:
            index = vocab.get(element, None)
            if index is not None:
                linker_mask[index] = 1
        linker_mask[-1] = 1
        # core_mask
        core_mask = [0] * len(file_names)
        for element in self.core:
            index = vocab.get(element, None)
            if index is not None:
                core_mask[index] = 1                
        # conector_mask
        conector_mask = [0] * len(file_names)
        for element in conector:
            index = vocab.get(element, None)
            if index is not None:
                conector_mask[index] = 1
        # func_mask
        func_mask = [0] * len(file_names)
        for element in self.func_groups:
            index = vocab.get(element, None)
            if index is not None:
                func_mask[index] = 1
        func_mask[-1] = 1

        return vocab, linker_mask, core_mask, conector_mask, func_mask
