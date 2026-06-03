'''
 # @ author: zhangyi
 # @ date: 2024-03-28 09:25:50
 # @ desc: 部署用预测数据集
'''

import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class PredictDataset(Dataset):
    def __init__(self, parentPath: str, ts: str):
        super(PredictDataset, self).__init__()
        self.ts = ts
        # 数据集根目录
        self.parentPath = parentPath
        # 晶胞体积
        self.cellVolumnDict = json.load(open(f"{parentPath}/cellVolumn_{self.ts}.json", mode="r"))
        # 数据ID列表
        self.cifNameList = sorted(self.cellVolumnDict.keys())
        # 拓扑类型
        self.topologyDict = json.load(open(f"{parentPath}/topology.json", mode="r"))
        # NOTE:部署修改
        self.topology2IndexDict = json.load(open("/home/tianyajun/MARL_for_COFs/cof_predictor/predictor/code/dataset/topology2D.json", mode="r"))
        
    def __len__(self) -> int:
        return len(self.cifNameList)
    
    def __getitem__(self, idx) -> (dict, dict):
        cifName = self.cifNameList[idx]
        x = {
            "graph_embed": np.load(f"{self.parentPath}/graphEmbed_{self.ts}/{cifName}_graph.npy"), 
            "graph_mask": np.load(f"{self.parentPath}/graphEmbed_{self.ts}/{cifName}_mask.npy")
        }
        # 体积归一化
        maxV = 1864581
        minV = 1233
        normalizedV = (self.cellVolumnDict[cifName] - minV) / (maxV - minV)
        y = {
            "cif_name": cifName,
            "topo_type_idx": self.topology2IndexDict[self.topologyDict[cifName]] if self.topologyDict[cifName] in self.topologyDict else 95,
            "cell_volumn": normalizedV
        }
        return (x, y)