'''
 # @ author: zhangyi
 # @ date: 2023-11-06 14:01:31
 # @ desc: 微调任务头
'''

import torch
import torch.nn as nn

from myutils.ConstructModelUtils import initWeights

class N2Absorption(nn.Module):
    """
    预测氮气吸附的下游任务头(氧气预测共用)
    """
    def __init__(self, inputDim, hidDim, layerNum):
        super(N2Absorption, self).__init__()
        self.layers = nn.Sequential()
        if layerNum == 1:
            self.layers.append(nn.Linear(inputDim, 1))
        else:
            for i in range(layerNum):
                if i == layerNum - 1:
                    self.layers.append(nn.Linear(hidDim, 1))
                elif i == 1:
                    self.layers.append(nn.Linear(inputDim, hidDim))
                    self.layers.append(nn.GELU())
                else:
                    self.layers.append(nn.Linear(hidDim, hidDim))
                    self.layers.append(nn.GELU())
        self.apply(initWeights)
        
    def forward(self, x):
        return self.layers(x)

class Bandgap(nn.Module):
    """
    预测带隙属性的下游任务头(0703部署用于多拓扑氧气吸附预测)
    """
    def __init__(self, inputDim, hidDim, layerNum):
        super(Bandgap, self).__init__()
        self.layers = nn.Sequential()
        if layerNum == 1:
            self.layers.append(nn.Linear(inputDim, 1))
        else:
            for i in range(layerNum):
                if i == layerNum - 1:
                    self.layers.append(nn.Linear(hidDim, 1))
                elif i == 1:
                    self.layers.append(nn.Linear(inputDim, hidDim))
                    self.layers.append(nn.GELU())
                else:
                    self.layers.append(nn.Linear(hidDim, hidDim))
                    self.layers.append(nn.GELU())
        self.apply(initWeights)
        
    def forward(self, x):
        return self.layers(x)