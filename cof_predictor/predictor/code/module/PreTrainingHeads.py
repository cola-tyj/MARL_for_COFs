'''
 # @ author: zhangyi
 # @ date: 2023-10-17 11:21:36
 # @ desc: 预训练任务头,包含化合物类型分类、孔隙率预测、通道数预测、最大孔直径预测
 '''
 
import torch
import torch.nn as nn

from myutils.ConstructModelUtils import initWeights


class CompoundTypeClassification(nn.Module):
    def __init__(self, dim, nClasses=11):
        super(CompoundTypeClassification, self).__init__()
        self.fc = nn.Linear(dim, nClasses)
        self.softmax = nn.Softmax()
        self.apply(initWeights)
    
    def forward(self, x):
        x = self.fc(x)  # [B, dim]->[B, nClasses]
        return self.softmax(x)
        
class VoidFractionRegression(nn.Module):
    def __init__(self, dim):
        super(VoidFractionRegression, self).__init__()
        self.fc = nn.Linear(dim, 1)
        self.apply(initWeights)
    
    def forward(self, x):
        return self.fc(x)
        
class ChannelNumRegression(nn.Module):
    def __init__(self, dim):
        super(ChannelNumRegression, self).__init__()
        self.fc = nn.Linear(dim, 1)
        self.apply(initWeights)
    
    def forward(self, x):
        return self.fc(x)
        
class LcdRegression(nn.Module):
    def __init__(self, dim):
        super(LcdRegression, self).__init__()
        self.fc = nn.Linear(dim, 1)
        self.apply(initWeights)
    
    def forward(self, x):
        return self.fc(x)