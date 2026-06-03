'''
 # @ author: zhangyi
 # @ date: 2023-10-23 13:58:38
 # @ desc: 结合了TransformerEncoder+Heads的模型,用于训练过程
'''

import torch
import torch.nn as nn

from module import PreTrainingHeads, FinetuningHeads, TransformerEncoder


class TrainingModel(nn.Module):
    def __init__(self, dim, layerNum, vis=False):
        super().__init__()
        self.transformerEncoder = TransformerEncoder.TransformerEncoder(dim=dim, layerNum=layerNum, vis=vis)
        self.compoundTypeClassificationHead = PreTrainingHeads.CompoundTypeClassification(dim=dim)
        self.voidFractionRegressionHead = PreTrainingHeads.VoidFractionRegression(dim=dim)
        # self.channelNumRegressionHead = PreTrainingHeads.ChannelNumRegression(dim=dim)
        self.lcdRegressionHead = PreTrainingHeads.LcdRegression(dim=dim)
        
    def forward(self, graphEmbed, graphMask, cellVolumn, topoTypeIdx):
        embed = self.transformerEncoder(graphEmbed=graphEmbed, graphMask=graphMask, cellVolumn=cellVolumn, topoTypeIdx=topoTypeIdx)
        ctPred = self.compoundTypeClassificationHead(embed)
        vfPred = self.voidFractionRegressionHead(embed)
        lcdPred = self.lcdRegressionHead(embed)
        predDict = {
            "ctPred": ctPred,
            "vfPred": vfPred,
            "lcdPred": lcdPred
        }
        return predDict
    
class FinetuneModel(nn.Module):
    def __init__(self, inputDim, hidDim, layerNum, pretrainModel: nn.Module):
        super().__init__()
        self.pretrainModel = pretrainModel
        # 预训练参数梯度不更新
        for name, param in self.pretrainModel.named_parameters():
            param.requires_grad = False
        self.downstreamHead = FinetuningHeads.Bandgap(inputDim, hidDim, layerNum)
        
    def forward(self, graphEmbed, graphMask, cellVolumn, topoTypeIdx):
        embed = self.pretrainModel.transformerEncoder(
            graphEmbed=graphEmbed, 
            graphMask=graphMask, 
            cellVolumn=cellVolumn, 
            topoTypeIdx=topoTypeIdx
        )
        pred = self.downstreamHead(embed)
        return pred
    
class N2FinetuneModel(nn.Module):
    """
    氮气吸附预测用
    """
    def __init__(self, inputDim, hidDim, layerNum, pretrainModel: nn.Module):
        super().__init__()
        self.pretrainModel = pretrainModel
        # 预训练参数梯度不更新
        for name, param in self.pretrainModel.named_parameters():
            param.requires_grad = False
        self.n2AbsorptionHead = FinetuningHeads.Bandgap(inputDim, hidDim, layerNum)
    def forward(self, graphEmbed, graphMask, cellVolumn, topoTypeIdx):
        embed = self.pretrainModel.transformerEncoder(
            graphEmbed=graphEmbed, 
            graphMask=graphMask, 
            cellVolumn=cellVolumn, 
            topoTypeIdx=topoTypeIdx
        )
        pred = self.n2AbsorptionHead(embed)
        return pred