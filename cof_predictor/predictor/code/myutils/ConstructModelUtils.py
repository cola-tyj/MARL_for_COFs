'''
 # @ author: zhangyi
 # @ date: 2023-10-16 15:33:44
 # @ desc: 模型训练时的工具方法
 '''

import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from module import Loss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset


def initWeights(module) -> None:
    """
    初始化模块权重
    linear/embedding/ln
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
        
def setSeed(seed: int) -> None:
    """
    设置随机数种子
    """
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
        
def updateMinLossDict(epoch: int, currentDict: dict, minDict: dict) -> None:
    """
    更新训练时的dev loss dict,预训练和微调共用
    """
    for k in currentDict.keys():
        if currentDict[k] < minDict[k]:
            minDict[k] = currentDict[k]
            ki = k.replace("Loss", "Idx")
            minDict[ki] = epoch

def saveModel(
        epoch: int, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        savePath: str, 
        modelName: str, 
        devMinLossDict: dict, 
        batchStep: int
    ) -> None:
    """
    保存模型训练结果,预训练和微调以及baseline共用
    """
    checkpoint_dict = {
        'epoch': epoch,  # 当前epoch数
        'model': model.state_dict(),  # 模型参数
        'optim': optimizer.state_dict(),  # 优化器参数
        "devMinLossDict": devMinLossDict,  # 目前开发集上loss最小的情况
        "batchStep": batchStep  # 当前训练的batch数,用于记录tensorBoard
    }
    torch.save(checkpoint_dict, f"{savePath}/{modelName}_epoch_{epoch}.pkl")
    print(f"{modelName}_epoch_{epoch} saved in {savePath}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
def saveLoss(writer: SummaryWriter, kv: dict, step: int) -> None:
    """
    写入一系列的loss,预训练和微调共用
    """
    for key in kv.keys():
        writer.add_scalar(tag=key, scalar_value=kv[key], global_step=step)
        
def savePrediction(result: list, taskList: list, savePath:str) -> None:
    """
    保存预训练任务上的预测结果,预训练和微调共用

    Args:
        result(list[list[dict]]): 包含每个batch的预测结果
        taskList(list[str]): 任务类型前缀
        savePath (str): 保存路径
    """
    # 建立保存的路径和保存文件
    os.makedirs(f"{savePath}", exist_ok=True)
    logDict = dict()
    for task in taskList:
        logDict[task] = open(f"{savePath}/{task}.csv", mode="a")
        if task == "ct":
            # 分类任务
            logDict[task].write("compound_type,cifName,pred,label,softmax\n")
        elif task in ["vf", "cn", "lcd"]:
            # 预训练回归任务
            logDict[task].write("compound_type,cifName,pred,label,diff\n")
        else:
            # 微调回归任务
            logDict[task].write("compound_type,cifName,pred,label,eps,diff\n")
        
    for batch in result:
        predDict, labelDict = batch
        compoundType = labelDict["compound_type"]
        cifName = labelDict["cif_name"]
        for task in taskList:
            pred = predDict[f"{task}Pred"]
            label = labelDict[f"{task}Label"]
            batchNum = len(pred)
            if task == "ct":
                # 分类任务: label[B], pred[B, nClasses]
                for i in range(batchNum):
                    # 获取预测结果
                    softmaxList = pred[i].tolist()
                    logDict[task].write(f"{compoundType[i]},{cifName[i]},{softmaxList.index(max(softmaxList))},{label[i].item()},{softmaxList}\n")
            elif task in ["vf", "cn", "lcd"]:
                # 预训练回归任务: label[B], pred[B, 1]
                for i in range(batchNum):
                    predVal, labelVal = pred[i][0].item(), label[i].item()
                    diff = abs(predVal - labelVal)
                    logDict[task].write(f"{compoundType[i]},{cifName[i]},{predVal},{labelVal},{diff}\n")
            else:
                # 微调回归任务: label[B], pred[2, B]
                for i in range(batchNum):
                    predVal, labelVal, eps = pred[i].item(), label[0][i].item(), label[1][i].item()
                    lower, upper = labelVal - eps, labelVal + eps
                    diff = 0 if predVal > lower and predVal < upper else min(abs(predVal - lower), abs(predVal - upper))
                    logDict[task].write(f"{compoundType[i]},{cifName[i]},{predVal},{labelVal},{eps},{diff}\n")
        
    for k in logDict.keys():
        logDict[k].close()

def savePredictionWithoutLabel(result: list, task: str, savePath:str, ts: str) -> str:
    """
    保存部署预测结果

    Args:
        result(list[list[dict]]): 包含每个batch的预测结果
        task(str): 任务类型
        savePath (str): 保存路径
        ts (str): 本次预测的唯一标识
    """
    # 建立保存的路径和保存文件
    os.makedirs(f"{savePath}", exist_ok=True)
    logDict = dict()
    resultFile = open(f"{savePath}/{task}_{ts}.csv", mode="a")
    resultFile.write("cifName,pred\n")
        
    for batch in result:
        predDict, labelDict = batch
        cifName = labelDict["cif_name"]
        pred = predDict[f"{task}Pred"]
        batchNum = len(pred)
        # 微调回归任务: label[B], pred[2, B]
        for i in range(batchNum):
            predVal = pred[i].item()
            resultFile.write(f"{cifName[i]},{predVal}\n")
            
    resultFile.close()
    return f"{savePath}/{task}_{ts}.csv"
        
def checkPretrainDev(
        devDataset: Dataset, 
        devDataLoader: DataLoader, 
        devBatchNum: int, 
        model: nn.Module
    ) -> dict:
    """
    在开发集上获取预训练任务的loss
    """
    ctDevLoss = .0
    vfDevLoss = .0
    lcdDevLoss = .0
    for devBatch in devDataLoader:
        # 解析输入和label
        x, y = devBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        ctLabel, vfLabel, lcdLabel = y["compound_type_idx"].cuda(), y["void_fraction"].cuda(), y["lcd"].cuda()
        # 获取预测结果
        predDict = model(graphEmbed=graphEmbed, graphMask=mask, cellVolumn=cellVolumn, topoTypeIdx=topoTypeIdx)
        # 计算当前batch预训练任务的loss和总loss
        ctLoss = Loss.batchClassificationLoss(predDict['ctPred'], ctLabel)
        vfLoss = Loss.batchRegressionLoss(predDict['vfPred'], vfLabel)
        lcdLoss = Loss.batchRegressionLoss(predDict['lcdPred'], lcdLabel)
        totalLoss = Loss.getMultiTaskTotalLossModified([ctLoss, vfLoss, lcdLoss])
        ctDevLoss += ctLoss.item()
        vfDevLoss += vfLoss.item()
        lcdDevLoss += lcdLoss.item()
    result = {
        "ctDevLoss": ctDevLoss / devBatchNum,
        "vfDevLoss": vfDevLoss / devBatchNum,
        "lcdDevLoss": lcdDevLoss / devBatchNum
    }
    return result

def checkFinetuneDev(
        devDataset: Dataset, 
        devDataLoader: DataLoader, 
        devBatchNum: int, 
        model: nn.Module
    ) -> dict:
    """
    在开发集上获取微调任务的loss
    """
    loss = .0
    for devBatch in devDataLoader:
        # 解析输入和label
        x, y = devBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        label, eps = y["n2_absorption"][0].cuda(), y["n2_absorption"][1].cuda()
        n2AbsorptionPred = model(
            graphEmbed=graphEmbed, 
            graphMask=mask, 
            cellVolumn=cellVolumn, 
            topoTypeIdx=topoTypeIdx
        )
        n2AbsorptionLoss = Loss.batchRegressionIntervalLoss(n2AbsorptionPred, label=label, eps=eps)
        loss = loss + n2AbsorptionLoss.item()
    result = {
        "n2AbsorptionDevLoss": loss / devBatchNum
    }
    return result
    
def checkBaselineDev(
        devDataset: Dataset, 
        devDataLoader: DataLoader, 
        devBatchNum: int, 
        model: nn.Module
    ) -> dict:
    """
    在开发集上获取baseline的loss
    """
    loss = .0
    for devBatch in devDataLoader:
        # 解析输入和label
        x, y = devBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        label, eps = y["n2_absorption"][0].cuda(), y["n2_absorption"][1].cuda()
        n2AbsorptionPred = model(
            graphEmbed=graphEmbed, 
            graphMask=mask
        )
        n2AbsorptionLoss = Loss.batchRegressionIntervalLoss(n2AbsorptionPred, label=label, eps=eps)
        loss = loss + n2AbsorptionLoss.item()
    result = {
        "n2AbsorptionDevLoss": loss / devBatchNum
    }
    return result