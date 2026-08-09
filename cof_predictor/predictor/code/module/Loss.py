'''
 # @ author: zhangyi
 # @ date: 2023-10-13 16:33:28
 # @ desc: 损失函数
'''

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def distillationLoss(outputEmbed: Tensor, targetEmbed: Tensor) -> torch.float16:
    """
    CGCNN->PointTransformer蒸馏损失函数,计算两组向量的欧氏距离的平均值

    Args:
        outputEmbed (Tensor fp16): student输出结果(300, 768)
        targetEmbed (Tensor fp16): teacher输出结果(300, 768)

    Returns:
        torch.float16: loss
    """
    # 最多独特原子数
    maxGraphLen = 300
    # 实际原子数
    actualAtomNum = maxGraphLen
    # 两组向量的欧氏距离和
    l2NormSum = torch.tensor(.0)
    
    for i in range(maxGraphLen):
        if np.all(outputEmbed[i].numpy() < 1e-6) and np.all(outputEmbed[i].numpy() > -1e-6):
            actualAtomNum = i
            break
        else:
            l2NormSum = l2NormSum + torch.sum(torch.sqrt(torch.square(outputEmbed[i].float() - targetEmbed[i].float())))
    return (l2NormSum / actualAtomNum).type(torch.float16)

# 分类任务选择损失函数
ceLoss = nn.CrossEntropyLoss(reduction="mean")

def batchClassificationLoss(output: Tensor, label: Tensor) -> torch.float16:
    """
    使用pytorch多分类损失函数计算loss

    Args:
        output (Tensor[B, nClasses]): softmax后的输出结果
        label (Tensor[B]): label
    """
    loss = ceLoss(input=output, target=label)
    return loss

# 回归任务选择损失函数
huberLoss = nn.SmoothL1Loss(reduction="mean")

def batchRegressionLoss(output: Tensor, label: list) -> torch.float16:
    """
    预训练任务回归任务损失函数

    Args:
        output (Tensor[B, 1]): 预测结果
        label (list[B]): label
    """
    batchSize = len(label)
    label = torch.tensor(label, dtype=torch.float16).to(output)
    output = output.squeeze(1)
    loss = huberLoss(output, label)
    return loss

def getMultiTaskTotalLossInit(lossList: list, weightList: list=None) -> torch.float16:
    """
    获取多任务训练时的总loss,第一种方法
    将每个任务的loss的梯度都归到同一个维度
    https://www.zhihu.com/question/375794498/answer/2292320194

    Args:
        lossList (list): loss列表
        weightList (list): 每个loss对应的权重,默认None
    Returns:
        torch.float16: 多任务训练时的总loss,数值无意义,梯度用于优化
    """
    eps = 1e-6
    totalLoss = torch.tensor(0, dtype=torch.float16).to(lossList[0])
    weightList = [1 for _ in range(len(lossList))] if weightList is None else weightList
    for i, loss in enumerate(lossList):
        totalLoss = totalLoss + weightList[i] * loss / (loss.detach() + eps)
    return totalLoss

def getMultiTaskTotalLossModified(lossList: list, weightList: list=None) -> torch.float16:
    """
    获取多任务训练时的总loss,改进方法
    将每个任务的loss的梯度都归到同一个维度,可选为每个任务赋权
    https://www.zhihu.com/question/375794498/answer/2292320194

    Args:
        lossList (list): loss列表
        weightList (list): 每个loss对应的权重,默认None

    Returns:
        torch.float16: 多任务训练时的总loss,数值无意义,梯度用于优化
    """
    taskNum = len(lossList)
    eps = 1e-6
    if not weightList:
        weightList = [1.0 for _ in range(taskNum)]
    loss0 = lossList[0]
    totalLoss = torch.tensor(lossList[0] * weightList[0], dtype=torch.float16).to(loss0)
    
    for i in range(1, taskNum):
        currLoss = lossList[i]
        totalLoss = totalLoss + weightList[i] * currLoss / ((currLoss / (loss0 + eps)).detach() + eps)
        
    return totalLoss

def getMultiTaskTotalLossSimple(lossList: list, weightList: list=None) -> torch.float16:
    """
    直接将多任务的loss加权求和

    Args:
        lossList (list): loss列表
        weightList (list): 每个loss对应的权重,总和为1,默认None

    Returns:
        torch.float16: 多任务训练时的总loss
    """
    totalLoss = torch.tensor(0, dtype=torch.float16).to(lossList[0])
    for i, loss in enumerate(lossList):
        totalLoss = totalLoss + weightList[i] * loss
    return totalLoss

def batchRegressionIntervalLoss(output: Tensor, label: list, eps: list) -> torch.float16:
    """
    微调任务回归损失函数
    气体吸附属性带有误差区间

    Args:
        output (Tensor[B, 1]): 预测结果
        label (list[B]): 标注
        eps (list[B]): 标注误差范围

    Returns:
        torch.float16: loss
    """
    label = torch.tensor(label).to(output)
    batchSize = len(label)
    totalLoss = torch.tensor(.0).to(output)
    for i in range(batchSize):
        l, r = label[i] - eps[i], label[i] + eps[i]
        if output[i][0].item() < l:
            totalLoss = totalLoss + huberLoss(output[i][0], r)
        elif output[i][0].item() > r:
            totalLoss = totalLoss + huberLoss(output[i][0], l)
    return totalLoss