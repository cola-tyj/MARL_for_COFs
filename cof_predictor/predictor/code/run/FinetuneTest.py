'''
 # @ author: zhangyi
 # @ date: 2023-11-30 10:37:33
 # @ desc: 测试微调结果
'''
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import CustomCommonDataset
from module import Model
from myutils import ConstructModelUtils, ProcessTestResult
from tqdm import tqdm

# 超参数
dim = 768
layerNum = 6
initLr = 1e-3
batchSize = 2
optimizerEps = 1e-3

# 微调部分超参数
hidDim = 768
finetuneLayerNum = 2

# 加载数据
testDataset = CustomCommonDataset.ReddCoffeeFromCgcnnWithTopologyAndVolumnN2Finetune0304(mode="test")
testDataLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False, drop_last=False, num_workers=1)
testBatchNum = testDataset.__len__() // batchSize + 1

# 构造模型
pretrainModel = Model.TrainingModel(dim=dim, layerNum=layerNum).cuda().half()
model = Model.FinetuneModel(inputDim=dim, hidDim=hidDim, layerNum=finetuneLayerNum, pretrainModel=pretrainModel).cuda().half()
model.eval()

def doPredict(comment: str) -> float:
    """
    执行测试,记录测试情况,读取并返回结果
    """
    logDir = f"/home/zhangyi/code/PyG/log/testFinetuneTask_{int(time.time())}_{comment}"
    os.makedirs(logDir, exist_ok=True)
    
    result = []
    for batchIdx, testBatch in enumerate(testDataLoader):
        x, y = testBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        labelDict = {
            "compound_type": y["compound_type"], 
            "cif_name": y["cif_name"],
            "n2aLabel": [y["n2_absorption"][0], y["n2_absorption"][1]]
        }
        predDict = {
            "n2aPred": model(
                graphEmbed=graphEmbed, 
                graphMask=mask, 
                cellVolumn=cellVolumn, 
                topoTypeIdx=topoTypeIdx
            )
        }
        result.append([predDict, labelDict])
        
    # 记录预测结果
    ConstructModelUtils.savePrediction(
        result, 
        taskList=["n2a"], 
        savePath=logDir
    )
    
    # 展示测试结果
    return ProcessTestResult.getFinetuneTestResult(logDir, comment=paramPath)
    
    
if __name__ == '__main__':
    # ! 指定加载模型参数
    paramPathList = [f"/home/zhangyi/code/PyG/log/[best]after_swap_train_weight_5_100_10/[0304_finetune_dataset]finetune_layer_2/finetune_dim_768_layerNum_6_lr_0.001_totalEpoch_7200_batchSize_1_weight_[5, 100, 10]_epoch_{i}.pkl" for i in range(3000, 7200)]
    
    # 记录最小mae
    minMaeEpoch = -1
    minMae = 1e8
    
    for paramPath in tqdm(paramPathList):
        # 加载模型参数
        param = torch.load(paramPath)
        model.load_state_dict(param["model"])
        model.eval()
        # 解析当前epoch
        currEpoch = paramPath.split("epoch_")[1].split(".")[0]
        # 执行测试
        currMae = doPredict(comment=currEpoch)
        if currMae < minMae:
            minMae = currMae
            minMaeEpoch = currEpoch
        time.sleep(1)
    
    # 最终结果
    print(f"minMae:{minMae}, epoch:{minMaeEpoch}")