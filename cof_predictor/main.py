'''
 # @ author: zhangyi
 # @ date: 2024-03-22 15:05:26
 # @ desc: 预测氧气吸附方法入口
'''

# NOTE:部署修改
import sys
sys.path.append("/home/liuhaoyu/code/rnd_1/cof_predictor")

import re
import os 
import csv
import time
import glob
import json
import math
import logging
import torch
import argparse

from torch.utils.data import DataLoader, Dataset
from pmtransformer.moft import entry
from predictor.code.dataset.PredictDataset import PredictDataset
from predictor.code.module import Model
from predictor.code.myutils import ConstructModelUtils

def doPredict(cifPath: str, logPath: str, ts: str):
    """
    指定CIF文件夹路径,执行氧气吸附/带隙属性预测,根据需要保留step4和step5
    要求文件夹和每个CIF文件除后缀(.cif)之外不包含符号"."和"/"

    Args:
        cifPath (str): 要预测的CIF文件的根目录
        logPath (str): 日志目录
        ts (str): 本次预测的唯一标识
    """
    # log配置
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{logPath}/predict_log_{ts}.log",
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parentPath = os.path.dirname(cifPath)
    
    # 1.获取图嵌入结果
    logging.debug(f"{ts}_step1: 获取图嵌入结果,保存路径{parentPath}/graphEmbed_{ts}")
    entry.getGraphEmbed(cifPath, ts)
    # 删除 bug cif
    cif_files = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(os.path.join(cifPath, '*.cif'))]
    graphEmbedPath = os.path.join(parentPath, "graphEmbed_"+ts)
    npy_files = [os.path.splitext(os.path.basename(file))[0].replace('_graph', '') for file in glob.glob(os.path.join(graphEmbedPath, '*.npy'))]
    files_to_delete = set(cif_files) - set(npy_files)
    for file_name in files_to_delete:  # 删除这些文件
        # 构造完整的.cif文件路径
        file_path = os.path.join(cifPath, file_name + '.cif')
        # 如果文件存在，则删除它
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found, skipped: {file_path}")
    
    # 2.计算晶胞体积并保存为json
    logging.debug(f"{ts}_step2: 计算晶胞体积并保存为json,保存路径{parentPath}/cellVolumn_{ts}.json")
    getCellVolumn(cifPath, ts)
    
    # 3.构建数据集
    logging.debug(f"{ts}_step3: 构建数据集")
    ds = PredictDataset(parentPath, ts)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 4.氧气吸附预测
    logging.debug(f"{ts}_step4: 执行氧气吸附预测并返回结果文件路径,保存路径{parentPath}/o2a_{ts}.csv")
    #return predictO2AdsorptionByModel(dl, savePath=parentPath, ts=ts)

    # 5.带隙预测
    logging.debug(f"{ts}_step5: 执行带隙预测并返回结果文件路径,保存路径{parentPath}/bandgap_{ts}.csv")
    #return predictBandgapByModel(dl, savePath=parentPath, ts=ts)

    # 6.氮气吸附预测
    logging.debug(f"{ts}_step6: 执行氮气吸附预测并返回结果文件路径,保存路径{parentPath}/n2a_{ts}.csv")
    return predictN2AdsorptionByModel(dl, savePath=parentPath, ts=ts)
    

def getCellVolumn(cifPath: str, ts: str):
    """
    指定cif文件的目录，计算所有晶胞体积并保存为json
    """
    # 所有cif文件目录
    cifPathList = glob.glob(os.path.join(cifPath, '*.cif'))
    
    # 正则表达式配置
    pattern = r'(_cell_length_a|_cell_length_b|_cell_length_c|_cell_angle_alpha|_cell_angle_beta|_cell_angle_gamma)\s+([0-9]+(\.[0-9]+)?)'
    
    # 匹配并计算晶胞体积
    cellVolumnDict = dict()
    for cif in cifPathList:
        cifName = cif.split("/")[-1].split(".")[0]
        with open(cif, mode="r") as f:
            text = f.read()
            matches = re.findall(pattern, text)
        cellValues = dict()
        for match in matches:
            key, value = match[0], match[1]
            cellValues[key] = float(value)
        cellVolumnDict[cifName] = calculateVolumn(cellValues)
    
    # 保存计算结果
    parentPath = os.path.dirname(cifPath)
    with open(f"{parentPath}/cellVolumn_{ts}.json", mode="a") as f:
        json.dump(cellVolumnDict, f, indent=4)

def calculateVolumn(cellParam: dict) -> float:
    """
    根据晶胞6参数计算体积
    """
    cAlpha = math.cos(math.radians(cellParam["_cell_angle_alpha"]))
    cBeta = math.cos(math.radians(cellParam["_cell_angle_beta"]))
    cGamma = math.cos(math.radians(cellParam["_cell_angle_gamma"]))
    volumn = cellParam["_cell_length_a"] * cellParam["_cell_length_b"] * cellParam["_cell_length_c"] * \
        math.sqrt(1 - math.pow(cAlpha, 2) - math.pow(cBeta, 2) - math.pow(cGamma, 2) + \
            abs(2 * cAlpha * cBeta * cGamma))
    return volumn

def predictO2AdsorptionByModel(dl: DataLoader, savePath: str, ts: str) -> str:
    """构建模型"""
    # 超参数
    dim = 768
    layerNum = 6
    initLr = 1e-3
    batchSize = 2
    optimizerEps = 1e-3
    # 微调部分超参数
    hidDim = 768
    finetuneLayerNum = 3
    # 构造模型并加载预训练数据
    pretrainModel = Model.TrainingModel(dim=dim, layerNum=layerNum).cuda().half()
    model = Model.FinetuneModel(inputDim=dim, hidDim=hidDim, layerNum=finetuneLayerNum, pretrainModel=pretrainModel).cuda().half()
    # 加载微调数据
    # NOTE:部署修改
    paramPath = "/home/liuhaoyu/code/rnd_1/cof_predictor/predictor/params/finetune_dim_768_layerNum_6_lr_0.001_totalEpoch_7200_batchSize_1_weight_[5, 100, 10]_epoch_4822.pkl"
    param = torch.load(paramPath)
    model.load_state_dict(param["model"])
    model.eval()
    
    """执行预测"""
    result = []
    for batchIdx, testBatch in enumerate(dl):
        x, y = testBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        labelDict = {
            "cif_name": y["cif_name"]
        }
        predDict = {
            "o2aPred": model(
                graphEmbed=graphEmbed, 
                graphMask=mask, 
                cellVolumn=cellVolumn, 
                topoTypeIdx=topoTypeIdx
            )
        }
        result.append([predDict, labelDict])
        
    # 记录并返回预测结果
    return ConstructModelUtils.savePredictionWithoutLabel(result, task="o2a", savePath=savePath, ts=ts)

def predictBandgapByModel(dl: DataLoader, savePath: str, ts: str) -> str:
    """构建模型"""
    # 超参数
    dim = 768
    layerNum = 6
    initLr = 1e-3
    batchSize = 2
    optimizerEps = 1e-3
    # 微调部分超参数
    hidDim = 768
    finetuneLayerNum = 3
    # 构造模型并加载预训练数据
    pretrainModel = Model.TrainingModel(dim=dim, layerNum=layerNum).cuda().half()
    model = Model.FinetuneModel(inputDim=dim, hidDim=hidDim, layerNum=finetuneLayerNum, pretrainModel=pretrainModel).cuda().half()
    # 加载微调数据
    # NOTE:部署修改
    paramPath = "/home/liuhaoyu/code/rnd_1/cof_predictor/predictor/params/bandgap_finetune_dim_768_layerNum_6_lr_0.001_totalEpoch_7200_batchSize_1_weight_[5, 100, 10]_finetuneLayerNum3_epoch_5507.pkl"
    param = torch.load(paramPath)
    model.load_state_dict(param["model"])
    model.eval()
    
    """执行预测"""
    result = []
    for batchIdx, testBatch in enumerate(dl):
        x, y = testBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        labelDict = {
            "cif_name": y["cif_name"]
        }
        predDict = {
            "bandgapPred": model(
                graphEmbed=graphEmbed, 
                graphMask=mask, 
                cellVolumn=cellVolumn, 
                topoTypeIdx=topoTypeIdx
            )
        }
        result.append([predDict, labelDict])
        
    # 记录并返回预测结果
    return ConstructModelUtils.savePredictionWithoutLabel(result, task="bandgap", savePath=savePath, ts=ts)

def predictN2AdsorptionByModel(dl: DataLoader, savePath: str, ts: str) -> str:
    """构建模型"""
    # 超参数
    dim = 768
    layerNum = 6
    initLr = 1e-3
    batchSize = 2
    optimizerEps = 1e-3
    # 微调部分超参数
    hidDim = 768
    finetuneLayerNum = 2
    # 构造模型并加载预训练数据
    pretrainModel = Model.TrainingModel(dim=dim, layerNum=layerNum).cuda().half()
    model = Model.N2FinetuneModel(inputDim=dim, hidDim=hidDim, layerNum=finetuneLayerNum, pretrainModel=pretrainModel).cuda().half()
    # 加载微调数据
    # NOTE:部署修改
    paramPath = "/home/liuhaoyu/code/rnd_1/cof_predictor/predictor/params/n2_finetune_dim_768_layerNum_6_lr_0.001_totalEpoch_7200_batchSize_1_weight_[5, 100, 10]_epoch_4300.pkl"
    param = torch.load(paramPath)
    model.load_state_dict(param["model"])
    model.eval()
    """执行预测"""
    result = []
    for batchIdx, testBatch in enumerate(dl):
        x, y = testBatch
        graphEmbed, mask = x["graph_embed"].cuda(), x["graph_mask"].cuda()
        topoTypeIdx, cellVolumn = y["topo_type_idx"].cuda(), torch.tensor(y["cell_volumn"], dtype=torch.float16).cuda()
        labelDict = {
            "cif_name": y["cif_name"]
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
    # 记录并返回预测结果
    return ConstructModelUtils.savePredictionWithoutLabel(result, task="n2a", savePath=savePath, ts=ts)

# 使用示例
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='O2吸附预测')

    # parser.add_argument('cifPath', type=str, help='cif文件夹目录')
    # parser.add_argument('logPath', type=str, help='日志保存目录')
    # parser.add_argument('ts', type=int, help='预测唯一标识')
    # args = parser.parse_args()

    # resultFilePath = doPredict(args.cifPath, logPath=args.logPath, ts=args.ts)
    # print(resultFilePath)
    resultFilePath = doPredict(cifPath="/home/liuhaoyu/code/rnd_1/cofs/cif", logPath="/home/liuhaoyu/code/rnd_1/logs", ts="0716")
    print(resultFilePath)