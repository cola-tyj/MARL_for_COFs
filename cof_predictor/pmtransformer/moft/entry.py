'''
 # @ author: zhangyi
 # @ date: 2023-09-26 19:01:09
 # @ desc: 获取REDD-COFFEE CGCNN嵌入结果
'''

import sys
# NOTE:部署修改
sys.path.append('/home/liuhaoyu/code/rnd_1/cof_predictor/pmtransformer/moft')

import json
import os
import time

# from moftransformer import predict
import predict
# from moftransformer.utils import prepare_data
from moftutils import prepare_data

# Get ckpt file
log_dir = './logs/'  # same directory make from training
seed = 0  # default seeds
version = 0  # version for model. It increases with the number of trains
mean = 0
std = 1
split = "train"  # 数据分在哪个数据集中

class PMTOperation:
    def __init__(self, originalPath, rootDir, ts):
        """
        构建文件结构

        Args:
            originalPath (str): 要预测的cif文件夹,例如/home/zhangyi/my_dataset
            rootDir (str): 临时文件夹
            ts (str): 本次预测的时间戳
        """
        # 构建文件结构
        self.ts = ts
        originalDataset = originalPath.split("/")[-1]
        self.dataset = f"{originalDataset}_{ts}"
        self.resultDir = rootDir
        os.makedirs(self.resultDir, exist_ok=True)
        if not os.path.exists(f"{self.resultDir}/{self.dataset}"):
            os.mkdir(f"{self.resultDir}/{self.dataset}")
            # 复制要预测的CIF
            os.system(f"cp -r {originalPath} {self.resultDir}/{self.dataset}")
            os.system(f"mv {self.resultDir}/{self.dataset}/{originalDataset} {self.resultDir}/{self.dataset}/raw")
            # 生成fake label
            self.generateRawJson(downstream="graph_embed")

    def processData(self, downstream="graph_embed"):
        """
        处理数据集CIF->graphdata, gridenergy...
        """
        root_cif = f"{self.resultDir}/{self.dataset}/raw"
        root_dataset = f"{self.resultDir}/{self.dataset}/dataset"
        prepare_data(root_cif, root_dataset, downstream=downstream)

    def predict(self, downstream="graph_embed"):
        """
        指定数据集和参数输出预测结果
        """
        # NOTE:部署修改
        load_path = "/home/liuhaoyu/code/rnd_1/cof_predictor/pmtransformer/params/pmtransformer.ckpt"
        root_dataset = f"{self.resultDir}/{self.dataset}/dataset"

        # 构造图嵌入结果保存目录
        graphEmbedSavePath = f"{os.path.dirname(self.resultDir)}/graphEmbed_{self.ts}"
        os.makedirs(graphEmbedSavePath, exist_ok=True)
        
        predict.predict(root_dataset, load_path=load_path, downstream=downstream, split=split, mean=mean, std=std, save_dir=f"{self.resultDir}/{self.dataset}", graphEmbedSavePath=graphEmbedSavePath)

    def doPredict(self, downstream="graph_embed"):
        # 执行预测并写入结果
        self.processData()
        self.predict()
        # 返回结果
        # print("result saved in:" + f"{self.resultDir}/{self.dataset}/{split}_{downstream}_prediction.csv")

    def generateRawJson(self, downstream="graph_embed"):
        """
        生成fake label
        """
        rawPath = f"{self.resultDir}/{self.dataset}/raw"
        jsonPath = f"{rawPath}/raw_{downstream}.json"
        if os.path.exists(jsonPath):
            return
        with open(jsonPath, mode="a") as f:
            dictt = dict()
            files = os.listdir(rawPath)
            for file in files:
                if file.endswith(".cif"):
                    dictt[file.split(".")[0]] = -1
            f.write(json.dumps(dictt))

def getGraphEmbed(cifPath, ts):
    parentPath = os.path.dirname(cifPath)
    # PMT临时目录
    tempPath = f"{parentPath}/temp_{ts}"
    pmt = PMTOperation(cifPath, rootDir=tempPath, ts=ts)
    pmt.doPredict()
    print("ge done")
    
# if __name__ == '__main__':
#     ts = str(int(time.time()))
#     getGraphEmbed("/home/zhangyi/dataset/cof_br/cif", ts)