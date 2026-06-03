'''
 # @ author: zhangyi
 # @ date: 2023-10-24 19:10:14
 # @ desc: 处理测试结果
'''

import csv

def getMAE(logPath: str, comment: str) -> float:
    """
    获取指定测试结果的MAE
    """
    cnt = 0
    diff = 0
    with open(logPath, mode="r") as f:
        f.readline()
        for line in f.readlines():
            split = line.split(",")
            cnt += 1
            diff += abs(float(split[-1]))
    mae = diff / cnt
    task = logPath.split("/")[-1]
    print(f"{task}-{comment}:{mae}")
    return mae

def getClassficationAccuracy(logPath: str, comment: str) -> float:
    """
    获取分类任务的准确率
    """
    cnt = 0
    trueCnt = 0
    with open(logPath, mode="r") as f:
        f.readline()
        for line in f.readlines():
            split = line.split(",")
            cnt += 1
            trueCnt += 1 if split[2] == split[3] else 0
    accuracy = trueCnt / cnt
    task = logPath.split("/")[-1]
    print(f"{task}-{comment}:{accuracy}")
    return accuracy

def getPretrainTestResult(path: str, comment: str):
    """
    获取预训练的测试结果
    """
    getClassficationAccuracy(f"{path}/ct.csv", comment)
    getMAE(f"{path}/vf.csv", comment)
    getMAE(f"{path}/lcd.csv", comment)

def getIntervalMAE(logPath: str, comment=None):
    """
    获取MAE(微调)
    如果预测值在误差区间则MAE=0,否则按区间较近的一侧计算
    """
    cnt = 0
    diff = 0
    with open(logPath, mode="r") as f:
        f.readline()
        for line in f.readlines():
            split = line.split(",")
            cnt += 1
            pred, label, eps = float(split[-4]), float(split[-3]), float(split[-2])
            l = label - eps
            r = label + eps
            if pred < l:
                diff += l - pred
            elif pred > r:
                diff += pred - r
    mae = diff / cnt
    task = logPath.split("/")[-1]
    print(f"{task}-{comment}:{mae}")
    return mae
    
def getFinetuneTestResult(path: str, comment=None) -> float:
    """
    获取微调结果的MAE
    """
    return getIntervalMAE(f"{path}/n2a.csv", comment)

def addAtomNumToTestResult(path:str) -> None:
    """
    改写测试结果文件,增加原子数统计
    """
    with open(path, 'r') as csv_file:  
        reader = csv.reader(csv_file) 
        rows = list(reader)
    baseDir = path.split(".")[0]
    cifDir = "/home/zhangyi/dataset/ReDD-COFFEE/data_cif"
    for i, row in enumerate(rows):
        if i ==  0:
            row.append("aton_num")
            continue
        ct = row[0]
        name = row[1]
        row.append(getAtomNumSingle(f"{cifDir}/{ct}/{name}.cif"))
    
    with open(f'{baseDir}_atom.csv', 'w', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerows(rows)

def getAtomNumSingle(inputFilePath: str) -> int:
    """
    获取单个cif文件的原子数
    """
    with open(inputFilePath, mode="r") as f:
        return len(f.readlines()) - 24
    
if __name__ == '__main__':
    addAtomNumToTestResult("/home/zhangyi/code/PyG/log/testFinetuneTask_0304_epoch_3213/n2a.csv")
    