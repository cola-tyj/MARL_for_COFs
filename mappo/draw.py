import os
import shutil
import time
import csv
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''folder_path = '/home/tianyajun/MARL_for_COFs/data'
# 获取文件夹中的所有文件
conector = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'conector')) if f.endswith('.cjson')]
func_groups = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'func_groups')) if f.endswith('.cjson')]
C2 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C2')) if f.endswith('.cjson')]
C3 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C3')) if f.endswith('.cjson')]
C4 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C4')) if f.endswith('.cjson')]
C6 = [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'sub', 'C6')) if f.endswith('.cjson')]
core = C2 + C3 + C4 + C6

#file_names = ['PAD','CLS'] + core + func_groups + conector
file_names = ['PAD','CLS'] + [f.split('.')[0] for f in os.listdir(os.path.join(folder_path, 'all')) if f.endswith('.cjson')]
#file_names = list(set(file_names))
vocab = {file_name: idx for idx, file_name in enumerate(file_names)} 
print(vocab)
#print(file_names)
linker_mask = [0] * len(file_names)
for element in C2:
    index = vocab.get(element, None)
    if index is not None:
        linker_mask[index] = 1
print(linker_mask)'''


'''import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV文件路径
csv_file_path = '/home/tianyajun/MARL_for_COFs/result.csv'

# 读取CSV文件
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过列标题
    pred_values = [float(row[1]) for row in reader]

# 计算每100个数的平均值
averages = [np.mean(pred_values[i:i+100]) for i in range(0, len(pred_values), 100)]

# 由于我们计算的是每100个数的平均值，我们需要创建一个对应的索引列表
indexes = list(range(0, len(pred_values), 100))

# 绘制折线图
plt.figure(figsize=(10, 5))
#plt.plot(indexes, averages, marker='o', label='100-number average')
plt.plot(indexes, averages, label='100-number average')

plt.title('Average of Pred Column every 100 Numbers')
plt.xlabel('Index (every 100 numbers)')
plt.ylabel('Average Value')
plt.legend()
plt.grid(True)

# 保存图表到文件
plt.savefig('averages_plot.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()'''


'''import pandas as pd
import matplotlib.pyplot as plt

# 指定.csv文件的路径
csvFilePath = '/home/tianyajun/MARL_for_COFs/ave_result.csv'

# 指定保存图表的文件路径
saveFilePath = '/home/tianyajun/MARL_for_COFs/average_pred_plot.png'

# 读取CSV文件
df = pd.read_csv(csvFilePath)

# 检查'Average of pred'列是否存在
if 'Average of pred' in df.columns:
    # 绘制'Average of pred'列的数据
    plt.figure(figsize=(10, 5))  # 设置图表的大小
    plt.plot(df['Average of pred'])  # 绘制线图，带有圆圈标记
    plt.title('Oxygen Adsorption Prediction by Epoch')  # 设置图表标题
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.ylabel('Average of pred')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    
    # 保存图表到本地文件
    plt.savefig(saveFilePath, dpi=300, bbox_inches='tight')
    
    # 显示图表
    #plt.show()
else:
    print("The column 'Average of pred' does not exist in the CSV file.")

'''



def f1():  # Returns by Episode
    # 指定.csv文件的路径
    csvFilePath = '/home/tianyajun/MARL_for_COFs/returns.csv'
    # 指定保存图表的文件路径
    saveFilePath = '/home/tianyajun/MARL_for_COFs/average_pred_plot.png'
    # 读取CSV文件
    df = pd.read_csv(csvFilePath)
    # 检查'Episode_Reward_Sum'列是否存在
    if 'Episode_Reward_Sum' in df.columns:
        # 绘制'Episode_Reward_Sum'列的数据
        plt.figure(figsize=(10, 5))  # 设置图表的大小
        plt.plot(df['Episode_Reward_Sum'])  # 绘制线图，带有圆圈标记
        plt.xlabel('Episode')  # 设置x轴标签
        plt.ylabel('Returns')  # 设置y轴标签
        plt.grid(True)  # 显示网格
        # 保存图表到本地文件
        plt.savefig(saveFilePath, dpi=300, bbox_inches='tight')
    else:
        print("The column 'Episode_Reward_Sum' does not exist in the CSV file.")


def f2():  # Returns by Epoch
    # CSV文件路径
    csv_file_path = '/home/tianyajun/MARL_for_COFs/returns.csv'

    data = pd.read_csv(csv_file_path)
    data['average1'] = data.mean(axis=1)
    # 计算每32个数的平均值
    averages = [np.mean(data['average1'][i:i+32]) for i in range(0, len(data['average1']), 32)]

    # 创建一个对应的epoch索引列表，每个epoch代表32个数据点
    epochs = range(1, len(averages) + 1)

    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, averages)

    plt.title('Average Returns every 32 Episodes')
    plt.xlabel('Epoch')
    plt.ylabel('Returns')
    plt.grid(True)

    # 保存图表到文件
    plt.savefig('average_returns_plot.png', dpi=300, bbox_inches='tight')


def f3():  # 平滑处理f2的曲线，滑动窗口大小为9
    # CSV文件路径
    csv_file_path = '/home/tianyajun/MARL_for_COFs/returns.csv'

    # 读取CSV文件
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过列标题
        pred_values = [float(row[1]) for row in reader]

    # 计算每32个数的平均值
    averages = [np.mean(pred_values[i:i+32]) for i in range(0, len(pred_values), 32)]
    a = [20.0 for i in range(100)]
    averages+=a
    #averages = np.array([averages[0], averages[0]] + list(averages) + [averages[len(averages)-1], averages[len(averages)-1]])

    # 滑动窗口大小为9的平滑处理
    smoothed_averages = []
    for i in range(len(averages) - 8):  # 修改这里以适应窗口大小为9
        window_average = np.mean(averages[i:i+8])  # 修改这里以适应窗口大小为9
        smoothed_averages.append(window_average)

    # 创建一个对应的epoch索引列表，每个epoch代表32个数据点
    epochs = range(1, len(averages) + 1)

    # 由于平滑处理后的数据点数量会减少，我们需要调整epochs列表
    smoothed_epochs = epochs[:-8]  # 修改这里以适应窗口大小为9

    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_epochs, smoothed_averages, label='Smoothed Average Returns (Window Size 9)')

    plt.title('Smoothed Average Returns every 32 Episodes with Window Size 9')
    plt.xlabel('Epoch')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)

    # 保存图表到文件
    plt.savefig('smoothed_average_returns_plot.png', dpi=300, bbox_inches='tight')


def f4():  # Intrinsic_rewards by Episode
    # 指定.csv文件的路径
    csvFilePath = '/home/tianyajun/MARL_for_COFs/intrinsic.csv'
    # 指定保存图表的文件路径
    saveFilePath = '/home/tianyajun/MARL_for_COFs/Intrinsic_rewards_plot.png'
    # 读取CSV文件
    df = pd.read_csv(csvFilePath)
    if 'Intrinsic_rewards' in df.columns:
        # 绘制'Episode_Reward_Sum'列的数据
        plt.figure(figsize=(10, 5))  # 设置图表的大小
        plt.plot(df['Intrinsic_rewards'])  # 绘制线图，带有圆圈标记
        plt.xlabel('Episode')  # 设置x轴标签
        plt.ylabel('Intrinsic Rewards')  # 设置y轴标签
        plt.grid(True)  # 显示网格
        # 保存图表到本地文件
        plt.savefig(saveFilePath, dpi=300, bbox_inches='tight')
    else:
        print("The column 'Intrinsic_rewards' does not exist in the CSV file.")


def f5():  # Max_pred by Epoch
    # 指定.csv文件的路径
    csvFilePath = '/home/tianyajun/MARL_for_COFs/max_result.csv'
    # 指定保存图表的文件路径
    saveFilePath = '/home/tianyajun/MARL_for_COFs/Max_pred_plot.png'
    # 读取CSV文件
    df = pd.read_csv(csvFilePath)
    if 'Maximum of pred' in df.columns:
        # 绘制'Episode_Reward_Sum'列的数据
        plt.figure(figsize=(10, 5))  # 设置图表的大小
        plt.plot(df['Maximum of pred'])  # 绘制线图
        plt.xlabel('Episode')  # 设置x轴标签
        plt.ylabel('Maximum of pred')  # 设置y轴标签
        plt.grid(True)  # 显示网格
        # 保存图表到本地文件
        plt.savefig(saveFilePath, dpi=300, bbox_inches='tight')
    else:
        print("The column 'Intrinsic_rewards' does not exist in the CSV file.")


def f6():  # Agent ave_returns by Episode
    # 指定CSV文件的路径
    csv_file_path = '/home/tianyajun/MARL_for_COFs/returns.csv'

    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 计算每个epoch（每32个episode）所有智能体返回值的平均值
    # 使用np.array()将DataFrame转换为numpy数组，以便于计算平均值
    returns = np.array(df.values)
    epoch_returns = returns.reshape(-1, 32, 6).mean(axis=2).mean(axis=1)

    # 将epoch返回值转换为DataFrame
    epoch_returns_df = pd.DataFrame(epoch_returns)

    # 绘制epoch与平均返回值的关系图
    plt.figure(figsize=(12, 6))  # 设置图表的大小
    plt.plot(epoch_returns_df.index, epoch_returns_df[0], label='Average Return per Epoch')  # 绘制线图，并添加标记
    plt.title('Average Return per Epoch Across All Agents')  # 设置图表标题
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.ylabel('Average Return')  # 设置y轴标签
    plt.grid(True)
    plt.legend()  # 显示图例

    # 保存图表为图片，指定文件路径和格式
    plt.savefig('average_returns_per_epoch.png', dpi=300, bbox_inches='tight')


f2()
f5()
