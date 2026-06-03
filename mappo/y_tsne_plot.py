import csv
import re
import torch
from rdkit.DataStructs import ConvertToNumpyArray
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.manifold import TSNE
from combine_substructures import combined
from transformer import Embedding_Layer
from transformer import TransformerEncoder
from train import Decoder

def get_data(file_path,max_count=100):
    Embedding = Embedding_Layer(dim=128,pad=32)
    Embedding.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/embedding_parameters.pth'))
    Transformer_model = TransformerEncoder(dim=128)
    Transformer_model.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/transformer_encoder.pth'))
    pad = 32
    vocab = {'PAD': 0, 'CLS': 1, 'F': 2, 'CN': 3, 'SO2H': 4, '307': 5, 'NH2': 6, 'NO2': 7, 'Cl': 8, 'HECO': 9, 'BENZ6': 10, 
             'OCOCH3': 11, 'OH': 12, '309': 13, 'CHO': 14, 'CH3': 15, 'PTCA': 16, 'L2':17, '205': 18, '308': 19, '210': 20, 
             'CH2CN': 21, 'Br': 22, '209': 23, '208': 24, '305': 25, '301': 26, 'H': 27, 'NHOH': 28, 'T3':29,'207': 30, 
             'S4':31, '302': 32, 'OHc': 33, '206': 34, 'OMe': 35, 'COOH': 36, 'PORP': 37, '310': 38, '202': 39, 'H6':40,
             'CHS': 41, 'EPO': 42, '306': 43, 'I': 44, '203': 45, 'SH': 46, '204': 47, '304': 48, 'NO': 49, 
             '201': 50, '303': 51}
    count = 0
    #max_count = 1000  # 你想要读取的最大行数
    hcb = []
    sql = []
    kgd = []
    hxl = []
    # 打开文件
    with open(file_path, newline='') as csvfile:
        # 创建CSV读取器
        csv_reader = csv.reader(csvfile)
        # 读取第一列
        for row in csv_reader:
            if count >= max_count:
                break  # 如果已经读取了足够的行数，就停止读取
            if 'SQL' in row[0]:
                sql.append(row[0])
            elif 'HCB' in row[0]:
                hcb.append(row[0])
            elif 'KGD' in row[0]:
                kgd.append(row[0])
            elif 'HXL' in row[0]:
                hxl.append(row[0])
            count += 1
    
    inputs = hcb + sql + kgd + hxl
    new_states = []
    # 分解输入字符串为部分
    for input in inputs:
        input = input.split('-')[0:-1]
        input = '-'.join(input)
        input_parts = re.split(r'[-_]', input)[0:-1]
        input_parts = [item for item in input_parts if item != 'CLS']
        input_parts = ['CLS'] + input_parts

        # 将输入字符串的每个部分映射到索引
        input_indices = [vocab[part] for part in input_parts]
        # 计算需要补齐的数量
        padding_size = pad - len(input_indices)
        # 补齐
        new_state = np.pad(input_indices, (0, padding_size), constant_values=0)
        new_states.append(new_state)

    with torch.no_grad():
        # 转换成 tensor
        observations = torch.tensor(np.array(new_states), dtype=torch.long)
        # 嵌入层  N x 32 -> N x 32 x 128
        observations = Embedding(observations).float()
        # Transformer Encoder 嵌入层  N x 32 x 128 -> N x 128
        observations = Transformer_model(observations).float()

    print(observations.size(),len(hcb),len(sql),len(kgd),len(hxl))
    #torch.save(Embedding.state_dict(), 'embedding_parameters.pth')
    #torch.save(Transformer_model.state_dict(), 'transformer_encoder.pth')
    return observations.numpy(),[len(hcb),len(sql),len(kgd),len(hxl)]

def get_data_rand(file_path,max_count=100):
    Embedding = Embedding_Layer(dim=128,pad=32)
    Embedding.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/embedding_parameters.pth'))
    Transformer_model = TransformerEncoder(dim=128)
    Transformer_model.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/transformer_encoder.pth'))
    decoder = Decoder(128, 32)
    decoder.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/mappo/model/decoder.pth'))

    pad = 32
    vocab = {'PAD': 0, 'CLS': 1, 'F': 2, 'CN': 3, 'SO2H': 4, '307': 5, 'NH2': 6, 'NO2': 7, 'Cl': 8, 'HECO': 9, 'BENZ6': 10, 
             'OCOCH3': 11, 'OH': 12, '309': 13, 'CHO': 14, 'CH3': 15, 'PTCA': 16, 'L2':17, '205': 18, '308': 19, '210': 20, 
             'CH2CN': 21, 'Br': 22, '209': 23, '208': 24, '305': 25, '301': 26, 'H': 27, 'NHOH': 28, 'T3':29,'207': 30, 
             'S4':31, '302': 32, 'OHc': 33, '206': 34, 'OMe': 35, 'COOH': 36, 'PORP': 37, '310': 38, '202': 39, 'H6':40,
             'CHS': 41, 'EPO': 42, '306': 43, 'I': 44, '203': 45, 'SH': 46, '204': 47, '304': 48, 'NO': 49, 
             '201': 50, '303': 51}
    
    vocab_weights = {'PAD': 0, 'CLS': 1, 'F': 1, 'CN': 1, 'SO2H': 1, '307': 5, 'NH2': 1, 'NO2': 1, 'Cl': 1, 'HECO': 3, 'BENZ6': 3, 
             'OCOCH3': 1, 'OH': 11, '309': 3, 'CHO': 1, 'CH3': 1, 'PTCA': 3, 'L2':3, '205': 3, '308': 3, '210': 3, 
             'CH2CN': 1, 'Br': 1, '209': 3, '208': 3, '305': 3, '301': 3, 'H': 1, 'NHOH': 1, 'T3': 5,'207': 3, 
             'S4': 5, '302': 3, 'OHc': 1, '206': 3, 'OMe': 1, 'COOH': 1, 'PORP': 3, '310': 3, '202': 3, 'H6':5,
             'CHS': 1, 'EPO': 1, '306': 3, 'I': 1, '203': 3, 'SH': 1, '204': 3, '304': 3, 'NO': 1, 
             '201': 3, '303': 3}
    count = 0
    sizes = []
    inputs = []

    # 打开文件
    with open(file_path, newline='') as csvfile:
        # 创建CSV读取器
        csv_reader = csv.reader(csvfile)
        # 读取第一列
        for row in csv_reader:
            if count >= max_count:
                break  # 如果已经读取了足够的行数，就停止读取
            else:
                count += 1
                inputs.append(row[0])
    sizes.append(count)

    new_states = []
    weights = []
    # 分解输入字符串为部分
    for input in inputs:
        input = input.split('-')[0:-1]
        input = '-'.join(input)
        input_parts = re.split(r'[-_]', input)[0:-1]
        input_parts = [item for item in input_parts if item != 'CLS']
        input_parts = ['CLS'] + input_parts

        # 将输入字符串的每个部分映射到索引
        input_indices = [vocab[part] for part in input_parts]
        weight = [vocab_weights[part] for part in input_parts]
        # 计算需要补齐的数量
        padding_size = pad - len(input_indices)
        # 补齐
        new_state = np.pad(input_indices, (0, padding_size), constant_values=0)
        new_states.append(new_state)
        weights.append(np.pad(weight, (0, padding_size), constant_values=1))

    with torch.no_grad():
        # 转换成 tensor
        observations = torch.tensor(np.array(new_states), dtype=torch.long)
        weights_tensor = torch.tensor(np.array(weights), dtype=torch.float)
        # 嵌入层  N x 32 -> N x 32 x 128
        observations = Embedding(observations).float()
        # 加权   N x 32 x 128  *  N x 32
        result = torch.zeros_like(observations)
        for i in range(observations.shape[0]):  # 遍历第1个维度
            for j in range(observations.shape[1]):  # 遍历第2个维度
                # 将B中的元素与A中对应的行相乘
                result[i, j, :] = observations[i, j, :] * weights_tensor[i, j]
        observations = result
        # Transformer Encoder 嵌入层  N x 32 x 128 -> N x 128
        observations = Transformer_model(observations).float()
        # Decoder  N x 128 -> N x 32
        #observations = decoder(observations).float()
    print(observations.size(),sizes)
    return observations.numpy(), sizes

def get_data_rnd(max_count=100):
    Embedding = Embedding_Layer(dim=128,pad=32)
    Embedding.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/parameters_embedding.pth'))
    Transformer_model = TransformerEncoder(dim=128)
    Transformer_model.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/parameters_transformer.pth'))
    decoder = Decoder(128, 32)
    decoder.load_state_dict(torch.load('/home/tianyajun/MARL_for_COFs/mappo/model/decoder.pth'))

    pad = 32
    vocab = {'PAD': 0, 'CLS': 1, 'F': 2, 'CN': 3, 'SO2H': 4, '307': 5, 'NH2': 6, 'NO2': 7, 'Cl': 8, 'HECO': 9, 'BENZ6': 10, 
             'OCOCH3': 11, 'OH': 12, '309': 13, 'CHO': 14, 'CH3': 15, 'PTCA': 16, 'L2':17, '205': 18, '308': 19, '210': 20, 
             'CH2CN': 21, 'Br': 22, '209': 23, '208': 24, '305': 25, '301': 26, 'H': 27, 'NHOH': 28, 'T3':29,'207': 30, 
             'S4':31, '302': 32, 'OHc': 33, '206': 34, 'OMe': 35, 'COOH': 36, 'PORP': 37, '310': 38, '202': 39, 'H6':40,
             'CHS': 41, 'EPO': 42, '306': 43, 'I': 44, '203': 45, 'SH': 46, '204': 47, '304': 48, 'NO': 49, 
             '201': 50, '303': 51}
    
    vocab_weights = {'PAD': 0, 'CLS': 1, 'F': 1, 'CN': 1, 'SO2H': 1, '307': 5, 'NH2': 1, 'NO2': 1, 'Cl': 1, 'HECO': 3, 'BENZ6': 3, 
             'OCOCH3': 1, 'OH': 11, '309': 3, 'CHO': 1, 'CH3': 1, 'PTCA': 3, 'L2':3, '205': 3, '308': 3, '210': 3, 
             'CH2CN': 1, 'Br': 1, '209': 3, '208': 3, '305': 3, '301': 3, 'H': 1, 'NHOH': 1, 'T3': 5,'207': 3, 
             'S4': 5, '302': 3, 'OHc': 1, '206': 3, 'OMe': 1, 'COOH': 1, 'PORP': 3, '310': 3, '202': 3, 'H6':5,
             'CHS': 1, 'EPO': 1, '306': 3, 'I': 1, '203': 3, 'SH': 1, '204': 3, '304': 3, 'NO': 1, 
             '201': 3, '303': 3}
    count = 0
    sizes = []
    inputs = []
    file_path_rand = '/home/tianyajun/MARL_for_COFs/mappo/data_train/cofs.csv'
    file_path_rnd = '/home/tianyajun/MARL_for_COFs/mappo/data_train/rnd.csv'
    file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/no_rnd.csv'

    # 打开文件
    with open(file_path_rand, newline='') as csvfile:
        # 创建CSV读取器
        csv_reader = csv.reader(csvfile)
        # 读取第一列
        for row in csv_reader:
            if count >= 25000:
                break  # 如果已经读取了足够的行数，就停止读取
            else:
                count += 1
                inputs.append(row[0])
    sizes.append(count)
    '''count = 0
    with open(file_path_rnd, newline='') as csvfile:
        # 创建CSV读取器
        csv_reader = csv.reader(csvfile)
        # 读取第一列
        for row in csv_reader:
            if count >= max_count:
                break  # 如果已经读取了足够的行数，就停止读取
            else:
                count += 1
                inputs.append(row[0])
    sizes.append(count)
    count = 0
    with open(file_path, newline='') as csvfile:
        # 创建CSV读取器
        csv_reader = csv.reader(csvfile)
        # 读取第一列
        for row in csv_reader:
            if count >= max_count:
                break  # 如果已经读取了足够的行数，就停止读取
            else:
                count += 1
                inputs.append(row[0])
    sizes.append(count)'''

    new_states = []
    weights = []
    # 分解输入字符串为部分
    for input in inputs:
        input = input.split('-')[0:-1]
        input = '-'.join(input)
        input_parts = re.split(r'[-_]', input)[0:-1]
        input_parts = [item for item in input_parts if item != 'CLS']
        input_parts = ['CLS'] + input_parts

        # 将输入字符串的每个部分映射到索引
        input_indices = [vocab[part] for part in input_parts]
        weight = [vocab_weights[part] for part in input_parts]
        # 计算需要补齐的数量
        padding_size = pad - len(input_indices)
        # 补齐
        new_state = np.pad(input_indices, (0, padding_size), constant_values=0)
        new_states.append(new_state)
        weights.append(np.pad(weight, (0, padding_size), constant_values=1))

    with torch.no_grad():
        # 转换成 tensor
        observations = torch.tensor(np.array(new_states), dtype=torch.long)
        weights_tensor = torch.tensor(np.array(weights), dtype=torch.float)
        # 嵌入层  N x 32 -> N x 32 x 128
        observations = Embedding(observations).float()
        # 加权   N x 32 x 128  *  N x 32
        result = torch.zeros_like(observations)
        for i in range(observations.shape[0]):  # 遍历第1个维度
            for j in range(observations.shape[1]):  # 遍历第2个维度
                # 将B中的元素与A中对应的行相乘
                result[i, j, :] = observations[i, j, :] * weights_tensor[i, j]
        observations = result
        # Transformer Encoder 嵌入层  N x 32 x 128 -> N x 128
        observations = Transformer_model(observations).float()
        # Decoder  N x 128 -> N x 32
        #observations = decoder(observations).float()
    print(observations.size(),sizes)
    return observations.numpy(), sizes

def tsne_plot(fps_np, sizes, labels):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    fps_2d = tsne.fit_transform(fps_np)

    # 绘制t-SNE图
    #plt.figure(figsize=(12, 8))
    
    start_index = 0

    colors = [plt.cm.Greys(0.5),plt.cm.jet(0.0),plt.cm.jet(0.25),plt.cm.jet(0.5),plt.cm.jet(0.75)]

    for i, size in enumerate(sizes):
        end_index = start_index + size
        # 获取批次数据
        batch_data = fps_np[start_index:end_index]
        # 绘制批次数据
        '''plt.scatter(fps_2d[start_index:end_index, 0], fps_2d[start_index:end_index, 1], 
                    alpha=0.5, label=labels[i], 
                    color=plt.cm.jet(i / (len(sizes)-1)))  # 使用颜色映射'''
        plt.scatter(fps_2d[start_index:end_index, 0], fps_2d[start_index:end_index, 1], 
                    alpha=0.5, label=labels[i], 
                    color=colors[i])  # 使用颜色映射
        start_index = end_index

    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    #plt.title('t-SNE Plot of Different Topological Networks')
    #plt.title('with rnd reward')
    plt.title('top 2.5%')
    plt.legend()
    #plt.show()
    plt.savefig('tsne_trend.png', bbox_inches='tight')

def tsne_plot1(fps_np, sizes, labels):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    fps_2d = tsne.fit_transform(fps_np)

    sizes = [22477,1024, 45, 75, 70]
    fps_2d = np.concatenate((fps_2d[:20505], fps_2d[-1972:], fps_2d[20505:-1972]), axis=0)


    # 绘制t-SNE图
    #plt.figure(figsize=(12, 8))
    
    start_index = 0

    colors = [plt.cm.Greys(0.5),plt.cm.jet(0.0),plt.cm.jet(0.25),plt.cm.jet(0.5),plt.cm.jet(0.75)]

    for i, size in enumerate(sizes):
        end_index = start_index + size
        # 获取批次数据
        batch_data = fps_np[start_index:end_index]
        # 绘制批次数据
        '''plt.scatter(fps_2d[start_index:end_index, 0], fps_2d[start_index:end_index, 1], 
                    alpha=0.5, label=labels[i], 
                    color=plt.cm.jet(i / (len(sizes)-1)))  # 使用颜色映射'''
        plt.scatter(fps_2d[start_index:end_index, 0], fps_2d[start_index:end_index, 1], 
                    alpha=0.5, label=labels[i], 
                    color=colors[i])  # 使用颜色映射
        start_index = end_index

    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    #plt.title('t-SNE Plot of Different Topological Networks')
    plt.title('without rnd reward')
    plt.legend()
    #plt.show()
    plt.savefig('tsne_trend.png', bbox_inches='tight')

def main1():
    file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/no.csv'
    fps_np, sizes = get_data(file_path,max_count=10000)
    
    # 定义批次标签
    labels = ['hcb', 'sql', 'kgd', 'hxl']
    colors = [plt.cm.jet(0.0),plt.cm.jet(0.25),plt.cm.jet(0.5),plt.cm.jet(0.75)]
    # 绘制并保存t-SNE图
    tsne_plot(fps_np, sizes, labels, colors)

def main():
    count=5000
    file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/cof.csv'
    a, b = get_data_rand(file_path,max_count=25000)
    file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/no.csv'
    c, d = get_data(file_path,max_count=count)
    file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/top5.csv'
    fps_np, sizes = get_data(file_path,max_count=count)

    #fps_np = np.concatenate((a, c, fps_np), axis=0)
    #sizes = b + d + sizes
    fps_np = np.concatenate((a, fps_np), axis=0)
    sizes = b + sizes
    print(sizes)
    labels = ['Material space', 'HCB:84.8%', 'SQL:1.2%', 'KGD:1.8%', 'HXL:12.2%']
    tsne_plot(fps_np, sizes, labels)


if __name__ == "__main__":
    main()