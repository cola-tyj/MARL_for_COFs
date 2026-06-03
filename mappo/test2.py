import subprocess
import os
import logging
import re
import csv
import shutil
import pandas as pd
import matplotlib.pyplot as plt

def randcof():
    # 定义源文件和目标文件的路径
    source_file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/cof.csv'
    target_file_path = '/home/tianyajun/MARL_for_COFs/mappo/data_train/rand2.5.csv'

    # 读取源文件
    with open(source_file_path, 'r') as source_file:
        lines = source_file.readlines()

    # 每隔40行取一行
    selected_lines = lines[::40]

    # 将选中的行写入目标文件
    with open(target_file_path, 'w') as target_file:
        target_file.writelines(selected_lines)

    print(f"完成！已将每隔40行的数据保存到{target_file_path}")

def Pore_diameters(cof): # zeo++计算孔径
    # 定义命令和参数
    command1 = [
        "/home/tianyajun/zeo++-0.3/network",
        "-ha",
        "-res"
    ]
    command2 = [
        "/home/tianyajun/zeo++-0.3/network",
        "-ha",
        "-psd",
        "1.2",
        "1.2",
        "50000"
    ]
    command3 = [
        "/home/tianyajun/zeo++-0.3/network",
        "-ha",
        "-sa",
        "1.2",
        "1.2",
        "2000"
    ]
    command4 = [
        "/home/tianyajun/zeo++-0.3/network",
        "-ha",
        "-vol",
        "1.2",
        "1.2",
        "50000"
    ]
    return subprocess.run(command2 + [cof], capture_output=True, text=True)

def processSingle(cifName:str) -> None:
    """
    计算一个cif的气体吸附属性
    """

    cifPath = f"/home/tianyajun/MARL_for_COFs/cofs/SQL/{cifName}.cif"
    # 指定base文件夹
    baseDir = "/home/tianyajun/MARL_for_COFs/cofs/gas_adsorption"

    # 构造输入文件
    targetDir = f"{baseDir}/{cifName}"
    os.makedirs(targetDir, exist_ok=True)

    # 如果处理过则跳过
    if os.path.exists(f"{targetDir}/Output"):
        logging.warning(f"{cifPath} has processed")
        return

    # 1.cif
    os.system(f"cp \'{cifPath}\' \'{targetDir}\'")
    # 2.run
    os.system(f"cp \'/home/tianyajun/RASPA2/examples/test/1/run\' \'{targetDir}\'")
    # 3.simulation.input
    os.system(f"cp \'/home/tianyajun/RASPA2/examples/test/1/simulation.input\' \'{targetDir}\'")
    vf = getVoidFraction(cifName) # 计算孔隙率
    fillSimulation(f"{targetDir}/simulation.input", cifName, vf) # 将孔隙率填到simulation.input对应位置

    # 执行计算
    os.chdir(targetDir)
    logging.debug(f"cif:{cifPath} start")
    os.system("bash run")

    # 读取计算结果
    result = readSingle(targetDir)
    return result

def fillSimulation(filePath: str, cifName: str, voidFraction: float):
    """
    将孔隙率,cif名填到simulation.input

    Args:
        filePath (str): simulation.input路径
        cifName (str): 晶体名
        voidFraction (float): 孔隙率
    """
    with open(filePath, 'r') as file:  
        lines = file.readlines()
    lines[11] = f"FrameworkName                 {cifName}\n"
    lines[12] = f"HeliumVoidFraction            {str(voidFraction)}\n"
    # 将修改后的内容写回到文件中  
    with open(filePath, 'w') as file:
        file.writelines(lines)

def readSingle(path: str):
    """
    根据路径读取计算结果

    Args:
        path (str): 路径
    """
    cifName = path.split("/")[-1]
    filePath = f"{path}/Output/System_0/output_{cifName}_2.2.3_298.000000_101325.data"
    with open(filePath, mode="r") as f:
        for line in f:
            if line.startswith("\tAverage loading absolute"):
                former, latter = line.split("+/-")[0], line.split("+/-")[1]
                val = float(re.findall(r'\d+\.\d+', former)[0])
                eps = float(re.findall(r'\d+\.\d+', latter)[0])
                if val is None or eps is None:
                    return None
                else:
                    return [val, eps]
                
def getVoidFraction(cifName:str, shuxing='Fraction'):
    
    # 初始化变量
    fraction = 0.0

    if shuxing == 'Fraction':
        match = 'Fraction of sample points in node spheres:'
        filepath = f'/home/tianyajun/MARL_for_COFs/cofs/KGD/{cifName}.psd_histo'    
    elif shuxing == 'Volume':
        match = 'AV_cm\^3/g:'
        filepath = f'/home/tianyajun/MARL_for_COFs/cofs/KGD/{cifName}.vol'
    elif shuxing == 'Surface':
        match = 'ASA_m\^2/g:'
        filepath = f'/home/tianyajun/MARL_for_COFs/cofs/KGD/{cifName}.sa'
    elif shuxing == 'Diameter':
        match = '.res'
        filepath = f'/home/tianyajun/MARL_for_COFs/cofs/KGD/{cifName}.res'
    elif shuxing == 'O2':
        match = cifName+','
        filepath = '/home/tianyajun/MARL_for_COFs/cofs/KGD/absolute.csv'
    
    # 读取文件内容
    with open(filepath, 'r') as file:
        file_content = file.read()  

    fraction = re.compile(f"{match}\s*([0-9.]+)").search(file_content).group(1)

    # 将提取的字符串转换为浮点数
    fraction = float(fraction) if fraction else None
    
    return fraction

def delete_all_files_and_directories(directory):
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)  # 构造完整文件路径
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 如果是文件或链接，则删除
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 如果是目录（无论是否为空），则递归删除
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def plot_and_save_scatter():
    # 读取CSV文件
    file_path = '/home/tianyajun/MARL_for_COFs/cofs/303/shuxing.csv'
    data = pd.read_csv(file_path)

    # 绘制散点图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.scatter(data['Fraction'], data['O2absolute'], color='blue')  # 绘制散点图
    plt.title('Fraction vs O2 absolute')  # 设置标题
    plt.xlabel('Fraction')  # 设置x轴标签
    plt.ylabel('O2 absolute')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.savefig('/home/tianyajun/MARL_for_COFs/imgs/scatter_plot.png')

def main1(): # 批处理算孔径
    cof_dir = '/home/tianyajun/MARL_for_COFs/cofs/SQL'
    cofs = []
    for file in os.listdir(cof_dir):
        # 检查文件扩展名是否为'.cif'
        if file.endswith('.cif'):
            cofs.append(os.path.join(cof_dir,file))
    print(len(cofs))

    for cof in cofs:
        # 执行命令
        result = Pore_diameters(cof)

def main2(): # 把孔径拼到氧气吸附值后面
    # cat *.res > final_results_table.txt
    csv_file_path = '/home/tianyajun/MARL_for_COFs/cofs/303/303.csv'
    txt_file_path = '/home/tianyajun/MARL_for_COFs/cofs/303/final_results_table.txt'

    # 读取txt文件
    with open(txt_file_path, 'r') as txt_file:
        txt_lines = txt_file.readlines()

    # 读取csv文件
    with open(csv_file_path, 'r') as csv_file:
        csv_lines = csv_file.readlines()
    
    # 添加浮点数到csv文件的每一行
    updated_lines = []
    for csv_line, txt_line in zip(csv_lines, txt_lines):
        csv_parts = csv_line.strip().split(',')
        txt_float = txt_line.strip().split()[1]
        updated_line = f"{csv_parts[0]},{csv_parts[1]},{txt_float}\n"
        updated_lines.append(updated_line)

    # 将更新后的内容写回csv文件
    with open(csv_file_path, 'w') as csv_file:
        csv_file.writelines(updated_lines)

def main3(): # raspa算氧气吸附
    directory = '/home/tianyajun/MARL_for_COFs/cofs/SQL/'
    # 初始化一个空列表来存储文件名
    cif_filenames = []
    
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件名是否以.cif结尾
        if filename.endswith('.cif'):
            # 移除.cif扩展名并添加到列表中
            cif_filenames.append(filename[:-4])  # 移除最后四个字符（.cif）
    
    # 指定CSV文件名
    csv_filename = '/home/tianyajun/MARL_for_COFs/cofs/SQL/absolute.csv'
    
    # 使用'w'模式打开CSV文件，准备写入
    with open(csv_filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        writer = csv.writer(csvfile)
        # 写入列标题
        writer.writerow(['cifname', 'value1', 'value2'])
        
        # 遍历所有.cif文件名
        for cifname in cif_filenames:
            # 调用processSingle函数获取结果
            result = processSingle(cifname)
            print(result)
            #result = [0,1]
            # 写入CSV文件
            writer.writerow([cifname] + result)
            delete_all_files_and_directories('/home/tianyajun/MARL_for_COFs/cofs/gas_adsorption')

def main4(): # 读取各种属性
    Fraction = 'Fraction'
    Surface = 'Surface'
    Volume = 'Volume'
    directory = '/home/tianyajun/MARL_for_COFs/cofs/KGD/'
    # 初始化一个空列表来存储文件名
    cif_filenames = []
    
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件名是否以.cif结尾
        if filename.endswith('.cif'):
            # 移除.cif扩展名并添加到列表中
            cif_filenames.append(filename[:-4])  # 移除最后四个字符（.cif）
    
    # 指定CSV文件名
    csv_filename = '/home/tianyajun/MARL_for_COFs/cofs/KGD/shuxing.csv'
    
    # 使用'w'模式打开CSV文件，准备写入
    with open(csv_filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        writer = csv.writer(csvfile)
        # 写入列标题
        writer.writerow(['cifname', 'Fraction', 'Surface', 'Volume', 'Diameter', 'O2absolute'])
        
        # 遍历所有.cif文件名
        for cifname in cif_filenames:
            Fraction = getVoidFraction(cifname,'Fraction')
            Surface = getVoidFraction(cifname,'Surface')
            Volume = getVoidFraction(cifname,'Volume')
            Diameter = getVoidFraction(cifname,'Diameter')
            O2absolute = getVoidFraction(cifname,'O2')
            # 写入CSV文件
            writer.writerow([cifname,Fraction,Surface,Volume,Diameter,O2absolute])



if __name__ == "__main__":
    main4()


