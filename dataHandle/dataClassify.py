import os
import shutil

# 指定目录
source_dir = r"D:\Game\laji\广州琶洲算法大赛新能源汽车电池健康度预测数据集\15台车运行数据"

# 创建目标文件夹
for i in range(1, 16):
    folder_name = f"CL{i}"
    os.makedirs(os.path.join(source_dir, folder_name), exist_ok=True)

# 移动文件
for file_name in os.listdir(source_dir):
    if os.path.isfile(os.path.join(source_dir, file_name)):
        prefix = file_name[:3]
        if prefix.startswith("CL") and prefix[2:].isdigit():
            folder_name = f"CL{int(prefix[2:])}"
            shutil.move(os.path.join(source_dir, file_name), os.path.join(source_dir, folder_name, file_name))
