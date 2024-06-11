
import os
import pandas as pd

# 指定文件夹路径
folder_path = './data/'

# 获取文件夹下所有 CSV 文件的文件名
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 读取第一个 CSV 文件并保留表头
merged_data = pd.read_csv(os.path.join(folder_path, csv_files[0]))

# 循环读取其余 CSV 文件,不保留表头,并将数据追加到合并的数据框中
for file in csv_files[1:]:
    data = pd.read_csv(os.path.join(folder_path, file), header=0)
    merged_data = merged_data.append(data, ignore_index=True)

# 将合并后的数据保存到新的 CSV 文件中
merged_data.to_csv('./data/alldata7000.csv', index=False)

print("CSV 文件合并完成!")
