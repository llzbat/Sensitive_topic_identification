import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
file_path = 'data1.csv'
data = pd.read_csv(file_path, header=None)

# 确保 CSV 文件有两列
assert data.shape[1] == 2, "CSV 文件应有两列"

# 去除第一列中的空格

# 打乱数据
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 拆分数据集，保持每个类别的比例一致
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[1])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data[1])

# 统计每个数据集中的类别数量
train_counts = train_data[1].value_counts()
val_counts = val_data[1].value_counts()
test_counts = test_data[1].value_counts()

# 打印类别统计
print("训练集类别统计：")
print(train_counts)
print("\n验证集类别统计：")
print(val_counts)
print("\n测试集类别统计：")
print(test_counts)

# 检查拆分比例
print(f"\n训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
print(f"测试集大小: {len(test_data)}")

# 保存拆分后的数据集为 .txt 文件，使用 '\t' 分隔
train_data.to_csv('dataset/data/train_data.txt', sep='\t', index=False, header=False)
val_data.to_csv('dataset/data/val_data.txt', sep='\t', index=False, header=False)
test_data.to_csv('dataset/data/test_data.txt', sep='\t', index=False, header=False)
