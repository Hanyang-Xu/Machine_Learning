import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 步骤 1: 加载数据
df = pd.read_csv('midterm_project/data.csv')
print("数据预览：")
print(df.head())

# 步骤 2: 计算相关性矩阵
df_without_failure = df.drop(columns=['Machine failure'])
correlation_matrix = df_without_failure.corr()

# 步骤 3: 提取与 'machine failure' 相关的特征
machine_failure_corr = df.corr()['Machine failure'].drop('Machine failure')
print("\n'机故障'与其他特征的相关性：")
print(machine_failure_corr)

# 步骤 4: 可视化相关性
machine_failure_corr_sorted = machine_failure_corr.abs().sort_values(ascending=False)
plt.figure(figsize=(12, 8))  # 增大图像尺寸
sns.barplot(x=machine_failure_corr_sorted.index, y=machine_failure_corr_sorted.values, palette='viridis')

# 增加字体大小和加粗
plt.title('Correlation with Machine Failure', fontsize=18, fontweight='bold')
plt.xlabel('Features', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.tight_layout()  # 确保布局紧凑

plt.savefig('correlation.png', transparent=True)
# 显示图像
plt.show()

# 步骤 6: 筛选出相关性较强的特征
high_corr_features = machine_failure_corr[machine_failure_corr.abs() > 0.5]
print("\n与 'machine failure' 相关性大于 0.5 的特征：")
print(high_corr_features)
