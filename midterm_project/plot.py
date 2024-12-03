import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 数据
categories = ['PCA', 'Linear-Autoencoder', 'Linear-Autoencoder', 'Nonlinear-Autoencoder', 'Nonlinear-Autoencoder']
subcategories = ['PCA', 'test', 'train', 'test', 'train']
values = [0.446, 0.468, 0.447, 0.44, 0.21]

# 创建 DataFrame
data = pd.DataFrame({
    'Category': categories,
    'Subcategory': subcategories,
    'Value': values
})

# 设置颜色（使用 Set2 调色板，避免使用红色）
colors = sns.color_palette("Set2", n_colors=5)

# 重新排列数据
order = ['PCA', 'Linear-Autoencoder', 'Nonlinear-Autoencoder']
data['Category'] = pd.Categorical(data['Category'], categories=order, ordered=True)

# 创建分组柱形图
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Value', hue='Subcategory', data=data, palette=colors, dodge=True)

# 调整柱子的排列方式，确保第一列不偏移
plt.ylabel('Reconstruction Error', fontsize=14)
plt.ylim(0, 0.6)
plt.legend(title='Subcategory', fontsize=12)

# 显示图形
plt.tight_layout()
plt.show()

