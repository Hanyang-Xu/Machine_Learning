import numpy as np

# 定义矩阵 A (2x4) 和 B (4x3)
A = np.array([[1, 2, 3, 4], 
              [5, 6, 7, 8]])

B = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9], 
              [10, 11, 12]])

# 确保 A 的列数等于 B 的行数
assert A.shape[1] == B.shape[0], "A 的列数必须等于 B 的行数"

# 将 A 和 B 变形，使得可以使用广播进行逐元素相乘
A_expanded = A[:, :, np.newaxis]  # (2, 4) -> (2, 4, 1)
B_expanded = B[np.newaxis, :, :]  # (4, 3) -> (1, 4, 3)

# 使用广播机制逐元素相乘
result_matrices = A_expanded * B_expanded  # (2, 4, 3) 结果为4个2x3矩阵

# 将结果按矩阵存储在一个列表中
result = [result_matrices[:, i, :] for i in range(result_matrices.shape[1])]

print(f"结果列表：{result}")
sum_matrix = np.sum(result_matrices, axis=1)  # (2, 3) 对所有矩阵求和
mean_matrix = np.mean(result_matrices, axis=1)  # (2, 3) 对所有矩阵取平均

print("四个矩阵求和的结果:\n", sum_matrix)
print("\n四个矩阵取平均的结果:\n", mean_matrix)

