import random
import numpy as np

pair = np.empty([100,2])
X_train = np.arange(100).reshape(100,1)
print(X_train.size)
a, b = 1, 10
y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
y_train = y_train.reshape(-1,1)

# print(f'X_train:{X_train}')
# print(f'y_train:{y_train}')
# print(f'pair:{pair}')

index = np.random.permutation(X_train.size)
print(index)
X_train_shuffled = X_train[index]
y_train_shuffled = y_train[index]
print(f'shuffled_pair:{X_train_shuffled[35]}')
print(f'shuffled_pair:{y_train_shuffled[35]}')
print(f'shuffled_pair:{y_train[X_train_shuffled[35]]}')