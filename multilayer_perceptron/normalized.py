import numpy as np

def load_data(file, ratio, random_state=None):
    dataset = np.loadtxt(file, delimiter=',')
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)

    split_index = int(dataset.shape[0] * (1 - ratio))
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    train_data = dataset[train_indices]
    test_data = dataset[test_indices]
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    y_train[y_train == 2] = 1.0000000001
    y_train[y_train == 1] = 0
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('separate_Perceptron/wine_formed_data', 0.3, True)
    _, n_feature = X_train.shape

    # 计算每个特征的最小值和最大值
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    print(min_val)
    print(max_val)

    # 应用最小-最大归一化
    normalized_X_train = (X_train - min_val) / (max_val - min_val)

    print("归一化后的训练数据：")
    print(normalized_X_train)
