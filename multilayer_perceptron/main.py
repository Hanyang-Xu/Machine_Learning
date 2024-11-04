import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron():
    def __init__(self, n_feature=1, n_iter=200, lr=1e-3, tol=None, train_mode='SGD', batch_size=10):
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.train_mode = train_mode
        self.best_loss = np.inf
        self.patience = 10
        self.layer_list = []
        self.H = []
        self.U = []
        self.loss = []

    class layer:
        def __init__(self, input_num, cell_num, activation=None):
            self.W = np.random.randn(cell_num, input_num + 1) * np.sqrt(1. / input_num)
            self.activation = activation

        def activate(self, U):
            if self.activation == 'sigmoid':
                out = 1. / (1. + np.exp(-U))
            elif self.activation == 'relu':
                out = np.maximum(0, U)
            return out

        def act_derivative(self, X):
            if self.activation == 'sigmoid':
                s = self.activate(X)
                out = s * (1 - s)
            elif self.activation == 'relu':
                out = np.where(X > 0, 1, 0)
            return out

    def _preprocess_data(self, X):
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1  
        X_[:, 1:] = X
        return X_
    
    def _cross_entropy(self, y, y_pred):
        epsilon = 1e-5
        loss = -(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return np.mean(loss)

    def _shuffle(self, X, y):
        y = y.reshape(-1, 1)
        index = np.random.permutation(X.shape[0])
        return X[index], y[index]

    def build_layer(self, input_num, cell_num, activation=None):
        layer = self.layer(input_num, cell_num, activation)
        self.layer_list.append(layer)
    
    def forward(self, X):
        self.H = []
        self.U = []
        layers_num = len(self.layer_list)
        for i in range(layers_num):
            if i == 0:
                self.H.append(X)
                X = self._preprocess_data(X)
                U = X @ self.layer_list[i].W.T
                self.U.append(U)
                H = self.layer_list[i].activate(U)
                self.H.append(H)
            else:
                H = self._preprocess_data(H)
                U = H @ self.layer_list[i].W.T
                self.U.append(U)
                H = self.layer_list[i].activate(U)
                self.H.append(H)
    
    def backward(self, y):
        W_grad = []
        delta_u = []
        U_reverse_list = self.U[::-1]
        H_reverse_list = self.H[::-1]
        layer_reverse_list = self.layer_list[::-1]

        for i in range(len(H_reverse_list) - 1):
            if i == 0:
                y_pred = H_reverse_list[i].reshape(-1, 1)
                E_o = (y_pred - y) / (y_pred * (1 - y_pred))
                delta_z = E_o * layer_reverse_list[i].act_derivative(U_reverse_list[i])
                delta_u.append(delta_z.T)
                W_grad_new = np.mean(delta_z.T[:, :, np.newaxis] * self._preprocess_data(H_reverse_list[i + 1])[np.newaxis, :, :], axis=1)
                W_grad.append(W_grad_new)
            else:
                delta_h = layer_reverse_list[i - 1].W[:, 1:].T @ delta_u[i - 1]
                delta_z = delta_h.T * layer_reverse_list[i].act_derivative(U_reverse_list[i])
                delta_u.append(delta_z.T)
                W_grad_new = np.mean(delta_z.T[:, :, np.newaxis] * self._preprocess_data(H_reverse_list[i + 1])[np.newaxis, :, :], axis=1)
                W_grad.append(W_grad_new)
        return W_grad[::-1]

    def stochastic_update(self, X, y):
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            X, y = self._shuffle(X, y)
            sample = X[0, :].reshape(-1, X.shape[1])
            y_sample = y[0, :].reshape(-1)
            self.forward(sample)
            y_pred = self.H[-1].reshape(-1)
            loss = self._cross_entropy(y_sample, y_pred)
            self.loss.append(loss)

            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                    if epoch_no_improve >= self.patience:
                        print("Early stopping triggered.")
                        break

            grad = self.backward(y_sample)
            for num, layer in enumerate(self.layer_list):
                layer.W -= self.lr * grad[num]
    
    def train(self, X, y):
        self.stochastic_update(X, y)

    def predict(self, X):
        self.forward(X)
        return (self.H[-1].reshape(-1) > 0.5).astype(int) + 1

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 2) & (y_pred == 1))
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 2))
        return TP / (TP + FN) if (TP + FN) != 0 else 0
    
    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    
def k_fold_cross_validation(X, y, k, model_class, **kwargs):
    fold_size = X.shape[0] // k
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    scores = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}

    for i in range(k):
        val_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model = model_class(**kwargs)
        model.build_layer(13, 16, 'relu')
        model.build_layer(16, 8, 'relu')
        model.build_layer(8, 4, 'relu')
        model.build_layer(4, 1, 'sigmoid')

        model.train(X_train, y_train)
        y_pred = model.predict(X_val)

        scores["accuracy"].append(model.accuracy(y_val, y_pred))
        scores["precision"].append(model.precision(y_val, y_pred))
        scores["recall"].append(model.recall(y_val, y_pred))
        scores["f1_score"].append(model.f1_score(y_val, y_pred))

    print("========== K-Fold Cross-Validation Results ==========")
    print(f"Average accuracy: {np.mean(scores['accuracy'])}")
    print(f"Average precision: {np.mean(scores['precision'])}")
    print(f"Average recall: {np.mean(scores['recall'])}")
    print(f"Average f1_score: {np.mean(scores['f1_score'])}")

def load_data(file, ratio, random_state=None):
    dataset = np.loadtxt(file, delimiter=',')
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)

    split_index = int(dataset.shape[0] * (1 - ratio))
    train_data = dataset[indices[:split_index]]
    test_data = dataset[indices[split_index:]]
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    y_train[y_train == 2] = 1.0000000001
    y_train[y_train == 1] = 0
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('separate_Perceptron/wine_formed_data', 0.3, True)
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train = (X_train - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)
    k = 5
    k_fold_cross_validation(X_train, y_train, k, MultilayerPerceptron, n_feature=X_train.shape[1], n_iter=300, lr=0.1, tol=0.01, train_mode='MBGD', batch_size=10)
