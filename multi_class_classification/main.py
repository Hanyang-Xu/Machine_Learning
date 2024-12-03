import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



class MultilayerPerceptron():
    def __init__(self, n_iter=200, lr=1e-3, tol=None, train_mode='SGD', batch_size = 10, type='classify', w0=0.5, w1=0.5, gate=0.5):
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.train_mode = train_mode
        self.type = type
        self.best_loss = np.inf
        self.patience = 10
        self.w0 = w0
        self.w1 = w1
        self.gate = gate
        self.layer_list = []
        self.H = []  
        self.U = []  
        self.loss = []
        self.layer_shape = []

    class layer:
        def __init__(self, input_num, cell_num, activation=None):
            self.W = np.random.randn(cell_num, input_num + 1) * np.sqrt(1. / input_num)
            self.activation = activation

        def activate(self, U):
            if self.activation == 'sigmoid':
                out = 1. / (1. + np.exp(-U))
            elif self.activation == 'relu':
                out = np.maximum(0, U)
            elif self.activation == 'linear':
                out = U
            elif self.activation == 'softmax':
                exp_U = np.exp(U - np.max(U))
                out = exp_U / np.sum(exp_U, axis=-1, keepdims=True)
            return out

        def act_derivative(self, X):
            if self.activation == 'sigmoid':
                s = self.activate(X)
                out = s * (1 - s)
            elif self.activation == 'relu':
                out = np.where(X > 0, 1, 0)
            elif self.activation == 'linear':
                out = np.ones_like(X)
            return out
        
        def return_act(self):
            return self.activation

    def _preprocess_data(self, X):
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1  
        X_[:, 1:] = X
        return X_
    
    def _loss(self, y, y_pred):
        if self.type == 'classify':
            # print("using cross entropy")
            epsilon = 1e-5
            # print(y)
            if y.ndim == 1:
                loss = -(self.w1*y*np.log(y_pred + epsilon) + self.w0*(1-y)*np.log(1-y_pred+epsilon))
                loss = np.mean(loss)
            else:
                loss = -np.sum(y * np.log(y_pred + epsilon)) / y.shape[0]
        elif self.type == 'regression':
            # print("using mse loss")
            loss = np.mean((y - y_pred) ** 2)
        return loss
    
    def _cal_E_o(self, y, y_pred):
        if self.type == "classify":
            epsilon = 1e-10  # 小常数，防止除零
            return (y_pred - y) / (y_pred * (1 - y_pred) + epsilon)
        elif self.type == "regression":
            return y_pred - y

    def _shuffle(self, X, y):
        # y = y.reshape(-1,1)
        index = np.random.permutation(X.shape[0])
        X_shuffled = X[index]
        y_shuffled = y[index]
        return X_shuffled, y_shuffled

    def build_layer(self, input_num, cell_num, activation=None):
        layer = self.layer(input_num, cell_num, activation)
        self.layer_shape.append(cell_num)
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

        for i in range(len(H_reverse_list)-1):
            if i == 0:
                y_pred = H_reverse_list[i]
                y_pred = y_pred
                y = y
                # print(f'y_pred = {y_pred}')
                # print(f'y = {y}')

                if layer_reverse_list[i].activation == 'softmax':
                    delta_z = y_pred - y
                    # print(f'delta_z = {delta_z}')
                else:
                    E_o = self._cal_E_o(y, y_pred)
                    delta_z = E_o * layer_reverse_list[i].act_derivative(U_reverse_list[i])

                delta_z = delta_z.T
                delta_u.append(delta_z)
                delta_z = delta_z[:,:,np.newaxis]
                H_reverse = self._preprocess_data(H_reverse_list[i+1])
                H_reverse = H_reverse[np.newaxis,:,:]
                W_grad_new = delta_z*H_reverse
                W_grad_new = np.mean(W_grad_new, axis=1)
                W_grad.append(W_grad_new)
            else:
                _delta_h = layer_reverse_list[i-1].W[:,1:].T @ delta_u[i-1]
                _delta_u = _delta_h.T*layer_reverse_list[i].act_derivative(U_reverse_list[i])
                _delta_u = _delta_u.T
                delta_u.append(_delta_u)
                _delta_u = _delta_u[:,:,np.newaxis]
                H_reverse = self._preprocess_data(H_reverse_list[i+1])
                H_reverse = H_reverse[np.newaxis,:,:]
                W_grad_new = _delta_u*H_reverse
                W_grad_new = np.mean(W_grad_new, axis=1)
                W_grad.append(W_grad_new)
        return W_grad

    def stochastic_update(self, init_X, init_y):
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            X, y = self._shuffle(init_X, init_y)
            sample = X[0,:].reshape(-1,X.shape[1])
            y = y[0,:].reshape(-1)
            self.forward(sample)
            y_pred = self.H[len(self.H)-1].reshape(-1)
            loss = self._loss(y, y_pred)
            self.loss.append(loss)

            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                    if epoch_no_improve >= self.patience:
                        print(f"Early stopping triggered due to the no improvement in loss.")
                        break
                else:
                    epoch_no_improve = 0
            grad = self.backward(y)[::-1]
            for num, layer in enumerate(self.layer_list):
                layer.W = layer.W - self.lr * grad[num]
            
    def mini_batch_update(self, init_X, init_y):
        epoch_no_improve = 0
        batch_size = self.batch_size

        for iter in range(self.n_iter):
            # print(f"[{iter}/{self.n_iter}]")
            X, y = self._shuffle(init_X, init_y)
            # print(f"shuffled_y={y}")
            sample = X[:batch_size,:].reshape(-1,X.shape[1])
            y_sample = y[:batch_size,:].reshape(-1,y.shape[1])
            # print(f"y_sample = {y_sample}")
            self.forward(sample)
            y_pred = self.H[len(self.H)-1]
            # print(f'y_pred = {y_pred}')
            loss = self._loss(y_sample, y_pred)
            self.loss.append(loss)

            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                    if epoch_no_improve >= self.patience:
                        print(f"Early stopping triggered due to the no improvement in loss.")
                        break
                else:
                    epoch_no_improve = 0
            
            grad = self.backward(y_sample)[::-1]
            for num, layer in enumerate(self.layer_list):
                layer.W = layer.W - self.lr * grad[num]

    def train(self, X_train, Y_train):
        if self.train_mode == 'SGD':
            self.stochastic_update(X_train, Y_train)
        elif self.train_mode == 'MBGD':
            self.mini_batch_update(X_train, Y_train)

    def predict(self, X):
        self.forward(X)
        if self.type == "classify":
            y_pred = self.H[-1]
            if y_pred.shape[1] > 1:  # 多分类
                # print(y_pred.shape)
                y_pred = np.argmax(y_pred, axis=1)
            else:  # 二分类
                y_pred = y_pred.reshape(-1)
                for i, y in enumerate(y_pred):
                    if y > self.gate:
                        y_pred[i] = 1
                    elif y < self.gate:
                        y_pred[i] = 0
        elif self.type == "regression":
            y_pred = self.H[-1].reshape(-1)
        return y_pred

    def plot_loss(self):
        plt.plot(self.loss)
        plt.title('loss')
        plt.grid()
        plt.show()

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    
    def precision(self, y_true, y_pred, average='macro'):
        n_classes = len(np.unique(y_true))
        precisions = []
        for i in range(n_classes):
            TP = np.sum((y_true == i) & (y_pred == i))
            FP = np.sum((y_true != i) & (y_pred == i))
            precisions.append(TP / (TP + FP) if (TP + FP) != 0 else 0)
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'weighted':
            weights = [np.sum(y_true == i) for i in range(n_classes)]
            return np.average(precisions, weights=weights)
        else:
            raise ValueError("Average must be 'macro' or 'weighted'")

    def recall(self, y_true, y_pred, average='macro'):
        n_classes = len(np.unique(y_true))  # 获取类别数量
        recalls = []
        for i in range(n_classes):
            TP = np.sum((y_true == i) & (y_pred == i))
            FN = np.sum((y_true == i) & (y_pred != i))
            recalls.append(TP / (TP + FN) if (TP + FN) != 0 else 0)
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'weighted':
            weights = [np.sum(y_true == i) for i in range(n_classes)]
            return np.average(recalls, weights=weights)
        else:
            raise ValueError("Average must be 'macro' or 'weighted'")

    def f1_score(self, y_true, y_pred, average='macro'):
        prec = self.precision(y_true, y_pred, average)
        rec = self.recall(y_true, y_pred, average)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

    def evaluate(self, y_true, y_pred, average='macro'):
        print("==========Evaluation of the model===========")
        print(f"Accuracy: {self.accuracy(y_true, y_pred)}")
        print(f"Precision ({average}): {self.precision(y_true, y_pred, average)}")
        print(f"Recall ({average}): {self.recall(y_true, y_pred, average)}")
        print(f"F1 Score ({average}): {self.f1_score(y_true, y_pred, average)}")

    def reset(self):
        """Reset the MLP model by reinitializing its weights and clearing history."""
        self.layer_list = []
        self.H = []
        self.U = []
        self.loss = []
        self.best_loss = np.inf  # Reset best loss

def load_data(file):
        dataset = np.loadtxt(file, delimiter=',')
        X = dataset[:, :dataset.shape[1]-1]
        y = dataset[:, -1]
        return X, y

if __name__ =="__main__":
    X_train, y_train= load_data('multi_class_classification/optdigits.tra')
    X_test, y_test = load_data('multi_class_classification/optdigits.tes')
    y_train = y_train.astype(int)
    y_train = np.eye(10)[y_train]
    _,n_feature = X_train.shape
    unique_values = np.unique(y_test)
    n_label = len(unique_values)

    # min_val = np.min(X_train, axis=0)
    # max_val = np.max(X_train, axis=0)
    # X_train = (X_train - min_val) / (max_val - min_val)
    # X_test = (X_test - min_val) / (max_val - min_val)

    model = MultilayerPerceptron(n_iter=100,lr=0.01,tol=1e-5,
                                 train_mode='MBGD',batch_size=1000,type='classify')
    model.build_layer(n_feature, 64, 'relu')
    model.build_layer(64, 128, 'relu')
    model.build_layer(128, 64, 'relu')
    model.build_layer(64, n_label, 'softmax')

    model.train(X_train, y_train)
    model.plot_loss()
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)
    model.evaluate(y_test,y_pred)
