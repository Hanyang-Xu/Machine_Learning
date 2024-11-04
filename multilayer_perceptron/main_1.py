import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron():
    def __init__(self, n_feature=1, n_iter=200, lr=1e-3, tol=None, train_mode='SGD', batch_size = 10, type='classify'):
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.train_mode = train_mode
        self.type = type
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
            elif self.activation == 'linear':
                out = U
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
    
    def _cross_entropy(self, y, y_pred):
        if self.type == 'classify':
            epsilon = 1e-5
            loss = -(y*np.log(y_pred + epsilon) + (1-y)*np.log(1-y_pred+epsilon))
            loss = np.mean(loss)
        elif self.type == 'regression':
            # print("using mse loss")
            loss = np.mean((y - y_pred) ** 2)
        return loss

    def _shuffle(self, X, y):
        y = y.reshape(-1,1)
        index = np.random.permutation(X.shape[0])
        X_shuffled = X[index]
        y_shuffled = y[index]
        return X_shuffled, y_shuffled

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

        for i in range(len(H_reverse_list)-1):
            if i == 0:
                y_pred = H_reverse_list[i]
                y_pred = y_pred.reshape(-1,1)
                y = y.reshape(-1,1)
                E_o = (y_pred-y)/(y_pred*(1-y_pred))
                delta_z = E_o*layer_reverse_list[i].act_derivative(U_reverse_list[i])
                # flag = layer_reverse_list[i].return_act()
                # print(f"act={flag}")
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
                # flag = layer_reverse_list[i].return_act()
                # print(f"act={flag}")
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
            loss = self._cross_entropy(y, y_pred)
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
        break_out = False
        batch_size = self.batch_size

        for iter in range(self.n_iter):
            X, y = self._shuffle(init_X, init_y)
            for j in range(int(X.shape[0]/batch_size)):
                sample = X[j*batch_size:(j+1)*batch_size,:].reshape(-1,X.shape[1])
                y_sample = y[j*batch_size:(j+1)*batch_size,:].reshape(-1)
                self.forward(sample)
                y_pred = self.H[len(self.H)-1].reshape(-1)
                loss = self._cross_entropy(y_sample, y_pred)
                self.loss.append(loss)

                if self.tol is not None:
                    if loss < self.best_loss - self.tol:
                        self.best_loss = loss
                        epoch_no_improve = 0
                    elif np.abs(loss - self.best_loss) < self.tol:
                        epoch_no_improve += 1
                        if epoch_no_improve >= self.patience:
                            print(f"Early stopping triggered due to the no improvement in loss.")
                            break_out = True
                            break
                    else:
                        epoch_no_improve = 0
                
                grad = self.backward(y_sample)[::-1]
                for num, layer in enumerate(self.layer_list):
                    layer.W = layer.W - self.lr * grad[num]
    
            if break_out:
                break_out = False
                break

    def train(self, X_train, Y_train):
        if self.train_mode == 'SGD':
            self.stochastic_update(X_train, Y_train)
        elif self.train_mode == 'MBGD':
            self.mini_batch_update(X_train, Y_train)

    def predict(self, X):
        self.forward(X)
        if self.type == "classify":
            y_pred = self.H[-1].reshape(-1)
            for i, y in enumerate(y_pred):
                if y > 0.5:
                    y_pred[i] = 2
                elif y < 0.5:
                    y_pred[i] = 1
        if self.type == "regression":
            y_pred = self.H[-1].reshape(-1)
        return y_pred

    def plot_loss(self):
        plt.plot(self.loss)
        plt.title('loss')
        plt.grid()
        plt.show()

    def accuracy(self,y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(self,y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives (1 correctly predicted as 1)
        FP = np.sum((y_true == 2) & (y_pred == 1))  # False Positives (2 incorrectly predicted as 1)
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(self,y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives (1 correctly predicted as 1)
        FN = np.sum((y_true == 1) & (y_pred == 2))  # False Negatives (1 incorrectly predicted as 2)
        return TP / (TP + FN) if (TP + FN) != 0 else 0
    
    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    
    def evaluate(self, y_true, y_pred):
        print("==========Evaluation of the model===========")
        print(f"accuracy={self.accuracy(y_true, y_pred)}")
        print(f"precision={self.precision(y_true, y_pred)}")
        print(f"recall={self.recall(y_true, y_pred)}")
        print(f"f1_score={self.f1_score(y_true, y_pred)}")

def load_data(file, ratio, random_state = None):
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
        # print(f"y_train:{y_train}")
        X_test = test_data[:, 1:]
        y_test = test_data[:, 0]
        y_train[y_train == 2] = 1.0000000001
        y_train[y_train == 1] = 0
        # print(f"formed_y_train:{y_train}")
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Classifier Performance Evaluation
    X_train, X_test, y_train, y_test= load_data('separate_Perceptron/wine_formed_data', 0.3, True)
    print(f"X:{X_train.shape}")
    print(f"y:{y_train.shape}")
    _,n_feature = X_train.shape
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train = (X_train - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)
    MLP = MultilayerPerceptron(n_feature=n_feature, n_iter=300, lr=0.1, tol=0.01, train_mode='MBGD')
    MLP.build_layer(13, 16, 'relu')
    MLP.build_layer(16, 8, 'relu')
    MLP.build_layer(8, 4, 'relu')
    MLP.build_layer(4, 1, 'sigmoid')
    MLP.train(X_train, y_train)
    MLP.plot_loss()
    y_pred = MLP.predict(X_test)
    print(f"y_true:{y_test}")
    print(f"y_pred:{y_pred}")
    MLP.evaluate(y_test, y_pred)

    # Nonlinear Function Approximation
    # X = np.arange(100).reshape(100,1)
    # y = 0.01 * (X**2) - 1.5 * X + 20+ np.random.normal(0, 5, size=X.shape)
    # y_train = y.reshape(-1)
    # _, n_feature = X.shape
    # print(X.shape)
    # print(y.shape)
    # print(n_feature)

    # min_val = np.min(X, axis=0)
    # max_val = np.max(X, axis=0)
    # X_train = (X - min_val) / (max_val - min_val)
    # X_test = (X - min_val) / (max_val - min_val)
    # y_min = np.min(y_train)
    # y_max = np.max(y_train)
    # y_train = (y_train - y_min) / (y_max - y_min)

    # MLP = MultilayerPerceptron(n_feature=n_feature, n_iter=300, lr=0.00001, tol=0.1, train_mode='MBGD', type='regression')
    # MLP.build_layer(1, 4, 'relu')
    # MLP.build_layer(4, 1, 'linear')
    # MLP.train(X_train, y_train)
    # MLP.plot_loss()
    # y_pred = MLP.predict(X_test)
    # print(f"y_true:{y_train}")
    # print(f"y_pred:{y_pred}")
    # MLP.evaluate(y_train, y_pred)