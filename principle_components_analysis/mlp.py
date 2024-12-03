import numpy as np
import matplotlib.pyplot as plt

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
            loss = -(self.w1*y*np.log(y_pred + epsilon) + self.w0*(1-y)*np.log(1-y_pred+epsilon))
            loss = np.mean(loss)
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
                E_o = self._cal_E_o(y, y_pred)
                delta_z = E_o*layer_reverse_list[i].act_derivative(U_reverse_list[i])
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
            print(f'X_hat:{y}')
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
        # print('========in the mini_batch========')

        for iter in range(self.n_iter):
            if iter%100 == 0:
                print(f"[{iter}/{self.n_iter}]")
            X, y = self._shuffle(init_X, init_y)
            sample = X[:batch_size,:].reshape(-1,X.shape[1])
            # print(f'X_shape:{sample.shape}')
            y_sample = y[:batch_size,:]
            # print(f'X_hat_shape:{y_sample.shape}')
            self.forward(sample)
            y_pred = self.H[len(self.H)-1]
            # print(f'y_pred_shape:{y_pred.shape}')
            loss = self._loss(y_sample, y_pred)
            # print(f'loss:{loss}')
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
            print(f'Y_train_shape:{Y_train.shape}')
            self.mini_batch_update(X_train, Y_train)


    def predict(self, X):
        self.forward(X)
        if self.type == "classify":
            y_pred = self.H[-1].reshape(-1)
            for i, y in enumerate(y_pred):
                if y > self.gate:
                    y_pred[i] = 1
                elif y < self.gate:
                    y_pred[i] = 0
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
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(self,y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
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

    def reset(self):
        """Reset the MLP model by reinitializing its weights and clearing history."""
        self.layer_list = []
        self.H = []
        self.U = []
        self.loss = []
        self.best_loss = np.inf  # Reset best loss
