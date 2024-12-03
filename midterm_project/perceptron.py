import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
import gc

class Perceptron:
    def __init__(self, n_feature=1, n_iter=200, lr=1e-3, tol=None, train_mode='BGD', batch_size=None, normalization=None):
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.norm_mode = normalization
        self.W = np.random.random(n_feature+1) * 0.5
        self.loss = []
        self.best_loss = np.inf
        self.patience = 10

    def _loss(self, y, y_pred):
        return -y_pred * y if y_pred * y < 0 else 0
    
    def _bgd_loss(self, y, y_pred):
        loss = -y * y_pred
        loss[loss < 0] = 0
        loss = np.sum(loss) / y.size
        return loss
    
    def _gradient(self, x_bar, y, y_pred):
        return -y * x_bar / y.size if y_pred * y <= 0 else 0
    
    def _bgd_gradient(self, x_bar, y, y_pred):
        loss = -y * y_pred
        negative_indices = np.where(loss < 0)[0]
        y = y[:, np.newaxis]
        grad = -y * x_bar
        grad[negative_indices, :] = 0
        grad = np.sum(grad, axis=0) / y.size
        return grad
    
    def _min_max_norm(self, X):
        self._max = np.max(X, axis=0)
        self._min = np.min(X, axis=0)
        self._range = self._max - self._min
        return (X - self._min) / self._range
    
    def _reverse_min_max_norm(self, x):
        return x * (self._max - self._min) + self._min

    def _mean_norm(self, x):
        self.mu = np.mean(x, axis=0)
        self.sigma = np.std(x, axis=0)
        return (x - self.mu) / self.sigma
    
    def _reverse_mean_norm(self, x):
        return x * self.sigma + self.mu
    
    def _preprocess_data(self, X):
        if self.norm_mode == 'min_max':
            X = self._min_max_norm(X)
        if self.norm_mode == 'mean':
            X = self._mean_norm(X)
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_
    
    def _shuffle(self, X, y):
        y = y.reshape(-1,1)
        index = np.random.permutation(X.shape[0])
        X_shuffled = X[index]
        y_shuffled = y[index]
        return X_shuffled, y_shuffled
    
    def _predict(self, X):
        return X @ self.W

    def predict(self, X):
        X = self._preprocess_data(X)
        y_pred = X @ self.W
        for i, y in enumerate(y_pred):
            if y > 0:
                y_pred[i] = 1
            elif y < 0:
                y_pred[i] = 0
        return y_pred

    def stochastic_update(self, X, y):
        break_out = False
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            for i, x in enumerate(X):
                y_pred = self._predict(x)
                loss =  self._loss(y[i], y_pred)
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
                        grad = self._gradient(x, y[i], y_pred)
                        self.W = self.W - self.lr * grad
            if break_out:
                break_out = False
                break
    
    def batch_update(self, X, y):
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            y_pred = self._predict(X)
            loss = self._bgd_loss(y, y_pred)
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

            grad = self._bgd_gradient(X, y, y_pred)
            self.W = self.W - self.lr * grad

    def mini_batch_update(self, init_X, init_y):
        epoch_no_improve = 0
        batch_size = self.batch_size

        for iter in range(self.n_iter):
            print(f"[{iter}/{self.n_iter}]")
            X, y = self._shuffle(init_X, init_y)
            sample = X[:batch_size,:].reshape(-1,X.shape[1])
            y_sample = y[:batch_size,:].reshape(-1)
            y_pred = self._predict(sample).reshape(-1)
            loss = self._bgd_loss(y_sample, y_pred)
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
            
            grad = self._bgd_gradient(sample, y_sample, y_pred)
            self.W = self.W - self.lr * grad
    

    def train(self, X_train, Y_train):
        X_train_bar = self._preprocess_data(X_train)
        if self.train_mode == 'BGD':
            self.batch_update(X_train_bar, Y_train)
        elif self.train_mode == 'SGD':
            self.stochastic_update(X_train_bar, Y_train)
        elif self.train_mode == 'MBGD':
            self.mini_batch_update(X_train_bar, Y_train)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.title('loss')
        plt.grid()
        plt.show()
    
    def accuracy(self,y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(self,y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives (1 correctly predicted as 1)
        FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives (2 incorrectly predicted as 1)
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(self,y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives (1 correctly predicted as 1)
        FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives (1 incorrectly predicted as 2)
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
        dataset = np.load(file, allow_pickle=True)
        dataset = np.array(dataset, dtype=float)
        # print(dataset.dtype)
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.arange(dataset.shape[0])
        np.random.shuffle(indices)

        split_index = int(dataset.shape[0] * (1 - ratio))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        train_data = dataset[train_indices]
        test_data = dataset[test_indices]
        X_train = train_data[:, :6]
        y_train = train_data[:, 6]
        # print(f"y_train:{y_train}")
        X_test = test_data[:, :6]
        y_test = test_data[:, 6]

        # zero_indices = np.where(y_train == 0)[0]
        # sample_indices = np.random.choice(zero_indices, size=244, replace=False)
        # one_indices = np.where(y_train==1)[0]
        # X_zeros = X_train[sample_indices]
        # y_zeros = y_train[sample_indices]
        # X_ones = X_train[one_indices]
        # y_ones = y_train[one_indices]
        # X_train = np.vstack((X_zeros,X_ones))
        # y_train = np.hstack((y_zeros,y_ones))

        smote = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        y_train[y_train == 1] = 1.000000000001
        y_train[y_train == 0] = -1
        # print(f"formed_y_train:{y_train}")

        
        return X_train, X_test, y_train, y_test
    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test= load_data('midterm_project/ai4i2020.npy', 0.3, True)
    _,n_feature = X_train.shape
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train = (X_train - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)
    for n_iter in [1000, 2000, 5000, 10000]:
        for batch_size in [16, 32, 50]:
            for lr in [0.1, 0.01, 0.001]:
                model = Perceptron(n_feature=n_feature, n_iter=n_iter, lr=lr, tol=0.0001, train_mode='MBGD', batch_size=batch_size)
                model.train(X_train, y_train)
                # plt.figure()
                # model.plot_loss()
                y_pred = model.predict(X_test)
                np.set_printoptions(threshold=np.inf)
                print(f"y_true:{y_test}")
                print(f"y_pred:{y_pred}")
                model.evaluate(y_test, y_pred)
                precision = model.precision(y_test, y_pred)
                recall = model.recall(y_test, y_pred)
                f1 = model.f1_score(y_test, y_pred)
                save_data = [[model.n_iter,model.lr,model.batch_size,precision,recall,f1]]
                df = pd.DataFrame(save_data, columns=['n_iter','learning_rate','batch_size','precision','recall','f1_score'])
                df.to_csv('./perceptron_results.csv', mode='a', header=False, index=False)
                del model  # 删除模型对象，释放内存
                gc.collect()  # 强制垃圾回收