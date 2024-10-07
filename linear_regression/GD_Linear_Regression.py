import numpy as np
import matplotlib.pyplot as plt

class GDLinearRegression:
    def __init__(self, n_feature=1, n_iter=200, lr=1e-3, tol=None, train_mode='BGD', batch_size=None, normalization=None):
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.norm_mode = normalization
        self.W = np.random.random(n_feature+1) * 0.05
        self.loss = []

    def _MSEloss(self, y, y_pred):
        return np.sum((y_pred - y) **2 ) / y.size
    
    def _gradient(self, X, y, y_pred):
        return (y_pred - y) @ X / y.size
    
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
        return y_pred

    def batch_update(self, X, y):

        if self.tol is not None:
            loss_old = np.inf

        for iter in range(self.n_iter):
            print(f'==========epoch_num={iter+1}===========')
            y_pred = self._predict(X)
            loss = self._MSEloss(y, y_pred)
            self.loss.append(loss)

            if self.tol is not None:
                if np.abs(loss_old - loss) < self.tol:
                    break
                loss_old = loss
        
            grad = self._gradient(X, y, y_pred)
            self.W = self.W - self.lr * grad
            print(f'current_weight={self.W}\n')

    def stochastic_update(self, init_X, init_y):

        if self.tol is not None:
            loss_old = np.inf

        for iter in range(self.n_iter):
            print(f'==========epoch_num={iter+1}===========')
            X, y = self._shuffle(init_X, init_y)
            sample = X[0,:].reshape(-1,2)
            y = y[0,:]
            y_pred = self._predict(sample).reshape(-1)
            loss = self._MSEloss(y, y_pred)
            self.loss.append(loss)

            if self.tol is not None:
                if np.abs(loss_old - loss) < self.tol:
                    break
                loss_old = loss
        
            grad = self._gradient(sample, y, y_pred)
            self.W = self.W - self.lr * grad
            print(f'current_weight={self.W}\n')
    
    def mini_batch_update(self, init_X, init_y):

        batch_size = self.batch_size
        print(f'batchsize={batch_size}')

        if self.tol is not None:
            loss_old = np.inf

        for iter in range(self.n_iter):
            X, y = self._shuffle(init_X, init_y)
            print(f'==========epoch_num={iter+1}===========')
            for j in range(int(X.shape[0]/batch_size)):
                print(f'----------batch_num={j+1}-----------')
                sample = X[j*batch_size:(j+1)*batch_size,:].reshape(-1,2)
                y_sample = y[j*batch_size:(j+1)*batch_size,:].reshape(-1)
                y_pred = self._predict(sample).reshape(-1)
                loss = self._MSEloss(y_sample, y_pred)
                self.loss.append(loss)

                if self.tol is not None:
                    if np.abs(loss_old - loss) < self.tol:
                        break
                    loss_old = loss
                grad = self._gradient(sample, y_sample, y_pred)
                self.W = self.W - self.lr * grad
                print(f'current_weight={self.W}\n')

    def train(self, X_train, Y_train):
        X_train = self._preprocess_data(X_train)
        if self.train_mode == 'BGD':
            self.batch_update(X_train, Y_train)
        elif self.train_mode == 'SGD':
            self.stochastic_update(X_train, Y_train)
        elif self.train_mode == 'MBGD':
            self.mini_batch_update(X_train, Y_train)

    def plot_loss(self):
        # print(self.loss)
        plt.plot(self.loss)
        plt.title('loss')
        plt.grid()
        plt.show()

if __name__ == '__main__':

    X_train = np.arange(100).reshape(100,1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
    y_train = y_train.reshape(-1)
    _, n_feature = X_train.shape
    # print(n_feature)
    # print(y_train.size)

    bgd_lreg = GDLinearRegression(n_feature=n_feature, n_iter=300, lr=0.0001, tol=0.01, train_mode='BGD', normalization='None')
    bgd_lreg.train(X_train, y_train)
    bgd_lreg.plot_loss()
    print(f'Learned weights are {bgd_lreg.W}')
    y_pred = bgd_lreg.predict(X_train)

    # sgd_lreg = GDLinearRegression(n_feature=n_feature, n_iter=300, lr=0.0001, tol=0.1, train_mode='SGD', normalization='None')
    # sgd_lreg.train(X_train, y_train)
    # sgd_lreg.plot_loss()
    # print(f'Learned weights are {sgd_lreg.W}')
    # y_pred = sgd_lreg.predict(X_train)

    # mbgd_lreg = GDLinearRegression(n_feature=n_feature, n_iter=300, lr=0.01, tol=0.1, train_mode='MBGD', batch_size=10, normalization='min_max')
    # mbgd_lreg.train(X_train, y_train)
    # mbgd_lreg.plot_loss()
    # print(f'Learned weights are {mbgd_lreg.W}')
    # y_pred = mbgd_lreg.predict(X_train)

    plt.figure()
    plt.scatter(X_train, y_train, label="Actual Data")
    plt.plot(X_train, y_pred, label="Predicted Data", c='red')
    plt.title("Linear Regression")
    plt.xlabel("X_train")
    plt.ylabel("y")
    plt.legend()
    plt.show()