import numpy as np
import matplotlib.pyplot as plt

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
        #print(f"loss={loss}")
        loss = np.sum(loss) / y.size
        print(loss)
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
        # y_pred = np.where(y_pred > 0, 1, 2)
        for i, y in enumerate(y_pred):
            if y > 0:
                y_pred[i] = 2
            elif y < 0:
                y_pred[i] = 1
        return y_pred

    def stochastic_update(self, X, y):
        break_out = False
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            for i, x in enumerate(X):
                y_pred = self._predict(x)
                print(f"y_pred={y_pred}")
                loss =  self._loss(y[i], y_pred)
                print(f"loss={loss}")
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
                        print(f"grad={grad}")
                        self.W = self.W - self.lr * grad
            if break_out:
                break_out = False
                break
    
    def batch_update(self, X, y):
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            print(f'==========epoch_num={iter+1}===========')
            # print(X.shape)
            y_pred = self._predict(X)
            print(f"y_pred_shape={y_pred.shape}")
            print(f"y_shape={y.shape}")
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
            print(f"current_weight={self.W}")
    

    def train(self, X_train, Y_train):
        X_train_bar = self._preprocess_data(X_train)
        if self.train_mode == 'BGD':
            self.batch_update(X_train_bar, Y_train)
        elif self.train_mode == 'SGD':
            self.stochastic_update(X_train_bar, Y_train)

    def plot_loss(self):
        # print(self.loss)for i, x in enumerate(X):
        plt.plot(self.loss)
        plt.title('loss')
        plt.grid()
        plt.show()
    
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives (1 correctly predicted as 1)
        FP = np.sum((y_true == 2) & (y_pred == 1))  # False Positives (2 incorrectly predicted as 1)
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives (1 correctly predicted as 1)
        FN = np.sum((y_true == 1) & (y_pred == 2))  # False Negatives (1 incorrectly predicted as 2)
        return TP / (TP + FN) if (TP + FN) != 0 else 0
    
    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0




def load_data(file, ratio, random_state = None):
        dataset = np.loadtxt(file, delimiter=',')
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.arange(dataset.shape[0])
        np.random.shuffle(indices)

        split_index = int(dataset.shape[0] * (1 - ratio))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        data_train = dataset[train_indices]
        data_test = dataset[test_indices]

        return data_train, data_test
    

if __name__ == '__main__':
    train_data, test_data = load_data('separate_Perceptron/wine_formed_data', 0.3, True)
    # print(f"train_data:{train_data}\n")
    # print(f"test_data:{test_data}\n")
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    for i,y in enumerate(y_train):
        if y == 2:
            y_train[i] = 1
    for i,y in enumerate(y_train):
        if y == 1:
            y_train[i] = -1
    print(f"y_train={y_train}")
    _,n_feature = X_train.shape
    model = Perceptron(n_feature=n_feature, n_iter=900, lr=0.001, tol=0.0001, train_mode='SGD')
    model.train(X_train, y_train)
    plt.figure()
    model.plot_loss()
    y_pred = model.predict(X_test)
    print(f"predicted_y = {y_pred}")
    print(f"true_y = {y_test}")
    print(y_pred - y_test)