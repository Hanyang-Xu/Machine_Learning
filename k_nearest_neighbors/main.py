from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.kdtree = None
    
    def _predict(self,x):
        dist,idx = self.kdtree.query(x,k=self.k,p=2)
        if idx.size == 1:
            idx = [idx]
        neighbors_labels =[self.y_train[i] for i in idx]
        prediction = max(set(neighbors_labels),key=neighbors_labels.count)
        return prediction
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
def load_data(file, ratio, random_state = None):
        df = pd.read_csv(file, delimiter=',')
        shuffled_df = df.sample(frac=1, random_state=random_state)
        split_index = int(shuffled_df.shape[0] * (1 - ratio))
        train_data = shuffled_df.iloc[:split_index]
        test_data = shuffled_df.iloc[split_index:]
        X_train = train_data.iloc[:,2:].to_numpy()
        y_train = train_data.iloc[:,1].to_numpy()
        X_test = test_data.iloc[:,2:].to_numpy()
        y_test = test_data.iloc[:,1].to_numpy()
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test= load_data('k_nearest_neighbors/wdbc.data', 0.3, 41)
    accuracy_list = []
    for i in range(10):
        knn = KNN(k = i+1)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        accuracy_list.append(accuracy)
        print(f"k={i+1} Accuracy: {accuracy * 100:.2f}%")

    print(accuracy_list)
    fig = plt.figure()
    plt.plot(accuracy_list)
    plt.xlabel("value of k")  
    plt.ylabel("accuracy")
    plt.ylim(0.8, 1)
    plt.show()
