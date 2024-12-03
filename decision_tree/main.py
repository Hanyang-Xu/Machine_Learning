import numpy as np
from plotter import DecisionTreePlotter

class DecisionTree:
    class Node:
        def __init__(self):
            self.value = None
            self.feature_index = None
            self.children = {}
        
        def __str__(self) -> str:
            if self.children:
                s = f"Internal node <{self.feature_index}>:\n"
                for fv, node in self.children.items():
                    ss = f"[{fv}]->{node}"
                    s += "\t" + ss.replace("\n", "\n\t") + "\n"
            else:
                s = f"Leaf node({self.value})"

            return s
        
    def __init__(self, gain_threshhold = 1e-2) -> None:
        self.gain_threshold = gain_threshhold
    
    def _entropy(self, y):
        count_y = np.bincount(y)
        prob_y = count_y[np.nonzero(count_y)] / y.size
        entropy_y = -np.sum(prob_y * np.log2(prob_y))
        return entropy_y
    
    def _conditional_entropy(self, feature, y):
        feature_values = np.unique(feature)
        h = 0.
        for v in feature_values:
            y_sub = y[feature == v]
            prob_y_sub = y_sub.size / y.size
            h += prob_y_sub * self._entropy(y_sub)
        return h
    
    def _information_gain(self, feature, y):
        ig_feature = self._entropy(y) - self._conditional_entropy(feature, y)
        return ig_feature
    
    def _select_feature(self, X, y, feature_list):
        if feature_list:
            gains = np.apply_along_axis(self._information_gain, 0, X[:, feature_list], y)
            index = np.argmax(gains)
            if gains[index] > self.gain_threshold:
                return index
        return None
    
    def _build_tree(self, X, y, feature_list):
        node = DecisionTree.Node()
        labels_count = np.bincount(y)
        node.value = np.argmax(np.bincount(y))

        if np.count_nonzero(labels_count) != 1:
            index = self._select_feature(X, y, feature_list)
            if index is not None:
                node.feature_index = feature_list.pop(index)
                feature_values = np.unique(X[:,node.feature_index])
                for v in feature_values:
                    idx = X[:, node.feature_index] == v
                    X_sub, y_sub = X[idx], y[idx]
                    node.children[v] = self._build_tree(X_sub, y_sub, feature_list.copy())
        return node
    
    def train(self, X_train, y_train):
        _, n = X_train.shape
        self.tree_ = self._build_tree(X_train, y_train, list(range(n)))

    def _predict_one(self, x):
        node = self.tree_
        while node.children:
            child = node.children.get(x[node.feature_index])
            if not child:
                break
            node = child
        return node.value
    
    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)
    
    def __str__(self):
        if hasattr(self, 'tree_'):
            return str(self.tree_)
        return ''
    
    def reset(self):
        self.tree_ = None
    
def load_data(file, ratio, random_state = None):
        dataset = np.loadtxt(file, dtype=int)
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.arange(dataset.shape[0])
        np.random.shuffle(indices)

        split_index = int(dataset.shape[0] * (1 - ratio))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        train_data = dataset[train_indices]
        test_data = dataset[test_indices]
        X_train = train_data[:, 1:-1]
        y_train = train_data[:, -1]
        X_test = test_data[:, 1:-1]
        y_test = test_data[:, -1]
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    # for n in range(5):
    #     ratio = 0.1 + 0.1*n
    #     # print(ratio)
    #     accuracy_list = []
    #     for i in range(1000):
            # X_train, X_test, y_train, y_test = load_data('decision_tree/lenses/lenses.data', ratio)
            # dt01 = DecisionTree()
            # dt01.reset()
            # dt01.train(X_train,y_train)
            # y_pred = dt01.predict(X_test)
            # accuracy = np.sum(y_pred == y_test) / len(y_test)
    #         accuracy_list.append(accuracy)
    #     print("ratio = {:.2f}, accuracy = {:.2f}".format(ratio, np.mean(accuracy_list)))
        
        
    X_train, X_test, y_train, y_test = load_data('decision_tree/lenses/lenses.data', 0.3)
    dt01 = DecisionTree()
    dt01.reset()
    dt01.train(X_train,y_train)
    y_pred = dt01.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(accuracy)
    feature_names = [
    {'name': 'age of the patient', 'value_names': {1: 'young', 2: 'pre-presbyopic', 3: 'presbyopic'}},
    {'name': 'spectacle prescription', 'value_names': {1: 'myope', 2: 'hypermetrope'}},
    {'name': 'astigmatic', 'value_names': {1: 'no', 2: 'yes'}},
    {'name': 'tear production rate', 'value_names': {1: 'reduced', 2: 'normal'}}]

    label_names = ['_','hard contact lenses', 'soft contact lenses', 'no contact lenses']
    plotter = DecisionTreePlotter(dt01.tree_, feature_names, label_names)
    plotter.plot()
          