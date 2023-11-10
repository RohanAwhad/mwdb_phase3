import torch
from tqdm import tqdm
import numpy as np

'''
Possible Optimizations for best_splits:
- Histogram-based Splits: Create a histogram or a discrete binning of continuous features and then only consider splitting on bin edges. This is the approach used by algorithms like H2O's Random Forest and LightGBM.
- Approximate Algorithms: For very large datasets, consider using approximate algorithms to find split points. These algorithms work by approximating the best split point rather than finding the exact best split through sorting and evaluating every possible threshold.
- Feature Sampling: At each node, randomly sample a subset of features to consider for splitting, rather than evaluating all features. This reduces the computation per node and can help with generalization by reducing the variance of the model.
- Variance Reduction: Evaluate the variance reduction for each feature as a potential split criterion. Features that contribute little to reducing the overall variance may be less likely to be good candidates for splitting.
'''

class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _gini(self, y):
        """
        Calculate the Gini impurity of a set of labels.

        The Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if
        it was randomly labeled according to the distribution of labels in the subset. It reaches its minimum (zero) when
        all cases in the node fall into a single target category.

        Parameters:
        y (array-like): The set of labels to calculate the Gini impurity for.

        Returns:
        float: The Gini impurity of the set of labels.
        """
        m = len(y)
        return 1.0 - sum((torch.sum(y == c).item() / m) ** 2 for c in torch.unique(y))  # sum of squares of probabilities of each class


    def _best_split(self, X, y):
        """
        Find the best split for a decision tree classifier.

        Parameters:
        X (np.ndarray): The input data.
        y (np.ndarray): The target labels.

        Returns:
        Tuple[int, float]: The index of the best feature and the threshold value for the best split.
        """
        # if X, and y have torch tensors or numpy arrays
        m = X.shape[0]  # Number of samples
        if m <= 1: return None, None

        # Unique classes and their counts
        class_counts = torch.zeros(self.n_classes_)
        for c in torch.unique(y): 
            class_counts[c] = torch.sum(y == c)

        best_gini = 1.0 - torch.sum((class_counts / m) ** 2)
        best_idx, best_thr = None, None

        # Loop through a random subset of features
        k = int(self.n_features_ ** 0.5)

        # find the k features with the highest variance
        # variances = torch.var(X, dim=0)
        # feat_indices = torch.argsort(variances)[-k:]
        # new_X = X[:, feat_indices]

        # find the k features randomly
        feat_indices = np.random.choice(self.n_features_, size=int(np.sqrt(self.n_features_)), replace=False)
        new_X = X[:, feat_indices]

        # Loop through all features
        m_range = torch.arange(1, m)
        unsq_m_range = m_range.reshape(-1, 1)
        gini_right_m_range = m - unsq_m_range
        gini_m_range = m - m_range
        sorted_idx = torch.argsort(new_X, axis=0)
        all_num_left = torch.zeros((k, m, self.n_classes_))
        # _tmp_all_gini_left = torch.zeros((self.n_features_, m-1))
        for idx in tqdm(range(k), desc='Finding Best Split', leave=False):
            # thresholds = X[sorted_idx[:, idx], idx]
            classes = y[sorted_idx[:, idx]]
            all_num_left[idx, m_range, classes[:-1]] = 1

        all_num_left = torch.cumsum(all_num_left, axis=1)
        #print(all_num_left.shape)
        all_num_right = class_counts - all_num_left
        all_num_left = all_num_left[:, 1:, :]
        all_num_right = all_num_right[:, 1:, :]
        all_gini_left = 1.0 - torch.sum((all_num_left / unsq_m_range) ** 2, axis=2)
        all_gini_right = 1.0 - torch.sum((all_num_right / gini_right_m_range) ** 2, axis=2)
        all_gini = (m_range * all_gini_left + gini_m_range * all_gini_right) / m
        #print(f'all_gini shape: {all_gini.shape}')

        x = torch.argmin(all_gini)
        i = x // (m-1)
        j = x % (m-1)
        if all_gini[i, j] < best_gini:
            best_thr = (new_X[sorted_idx[j, i], i] + new_X[sorted_idx[j - 1, i], i]) / 2
            best_idx = feat_indices[i]
        return best_idx, best_thr


    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows a decision tree from the given training data.
        Args:
            X (torch.Tensor): The training data features.
            y (torch.Tensor): The training data labels.
            depth (int): The current depth of the tree.

        Returns:
            DecisionTreeNode: The root node of the decision tree.
        """
        # Population for each class in current node. The predicted class is the one with largest population.
        num_samples_per_class = [torch.sum(y == i).item() for i in range(self.n_classes_)]
        predicted_class = torch.argmax(torch.tensor(num_samples_per_class)).item()
        node = DecisionTreeNode(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            # Determine the best split over all features and thresholds of the current node
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # Split the data
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                # Grow the left and right child recursively
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        else:
            print(f'leaf node at depth {depth} with {len(y)} samples')

        return node

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """
        Predict the class for a given input using the decision tree.

        Parameters:
        inputs (torch.Tensor): The input data.

        Returns:
        int: The predicted class.
        """
        # Start from the root of the decision tree
        current_node = self.tree_

        # Traverse the tree until we reach a leaf node
        while current_node.left:
            # Check the feature at the current node's feature index
            feature_value = inputs[current_node.feature_index]

            # If the feature value is less than the node's threshold, go to the left child.
            # Otherwise, go to the right child.
            if feature_value < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right

        # Once we've reached a leaf node, return its predicted class
        return current_node.predicted_class

if __name__ == '__main__':
    # Example usage:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier as DTC
    from sklearn.metrics import accuracy_score

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print('Sklearn Decision Tree Classifier:')
    clf = DTC(max_depth=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

    print('Custom Decision Tree Classifier:')
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(torch.tensor(X_train), torch.tensor(y_train))
    my_predictions = clf.predict(torch.tensor(X_test))
    print(accuracy_score(y_test, my_predictions))
    # check how many of my_predicitons were for class 0
    print(np.sum(np.array(my_predictions) == 0))
    print(len(my_predictions))