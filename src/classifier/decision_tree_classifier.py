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
        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(y, np.ndarray): y = np.array(y)
        m = X.shape[0]  # Number of samples
        if m <= 1: return None, None

        # Unique classes and their counts
        class_counts = np.zeros(self.n_classes_)
        for c in np.unique(y): class_counts[c] = np.sum(y == c)

        best_gini = 1.0 - sum((n / m) ** 2 for n in class_counts)
        best_idx, best_thr = None, None

        # Loop through a random subset of features
        k = int(np.sqrt(self.n_features_))
        variances = np.var(X, axis=0)
        top_k_indices = np.argsort(variances)[-k:]
        new_X = X[:, top_k_indices]

        #feature_indices = np.random.choice(self.n_features_, size=int(np.sqrt(self.n_features_)), replace=False)
        for idx in tqdm(range(k), desc='Finding Best Split', leave=False):
            feature_values = new_X[:, idx]
            sorted_idx = np.argsort(feature_values)
            class_counts_left = np.zeros(self.n_classes_)
            class_counts_right = class_counts.copy()
            num_left = 0
            num_right = m
            for i in range(1, m):
                c = y[sorted_idx[i - 1]]
                class_counts_left[c] += 1
                class_counts_right[c] -= 1
                num_left += 1
                num_right -= 1
                if feature_values[sorted_idx[i]] == feature_values[sorted_idx[i - 1]]:
                    continue

                gini_left = 1.0 - sum((class_counts_left[x] / num_left) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((class_counts_right[x] / num_right) ** 2 for x in range(self.n_classes_))
                gini = (num_left * gini_left + num_right * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = top_k_indices[idx]
                    best_thr = (feature_values[sorted_idx[i]] + feature_values[sorted_idx[i - 1]]) / 2

        return best_idx, best_thr
    # def _best_split(self, X, y, percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
    #     class_counts = torch.zeros(self.n_classes_)
    #     for c in torch.unique(y):
    #         class_counts[c] = torch.sum(y == c)
        
    #     best_gini = 1.0 - sum((n / len(y)) ** 2 for n in class_counts)
    #     best_idx, best_thr = None, None
        
    #     # Loop through all features
    #     for idx in tqdm(range(self.n_features_)):
    #         feature_values = X[:, idx]
    #         unique_values = torch.unique(feature_values)

    #         # Get threshold candidates based on percentiles
    #         k_list = list(map(int, len(unique_values) * (torch.tensor(percentiles) / 100.0)))
    #         threshold_candidates = unique_values[k_list]
            
    #         for threshold in threshold_candidates:
    #             left_mask = feature_values <= threshold
    #             right_mask = feature_values > threshold
                
    #             left_classes = y[left_mask]
    #             right_classes = y[right_mask]
                
    #             # Compute class counts for left and right splits
    #             left_counts = torch.zeros(self.n_classes_)
    #             right_counts = torch.zeros(self.n_classes_)
    #             for c in torch.unique(y):
    #                 left_counts[c] = torch.sum(left_classes == c)
    #                 right_counts[c] = torch.sum(right_classes == c)
                
    #             # Compute Gini for left and right
    #             m_left = left_classes.size(0)
    #             m_right = right_classes.size(0)
    #             gini_left = 1.0 - sum((n / m_left) ** 2 for n in left_counts if m_left > 0)
    #             gini_right = 1.0 - sum((n / m_right) ** 2 for n in right_counts if m_right > 0)
                
    #             # Weighted Gini for this split
    #             gini = (m_left * gini_left + m_right * gini_right) / (m_left + m_right)
                
    #             # Check if this is the best split so far
    #             if gini < best_gini:
    #                 best_gini = gini
    #                 best_idx = idx
    #                 best_thr = threshold
                    
    #     return best_idx, best_thr


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