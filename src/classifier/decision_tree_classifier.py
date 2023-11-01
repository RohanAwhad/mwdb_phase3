import numpy as np

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
    def __init__(self, max_depth=None):
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
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))  # sum of squares of probabilities of each class

    def _best_split(self, X, y):
        """
        Find the best split for a decision tree classifier.

        Parameters:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The target labels.

        Returns:
        Tuple[int, float]: The index of the best feature and the threshold value for the best split.
        """
        m = X.shape[0]  # Number of samples
        if m <= 1:
            return None, None

        # Unique classes and their counts
        class_counts = {c: np.sum(y == c) for c in np.unique(y)}
        best_gini = 1.0 - sum((n / m) ** 2 for n in class_counts.values())
        best_idx, best_thr = None, None

        # Loop through all features
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # Naive algorithm: Iterate through all thresholds and count how many samples would be
            num_left = {c: 0 for c in class_counts}
            num_right = class_counts.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # gini impurity calculation
                gini_left = 1.0 - sum((num_left[cl] / i) ** 2 for cl in num_left)
                gini_right = 1.0 - sum((num_right[cr] / (m - i)) ** 2 for cr in num_right)
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows a decision tree from the given training data.

        Args:
            X (numpy.ndarray): The training data features.
            y (numpy.ndarray): The training data labels.
            depth (int): The current depth of the tree.

        Returns:
            DecisionTreeNode: The root node of the decision tree.
        """
        # Population for each class in current node. The predicted class is the one with largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
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
        inputs (numpy.ndarray): The input data.

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
    clf.fit(X_train, y_train)
    my_predictions = clf.predict(X_test)
    print(accuracy_score(y_test, my_predictions))

    print(list(zip(predictions, my_predictions)))