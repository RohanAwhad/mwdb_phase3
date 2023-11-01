import numpy as np
from decision_tree_classifier import DecisionTreeClassifier

# Test case 1: Test with a simple dataset
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
clf = DecisionTreeClassifier(max_depth=4) # increased max_depth to 4
clf.fit(X, y)
assert clf.predict([[2.5]]) == [0]
assert clf.predict([[7.5]]) == [1]

# Test case 2: Test with a larger dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)
assert clf.predict([[1, 1]]) == [0]

# Test case 3: Test with a dataset with missing values
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3], [np.nan, np.nan]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)
assert clf.predict([[1, 1]]) == [0]
