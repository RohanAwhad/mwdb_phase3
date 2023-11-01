import torch

class NearestNeighborClassifier:
    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        self.X = X.clone()
        self.y = y.clone()

    def predict(self, X_test):
        if not isinstance(X_test, torch.Tensor): X_test = torch.tensor(X_test)
        y_pred = []
        for x_test in X_test:
            if x_test.ndimension() == 1: x_test = x_test.unsqueeze(0)
            # Find the nearest neighbor
            nn = self.find_nearest_neighbor(x_test)
            # Predict the label of the nearest neighbor
            y_nn = self.y[nn]
            # Find the most frequent label
            label = torch.argmax(torch.bincount(y_nn))
            y_pred.append(label)
        return y_pred

    def find_nearest_neighbor(self, x_test):
        # Calculate the distance between x_test and each row in X
        distances = torch.norm(self.X - x_test, dim=1)
        # Find the m-nearest neighbor
        return torch.argsort(distances)[:self.m]

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # test with sklearn nn classifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print('Sklearn Nearest Neighbor Classifier:')
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

    print('Custom Nearest Neighbor Classifier:')
    clf = NearestNeighborClassifier(m=3)
    clf.fit(torch.tensor(X_train), torch.tensor(y_train))
    my_predictions = clf.predict(torch.tensor(X_test))
    print(accuracy_score(y_test, my_predictions))
