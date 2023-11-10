import torch
from tqdm import tqdm

class PersonalizedPageRankClassifier:
    def __init__(self, alpha=0.85, max_iter=100, tol=1e-6):
        self.alpha = alpha  # Damping factor for PageRank
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.graph = None  # Placeholder for the graph
        self.personalization_vectors = None  # Personalization vectors for each class
        self.classes_ = None  # Unique classes in the dataset
        self.page_rank_matrix = None  # Store PageRank scores for each class
        self.X_train = None  # Store the training data


    def fit(self, X, y):
        # Create a graph based on feature similarity
        self.X_train = X
        self.classes_ = torch.unique(y)
        self.graph = self._create_graph(X)
        self.personalization_vectors = self._create_personalization_vectors(y)


    def predict(self, X):
        # Add new points to the graph and compute their PageRank scores
        predictions = []
        for x in tqdm(X, desc='Predicting', leave=False):
            # Compute similarity to existing points and update graph
            updated_graph = self._update_graph(self.graph, x)
            scores = torch.stack([self._page_rank(updated_graph, torch.concatenate((personal_vector, torch.tensor([0]))))
                               for personal_vector in self.personalization_vectors])
            # Take the class with the highest PageRank score as the prediction
            predicted_class = self.classes_[scores[:, -1].argmax()]
            predictions.append(predicted_class)
        return predictions

    def _create_graph(self, X):
        # Calculate the cosine similarity matrix which will serve as our adjacency matrix
        similarity_matrix = torch.mm(X, X.t()) / (torch.norm(X, dim=1, keepdim=True) * torch.norm(X, dim=1, keepdim=True).t())

        # For the PageRank algorithm, we don't want self-loops, 
        # so we set the diagonal to zero.
        similarity_matrix.fill_diagonal_(0)

        # Return the stochastic matrix as the adjacency matrix representation of the graph
        return similarity_matrix

    def _create_personalization_vectors(self, y):
        # Initialize an array to hold the personalization vectors, one for each class
        personalization_vectors = torch.zeros((len(self.classes_), len(y)))

        # Assign higher weights for nodes belonging to the corresponding class
        for i, class_label in enumerate(self.classes_):
            personalization_vectors[i] = (y == class_label).int()

        # Normalize the vectors so they sum up to 1, making them proper probability distributions
        personalization_vectors = personalization_vectors / personalization_vectors.sum(dim=1, keepdim=True)
        
        return personalization_vectors

    def _page_rank(self, graph, personalization_vector):
        # Get the transition matrix
        M = graph / graph.sum(dim=1, keepdim=True)

        # Handle division by zero in case there are rows with all zeros
        M[torch.isnan(M)] = 1.0 / graph.shape[1]

        # convert dtype to float32
        M = M.float()

        # Number of nodes
        N = graph.shape[0]

        # Initialize PageRank vector
        R = torch.ones(N) / N

        # Compute the teleporting vector based on personalization
        S = personalization_vector

        for _ in range(self.max_iter):
            R_next = self.alpha * torch.matmul(M.T, R) + (1 - self.alpha) * S

            # Check for convergence
            if torch.norm(R_next - R, 1) <= self.tol:
                break

            R = R_next
        return R

    def _update_graph(self, graph, x):
        x_new = x.reshape(1, -1)  # Reshape x_new to fit the expected input shape for similarity computation
        #new_similarities = cosine_similarity(x_new, self.X_train_).flatten()
        new_similarities = torch.mm(x_new, self.X_train.t()) / (torch.norm(x_new, dim=1, keepdim=True) * torch.norm(self.X_train, dim=1, keepdim=True).t())
        new_similarities = new_similarities.flatten()

        # Create a new row for the extended adjacency matrix
        new_row = torch.zeros((1, graph.shape[0] + 1))  # Plus one for the new point itself
        new_row[0, :-1] = new_similarities  # Set all but the last column of the new row

        # Create a new column for the extended adjacency matrix, which also includes the self-link (0)
        new_column = torch.zeros((graph.shape[0], 1))
        new_column[:, 0] = new_similarities  # Set all but the last row of the new column

        # Add the new row and column to the existing graph to form the updated graph
        updated_graph = torch.cat((torch.cat((graph, new_column), dim=1), new_row), dim=0)

        # Make sure the last entry (new self-link) is zero
        updated_graph[-1, -1] = 0

        return updated_graph


if __name__ == '__main__':
    # Usage Example:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # convert to tensors
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)


    # Create and train the classifier
    ppr_classifier = PersonalizedPageRankClassifier()
    ppr_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = ppr_classifier.predict(X_test)
    print(accuracy_score(y_test, predictions))
