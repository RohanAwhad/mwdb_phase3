"""
Task 2: Implement a program which,

  - for each unique label l, computes the correspending c most significant clusters associated with the even numbered Caltec101 images (using DBScan algorithm); the resulting clusters should be visualized both
    * as differently colored point clouds in a 2-dimensional MDS space, and
    * as groups of image thumbnails. and 
  - for the odd numbered images, predicts the most likely labels using the c label-specific clusters.

The system should also output per-label precision, recall, and F1-score values as well as output an overall accuracy value.

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import seaborn as sns
import torch
import torchvision

from collections import Counter, defaultdict
from torchvision.transforms import functional as TF
from tqdm import tqdm

N_CLUSTERS = int(input("Enter number of clusters: "))
K = 1024
VERSION = 6
ARTIFACTS_DIR = f"artifacts_{VERSION}"
OUTPUTS_DIR = f"outputs_{VERSION}"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(f"{OUTPUTS_DIR}/task_2", exist_ok=True)
os.makedirs(f"{OUTPUTS_DIR}/task_2_svd_{K}_space_C_{N_CLUSTERS}", exist_ok=True)

TORCH_HUB = "./models/"
torch.set_grad_enabled(False)
torch.hub.set_dir(TORCH_HUB)

DATA_DIR = "./data/caltech101"
DATASET = torchvision.datasets.Caltech101(DATA_DIR, download=False)

MODEL_NAME = "ResNet50_Weights.DEFAULT"
RESNET_MODEL = torchvision.models.resnet50(weights=MODEL_NAME).eval()

LABEL_TO_CLUSTER_CENTROIDS_PATH = (
    f"{ARTIFACTS_DIR}/label_to_cluster_centroids_C_{N_CLUSTERS}_svd_{K}.pkl"
)
LATENT_FEAT_PATH = f"{ARTIFACTS_DIR}/resnet50_avgpool_features_svd_{K}.pkl"
LABEL_TO_MDS_PATH = f"{ARTIFACTS_DIR}/label_to_mds_C_{N_CLUSTERS}_svd_{K}.pkl"
LABEL_TO_DBSCAN_PARAMS_PATH = (
    f"{ARTIFACTS_DIR}/label_to_dbscan_params_grid_search_C_{N_CLUSTERS}_svd_{K}.pkl"
)
LABEL_TO_CLUSTERS_PATH = f"{ARTIFACTS_DIR}/label_to_clusters_C_{N_CLUSTERS}_svd_{K}.pkl"
ODD_IMG_FEAT_PATH = f"{ARTIFACTS_DIR}/odd_img_features_svd_{K}.pkl"
ODD_IMG_LABELS_PATH = f"{ARTIFACTS_DIR}/odd_image_labels.pkl"


class SaveOutput:
    def __init__(self):
        self.output = []

    def __call__(self, module, inp, out):
        self.output.append(out)

    def __getitem__(self, index):
        return self.output[index]

    def clear(self):
        self.output = []


class FeatureExtractor:
    def __init__(self):
        self.model = RESNET_MODEL
        self.hook = SaveOutput()
        self.handle = self.model.get_submodule("avgpool").register_forward_hook(
            self.hook
        )

    def __call__(self, img):
        self.hook.clear()
        x = TF.resize(img, size=(224, 224))
        x = TF.to_tensor(x).unsqueeze(0)
        self.model(x)
        avgpool_features = self.hook[0].view(-1, 2).mean(-1).flatten()
        self.hook.clear()
        return avgpool_features


class SVD:
    def __init__(self):
        self.components_ = None

    def fit(self, A: torch.Tensor, K: int):
        eigen_values2, V = map(torch.tensor, np.linalg.eig(A.T @ A))
        tmp_2, v_tmp = [], []
        for ev, v in sorted(zip(eigen_values2, V.T), key=lambda x: x[0], reverse=True):
            tmp_2.append(ev)
            v_tmp.append(v)

        eigen_values2, V = torch.tensor(tmp_2), torch.stack(v_tmp).T
        eigen_values2, V = eigen_values2[:K], V[:, :K]
        self.components_ = V.T

    def transform(self, feats: torch.Tensor):
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        return feats @ self.components_.T

    def fit_transform(self, A: torch.Tensor, K: int):
        self.fit(A, K)
        return self.transform(A)


extractor = FeatureExtractor()
# svd = SVD()

if os.path.exists("artifacts/resnet50_avgpool_features.pkl"):
    features = torch.load("artifacts/resnet50_avgpool_features.pkl")
    labels = torch.load("artifacts/labels.pkl")
    image_ids = torch.load("artifacts/image_ids.pkl")
else:
    features = []
    labels = []
    image_ids = []

    for idx, (image, label) in tqdm(
        enumerate(DATASET), desc="Extracting Features", total=len(DATASET), leave=False
    ):
        if idx % 2:
            continue  # Skipping odd images

        if image.mode != "RGB":
            image = image.convert("RGB")
        feature = extractor(image)  # Assuming extractor expects batched input
        features.append(feature)  # Flattening and converting to numpy
        labels.append(label)
        image_ids.append(idx)

    features = torch.stack(features)
    torch.save(features, "artifacts/resnet50_avgpool_features.pkl")
    torch.save(labels, "artifacts/labels.pkl")
    torch.save(image_ids, "artifacts/image_ids.pkl")

# if os.path.exists(LATENT_FEAT_PATH):
#     features, svd = torch.load(LATENT_FEAT_PATH)
# else:
#     features = svd.fit_transform(features, K=K)
#     torch.save((features, svd), LATENT_FEAT_PATH)

label_to_feat_idx = defaultdict(list)
for idx, label in enumerate(labels):
    label_to_feat_idx[label].append(idx)

label_to_dataset_idx = defaultdict(list)
for label, idx in zip(labels, image_ids):
    label_to_dataset_idx[label].append(idx)


def z_score_normalization(features):
    # Calculate the mean and standard deviation for each feature
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)

    # Apply the normalization: (feature - mean) / std_dev
    normalized_features = (features - means) / std_devs
    return normalized_features, means, std_devs


def dbscan(X, eps, min_samples):
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise

    Parameters:
    X: ndarray, shape (n_samples, n_features)
        The input samples.
    eps: float
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples: int
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    Returns:
    labels: ndarray, shape (n_samples,)
        Cluster labels for each point. Noisy samples are given the label -1.
    """

    # Initialize labels to -1 (indicating noise)
    labels = -1 * np.ones(X.shape[0], dtype=int)

    # Cluster label
    C = -1

    # Iterate over all points
    for i in range(X.shape[0]):
        if labels[i] != -1:
            continue

        # Find neighbors using euclidean distance
        neighbors = np.where(np.linalg.norm(X - X[i], axis=1) < eps)[0]

        # Mark as noise and continue if not enough neighbors
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        # Start a new cluster
        C += 1
        labels[i] = C

        # Process every point in the neighborhood
        k = 0
        while k < len(neighbors):
            p = neighbors[k]

            if labels[p] == -1:
                labels[p] = C  # Change noise to border point
            elif labels[p] != -1:
                k += 1
                continue
            # Add point to cluster
            labels[p] = C

            # Get new neighbors and add them to the list
            p_neighbors = np.where(np.linalg.norm(X - X[p], axis=1) < eps)[0]
            if len(p_neighbors) >= min_samples:
                # remove duplicates
                p_neighbors = np.setdiff1d(p_neighbors, neighbors)
                neighbors = np.append(neighbors, p_neighbors)

            k += 1

    return labels


def find_dbscan_params(
    X,
    initial_eps,
    initial_min_samples,
    target_clusters,
    tolerance=1,
    eps_step=0.05,
    min_samples_step=1,
    max_iterations=100,
):
    """
    Adjust DBSCAN parameters to reach the target number of clusters within a tolerance.

    Parameters:
    X (array-like): The input data for clustering.
    initial_eps (float): Initial epsilon value for DBSCAN.
    initial_min_samples (int): Initial min_samples value for DBSCAN.
    target_clusters (int): Target number of clusters.
    tolerance (int): Tolerance for the number of clusters.
    eps_step (float): Step size for adjusting epsilon.
    min_samples_step (int): Step size for adjusting min_samples.
    max_iterations (int): Maximum number of iterations for adjustment.

    Returns:
    tuple: (best_eps, best_min_samples) after adjustment.
    """
    eps = initial_eps
    min_samples = initial_min_samples
    iteration = 0
    update_eps_flag = True

    while iteration < max_iterations:
        labels = dbscan(X, eps=eps, min_samples=min_samples)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if target_clusters - tolerance <= n_clusters <= target_clusters + tolerance:
            return eps, min_samples

        if n_clusters < target_clusters:
            if update_eps_flag:
                eps -= eps_step
                eps = max(eps, 0.01)  # Prevent eps from becoming non-positive
            else:
                min_samples = max(1, min_samples - min_samples_step)
        else:
            if update_eps_flag:
                eps += eps_step
            else:
                min_samples += min_samples_step

        update_eps_flag = not update_eps_flag  # Toggle the flag
        iteration += 1

    return eps, min_samples


def random_search_dbscan_params(
    X, target_clusters, iterations=100, eps_range=(0.1, 1.0), min_samples_range=(2, 10)
):
    """
    Perform a random search to find suitable DBSCAN parameters for a target number of clusters.

    Parameters:
    X: ndarray
        Input data for clustering.
    target_clusters: int
        Desired number of clusters.
    iterations: int
        Number of iterations for random search.
    eps_range: tuple
        Range (min, max) for eps.
    min_samples_range: tuple
        Range (min, max) for min_samples.

    Returns:
    best_eps: float
        Best found eps value.
    best_min_samples: int
        Best found min_samples value.
    """

    best_eps = None
    best_min_samples = None
    best_diff = float("inf")
    best_noise = float("inf")
    best_score = float("-inf")
    best_num_clusters = 0

    for _ in range(iterations):
        eps = random.uniform(*eps_range)
        min_samples = random.randint(*min_samples_range)

        labels = dbscan(X, eps=eps, min_samples=min_samples)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        (
            best_eps,
            best_min_samples,
            best_score,
            best_diff,
            best_num_clusters,
        ) = calculate_best_params(
            X,
            labels,
            num_clusters,
            target_clusters,
            eps,
            min_samples,
            best_eps,
            best_min_samples,
            best_score,
            best_diff,
            best_num_clusters,
        )
        # get count of -1 labels
        num_noise = np.sum(labels == -1)

        # diff = abs(num_clusters - target_clusters)
        # if diff < best_diff:
        #     best_eps, best_min_samples, best_diff, best_noise = (
        #         eps,
        #         min_samples,
        #         diff,
        #         num_noise,
        #     )
        # elif diff == best_diff:
        #     if num_noise < best_noise:
        #         best_eps, best_min_samples, best_diff, best_noise = (
        #             eps,
        #             min_samples,
        #             diff,
        #             num_noise,
        #         )
        #     elif num_clusters > target_clusters:
        #         best_eps, best_min_samples, best_diff, best_noise = (
        #             eps,
        #             min_samples,
        #             diff,
        #             num_noise,
        #         )
        #     elif min_samples > best_min_samples:
        #         best_eps, best_min_samples, best_diff, best_noise = (
        #             eps,
        #             min_samples,
        #             diff,
        #             num_noise,
        #         )

    print(
        f"the selected params have a silhouette score of {best_score}, and num_clusters: {best_num_clusters}"
    )
    return best_eps, best_min_samples


from sklearn.metrics.pairwise import euclidean_distances


def dunn_index(X, labels):
    """
    Compute the Dunn Index for a given set of points and cluster labels.

    Parameters:
    X (array-like): The input data points.
    labels (array-like): The cluster labels for each point in X.

    Returns:
    float: The Dunn Index.
    """
    unique_clusters = np.unique(labels)
    if len(unique_clusters) == 1:
        return np.inf  # Only one cluster exists

    # Compute intra-cluster distances (maximum distance within each cluster)
    intra_cluster_dists = np.array(
        [np.max(euclidean_distances(X[labels == k])) for k in unique_clusters]
    )

    # Compute inter-cluster distances (minimum distance between clusters)
    inter_cluster_dists = np.inf
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            dist = np.min(
                euclidean_distances(
                    X[labels == unique_clusters[i]], X[labels == unique_clusters[j]]
                )
            )
            if dist < inter_cluster_dists:
                inter_cluster_dists = dist

    return inter_cluster_dists / np.max(intra_cluster_dists)


from sklearn.metrics import davies_bouldin_score


def compute_davies_bouldin_index(X, labels):
    """
    Compute the Davies-Bouldin Index for a given set of points and a DBSCAN model.

    Parameters:
    X (array-like): The input data points.

    Returns:
    float: The Davies-Bouldin Index.
    """
    # Handling the case of a single cluster or no clusters
    if len(np.unique(labels)) < 2:
        return np.inf

    return -1 * davies_bouldin_score(X, labels)


from sklearn.metrics import silhouette_score


def silhouette_score_(X, labels):
    """
    Compute the silhouette score for each sample of X given the cluster labels.

    Parameters:
    X (array-like): NumPy array of data points.
    labels (array-like): Cluster labels for each point in X.

    Returns:
    float: The mean silhouette score for all samples.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        # Silhouette score is not well-defined for a single cluster or if each data point is its own cluster
        raise ValueError("Silhouette score is not defined for a single cluster.")

    # Calculate pairwise distances
    dist_matrix = np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))

    # Calculate a(i)
    a_vals = np.array(
        [
            np.mean(
                [
                    dist_matrix[i, j]
                    for j in range(len(X))
                    if labels[i] == labels[j] and i != j
                ]
            )
            for i in range(len(X))
        ]
    )

    # Calculate b(i)
    b_vals = np.array(
        [
            np.min(
                [
                    np.mean(dist_matrix[i, labels == label])
                    for label in unique_labels
                    if label != labels[i]
                ]
            )
            for i in range(len(X))
        ]
    )

    # Calculate silhouette values
    s_vals = (b_vals - a_vals) / np.maximum(a_vals, b_vals)

    # Return the mean silhouette score
    return np.mean(s_vals)


# def calculate_best_params(
#     X,
#     labels,
#     num_clusters,
#     target_clusters,
#     eps,
#     min_samples,
#     best_eps,
#     best_min_samples,
#     best_score,
#     best_diff,
#     best_num_clusters,
# ):
#     # print(f"best_score: {best_score}")
#     if best_eps is None and best_min_samples is None:
#         return eps, min_samples, best_score, best_diff, num_clusters
#     # Adjust if the number of clusters is less than the target, ignore this run
#     diff = abs(num_clusters - target_clusters)
#     print(f"diff: {diff}")

#     if num_clusters < target_clusters and best_num_clusters >= num_clusters:
#         return best_eps, best_min_samples, best_score, best_diff, best_num_clusters

#     # Prefer solutions that meet or exceed the target number of clusters
#     if num_clusters >= target_clusters:
#         if diff < best_diff or (diff == best_diff and num_clusters > target_clusters):
#             best_eps, best_min_samples, best_diff, best_num_clusters = (
#                 eps,
#                 min_samples,
#                 diff,
#                 num_clusters,
#             )

#         if num_clusters > 1:  # Silhouette score is only valid if num_clusters > 1
#             try:
#                 score = silhouette_score(X, labels)
#             except ValueError:
#                 return (
#                     best_eps,
#                     best_min_samples,
#                     best_score,
#                     best_diff,
#                     best_num_clusters,
#                 )

#             # Prefer higher silhouette score among those that meet or exceed target clusters
#             if num_clusters >= target_clusters and (
#                 diff < best_diff or (diff == best_diff and score > best_score)
#             ):
#                 best_eps, best_min_samples, best_score, best_diff, best_num_clusters = (
#                     eps,
#                     min_samples,
#                     score,
#                     diff,
#                     num_clusters,
#                 )

#     return best_eps, best_min_samples, best_score, best_diff, best_num_clusters


def calculate_best_params(
    X,
    labels,
    num_clusters,
    target_clusters,
    eps,
    min_samples,
    best_eps,
    best_min_samples,
    best_score,
    best_diff,
    best_num_clusters,
):
    # only calculate silhouette score if num_clusters > 1
    if num_clusters > 1:
        try:
            score = silhouette_score(X, labels)
            # score = dunn_index(X, labels)
            # score = -1 * davies_bouldin_score(X, labels)
            if score > best_score:
                return (
                    eps,
                    min_samples,
                    score,
                    best_diff,
                    num_clusters,
                )
        except ValueError:
            pass
    return (
        best_eps,
        best_min_samples,
        best_score,
        best_diff,
        best_num_clusters,
    )


def grid_search_dbscan_params(
    X, target_clusters, eps_range=(0.1, 1.0), min_samples_range=(2, 10), iterations=100
):
    """
    Perform a grid search to find suitable DBSCAN parameters for a target number of clusters.

    Parameters:
    X: ndarray
        Input data for clustering.
    target_clusters: int
        Desired number of clusters.
    eps_range: tuple
        Range (min, max) for eps.
    min_samples_range: tuple
        Range (min, max) for min_samples.

    Returns:
    best_eps: float
        Best found eps value.
    best_min_samples: int
        Best found min_samples value.
    """

    best_eps = None
    best_min_samples = None
    best_diff = float("inf")
    best_score = float("-inf")
    best_num_clusters = 0

    min_samples_values = list(range(min_samples_range[0], min_samples_range[1] + 1))
    eps_values = np.linspace(
        eps_range[0], eps_range[1], num=iterations // len(min_samples_values)
    )

    # Call the function inside the loop
    for eps in tqdm(eps_values, desc="Grid Search", leave=False):
        for min_samples in min_samples_values:
            labels = dbscan(X, eps=eps, min_samples=min_samples)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            (
                best_eps,
                best_min_samples,
                best_score,
                best_diff,
                best_num_clusters,
            ) = calculate_best_params(
                X,
                labels,
                num_clusters,
                target_clusters,
                eps,
                min_samples,
                best_eps,
                best_min_samples,
                best_score,
                best_diff,
                best_num_clusters,
            )

            # # diff = abs(num_clusters - target_clusters)
            # # if diff < best_diff:
            # #     best_eps, best_min_samples, best_diff, best_noise = (
            # #         eps,
            # #         min_samples,
            # #         diff,
            # #         num_noise,
            # #     )
            # # elif diff == best_diff:
            #     if num_noise < best_noise:
            #         best_eps, best_min_samples, best_diff, best_noise = (
            #             eps,
            #             min_samples,
            #             diff,
            #             num_noise,
            #         )
            #     elif num_clusters > target_clusters and num_noise == best_noise:
            #         best_eps, best_min_samples, best_diff, best_noise = (
            #             eps,
            #             min_samples,
            #             diff,
            #             num_noise,
            #         )
            #     elif min_samples > best_min_samples and num_noise == best_noise:
            #         best_eps, best_min_samples, best_diff, best_noise = (
            #             eps,
            #             min_samples,
            #             diff,
            #             num_noise,
            #         )

    print(
        f"the selected params have a cluster score of {best_score}, and num_clusters: {best_num_clusters}"
    )
    return best_eps, best_min_samples


# z_normalized_features, z_mean, z_std = z_score_normalization(features.numpy())
z_normalized_features = features.numpy()
z_mean = 0
z_std = 1


class MDS:
    def __init__(self, distance_matrix, dimensions=2):
        self.distance_matrix = distance_matrix
        self.dimensions = dimensions
        self.eigvecs = None

    def fit_transform(self):
        n = self.distance_matrix.shape[0]
        centering_matrix = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * centering_matrix @ self.distance_matrix**2 @ centering_matrix

        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals = np.sqrt(np.abs(eigvals[: self.dimensions]))
        eigvecs = eigvecs[:, : self.dimensions]
        self.eigvecs = eigvecs
        coordinates = eigvecs * eigvals
        return coordinates

    def transform(self, new_point):
        new_point = new_point - np.mean(self.distance_matrix, axis=0)
        new_point = new_point @ self.eigvecs
        return new_point


# if os.path.exists(LABEL_TO_MDS_PATH):
#     with open(LABEL_TO_MDS_PATH, "rb") as f:
#         label_to_mds = pickle.load(f)
# else:
#     label_to_mds = {}
#     for label, indices in tqdm(label_to_feat_idx.items()):
#         _tmp = z_normalized_features[indices]

#         # find distance matrix
#         distance_matrix = np.linalg.norm(_tmp[:, np.newaxis] - _tmp, axis=2)

#         # find coordinates
#         mds_obj = MDS(distance_matrix)
#         new_feats = mds_obj.fit_transform()
#         label_to_mds[label] = (new_feats, mds_obj)

#     # save label_to_mds
#     with open(LABEL_TO_MDS_PATH, "wb") as f:
#         pickle.dump(label_to_mds, f)


# label_to_features
label_to_features = {}
for label, indices in tqdm(label_to_feat_idx.items()):
    _tmp = z_normalized_features[indices]
    label_to_features[label] = _tmp


if os.path.exists(LABEL_TO_DBSCAN_PARAMS_PATH):
    with open(LABEL_TO_DBSCAN_PARAMS_PATH, "rb") as f:
        label_to_dbscan_params = pickle.load(f)
else:
    label_to_dbscan_params = {}
    for label, _ in tqdm(label_to_feat_idx.items()):
        # _tmp = z_normalized_features[indices]
        # _tmp = label_to_mds[label][0]
        _tmp = label_to_features[label]

        # find range of eps
        distance_matrix = np.zeros((len(_tmp), len(_tmp)))
        for i in range(len(_tmp)):
            for j in range(len(_tmp)):
                distance_matrix[i, j] = np.linalg.norm(_tmp[i] - _tmp[j])

        eps_range = (np.min(distance_matrix), np.max(distance_matrix))
        # set eps_range to first and third quartile
        # eps_range = np.percentile(distance_matrix, [25, 75])
        print(f"eps_range: {eps_range}")

        min_samples_range = (1, 20)
        iterations = 500

        # find eps and min_samples
        eps, min_samples = None, None
        eps, min_samples = grid_search_dbscan_params(
            _tmp,
            target_clusters=N_CLUSTERS,
            eps_range=eps_range,
            min_samples_range=min_samples_range,
            iterations=iterations,
        )
        while eps is None and min_samples is None:
            eps, min_samples = random_search_dbscan_params(
                _tmp,
                target_clusters=N_CLUSTERS,
                eps_range=eps_range,
                min_samples_range=min_samples_range,
                iterations=iterations,
            )

        # further refine eps and min_samples for target_clusters
        eps, min_samples = find_dbscan_params(
            _tmp,
            initial_eps=eps,
            initial_min_samples=min_samples,
            target_clusters=N_CLUSTERS,
            tolerance=1,
            eps_step=0.1,
            min_samples_step=1,
            max_iterations=200,
        )

        label_to_dbscan_params[label] = (eps, min_samples, N_CLUSTERS)

    # save label_to_dbscan_params
    with open(LABEL_TO_DBSCAN_PARAMS_PATH, "wb") as f:
        pickle.dump(label_to_dbscan_params, f)

# find clusters for each label
if os.path.exists(LABEL_TO_CLUSTERS_PATH):
    with open(LABEL_TO_CLUSTERS_PATH, "rb") as f:
        label_to_clusters = pickle.load(f)
else:
    label_to_clusters = {}
    for label, indices in tqdm(label_to_feat_idx.items()):
        _tmp = label_to_features[label]
        eps, min_samples, _ = label_to_dbscan_params[label]
        labels = dbscan(_tmp, eps=eps, min_samples=min_samples)
        label_to_clusters[label] = labels

    # save label_to_clusters
    with open(LABEL_TO_CLUSTERS_PATH, "wb") as f:
        pickle.dump(label_to_clusters, f)


# for each label-specific cluster, reduce dimensions using MDS (Multi-Dimensional Scaling)
def mds(distance_matrix, dimensions=2):
    n = distance_matrix.shape[0]
    centering_matrix = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * centering_matrix @ distance_matrix**2 @ centering_matrix

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.sqrt(np.abs(eigvals[:dimensions]))
    eigvecs = eigvecs[:, :dimensions]
    coordinates = eigvecs * eigvals
    return coordinates


# plot clusters in 2D MDS space
# for label, (coordinates, _) in tqdm(
#     label_to_mds.items(), desc="Plotting Clusters", leave=False
# ):
for label, indices in tqdm(
    label_to_feat_idx.items(), desc="Plotting Clusters", leave=False
):
    _tmp = z_normalized_features[indices]
    # calculate distances between the _tmp points
    distance_matrix = np.linalg.norm(_tmp[:, np.newaxis] - _tmp, axis=2)
    coordinates = mds(distance_matrix)
    # check if the image is already saved
    img_fn = f"{OUTPUTS_DIR}/task_2_svd_{K}_space_C_{N_CLUSTERS}/task2_label_{label}_clusters.png"
    if os.path.exists(img_fn):
        continue
    cluster_labels = label_to_clusters[label]
    unique_clusters = sorted(list(set(cluster_labels)))
    plt.figure(figsize=(10, 10))
    eps, min_samples, _ = label_to_dbscan_params[label]
    plt.title(f"Label {label} Clusters | eps: {eps:.2f} | min_samples: {min_samples}")
    # for each cluster, select a color
    colors = sns.color_palette("bright", len(unique_clusters))
    # plot clusters
    for i in unique_clusters:
        cluster_coordinates = coordinates[cluster_labels == i]
        plt.scatter(
            cluster_coordinates[:, 0],
            cluster_coordinates[:, 1],
            color=colors[i],
        )
    # add legend
    plt.legend(unique_clusters)
    plt.savefig(img_fn)
    plt.close()

# for each cluster, plot the images
for label, indices in tqdm(
    label_to_feat_idx.items(), desc="Plotting Images", leave=False
):
    indices = np.array(indices)
    # check if the image is already saved
    img_fn = f"{OUTPUTS_DIR}/task_2_svd_{K}_space_C_{N_CLUSTERS}/task2_label_{label}_images.png"
    if os.path.exists(img_fn):
        continue
    cluster_labels = label_to_clusters[label]
    unique_clusters = sorted(list(set(cluster_labels)))
    # if -1 in unique_clusters:
    #     unique_clusters.remove(-1)
    fig, axs = plt.subplots(len(unique_clusters), 10, figsize=(10, 10))
    # for all axes, remove axis
    for ax in axs.flatten():
        ax.axis("off")
    fig.suptitle(f"Label {label} Images")
    # plot clusters
    for i in unique_clusters:
        # cluster_indices = indices[cluster_labels == i]
        cluster_indices = np.array(label_to_dataset_idx[label])[cluster_labels == i]
        # select 10 random images from the cluster
        cluster_indices = np.random.choice(
            cluster_indices, size=min(len(cluster_indices), 10), replace=False
        )
        for j, idx in enumerate(cluster_indices):
            image = DATASET[idx][0]
            ax = axs[i, j] if len(unique_clusters) > 1 else axs[j]
            ax.imshow(image)
            ax.axis("off")
            # have cluster id as axis label
            ax.set_title(f"Cluster {i}")
    plt.savefig(img_fn)
    plt.close()


if os.path.exists(ODD_IMG_FEAT_PATH):
    odd_image_features = torch.load(ODD_IMG_FEAT_PATH)
    trues = torch.load(ODD_IMG_LABELS_PATH)
else:
    odd_image_features = []
    trues = []
    for idx, (image, label) in tqdm(
        enumerate(DATASET),
        desc="Calculating Odd Image Features",
        total=len(DATASET),
        leave=False,
    ):
        if idx % 2 == 0:
            continue  # Skipping even images

        if image.mode != "RGB":
            image = image.convert("RGB")
        feature = extractor(image)
        # feature = svd.transform(feature)
        odd_image_features.append(feature.squeeze())
        trues.append(label)

    odd_image_features = torch.stack(odd_image_features)
    trues = torch.tensor(trues)
    torch.save(odd_image_features, ODD_IMG_FEAT_PATH)
    torch.save(trues, ODD_IMG_LABELS_PATH)


# for each cluster in each label, find the centroid

if os.path.exists(LABEL_TO_CLUSTER_CENTROIDS_PATH):
    with open(LABEL_TO_CLUSTER_CENTROIDS_PATH, "rb") as f:
        label_to_cluster_centroids = pickle.load(f)
else:
    label_to_cluster_centroids = {}
    for label, indices in tqdm(label_to_feat_idx.items()):
        _tmp = z_normalized_features[indices]
        cluster_labels = label_to_clusters[label]
        unique_clusters = sorted(list(set(cluster_labels)))
        centroids = []
        for i in unique_clusters:
            if i == -1:
                continue
            cluster_features = _tmp[cluster_labels == i]
            centroids.append(np.mean(cluster_features, axis=0))
        label_to_cluster_centroids[label] = np.stack(centroids)

    # save label_to_cluster_centroids
    with open(LABEL_TO_CLUSTER_CENTROIDS_PATH, "wb") as f:
        pickle.dump(label_to_cluster_centroids, f)


def predict_label(
    image_feature,
    label_clusters,
    label_dbscan_params,
    label_idx,
    train_features,
    label_to_mds=None,
):
    """
    Predict the label of an image based on the normalized number of the clusters that the image could be a part of.
    """

    # find the number of clusters that the image could be a part of
    # predicted_labels = []
    # for label, clusters in label_clusters.items():
    #     transformed_train_x, mds_obj = label_to_mds[label]
    #     transformed_point = mds_obj.transform(image_feature)
    #     eps, _, _ = label_dbscan_params[label]
    #     cluster_features = transformed_train_x[clusters != -1]
    #     # calculate distance using euclidean distance
    #     distances = np.linalg.norm(cluster_features - transformed_point, axis=1)
    #     if np.min(distances) < eps:
    #         # get number of points whose distances are less than eps
    #         _tmp = np.mean(distances[distances < eps])
    #         predicted_labels.append((label, _tmp))
    #     else:
    #         predicted_labels.append((label, float("inf")))

    # return sorted(predicted_labels, key=lambda x: x[1], reverse=False)[0][0]
    predicted_label = None
    all_distances = []
    all_labels = []
    for label, clusters in label_clusters.items():
        eps = label_dbscan_params[label][0]
        db_feat = train_features[label_idx[label]]
        cluster_features = db_feat[clusters != -1]
        # calculate distance using euclidean distance, normalize by eps
        distances = np.linalg.norm(cluster_features - image_feature, axis=1) / eps
        all_distances.append(distances)
        all_labels.extend([label] * len(distances))

    all_distances = np.concatenate(all_distances)
    all_labels = np.array(all_labels)

    # argsort and select top 10 labels
    top_10_indices = np.argsort(all_distances)[:10]
    top_10_labels = all_labels[top_10_indices]

    # find the most common label
    label_counts = Counter(top_10_labels)
    predicted_label = label_counts.most_common(1)[0][0]

    # predicted_label = None
    # all_distances = []
    # all_labels = []
    # for label, clusters in label_clusters.items():
    #     eps = label_dbscan_params[label][0]
    #     transformed_train_x, mds_obj = label_to_mds[label]
    #     transformed_point = mds_obj.transform(image_feature)
    #     cluster_features = transformed_train_x[clusters != -1]
    #     # calculate distance using euclidean distance, normalize by eps
    #     distances = np.linalg.norm(cluster_features - transformed_point, axis=1) / eps
    #     all_distances.append(distances)
    #     all_labels.extend([label] * len(distances))

    # all_distances = np.concatenate(all_distances)
    # all_labels = np.array(all_labels)

    # # argsort and select top 10 labels
    # top_10_indices = np.argsort(all_distances)[:10]
    # top_10_labels = all_labels[top_10_indices]

    # # find the most common label
    # label_counts = Counter(top_10_labels)
    # predicted_label = label_counts.most_common(1)[0][0]

    # predicted_label = None
    # all_distances = []
    # all_labels = []
    # for label, clusters in label_clusters.items():
    #     transformed_train_x, mds_obj = label_to_mds[label]
    #     transformed_point = mds_obj.transform(image_feature)
    #     cluster_features = transformed_train_x[clusters != -1]
    #     # calculate distance using euclidean distance
    #     distances = np.linalg.norm(cluster_features - transformed_point, axis=1)
    #     all_distances.append(distances)
    #     all_labels.extend([label] * len(distances))

    # all_distances = np.concatenate(all_distances)
    # all_labels = np.array(all_labels)

    # # argsort and select top 10 labels
    # top_10_indices = np.argsort(all_distances)[:10]
    # top_10_labels = all_labels[top_10_indices]

    # # find the most common label
    # label_counts = Counter(top_10_labels)
    # predicted_label = label_counts.most_common(1)[0][0]
    # print(predicted_label)

    # # find the number of clusters that the image could be a part of
    # predicted_labels = []
    # for label, clusters in label_clusters.items():
    #     eps, _, _ = label_dbscan_params[label]
    #     train_x = train_features[label_idx[label]]
    #     _tmp = 0
    #     for i in set(clusters):
    #         if i == -1:
    #             continue

    #         cluster_features = train_x[clusters == i]
    #         # calculate distance using euclidean distance
    #         distances = np.linalg.norm(cluster_features - image_feature, axis=1)
    #         if np.min(distances) < eps:
    #             _tmp += 1

    #     predicted_labels.append((label, _tmp))

    # return sorted(predicted_labels, key=lambda x: x[1], reverse=True)[0][0]
    # find the number of clusters that the image could be a part of

    # distances = []
    # predicted_labels = []

    # for label, ccentroids in cluster_centroids.items():
    #     # calculate distance using euclidean distance
    #     distances_ = np.mean(np.linalg.norm(ccentroids - image_feature, axis=1))
    #     distances.append(distances_)
    #     predicted_labels.append(label)

    # distances = np.array(distances)
    # predicted_labels = np.array(predicted_labels)

    # return predicted_labels[np.argmin(distances)]

    # # take the top 10 smallest distances
    # top_10_indices = np.argsort(distances)[:10]
    # print(top_10_indices)

    # # get the predicted labels for the top 10 smallest distances
    # top_10_predicted_labels = np.array(predicted_labels)[top_10_indices]
    # print(top_10_predicted_labels)
    # # input()

    # # find the most common label
    # label_counts = Counter(top_10_predicted_labels)
    # predicted_label = label_counts.most_common(1)[0][0]

    return predicted_label


# predict labels for odd images
preds = []
for y_true, img_feature in tqdm(
    zip(odd_image_features, trues),
    desc="Predicting Labels",
    total=len(odd_image_features),
    leave=False,
):
    normalized_feat = (img_feature.numpy() - z_mean) / z_std
    y_pred = predict_label(
        normalized_feat,
        label_to_clusters,
        label_to_dbscan_params,
        label_to_feat_idx,
        features,
        # label_to_mds,
    )
    preds.append(y_pred)

preds = torch.tensor(preds)

# Calculating per-label metrics
per_label_metrics = {}

for label in tqdm(
    set(label_to_cluster_centroids.keys()),
    desc="Calculating Per-Label Metrics",
    leave=False,
):
    tp = torch.sum((preds == label) & (trues == label))
    fp = torch.sum((preds == label) & (trues != label))
    fn = torch.sum((preds != label) & (trues == label))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    per_label_metrics[label] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

# Calculating overall metrics
tp = torch.sum(preds == trues)
fp = torch.sum(preds != trues)
precision = tp / (tp + fp)
accuracy = tp / len(preds)


with open(f"{OUTPUTS_DIR}/task2_C_{N_CLUSTERS}.csv", "w") as f:
    f.write("label,precision,recall,f1_score\n")
    for label, metrics in per_label_metrics.items():
        f.write(
            f'{label},{metrics["precision"]},{metrics["recall"]},{metrics["f1_score"]}\n'
        )

    f.write(f"\noverall accuracy,{accuracy}\n")

# pretty print metrics
print("per-label metrics:")
print(
    f'{"label".rjust(6)}{"precision".rjust(12)}{"recall".rjust(12)}{"f1_score".rjust(12)}'
)
for label in sorted(per_label_metrics.keys()):
    metrics = per_label_metrics[label]
    print(
        f'{label:6d}{metrics["precision"]:12.3f}{metrics["recall"]:12.3f}{metrics["f1_score"]:12.3f}'
    )

print()
print(f"overall accuracy: {accuracy:0.3f}")


# # print all predicted labels
# print("Predicted Labels:")
# print(f'{"image_id".rjust(8)}{"predicted_label".rjust(16)}')
# for image_id, pred in enumerate(preds[:1000]):
#     print(f"{(image_id*2)+1:8d}{pred:16d}")
