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

from collections import defaultdict
from torchvision.transforms import functional as TF
from tqdm import tqdm

# N_CLUSTERS = int(input("Enter number of clusters: "))
N_CLUSTERS = 5

os.makedirs("artifacts", exist_ok=True)
os.makedirs("outputs/task_2", exist_ok=True)

TORCH_HUB = "./models/"
torch.set_grad_enabled(False)
torch.hub.set_dir(TORCH_HUB)

DATA_DIR = "./data/caltech101"
DATASET = torchvision.datasets.Caltech101(DATA_DIR, download=False)

MODEL_NAME = "ResNet50_Weights.DEFAULT"
RESNET_MODEL = torchvision.models.resnet50(weights=MODEL_NAME).eval()


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


extractor = FeatureExtractor()

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


label_to_idx = defaultdict(list)
for idx, label in enumerate(labels):
    label_to_idx[label].append(idx)


def z_score_normalization(features):
    # Calculate the mean and standard deviation for each feature
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)

    # Apply the normalization: (feature - mean) / std_dev
    normalized_features = (features - means) / std_devs
    return normalized_features


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

            if labels[p] != -1:
                k += 1
                continue
            print("hitting post if condition")
            # Add point to cluster
            labels[p] = C

            # Get new neighbors and add them to the list
            p_neighbors = np.where(np.linalg.norm(X - X[p], axis=1) < eps)[0]
            if len(p_neighbors) >= min_samples:
                neighbors = np.append(neighbors, p_neighbors)

            k += 1

    return labels


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

    for _ in range(iterations):
        eps = random.uniform(*eps_range)
        min_samples = random.randint(*min_samples_range)

        labels = dbscan(X, eps=eps, min_samples=min_samples)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        diff = abs(num_clusters - target_clusters)
        if diff < best_diff:
            best_eps, best_min_samples, best_diff = eps, min_samples, diff
        elif diff == best_diff:
            if num_clusters > target_clusters:
                best_eps, best_min_samples, best_diff = eps, min_samples, diff
            if min_samples > best_min_samples:
                best_eps, best_min_samples, best_diff = eps, min_samples, diff

    return best_eps, best_min_samples


z_normalized_features = z_score_normalization(features.numpy())
label_to_dbscan_params_path = f"artifacts/label_to_dbscan_params_C_{N_CLUSTERS}.pkl"
if os.path.exists(label_to_dbscan_params_path):
    with open(label_to_dbscan_params_path, "rb") as f:
        label_to_dbscan_params = pickle.load(f)
else:
    label_to_dbscan_params = {}
    for label, indices in tqdm(label_to_idx.items()):
        print(f"Finding DBSCAN parameters for label {label}...")
        _tmp = z_normalized_features[indices]

        # find range of eps
        distance_matrix = np.zeros((len(_tmp), len(_tmp)))
        for i in range(len(_tmp)):
            for j in range(len(_tmp)):
                distance_matrix[i, j] = np.linalg.norm(_tmp[i] - _tmp[j])

        eps_range = (np.min(distance_matrix), np.max(distance_matrix))

        # find eps and min_samples
        eps, min_samples = None, None
        while eps is None and min_samples is None:
            eps, min_samples = random_search_dbscan_params(
                _tmp,
                target_clusters=N_CLUSTERS,
                eps_range=eps_range,
                min_samples_range=(3, 15),
            )
        label_to_dbscan_params[label] = (eps, min_samples, N_CLUSTERS)

    # save label_to_dbscan_params
    with open(label_to_dbscan_params_path, "wb") as f:
        pickle.dump(label_to_dbscan_params, f)

# find clusters for each label
label_to_clusters_path = f"artifacts/label_to_clusters_C_{N_CLUSTERS}.pkl"
if os.path.exists(label_to_clusters_path):
    with open(label_to_clusters_path, "rb") as f:
        label_to_clusters = pickle.load(f)
else:
    label_to_clusters = {}
    for label, indices in tqdm(label_to_idx.items()):
        print(f"Finding clusters for label {label}...")
        _tmp = z_normalized_features[indices]
        eps, min_samples, _ = label_to_dbscan_params[label]
        labels = dbscan(_tmp, eps=eps, min_samples=min_samples)
        label_to_clusters[label] = labels

    # save label_to_clusters
    with open(label_to_clusters_path, "wb") as f:
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


label_to_mds_path = f"artifacts/label_to_mds_C_{N_CLUSTERS}.pkl"
if os.path.exists(label_to_mds_path):
    with open(label_to_mds_path, "rb") as f:
        label_to_mds = pickle.load(f)
else:
    label_to_mds = {}
    for label, indices in tqdm(label_to_idx.items()):
        print(f"Finding MDS coordinates for label {label}...")
        _tmp = z_normalized_features[indices]

        # find distance matrix
        distance_matrix = np.zeros((len(_tmp), len(_tmp)))
        for i in range(len(_tmp)):
            for j in range(len(_tmp)):
                distance_matrix[i, j] = np.linalg.norm(_tmp[i] - _tmp[j])

        # find coordinates
        coordinates = mds(distance_matrix)
        label_to_mds[label] = coordinates

    # save label_to_mds
    with open(label_to_mds_path, "wb") as f:
        pickle.dump(label_to_mds, f)


# plot clusters in 2D MDS space
for label, coordinates in tqdm(
    label_to_mds.items(), desc="Plotting Clusters", leave=False
):
    # check if the image is already saved
    img_fn = f"outputs/task_2/task2_label_{label}_clusters.png"
    if os.path.exists(img_fn):
        continue
    cluster_labels = label_to_clusters[label]
    unique_clusters = sorted(list(set(cluster_labels)))
    plt.figure(figsize=(10, 10))
    plt.title(f"Label {label} Clusters")
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


ODD_IMG_FEAT_PATH = "artifacts/odd_img_features.pkl"
ODD_IMG_LABELS_PATH = "artifacts/odd_image_labels.pkl"
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
        odd_image_features.append(feature.squeeze())
        trues.append(label)

    odd_image_features = torch.stack(odd_image_features)
    trues = torch.tensor(trues)
    torch.save(odd_image_features, ODD_IMG_FEAT_PATH)
    torch.save(trues, ODD_IMG_LABELS_PATH)


# for each cluster in each label, find the centroid
label_to_cluster_centroids_path = (
    f"artifacts/label_to_cluster_centroids_C_{N_CLUSTERS}.pkl"
)
if os.path.exists(label_to_cluster_centroids_path):
    with open(label_to_cluster_centroids_path, "rb") as f:
        label_to_cluster_centroids = pickle.load(f)
else:
    label_to_cluster_centroids = {}
    for label, indices in tqdm(label_to_idx.items()):
        print(f"Finding cluster centroids for label {label}...")
        _tmp = z_normalized_features[indices]
        cluster_labels = label_to_clusters[label]
        unique_clusters = sorted(list(set(cluster_labels)))
        centroids = []
        for i in unique_clusters:
            cluster_features = _tmp[cluster_labels == i]
            centroids.append(np.mean(cluster_features, axis=0))
        label_to_cluster_centroids[label] = np.stack(centroids)

    # save label_to_cluster_centroids
    with open(label_to_cluster_centroids_path, "wb") as f:
        pickle.dump(label_to_cluster_centroids, f)


def predict_label(image_feature, cluster_centroids):
    """
    Predict the label of an image based on the closest DBSCAN cluster centroid.

    Parameters:
    image_feature: ndarray
        Feature vector of the image.
    cluster_centroids: dict
        A dictionary where keys are labels and values are arrays of centroids of clusters for that label.

    Returns:
    predicted_label: int or str
        The predicted label for the image.
    """

    min_distance = float("inf")
    predicted_label = None

    for label, ccentroids in cluster_centroids.items():
        # calculate distance using euclidean distance
        distances = np.linalg.norm(ccentroids - image_feature, axis=1)

        closest_cluster_distance = np.min(distances)

        if closest_cluster_distance < min_distance:
            min_distance = closest_cluster_distance
            predicted_label = label

    return predicted_label


# predict labels for odd images
preds = []
for y_true, feature in tqdm(
    zip(odd_image_features, trues),
    desc="Predicting Labels",
    total=len(odd_image_features),
    leave=False,
):
    y_pred = predict_label(feature.numpy(), label_to_cluster_centroids)
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


with open(f"outputs/task2_C_{N_CLUSTERS}.csv", "w") as f:
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


print(preds)
