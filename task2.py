"""
Task 2: Implement a program which,

  - for each unique label l, computes the correspending c most significant clusters associated with the even numbered Caltec101 images (using DBScan algorithm); the resulting clusters should be visualized both
    * as differently colored point clouds in a 2-dimensional MDS space, and
    * as groups of image thumbnails. and 
  - for the odd numbered images, predicts the most likely labels using the c label-specific clusters.

The system should also output per-label precision, recall, and F1-score values as well as output an overall accuracy value.

"""
import queue
import numpy as np
import os
import torch
import torchvision

from collections import defaultdict
from torchvision.transforms import functional as TF
from tqdm import tqdm

os.makedirs("artifacts", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

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


def dbscan_2(X, eps, min_samples):
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


z_normalized_features = z_score_normalization(features.numpy())
_tmp = z_normalized_features[label_to_idx[0]]
my_clabels = dbscan_2(_tmp, eps=20, min_samples=2)
