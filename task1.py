"""
Task 1: Implement a program which,

- for each unique label l, computes the corresponding k latent semantics (of your choice) associated with the even numbered Caltec101 images, and
- for the odd numbered images, predicts the most likely labels using distances/similarities computed under the label-specific latent semantics.

The system should also output per-label precision, recall, and F1-score values as well as output an overall accuracy value.

---
- Use TorchVision library to download caltech101 dataset
---
Latent Semantics

- Use PyTorch pretrained ResNet50 Model and add a hook at AvgPool layer to use them as image features
- Maintain a dictionary that maps index of the feature vector in the np matrix to the actual image id
- Use a handwritten SVD to convert it those features to k latent dimensions
- Select the indices which have even image id
- AvgPool image latent features belonging to a specific label into one feature vector, which we will use as label latent feature vector
- Do this for all feature vectors.
---
- Looping over all the odd id latent image feature vectors, use cosine similarity with latent label feature matrix to classify the image
- Store the classification results with true results
- Calculate per-label metrics and overall accuracy, with numpy. Sklearn cannot be used.

---
Questions:

1. Do you want the output to be csv or print it on terminal?
"""

from collections import Counter
import numpy as np
import os
import torch
import torchvision

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
K = input("Enter the number of latent dimensions (K. Default is 5): ")
K = 5 if K == "" else int(K)


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


IMG_LATENT_FEAT_MAP_PATH = f"artifacts/task_1_new_img_latent_feat_map_K_{K}.pkl"
IMG_LATENT_FEAT_COMPONENTS_PATH = (
    f"artifacts/task_1_new_img_latent_feat_components_K_{K}.pkl"
)
if os.path.exists(IMG_LATENT_FEAT_MAP_PATH):
    label_to_latent_features = torch.load(IMG_LATENT_FEAT_MAP_PATH)
    label_to_svd = torch.load(IMG_LATENT_FEAT_COMPONENTS_PATH)
else:
    # get indices for each label
    label_indices = {}
    for label in tqdm(set(labels), desc="Getting Label Indices", leave=False):
        label_indices[label] = [i for i, y in enumerate(labels) if y == label]

    # get latent features for each label
    label_to_latent_features = {}
    label_to_svd = {}
    for label, indices in tqdm(
        label_indices.items(), desc="Computing Latent Features", leave=False
    ):
        _tmp = features[indices]
        svd = SVD()
        label_to_latent_features[label] = svd.fit_transform(_tmp, K)
        label_to_svd[label] = svd

    torch.save(label_to_latent_features, IMG_LATENT_FEAT_MAP_PATH)
    torch.save(label_to_svd, IMG_LATENT_FEAT_COMPONENTS_PATH)


def cosine_similarity(x, y):
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    return (x @ y.T) / (torch.norm(x) * torch.norm(y))


def predict_label(img_feat, label_to_svd, label_to_latent_features):
    similarities = []
    all_labels = []
    for label, svd in label_to_svd.items():
        latent_features = label_to_latent_features[label]
        simi = cosine_similarity(svd.transform(img_feat), latent_features).squeeze()
        similarities.append(simi)
        all_labels.extend([label] * len(simi))

    similarities = torch.cat(similarities)
    all_labels = torch.tensor(all_labels)

    # argsort and select top 10 labels
    top_10_indices = torch.argsort(similarities)[-10:]
    top_10_labels = all_labels[top_10_indices]

    # find the most common label
    label_counts = Counter(top_10_labels)
    predicted_label = label_counts.most_common(1)[0][0]
    return predicted_label


PRED_FN = f"artifacts/task_1_new_preds_K_{K}.pkl"
TRUES_FN = f"artifacts/task_1_new_trues_K_{K}.pkl"
if os.path.exists(PRED_FN) and os.path.exists(TRUES_FN):
    preds = torch.load(PRED_FN)
    trues = torch.load(TRUES_FN)
else:
    preds = []
    trues = []
    # calculate and save odd image latent features
    for idx, (image, label) in tqdm(
        enumerate(DATASET),
        desc="Calculating Odd Image Latent Features",
        total=len(DATASET),
        leave=False,
    ):
        if idx % 2 == 0:
            continue  # Skipping even images

        if image.mode != "RGB":
            image = image.convert("RGB")
        feature = extractor(image)
        preds.append(predict_label(feature, label_to_svd, label_to_latent_features))
        trues.append(label)

    trues = torch.tensor(trues)
    preds = torch.tensor(preds)

    torch.save(preds, PRED_FN)
    torch.save(trues, TRUES_FN)


# Calculating per-label metrics
per_label_metrics = {}

for label in tqdm(set(labels), desc="Calculating Per-Label Metrics", leave=False):
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


with open(f"outputs/task1_new_K_{K}.csv", "w") as f:
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
