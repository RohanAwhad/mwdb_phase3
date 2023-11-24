"""
• Task 3: Implement a program which,
    - given even-numbered Caltec101 images,
        * creates an m-NN classifer (for a user specified m),
        * creates a decision-tree classifier,
        * creates a PPR based clasifier.
For this task, you can use feature space of your choice.
    - for the odd numbered images, predicts the most likely labels using the user selected classifier.

The system should also output per-label precision, recall, and F1-score values as well as output an overall accuracy value.
"""
from src.classifier import (
    DecisionTreeClassifier,
    NearestNeighborClassifier,
    PPRClassifier,
)
import os
import torch
import torchvision

from torchvision.transforms import functional as TF
from tqdm import tqdm
import time

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

labels = torch.tensor(labels)

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


def main(classifier, output_fn):
    global features, labels, odd_image_features, trues
    indices = torch.randperm(len(features))  # Randomly shuffle indices
    features = features[indices]
    labels = labels[indices]

    # fit classifier
    start_time = time.time()
    classifier.fit(features, labels)
    end_time = time.time()
    print(f"Time taken to fit classifier: {end_time - start_time:.2f} seconds")
    predictions = torch.tensor(classifier.predict(odd_image_features))
    per_label_metrics = {}

    for label in tqdm(
        labels.unique(), desc="Calculating Per-Label Metrics", leave=False
    ):
        tp = torch.sum((predictions == label) & (trues == label))
        fp = torch.sum((predictions == label) & (trues != label))
        fn = torch.sum((predictions != label) & (trues == label))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        per_label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    # Calculating overall metrics
    tp = torch.sum(predictions == trues)
    fp = torch.sum(predictions != trues)
    precision = tp / (tp + fp)
    accuracy = tp / len(predictions)

    # output filename

    with open(output_fn, "w") as f:
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
    print(f"\noverall accuracy: {accuracy:0.3f}")


if __name__ == "__main__":
    try:
        while True:
            classifier = int(
                input(
                    "Enter the classifier type:\n1. m-NN\n2. Decision Tree\n3. PPR\n> "
                )
            )
            if classifier == 1:
                classifier = NearestNeighborClassifier(m=int(input("Enter m: ")))
                output_fn = f"outputs/task3_{classifier.__class__.__name__}_m_{classifier.m}.csv"
            elif classifier == 2:
                classifier = DecisionTreeClassifier(max_depth=150)
                output_fn = f"outputs/task3_{classifier.__class__.__name__}.csv"
            elif classifier == 3:
                classifier = PPRClassifier(
                    alpha=1.0 - float(input("Enter random jump probability: "))
                )
                output_fn = f"outputs/task3_{classifier.__class__.__name__}_alpha_{classifier.alpha}.csv"
            else:
                raise ValueError("Invalid classifier type")

            # call main
            print("Running classifier...")
            main(classifier, output_fn)
    except KeyboardInterrupt:
        print("Exiting...")
