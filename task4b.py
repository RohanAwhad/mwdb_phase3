
import os
import torchvision
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import config
import helper
import argparse

from feature_descriptor import FeatureDescriptor
feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)
from task4a import MultiLayerLSH
_tmp = config.FEAT_DESC_FUNCS['resnet_fc']
feat_db, idx_dict, similarity_metric = _tmp[config.FEAT_DB], _tmp[config.IDX], _tmp[config.SIMILARITY_METRIC]

def l2_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2, ord=2)
def find_k_nearest(new_image_vector, loaded_lsh, k=5):
        # Compute hash values for the new image
        new_hash = loaded_lsh.compute_hash(new_image_vector)

        # Track distances and corresponding images
        distances = []
        all_images = []
        for layer, hash_layer in enumerate(new_hash):
            # Calculate L2 distance between new hash and stored hash
            value = loaded_lsh.hash_values.get((layer,tuple(hash_layer)), None)
            if value is not None:
                all_images += value
            
        unique_images = list(dict.fromkeys(all_images))
        # print(all_images)
        distances = [(image_id, l2_distance(new_image_vector, feat_db[int(image_id/2)])) for image_id in unique_images]
        # Sort distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Select the top k image IDs
        top_k_image_ids = [image_id for image_id, _ in distances[:k]]

        return top_k_image_ids,len(all_images), len(unique_images)


# Function to display images based on their IDs in a grid
def display_images_grid(image_ids, dataset, grid_cols=3):
    num_images = len(image_ids)
    grid_rows = math.ceil(num_images / grid_cols)

    _, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 5))

    for i, image_id in enumerate(image_ids):
        row, col = divmod(i, grid_cols)
        image, _ = dataset[image_id]
        if image.mode != "RGB": image = image.convert('RGB')
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

    # Hide empty subplots if there are fewer images than expected
    for i in range(num_images, grid_cols * grid_rows):
        row, col = divmod(i, grid_cols)
        axes[row, col].axis('off')

    plt.show()

def display_images(query_image_id, top_k_images):
    # Create a figure with k+1 subplots (for query image and top k images)
    num_images = len(top_k_images) + 1
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    query_image, _ = dataset[query_image_id]
    if query_image.mode != "RGB": query_image = query_image.convert('RGB')


    # Display the query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # Display the top k images
    for i, image_id in enumerate(top_k_images, start=1):
        image, _ = dataset[image_id]
        if image.mode != "RGB": image = image.convert('RGB')
        
        axes[i].imshow(image)
        axes[i].set_title(f"Top {i} Image")
        axes[i].axis('off')

    plt.show()

# Create an ArgumentParser
parser = argparse.ArgumentParser(description='Store hash.')

# Define command-line arguments
parser.add_argument('knn', type=str, help='K for Knn')
parser.add_argument('img_id', type=str, help='Query image id')

# Parse the arguments
args = parser.parse_args()
knn = int(args.knn) 
query_image = int(args.img_id)

print('loading caltech101 dataset ...')
DATA_DIR = './data/caltech101'
dataset = torchvision.datasets.Caltech101(DATA_DIR, download=True)

with open('./output/lsh.pkl', 'rb') as file:
    loaded_lsh = pickle.load(file)


image,_ = dataset[query_image]
if image.mode != "RGB": image = image.convert('RGB')
image_vector = feature_descriptor.extract_features(image, 'resnet_fc')


k_near_images,total_images,unique_images = find_k_nearest(image_vector,loaded_lsh,knn)



# Example: Display images with IDs 0, 1, and 2
# display_images_grid(near_images, dataset)
print("Total Images: ", total_images)
print("Unique Images: ", unique_images)
if(len(k_near_images)==0): 
    print("No near image found using LSH")   
else:
    display_images(query_image,k_near_images)

