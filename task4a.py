import os
import torchvision
import torch
import math
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import config
import helper
from feature_descriptor import FeatureDescriptor
import argparse
feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)
np.random.seed(0)




class MultiLayerLSH:
    def __init__(self, num_layers, num_random_vectors, input_dim,projection_vectors,random_width_vectors):
        self.num_layers = num_layers
        self.num_random_vectors = num_random_vectors
        self.input_dim = input_dim
        self.segment_size = 0.5
        # Initialize random projection vectors for each layer
        self.projection_vectors = projection_vectors
        # Store hash values for each input vector
        self.hash_values = {}
        self.data_vectors = {}
        self.width_vectors = random_width_vectors
        self.max1 = -111111
        self.min1 = 11111

    def hash_function(self, vector, projection_vector,width):
        # print(vector.shape,projection_vector.shape)
        vector = vector / np.linalg.norm(vector)
        projection = np.dot(vector, projection_vector)
        self.max1 = max(self.max1,projection)
        self.min1 = min(projection, self.min1)
        # print(math.floor(projection/width))
        # self.segment_size = np.linalg.norm(projection_vector) / num_segments
        # print(segment_size)
        # if(projection>0): return 1
        # else: return 0
        return math.floor(projection/width)
    

    def projection_vec(self):
        return self.projection_vectors
    
    def length(self):
        for layer in range(num_layers):
            for i in range(num_random_vectors):
                proj = self.projection_vectors[layer, i, :]
                # print(np.linalg.norm(proj)/ num_segments)

    def hash_data(self, data_vector, layer):
        hashes = []
        for i in range(self.num_random_vectors):
            hash_value = self.hash_function(data_vector, self.projection_vectors[layer, i, :],self.width_vectors[layer,i])
            hashes.append(hash_value)
        return hashes

    def compute_hash(self, data_vector):
        hash_values = []
        for layer in range(self.num_layers):
            layer_hash = self.hash_data(data_vector, layer)
            hash_values.append(layer_hash)
        return np.array(hash_values)
    
    
    def store_hash(self, data_id, data_vector):
        hash_value = self.compute_hash(data_vector)
        for layer, layer_hash in enumerate(hash_value):
            key = (layer, tuple(layer_hash))
            if key not in self.hash_values:
                self.hash_values[key] = []
            self.hash_values[key].append(data_id)

    def store_data_vector(self, data_id, data_vector):
        hash_value = self.compute_hash(data_vector)
        self.hash_values[data_id] = hash_value

    def save_hash_values(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.hash_values, file)

    def load_hash_values(self, file_path):
        with open(file_path, 'rb') as file:
            self.hash_values = pickle.load(file)
    


# Example usage
def main():

    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description='Store hash.')

    # Define command-line arguments
    parser.add_argument('num_layers', type=str, help='Number of layers1')
    parser.add_argument('num_hash', type=int, help='Number of hashes')

    # Parse the arguments
    args = parser.parse_args()
    num_layers = int(args.num_layers)
    num_random_vectors = int(args.num_hash)
    input_dim = 1000

    _tmp = config.FEAT_DESC_FUNCS['resnet_fc']
    feat_db, idx_dict, _ = _tmp[config.FEAT_DB], _tmp[config.IDX], _tmp[config.SIMILARITY_METRIC]

    if(True):
        # Initialize random projection vectors for each layer
        projection_vectors = 2*np.random.rand(num_layers, num_random_vectors, input_dim)-1
        projection_vectors /= np.linalg.norm(projection_vectors, axis=-1, keepdims=True)
        random_width_vector = np.random.rand(num_layers, num_random_vectors)*0.1 + 0.001*num_layers*num_random_vectors
        # Dump the NumPy array to a file using pickle
        # with open('./output/projection_vectors.pkl', 'wb') as file:
        #     pickle.dump(projection_vectors, file)
        lsh = MultiLayerLSH(num_layers, num_random_vectors, input_dim,projection_vectors,random_width_vector)
        
        for i in (tqdm(range(len(feat_db)))):
            image_vector = feat_db[i]
            lsh.store_hash(idx_dict[i][0], image_vector)
        # lsh.save_hash_values('./output/hash_values.pkl')
        # Serialize and save the object to a file
        # print("Max", lsh.max1)
        # print("Min1", lsh.min1)
        with open('./output/lsh.pkl', 'wb') as file:
            pickle.dump(lsh, file)

if __name__ == '__main__':
    main()
