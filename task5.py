import os
import torchvision
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm  # Import tqdm for the progress bar
import pickle
import matplotlib.pyplot as plt
import math
import config
import helper
import numpy as np
from task4a import MultiLayerLSH
import math
from feature_descriptor import FeatureDescriptor
from task4b import find_k_nearest, display_images
feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)
from task4a import MultiLayerLSH

_tmp = config.FEAT_DESC_FUNCS['resnet_fc']
feat_db, idx_dict, similarity_metric = _tmp[config.FEAT_DB], _tmp[config.IDX], _tmp[config.SIMILARITY_METRIC]
print('loading caltech101 dataset ...')
DATA_DIR = './data/caltech101'
dataset = torchvision.datasets.Caltech101(DATA_DIR, download=True)
def l2_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2, ord=2)

def process_feedback_using_probabilities(feedback_list, k_val, k_near_images, threshold_percent=5):
    pass

    R = sum(1 for d in feedback_list if 'R' == d['feedback'])
    RPlus = sum(1 for d in feedback_list if 'R+' == d['feedback'])
    I = sum(1 for d in feedback_list if 'I' == d['feedback'])
    IMinus = sum(1 for d in feedback_list if 'I-' == d['feedback'])

    N = len(feedback_list)
    # Get Top K Images from LSH
    k_near_images = k_near_images
    # For Query image and all other images get latent semantics
    latent_space = helper.load_semantics("task3_resnet_fc_svd_5.pkl")
    feat_db_latent_space = np.dot(feat_db, np.transpose(latent_space))

    # creare a list for scores
    scores = []
    R_Overall = sum(1 for d in feedback_list if 'R' in d['feedback'])
    # For each image in all the top K images
    for imageId in k_near_images:
        score = 0
        for i,feature in enumerate(feat_db_latent_space[int(imageId/2)]):
            ri = 0
            ni = 0
            for feedback in feedback_list:
                labeled_image_feature = feat_db_latent_space[int(feedback['id']/2)][i]
                diff = labeled_image_feature - feature
                if diff <= (threshold_percent/100)*(labeled_image_feature):
                    ni += 1
                    if 'R' in feedback["feedback"]:
                        ri += 1
            pi = ( ri + 0.5 ) / ( R_Overall + 1 )
            ui = (ni - ri + 0.5) / (N - R_Overall + 1)
            score += feature * (math.log((pi*(1-ui)) / (ui*(1-pi))))
            score += ( (R/(R+RPlus)) + 2*(RPlus/(R+RPlus)))
            score -= ( (I/(I+IMinus)) + 2*(IMinus/(I+IMinus)))
        scores.append((imageId, score))
    print(scores)
    scores.sort(key=lambda x: x[1], reverse=True)
    print(scores)
    top_k_image_ids = [image_id for image_id, _ in scores[:k_val]]
    return top_k_image_ids

def process_feedback_using_svm(feedback_list, k, query_image, unique_images, k_near_images):
    print(feedback_list)
    imageIds = [feedback['id'] for feedback in feedback_list]
    relevance = [feedback['feedback'] for feedback in feedback_list]
    X_list = []
    for imageId in imageIds:
        image_vector = feat_db[imageId]
        X_list.append(image_vector.numpy())
    X_list = np.array(X_list)
    print(X_list.shape)
    svm = SVM()
    svm.fit(X_list,relevance)
    X = []
    pred_imageIds = list(set(unique_images) - set(imageIds))
    #imageIds = [11,23,34]
    for imageId in tqdm(pred_imageIds):
        image_vector = feat_db[int(imageId/2)]
        X.append(image_vector.numpy())

    X = np.array(X)
    p1 = svm.predict(X)
    image,_ = dataset[query_image]
    if image.mode != "RGB": image = image.convert('RGB')
    image_vector = feature_descriptor.extract_features(image, 'resnet_fc')
    print(len(p1))
    relevant_indices = [(s, n) for s, n in zip(pred_imageIds, p1) if n > 1]
    target_values = ['R+', 'R']

    print(len(relevant_indices))
    distances = [(image_id, l2_distance(image_vector, feat_db[int(image_id/2)])) for image_id,n in relevant_indices]
    
    distances.sort(key=lambda x: x[1])

    top_k_image_ids = [image_id for image_id, _ in distances[:k]]
    return top_k_image_ids

class SVM:

    def __init__(self, alpha = 0.001, lambda_ = 0.01, n_iterations = 1000):
        self.alpha = alpha 
        self.lambda_ = lambda_ 
        self.n_iterations = n_iterations 
        self.weights_list = []

    def fit(self, X, y):
        a = np.array([1 if "I-" == rel else -1 for rel in y])
        b= np.array([1 if "R+" == rel else -1 for rel in y])
        c =  np.array([1 if "R" == rel else -1 for rel in y])
        d = np.array([1 if "I" == rel else -1 for rel in y])
        
        y_list = [a, d, c, b]
        for _y in y_list:
            n_samples, n_features = X.shape        
            w = np.zeros(n_features) 
            b = 0 
            for iteration in range(self.n_iterations):
                for i, Xi in enumerate(X):
                    if _y[i] * (np.dot(Xi, w) - b) >= 1 : 
                        w -= self.alpha * (2 * self.lambda_ * w) 
                    else:
                        w -= self.alpha * (2 * self.lambda_ * w - np.dot(Xi, _y[i])) 
                        b -= self.alpha * _y[i] 
            self.weights_list.append([w, b])

    def predict(self, X):
        pred_list = []
        
        for weights in self.weights_list:
            pred = np.dot(X, weights[0]) - weights[1] 
            
            pred_list.append([val for val in pred])
        return [index for value, index in [max((value, index) for index, value in enumerate(column)) for column in zip(*pred_list)]]
    

import matplotlib.pyplot as plt
from PIL import Image

def get_user_feedback(img_relevance_dict):
    feedback_list = []
    for image_id, image_array in img_relevance_dict.items():
        
        plt.imshow(image_array)
        plt.title(f"Image {image_id}")
        plt.axis('off')
        plt.pause(0.1)  
        plt.show()
        feedback = input("Is the image Very Irrelevant (I-) or Very Relevant (R+) or Relevant (R) or Irrelevant (I)? (I- / I / R/ R+): ")
        feedback_list.append({'id': image_id, 'feedback': feedback})
    return feedback_list

inp = helper.get_user_input('regime,query_image_id,topK')

# Parse the arguments
k_val = inp['topK']
query_image = inp['query_image_id']
regime = inp['regime']



with open('./output/lsh.pkl', 'rb') as file:
    loaded_lsh = pickle.load(file)
image,_ = dataset[query_image]
if image.mode != "RGB": image = image.convert('RGB')
image_vector = feature_descriptor.extract_features(image, 'resnet_fc')

k_near_images,total_images,unique_images = find_k_nearest(image_vector,loaded_lsh, k_val)
print()
print("Total Images: ", total_images)
print("Unique Images: ", len(unique_images))
display_images(query_image,k_near_images)
img_relevance = {}
counter = 10
for i,img in enumerate(k_near_images):
    # Getting user feedback for a maximum 10 images.
    if i < 10:
        tmpImage,_ = dataset[img]
        if tmpImage.mode != "RGB": tmpImage = tmpImage.convert('RGB')
        img_relevance[img] = np.array(tmpImage)

feedback_list = get_user_feedback(img_relevance)
#print(unique_images)
if regime == 1:
    processed_images = process_feedback_using_svm(feedback_list, k_val, query_image, unique_images, k_near_images)
else:
    latent_space = helper.load_semantics("task3_resnet_fc_svd_5.pkl")
    feat_db_latent_space = np.dot(feat_db, np.transpose(latent_space))
    processed_images = process_feedback_using_probabilities(feedback_list, k_val, k_near_images)

display_images(query_image,processed_images)