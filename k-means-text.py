import numpy as np
import requests
import warnings
import scipy.sparse as sp
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer



#################################################################
# Load Dataset
#################################################################

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    shuffle=True,
    random_state=42,
)

#################################################################
# REDUCED DATA FOR HARDWARE-LIMITED SYSTEMS

# If you are running this code on a hardware-limited system,
# you can use the following code snippet to reduce the dataset.
#################################################################
categories = [
     "alt.atheism",
     "talk.religion.misc",
     "comp.graphics",
     "sci.space",
 ]
 
dataset = fetch_20newsgroups(
     remove=("headers", "footers", "quotes"),
     subset="all",
     categories=categories,
     shuffle=True,
     random_state=42,
 )
################################################################# 


labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

#################################################################
# Evaluate Fitness
#################################################################
def fit_and_evaluate(km, X, n_runs=5):

    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

#################################################################
# Vectorize 
#################################################################
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(dataset.data)

#################################################################
# (TODO): Implement K-Means  
#################################################################

# Used https://towardsdatascience.com/the-math-and-code-behind-k-means-clustering-795582423666#7dd0
# for reference on how to code KMeans from scratch
clusters = 8
max_iter = 300
random_state = None 
class KMeans:
    labels_ = [] #TODO
    
    def __init__(self, X):
        self.clusters = clusters
        self.max_iter = max_iter
        self.random_state = 0.5 # random seed for reproducible results 
        self.centroids = None 
        self.labels_ = np.zeros(X.shape[0])

    # Used ChatGPT to determine meaning of centroids + find the distances 
    # Used ChatGPT to debug errors 
    def fit(self, X):
        np.random.seed(self.random_state)
        if sp.issparse(X): # ChatGPT to debug
            X = X.toarray()
        self.centroids = X[np.random.choice(X.shape[0], self.clusters, replace=False)]
        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.clusters):
                if np.any(self.labels_ == i):
                    new_centroids[i] = np.mean(X[self.labels_ == i], axis=0)
                else:
                    new_centroids[i] = self.centroids[i]
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

    # Used ChatGPT
    def set_params(self, random_state):
        self.random_state = random_state


kmeans = KMeans(X_tfidf)
# Feel free to change the number of runs
fit_and_evaluate(kmeans, X_tfidf)
# print(kmeans)
