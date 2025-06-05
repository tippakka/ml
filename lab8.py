import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
data = pd.read_csv("8.csv")  # Ensure 'em.csv' exists in the working directory

print("Input Data and Shape:")
print(data.shape)
print(data.head())

# Extract features
X = data[['v1', 'v2']].values
print("Shape of X:", X.shape)

# Plot original data
print("Graph for whole dataset")
plt.scatter(X[:, 0], X[:, 1], c='black', s=10)
plt.title("Original Data")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
print("KMeans Labels:", kmeans_labels)
print("KMeans Centroids:", centroids)

# Plot KMeans clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red')
plt.title("KMeans Clustering")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

# EM using Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X)
probs = gmm.predict_proba(X)
sizes = 10 + probs.max(axis=1) * 200  # Size = confidence based

print("GMM Labels:", gmm_labels)

# Plot GMM clustering
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, s=sizes, cmap='viridis')
plt.title("EM Clustering (Gaussian Mixture Model)")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()