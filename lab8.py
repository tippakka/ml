import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

data = pd.read_csv("8.csv")
X = data[['v1', 'v2']].values
print("Dataset:\n", X[:5])

plt.scatter(X[:, 0], X[:, 1], c='black', s=20)
plt.title("Original Data")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
labels_km = kmeans.fit_predict(X)
print("KMeans Labels:\n", labels_km)
print("Centroids:\n", kmeans.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis', s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, marker='*')
plt.title("KMeans Clustering")
plt.show()

gmm = GaussianMixture(n_components=3, random_state=0)
labels_gmm = gmm.fit(X).predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='plasma', s=30)
plt.title("GMM Clustering")
plt.show()
