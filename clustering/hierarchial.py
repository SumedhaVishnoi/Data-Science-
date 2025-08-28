''' Build a tree of clusters 
1) Agglomerative (bottom-up)
2) Divisive (top-down)'''

''' DBSCAN
Density-Based Spatial Clustering of Applications with Noise
Groups together points that are closely packed.

Points in low-density regions are marked as outliers.

Does not need k (number of clusters) in advance.'''

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load example data; replace with your own data as needed
data = load_iris()
X = data.data

kmeans = KMeans(n_clusters=3, random_state=42)  # You can change n_clusters as needed
kmeans.fit(X)

score = silhouette_score(X, kmeans.labels_)
print("Silhouette Score:", score)
