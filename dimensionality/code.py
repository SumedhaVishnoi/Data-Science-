from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load sample dataset
digits = load_digits()
X = digits.data
y = digits.target

# PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10')
plt.title("PCA Visualization")
plt.show()

# t-SNE (reduce to 2D)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='tab10')
plt.title("t-SNE Visualization")
plt.show()
