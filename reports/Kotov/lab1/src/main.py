import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('Wholesale customers data.csv')
df = df.fillna(df.mean())

y = df['Region']
X = df.drop(columns=['Region'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sort_index = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[sort_index]
eigenvectors = eigenvectors[:, sort_index]

PC2 = eigenvectors[:, :2]
X_manual_2 = np.dot(X_scaled, PC2)

PC3 = eigenvectors[:, :3]
X_manual_3 = np.dot(X_scaled, PC3)

pca2 = PCA(n_components=2)
X_sklearn_2 = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
X_sklearn_3 = pca3.fit_transform(X_scaled)

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(221)
ax1.scatter(X_manual_2[:, 0], X_manual_2[:, 1], c=y, cmap='brg')
ax1.set_title('Manual PCA (2D)')

ax2 = fig.add_subplot(222)
ax2.scatter(X_sklearn_2[:, 0], X_sklearn_2[:, 1], c=y, cmap='brg')
ax2.set_title('Sklearn PCA (2D)')

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_manual_3[:, 0], X_manual_3[:, 1], X_manual_3[:, 2], c=y, cmap='brg')
ax3.set_title('Manual PCA (3D)')

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_sklearn_3[:, 0], X_sklearn_3[:, 1], X_sklearn_3[:, 2], c=y, cmap='brg')
ax4.set_title('Sklearn PCA (3D)')

plt.tight_layout()
plt.show()

full_info = np.sum(eigenvalues)
reduce_info_2 = np.sum(eigenvalues[:2])
reduce_info_3 = np.sum(eigenvalues[:3])

print(f"Потери после применения PCA (2 компоненты): {100 - (reduce_info_2 / full_info * 100):.2f}%")
print(f"Потери после применения PCA (3 компоненты): {100 - (reduce_info_3 / full_info * 100):.2f}%")