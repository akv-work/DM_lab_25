import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FILE_NAME = 'heart_failure_clinical_records_dataset.csv'
TARGET_COL = 'DEATH_EVENT'

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, FILE_NAME)

df = pd.read_csv(file_path)

df = df.fillna(df.mean(numeric_only=True))

y = df[TARGET_COL].values
X_raw = df.drop(TARGET_COL, axis=1).values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

def manual_pca(data, n_components):
    cov_matrix = np.cov(data.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    sort_idx = np.argsort(-eig_vals)
    eig_vals_sorted = eig_vals[sort_idx]
    eig_vecs_sorted = eig_vecs[:, sort_idx]

    reduced = data @ eig_vecs_sorted[:, :n_components]

    full_info = eig_vals_sorted.sum()
    reduced_info = eig_vals_sorted[:n_components].sum()
    loss = 100 * (1 - reduced_info / full_info)

    return reduced, loss, eig_vals_sorted

X_manual_2d, loss_2d, eig_vals = manual_pca(X, 2)
X_manual_3d, loss_3d, _ = manual_pca(X, 3)

# 3. Способ 2: PCA через sklearn
pca_2d = PCA(n_components=2)
X_sklearn_2d = pca_2d.fit_transform(X)

pca_3d = PCA(n_components=3)
X_sklearn_3d = pca_3d.fit_transform(X)

print(f"Потери информативности при сжатии до 2-х компонент: {loss_2d:.2f}%")
print(f"Потери информативности при сжатии до 3-х компонент: {loss_3d:.2f}%")
print("Собственные значения (по убыванию):")
print(np.round(eig_vals, 4))

fig = plt.figure(figsize=(16, 12))

unique_labels = np.unique(y)
colors_map = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}
colors = [colors_map[label] for label in y]

ax1 = fig.add_subplot(221)
ax1.scatter(X_manual_2d[:, 0], X_manual_2d[:, 1], c=colors, alpha=0.7)
ax1.set_title('PCA 2D (вручную)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

ax2 = fig.add_subplot(222)
ax2.scatter(X_sklearn_2d[:, 0], X_sklearn_2d[:, 1], c=colors, alpha=0.7)
ax2.set_title('PCA 2D (scikit-learn)')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_manual_3d[:, 0], X_manual_3d[:, 1], X_manual_3d[:, 2], c=colors, alpha=0.7)
ax3.set_title('PCA 3D (вручную)')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_sklearn_3d[:, 0], X_sklearn_3d[:, 1], X_sklearn_3d[:, 2], c=colors, alpha=0.7)
ax4.set_title('PCA 3D (scikit-learn)')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')

plt.tight_layout()
plt.show()