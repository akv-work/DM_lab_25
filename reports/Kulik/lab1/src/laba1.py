import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('Exasens.csv', header=[0, 1])
clean_columns = []
for col in df.columns:
    c1 = "" if "Unnamed" in str(col[0]) else str(col[0]).strip()
    c2 = "" if "Unnamed" in str(col[1]) else str(col[1]).strip()
    name = f"{c1}_{c2}".strip("_")
    clean_columns.append(name)
df.columns = clean_columns
df = df.iloc[:, :9]
target_col = 'Diagnosis'
y = df[target_col]
X = df.drop(columns=[target_col, 'ID'])
X = X.apply(pd.to_numeric, errors='coerce')
valid_indices = X.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cov_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sort_index = np.argsort(-1 * eigenvalues)
sorted_eigenvalues = eigenvalues[sort_index]

total_variance = np.sum(sorted_eigenvalues)
info_2d = np.sum(sorted_eigenvalues[:2]) / total_variance * 100
info_3d = np.sum(sorted_eigenvalues[:3]) / total_variance * 100

loss_2d = 100 - info_2d
loss_3d = 100 - info_3d

print(f"Сохраненная информация (2D): {info_2d:.2f}%, Потери: {loss_2d:.2f}%")
print(f"Сохраненная информация (3D): {info_3d:.2f}%, Потери: {loss_3d:.2f}%")
PC_sorted = eigenvectors[:, sort_index]

W_2d = PC_sorted[:, :2]
X_pca_manual_2d = np.dot(X_scaled, W_2d)

W_3d = PC_sorted[:, :3]
X_pca_manual_3d = np.dot(X_scaled, W_3d)

pca_2d = PCA(n_components=2)
X_pca_sklearn_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_sklearn_3d = pca_3d.fit_transform(X_scaled)

unique_labels = y.unique()
colors = ['red', 'blue', 'green', 'orange', 'purple']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(2, 2, 1)
for label in unique_labels:
    mask = (y == label)
    ax1.scatter(X_pca_manual_2d[mask, 0], X_pca_manual_2d[mask, 1], 
                c=color_map[label], label=label, alpha=0.7)
ax1.set_title("2D PCA (NumPy Manual)")
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(2, 2, 2)
for label in unique_labels:
    mask = (y == label)
    ax2.scatter(X_pca_sklearn_2d[mask, 0], X_pca_sklearn_2d[mask, 1], 
                c=color_map[label], label=label, alpha=0.7)
ax2.set_title("2D PCA (Scikit-Learn)")
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
for label in unique_labels:
    mask = (y == label)
    ax3.scatter(X_pca_manual_3d[mask, 0], X_pca_manual_3d[mask, 1], X_pca_manual_3d[mask, 2], 
                c=color_map[label], label=label, alpha=0.7)
ax3.set_title("3D PCA (NumPy Manual)")
ax3.set_xlabel("PC 1")
ax3.set_ylabel("PC 2")
ax3.set_zlabel("PC 3")
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
for label in unique_labels:
    mask = (y == label)
    ax4.scatter(X_pca_sklearn_3d[mask, 0], X_pca_sklearn_3d[mask, 1], X_pca_sklearn_3d[mask, 2], 
                c=color_map[label], label=label, alpha=0.7)
ax4.set_title("3D PCA (Scikit-Learn)")
ax4.set_xlabel("PC 1")
ax4.set_ylabel("PC 2")
ax4.set_zlabel("PC 3")
ax4.legend()

plt.tight_layout()
plt.show()