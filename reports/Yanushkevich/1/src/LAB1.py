import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

columns = [
    "area", "perimeter", "compactness",
    "kernel_length", "kernel_width",
    "asymmetry_coefficient", "kernel_groove_length",
    "class",
]

df = pd.read_csv(
    "seeds_dataset.txt",
    sep=r"\s+",
    header=None,
    names=columns,
)

df = df.fillna(df.mean(numeric_only=True))

target = df["class"].values
X = df.drop(columns=["class"]).values

X_scaled = StandardScaler().fit_transform(X)

colors = {1: "red", 2: "blue", 3: "green"}
labels = {1: "Класс 1", 2: "Класс 2", 3: "Класс 3"}


def pca_manual(data, n_components):
    L = data.shape[0]

    mean_vector = np.sum(data, axis=0) / L
    X_centered = data - mean_vector

    K = (X_centered.T @ X_centered) / L

    eig_values, eig_vectors = np.linalg.eig(K)
    eig_values = eig_values.real
    eig_vectors = eig_vectors.real

    sort_index = np.argsort(-1 * eig_values)
    eig_values_sorted = eig_values[sort_index]
    eig_vectors_sorted = eig_vectors[:, sort_index]

    eig_vectors_sorted = eig_vectors_sorted / np.linalg.norm(eig_vectors_sorted, axis=0)

    W = eig_vectors_sorted[:, :n_components]
    Y = X_centered @ W

    return Y, eig_values_sorted


X_manual_2d, eig_values_sorted = pca_manual(X_scaled, n_components=2)
X_manual_3d, _ = pca_manual(X_scaled, n_components=3)


pca_2d = PCA(n_components=2)
X_sklearn_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_sklearn_3d = pca_3d.fit_transform(X_scaled)


def scatter_2d(ax, data, title):
    for cls in colors:
        mask = target == cls
        ax.scatter(data[mask, 0], data[mask, 1],
                    c=colors[cls], label=labels[cls], edgecolor="k", s=40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()


def scatter_3d(ax, data, title):
    for cls in colors:
        mask = target == cls
        ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                    c=colors[cls], label=labels[cls], edgecolor="k", s=40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()


fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(2, 2, 1)
scatter_2d(ax1, X_manual_2d, "Вручную (numpy.linalg.eig), 2 компоненты")

ax2 = fig.add_subplot(2, 2, 2)
scatter_2d(ax2, X_sklearn_2d, "sklearn.PCA, 2 компоненты")

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
scatter_3d(ax3, X_manual_3d, "Вручную (numpy.linalg.eig), 3 компоненты")

ax4 = fig.add_subplot(2, 2, 4, projection="3d")
scatter_3d(ax4, X_sklearn_3d, "sklearn.PCA, 3 компоненты")

plt.tight_layout()
plt.savefig("pca_seeds_visualization.png", dpi=150)
plt.show()


def criterion_J(eig_values, n_components):
    return eig_values[:n_components].sum() / eig_values.sum()


loss_2d = 100 - criterion_J(eig_values_sorted, 2) * 100
loss_3d = 100 - criterion_J(eig_values_sorted, 3) * 100

loss_2d_sklearn = 100 - pca_2d.explained_variance_ratio_.sum() * 100
loss_3d_sklearn = 100 - pca_3d.explained_variance_ratio_.sum() * 100

print("Собственные значения (по убыванию):", np.round(eig_values_sorted, 4))
print(f"Потери информативности при 2 компонентах (вручную): {loss_2d:.2f}%")
print(f"Потери информативности при 3 компонентах (вручную): {loss_3d:.2f}%")
print(f"Потери информативности при 2 компонентах (sklearn): {loss_2d_sklearn:.2f}%")
print(f"Потери информативности при 3 компонентах (sklearn): {loss_3d_sklearn:.2f}%")