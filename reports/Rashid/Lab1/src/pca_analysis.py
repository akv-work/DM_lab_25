import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

# ============================================================
# Variant 12: hcv+data.zip / hcvdat0.csv, class = Category
# ============================================================

DATA_PATH = "hcvdat0.csv"

# 1. Load data
df = pd.read_csv(DATA_PATH, index_col=0, na_values=["NA", ""])

print("Data shape:", df.shape)
print("Classes:")
print(df["Category"].value_counts())

# Target column is used only for coloring.
# It must NOT be included in PCA.
y = df["Category"].astype(str).copy()
X = df.drop(columns=["Category"]).copy()

# 2. Preprocessing
# Encode Sex: m -> 1, f -> 0
X["Sex"] = X["Sex"].map({"m": 1, "f": 0})

# Fill missing values in numerical features with the median
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

X = X.astype(float)

# Standardize features manually.
# PCA is sensitive to scale, so z-score normalization is used.
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
X_scaled = ((X - X_mean) / X_std).values

y_arr = y.values
classes = list(pd.unique(y_arr))

# ============================================================
# 3. Manual PCA using numpy.linalg.eig
# ============================================================

def manual_pca(Xs, n_components):
    """
    Manual PCA:
    1) center the data,
    2) compute the covariance matrix,
    3) compute eigenvalues/eigenvectors with np.linalg.eig,
    4) sort them in descending order,
    5) project the data.
    """
    Xc = Xs - Xs.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)

    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    scores = Xc @ eigvecs[:, :n_components]
    return scores, eigvals, eigvecs


# Compute all components first, then take 2 and 3
scores_manual_all, eigvals, eigvecs = manual_pca(
    X_scaled, n_components=X_scaled.shape[1]
)

scores_manual_2 = scores_manual_all[:, :2]
scores_manual_3 = scores_manual_all[:, :3]

# ============================================================
# 4. PCA using sklearn
# ============================================================

pca2 = PCA(n_components=2, random_state=42)
scores_sklearn_2 = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3, random_state=42)
scores_sklearn_3 = pca3.fit_transform(X_scaled)

# ============================================================
# 5. Losses based on eigenvalues
# ============================================================

total_variance = eigvals.sum()

print("\nEigenvalues:")
print(np.round(eigvals, 6))

print("\nPCA losses:")
for k in [2, 3]:
    retained = eigvals[:k].sum() / total_variance
    loss = 1.0 - retained
    discarded = total_variance - eigvals[:k].sum()

    print(f"k = {k}:")
    print(f"  retained variance ratio = {retained:.6f}")
    print(f"  relative loss           = {loss:.6f} ({loss * 100:.2f}%)")
    print(f"  discarded variance      = {discarded:.6f}")

print("\nCheck with sklearn:")
print("2 components, explained_variance_ratio_:")
print(pca2.explained_variance_ratio_)
print("Sum:", pca2.explained_variance_ratio_.sum())

print("3 components, explained_variance_ratio_:")
print(pca3.explained_variance_ratio_)
print("Sum:", pca3.explained_variance_ratio_.sum())

# ============================================================
# 6. Visualization
# ============================================================

colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_2d(scores, title, out_path):
    plt.figure(figsize=(9, 7))

    for i, cls in enumerate(classes):
        mask = y_arr == cls
        plt.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=[colors[i]],
            marker=markers[i % len(markers)],
            label=cls,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.3,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def plot_3d(scores, title, out_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, cls in enumerate(classes):
        mask = y_arr == cls
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            scores[mask, 2],
            c=[colors[i]],
            marker=markers[i % len(markers)],
            label=cls,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.2,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


plot_2d(
    scores_manual_2,
    "Manual PCA via np.linalg.eig: PC1-PC2",
    "pca_manual_2d.png",
)

plot_3d(
    scores_manual_3,
    "Manual PCA via np.linalg.eig: PC1-PC3",
    "pca_manual_3d.png",
)

plot_2d(
    scores_sklearn_2,
    "PCA via sklearn: PC1-PC2",
    "pca_sklearn_2d.png",
)

plot_3d(
    scores_sklearn_3,
    "PCA via sklearn: PC1-PC3",
    "pca_sklearn_3d.png",
)