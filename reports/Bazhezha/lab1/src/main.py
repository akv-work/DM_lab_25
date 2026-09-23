import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.genfromtxt("seeds_dataset.txt")
X, y = data[:, :-1], data[:, -1].astype(int)
X = (X - X.mean(0)) / X.std(0)

vals, vecs = np.linalg.eig(np.cov(X, rowvar=False))
order = np.argsort(vals.real)[::-1]
vals, vecs = vals.real[order], vecs.real[:, order]
X_eig2, X_eig3 = X @ vecs[:, :2], X @ vecs[:, :3]

X_sk2 = PCA(2).fit_transform(X)
X_sk3 = PCA(3).fit_transform(X)

share = vals / vals.sum()
print("Собственные значения:", np.round(vals, 4))
print("Доля объяснённой дисперсии:", np.round(share, 4))
for k in (2, 3):
    kept = share[:k].sum()
    print(f"{k} ГК: сохранено {kept:.2%}, потери {1 - kept:.2%}")

colors = {1: "tab:red", 2: "tab:green", 3: "tab:blue"}
names = {1: "Kama", 2: "Rosa", 3: "Canadian"}


def scatter(ax, P, title, dim):
    for c in np.unique(y):
        m = y == c
        if dim == 2:
            ax.scatter(P[m, 0], P[m, 1], c=colors[c], label=names[c], s=22)
        else:
            ax.scatter(P[m, 0], P[m, 1], P[m, 2], c=colors[c], label=names[c], s=22)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    if dim == 3:
        ax.set_zlabel("PC3")
    ax.legend()


fig = plt.figure(figsize=(12, 10))
scatter(fig.add_subplot(221), X_eig2, "numpy.linalg.eig — 2 ГК", 2)
scatter(fig.add_subplot(222), X_sk2, "sklearn.PCA — 2 ГК", 2)
scatter(fig.add_subplot(223, projection="3d"), X_eig3, "numpy.linalg.eig — 3 ГК", 3)
scatter(fig.add_subplot(224, projection="3d"), X_sk3, "sklearn.PCA — 3 ГК", 3)
plt.tight_layout()
plt.savefig("pca_visualization.png", dpi=150)
plt.show()
