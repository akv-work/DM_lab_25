import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"


df = pd.read_csv(CSV_PATH)
feature_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X = df[feature_cols].values.astype(float)
y = df["Region"].values
classes = np.unique(y)

X_std = StandardScaler().fit_transform(X)
n, p = X_std.shape

cov = np.cov(X_std, rowvar=False)
eig_vals, eig_vecs = np.linalg.eig(cov)
eig_vals, eig_vecs = eig_vals.real, eig_vecs.real

order = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[order]
eig_vecs = eig_vecs[:, order]

W2 = eig_vecs[:, :2]
W3 = eig_vecs[:, :3]
Z2_manual = X_std @ W2
Z3_manual = X_std @ W3


pca2 = PCA(n_components=2).fit(X_std)
pca3 = PCA(n_components=3).fit(X_std)
Z2_sk = pca2.transform(X_std)
Z3_sk = pca3.transform(X_std)


total_var = eig_vals.sum()
explained_ratio = eig_vals / total_var
cumulative = np.cumsum(explained_ratio)

loss2 = eig_vals[2:].sum() / total_var
loss3 = eig_vals[3:].sum() / total_var


np.set_printoptions(precision=4, suppress=True)
print("Размер выборки:", X.shape, "| классы Region:", dict(zip(*np.unique(y, return_counts=True))))
print("\n--- Способ 1: numpy.linalg.eig ---")
print("Собственные значения:", eig_vals)
print("Доля дисперсии      :", explained_ratio)
print("Накопленная доля    :", cumulative)
print("Собственные векторы (столбцы = ГК):\n", eig_vecs)

print("\n--- Способ 2: sklearn.decomposition.PCA ---")
print("explained_variance_      :", pca3.explained_variance_)
print("explained_variance_ratio_:", pca3.explained_variance_ratio_)


def align(a, b):
    s = np.sign(np.sum(a * b, axis=0))
    return a * s

print("\nСверка способов:")
print("  eigvals совпадают с sklearn:", np.allclose(eig_vals[:3], pca3.explained_variance_))
print("  проекции 2 ГК совпадают (с точностью до знака):", np.allclose(align(Z2_manual, Z2_sk), Z2_sk, atol=1e-6))
print("  проекции 3 ГК совпадают (с точностью до знака):", np.allclose(align(Z3_manual, Z3_sk), Z3_sk, atol=1e-6))

print("\n--- Потери при преобразовании PCA ---")
print(f"k = 2: сохранено {cumulative[1]*100:.2f}% дисперсии, потери = {loss2*100:.2f}% "
      f"(сумма отброшенных λ = {eig_vals[2:].sum():.4f} из {total_var:.4f})")
print(f"k = 3: сохранено {cumulative[2]*100:.2f}% дисперсии, потери = {loss3*100:.2f}% "
      f"(сумма отброшенных λ = {eig_vals[3:].sum():.4f} из {total_var:.4f})")


colors = {1: "tab:red", 2: "tab:green", 3: "tab:blue"}
markers = {1: "o", 2: "s", 3: "^"}
names = {1: "Region 1 (Lisbon)", 2: "Region 2 (Oporto)", 3: "Region 3 (Other)"}


def plot2d(ax, Z, title):
    for c in classes:
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], c=colors[c], marker=markers[c], s=25,
                   alpha=0.7, edgecolors="k", linewidths=0.3, label=names[c])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()


def plot3d(ax, Z, title):
    for c in classes:
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], c=colors[c], marker=markers[c],
                   s=25, alpha=0.7, label=names[c])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()



fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot2d(axes[0], Z2_manual, f"Вручную (numpy.linalg.eig), 2 ГК\nпотери {loss2*100:.1f}%")
plot2d(axes[1], Z2_sk, f"sklearn PCA, 2 ГК\nпотери {loss2*100:.1f}%")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=150)


fig = plt.figure(figsize=(15, 7))
plot3d(fig.add_subplot(121, projection="3d"), Z3_manual,
       f"Вручную (numpy.linalg.eig), 3 ГК\nпотери {loss3*100:.1f}%")
plot3d(fig.add_subplot(122, projection="3d"), Z3_sk,
       f"sklearn PCA, 3 ГК\nпотери {loss3*100:.1f}%")
plt.tight_layout()
plt.savefig("pca_3d.png", dpi=150)


fig, ax = plt.subplots(figsize=(7, 5))
idx = np.arange(1, p + 1)
ax.bar(idx, explained_ratio, alpha=0.7, label="Доля дисперсии")
ax.step(idx, cumulative, where="mid", color="k", label="Накопленная доля")
ax.set_xlabel("Номер главной компоненты")
ax.set_ylabel("Доля дисперсии")
ax.set_title("Собственные значения и потери")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150)

if "agg" not in plt.get_backend().lower():
    plt.show()