import os
import re
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PATH = sys.argv[1] if len(sys.argv) > 1 else "orig.cv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SEED = 0
os.makedirs(OUT, exist_ok=True)


BITMAP = re.compile(r"^[01]{32}$")


def load_optdigits(path):
    bitmaps, labels, rows = [], [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if BITMAP.match(s):
                rows.append([int(ch) for ch in s])
            elif len(rows) == 32 and s.isdigit():
                bitmaps.append(rows)
                labels.append(int(s))
                rows = []
    if rows:
        raise ValueError("файл оборвался посреди карты")
    return np.array(bitmaps, dtype=np.uint8), np.array(labels)


B, y = load_optdigits(PATH)
n = len(B)
# свёртка 32x32 -> 8x8: сумма единиц в блоках 4x4 (значения 0..16)
X = B.reshape(n, 8, 4, 8, 4).sum(axis=(2, 4)).reshape(n, 64).astype(float)
Xs = X / 16.0                       
classes = np.unique(y)

print(f"Загружено: {n} объектов, исходно 32x32=1024 пикселя, после свёртки {X.shape[1]} признака")
print("Классы (цифры):", dict(zip(*np.unique(y, return_counts=True))))


def train_autoencoder(data, k, seed=SEED):
    ae = MLPRegressor(hidden_layer_sizes=(32, 16, k, 16, 32), activation="tanh",
                      solver="adam", learning_rate_init=1e-3, max_iter=2000,
                      tol=1e-6, n_iter_no_change=40, random_state=seed)
    ae.fit(data, data)
    return ae


def encode(ae, data):
    a = data
    for i in range(3):                      
        a = np.tanh(a @ ae.coefs_[i] + ae.intercepts_[i])
    return a


def mse(a, b):
    return float(np.mean((a - b) ** 2))


ae_Z, ae_mse = {}, {}
for k in (2, 3):
    ae = train_autoencoder(Xs, k)
    ae_Z[k] = encode(ae, Xs)
    ae_mse[k] = mse(ae.predict(Xs), Xs)
    print(f"Автоэнкодер, {k} нейрона в среднем слое: эпох {ae.n_iter_}, MSE реконструкции {ae_mse[k]:.5f}")


mu = Xs.mean(axis=0)
Xc = Xs - mu
cov = np.cov(Xc, rowvar=False)
eig_vals, eig_vecs = np.linalg.eig(cov)
eig_vals, eig_vecs = eig_vals.real, eig_vecs.real
order = np.argsort(eig_vals)[::-1]
eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]

total = eig_vals.sum()
ratio = eig_vals / total
pca_Z, pca_mse, pca_loss = {}, {}, {}
for k in (2, 3):
    W = eig_vecs[:, :k]
    pca_Z[k] = Xc @ W
    recon = pca_Z[k] @ W.T + mu
    pca_mse[k] = mse(recon, Xs)
    pca_loss[k] = eig_vals[k:].sum() / total
    print(f"PCA, {k} компоненты: сохранено {ratio[:k].sum()*100:.2f}% дисперсии, "
          f"потери {pca_loss[k]*100:.2f}%, MSE реконструкции {pca_mse[k]:.5f}")


sk = PCA(n_components=3).fit(Xs)
print("Сверка с sklearn.PCA: собственные значения совпадают:",
      np.allclose(eig_vals[:3], sk.explained_variance_))
print("Первые 5 собственных значений:", np.round(eig_vals[:5], 4))


tsne_Z = {}
for k in (2, 3):
    tsne_Z[k] = TSNE(n_components=k, perplexity=30, init="pca",
                     learning_rate="auto", random_state=SEED).fit_transform(Xs)
    print(f"t-SNE, {k} компоненты: готово")


colors = plt.cm.tab10(np.arange(10))
markers = ["o", "s", "^", "v", "D", "<", ">", "p", "P", "X"]


def scatter2d(ax, Z, labels_xy):
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], color=colors[i], marker=markers[i], s=22,
                   alpha=0.75, edgecolors="k", linewidths=0.25, label=str(c))
    ax.set_xlabel(labels_xy[0])
    ax.set_ylabel(labels_xy[1])
    ax.grid(alpha=0.3)


def scatter3d(ax, Z, labels_xyz):
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], color=colors[i], marker=markers[i],
                   s=18, alpha=0.75, edgecolors="k", linewidths=0.2, label=str(c))
    ax.set_xlabel(labels_xyz[0])
    ax.set_ylabel(labels_xyz[1])
    ax.set_zlabel(labels_xyz[2])
    ax.view_init(elev=22, azim=45)


def make_figure(name, title, Z2, Z3, axis_name):
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    scatter2d(ax1, Z2, [f"{axis_name}1", f"{axis_name}2"])
    ax1.set_title(f"{title}: 2 компоненты")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    scatter3d(ax2, Z3, [f"{axis_name}1", f"{axis_name}2", f"{axis_name}3"])
    ax2.set_title(f"{title}: 3 компоненты")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, title="Цифра", loc="center right", markerscale=1.6)
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("сохранено:", path)


make_figure("ae.png", "Автоэнкодер", ae_Z[2], ae_Z[3], "AE")
make_figure("tsne.png", "t-SNE", tsne_Z[2], tsne_Z[3], "t-SNE")
make_figure("pca.png", "PCA", pca_Z[2], pca_Z[3], "PC")

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def knn_acc(Z):
    return cross_val_score(KNeighborsClassifier(5), Z, y, cv=cv).mean()


embeddings = {"PCA": pca_Z, "Автоэнкодер": ae_Z, "t-SNE": tsne_Z}
lines = ["Метод          | k | kNN-точность (5-fold) | silhouette"]
lines.append("-" * len(lines[0]))
for name, d in embeddings.items():
    for k in (2, 3):
        lines.append(f"{name:<14} | {k} | {knn_acc(d[k]):.3f}                 | {silhouette_score(d[k], y):.3f}")
lines.append("")
lines.append("Полная размерность (64 признака): kNN-точность "
             f"{knn_acc(Xs):.3f} (потолок для сравнения)")
lines.append("")
lines.append("Ошибка реконструкции (MSE на масштабе [0,1]):")
for k in (2, 3):
    lines.append(f"  k={k}: автоэнкодер {ae_mse[k]:.5f} | PCA {pca_mse[k]:.5f}")
text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(OUT, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(text + "\n")