from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch import nn

torch.manual_seed(0)
OUT = Path(__file__).parent
df = pd.read_csv(OUT / "rice.csv")
y = df.pop("Class").to_numpy()
X = df.to_numpy(float)
X = (X - X.mean(0)) / X.std(0)
labels = pd.factorize(y)[0]
MARK = dict(zip(np.unique(y), "os^vD"))


class AE(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, k))
        self.dec = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, n))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def fit_ae(k, epochs=500):
    x = torch.tensor(X, dtype=torch.float32)
    model = AE(x.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        mse(model(x)[0], x).backward()
        opt.step()
    with torch.no_grad():
        rec, z = model(x)
    return z.numpy(), mse(rec, x).item()


def pca(data):
    vals, vecs = np.linalg.eig(np.cov(data, rowvar=False))
    order = np.argsort(vals.real)[::-1]
    return data @ vecs.real[:, order], vals.real[order]


def draw(ax, Z, title):
    for cls in np.unique(y):
        m = y == cls
        ax.scatter(*Z[m].T, s=10, alpha=0.55, marker=MARK[cls], label=cls)
    ax.set_title(title, fontsize=9)
    ax.legend(markerscale=2, fontsize=7)


def show(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.show()
    plt.close(fig)


ae = {}
fig = plt.figure(figsize=(10, 4))
for i, k in enumerate((2, 3), 1):
    Z, err = fit_ae(k)
    ae[k] = err
    ax = fig.add_subplot(1, 2, i, projection="3d" if k == 3 else None)
    draw(ax, Z, f"Автоэнкодер, {k} нейрона\nMSE = {err:.4f}")
    print(f"AE {k}: MSE = {err:.6f}")
show(fig, "ae.png")

Z, vals = pca(X)
loss = {k: 1 - vals[:k].sum() / vals.sum() for k in (2, 3)}
fig = plt.figure(figsize=(10, 4))
for i, k in enumerate((2, 3), 1):
    ax = fig.add_subplot(1, 2, i, projection="3d" if k == 3 else None)
    draw(ax, Z[:, :k], f"PCA, {k} ГК\nпотери = {loss[k]:.2%}")
    print(f"PCA {k}: потери = {loss[k]:.4%}")
show(fig, "pca.png")

perps = (20, 40, 60)
tsne = {}
for k in (2, 3):
    for p in perps:
        emb = TSNE(n_components=k, perplexity=p, init="pca", random_state=0).fit_transform(X)
        tsne[(k, p)] = emb, silhouette_score(emb, labels)
        print(f"t-SNE {k}D, perplexity={p}: silhouette = {tsne[(k, p)][1]:.4f}")
best = {k: max(perps, key=lambda p: tsne[(k, p)][1]) for k in (2, 3)}

fig = plt.figure(figsize=(12, 7))
for i, (k, p) in enumerate((k, p) for k in (2, 3) for p in perps):
    emb, sil = tsne[(k, p)]
    tag = ", лучшая" if p == best[k] else ""
    ax = fig.add_subplot(2, 3, i + 1, projection="3d" if k == 3 else None)
    draw(ax, emb, f"t-SNE {k}D, perplexity={p}{tag}\nsilhouette = {sil:.3f}")
show(fig, "tsne.png")

print("\nВыводы:")
print(f"1. Автоэнкодер (ReLU, Adam) проецирует 7 признаков риса в узкий слой. "
      f"MSE = {ae[2]:.4f} при 2 нейронах и {ae[3]:.4f} при 3: третья компонента уменьшает ошибку восстановления.")
print(f"2. PCA из ЛР1 (eig ковариационной матрицы) теряет {loss[2]:.2%} дисперсии на 2 ГК и {loss[3]:.2%} на 3 ГК.")
print(f"3. t-SNE (init=PCA) построен при perplexity {', '.join(map(str, perps))}. "
      f"Лучшее разделение классов по silhouette: perplexity {best[2]} (2D) и {best[3]} (3D).")
print(f"4. Cammeo и Osmancik сдвигаются вдоль первой оси и у PCA, и у автоэнкодера, с перекрытием на границе. "
      f"Трёх компонент для 7 признаков риса хватает: PCA теряет {loss[3]:.2%} дисперсии, автоэнкодер снижает MSE до {ae[3]:.4f}. "
      f"t-SNE при perplexity {best[2]} чуть лучше отделяет сорта, но хранит соседство точек, а не долю дисперсии — её по-прежнему показывает PCA.")
