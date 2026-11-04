from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch import nn
from threadpoolctl import threadpool_limits

torch.manual_seed(42)
torch.set_num_threads(4)
path = Path(__file__).parent
df = pd.read_csv(path / "winequality-white.csv", sep=";")
y = df.pop("quality").to_numpy()
X = df.to_numpy(float)
X = (X - X.mean(0)) / X.std(0)
x = torch.tensor(X, dtype=torch.float32)


def draw(ax, Z, title):
    for i, c in enumerate(np.unique(y)):
        ax.scatter(*Z[y == c].T, s=6, alpha=0.45, color=plt.cm.tab10(i), label=str(c))
    ax.set(title=title, xlabel="Компонента 1", ylabel="Компонента 2")
    if Z.shape[1] == 3:
        ax.set_zlabel("Компонента 3")
    ax.legend(title="quality", fontsize=6, markerscale=2)


values, vectors = np.linalg.eig(np.cov(X, rowvar=False))
order = np.argsort(values)[::-1]
values, vectors = values.real[order], vectors.real[:, order]
ae_fig = plt.figure(figsize=(10, 4))
pca_fig = plt.figure(figsize=(10, 4))
for col, k in enumerate((2, 3)):
    encoder = nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, k))
    decoder = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, 11))
    model = nn.Sequential(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(1500):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), x)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        Z = encoder(x).numpy()
        error = nn.functional.mse_loss(model(x), x).item()
    print(f"Автоэнкодер {k}D: MSE = {error:.6f}")
    ax = ae_fig.add_subplot(1, 2, col + 1, projection="3d" if k == 3 else None)
    draw(ax, Z, f"Автоэнкодер, {k}D, MSE = {error:.4f}")
    Z = X @ vectors[:, :k]
    lost = 1 - values[:k].sum() / values.sum()
    print(f"PCA {k}D: потери дисперсии = {lost:.2%}")
    ax = pca_fig.add_subplot(1, 2, col + 1, projection="3d" if k == 3 else None)
    draw(ax, Z, f"PCA, {k}D, потери = {lost:.2%}")
for fig, name in ((ae_fig, "ae.png"), (pca_fig, "pca.png")):
    fig.tight_layout()
    fig.savefig(path / name, dpi=150)

fig = plt.figure(figsize=(15, 9))
for row, k in enumerate((2, 3)):
    for col, p in enumerate((20, 40, 60)):
        with threadpool_limits(limits=2):
            Z = TSNE(n_components=k, perplexity=p, init="pca", random_state=42, n_jobs=1).fit_transform(X)
        print(f"t-SNE {k}D, perplexity={p}: проекция построена")
        ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d" if k == 3 else None)
        draw(ax, Z, f"t-SNE, {k}D, perplexity = {p}")
fig.tight_layout()
fig.savefig(path / "tsne.png", dpi=180)
plt.show()
