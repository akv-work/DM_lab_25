from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch import nn

torch.manual_seed(0)
np.random.seed(0)

OUT = Path(__file__).parent
DATA_FILE = OUT / "wdbc.data"

EPOCHS = 500
LR = 1e-3
DIMS = (2, 3)
PERPS = (20, 40, 60)

COLORS = {"M": "#e4572e", "B": "#2e86ab"}
MARKERS = {"M": "o", "B": "^"}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.facecolor": "#f7f7fb",
    "axes.facecolor": "#ffffff",
})


def load_data(path: Path):
    df = pd.read_csv(path, header=None)
    y_raw = df[1].to_numpy()
    X = df.iloc[:, 2:].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    y_labels = pd.factorize(y_raw)[0]
    return X, y_raw, y_labels


class AE(nn.Module):
    def __init__(self, n_in: int, n_lat: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, n_lat),
        )
        self.dec = nn.Sequential(
            nn.Linear(n_lat, 32), nn.ReLU(),
            nn.Linear(32, n_in),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def fit_ae(X, k, epochs=EPOCHS, lr=LR):
    x = torch.tensor(X, dtype=torch.float32)
    model = AE(x.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = mse(model(x)[0], x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec, z = model(x)
    return z.numpy(), mse(rec, x).item()


def pca_eig(data):
    vals, vecs = np.linalg.eig(np.cov(data, rowvar=False))
    order = np.argsort(vals.real)[::-1]
    return data @ vecs.real[:, order], vals.real[order]


def scatter_by_class(ax, Z, y_raw, axis_names):
    for cls in ("B", "M"):
        m = y_raw == cls
        if not m.any():
            continue
        ax.scatter(
            *Z[m].T,
            s=22 if ax.name != "3d" else 18,
            alpha=0.75,
            marker=MARKERS[cls],
            c=COLORS[cls],
            edgecolors="white",
            linewidths=0.4,
            label=cls,
        )
    if ax.name == "3d":
        ax.set_xlabel(axis_names[0], labelpad=-4)
        ax.set_ylabel(axis_names[1], labelpad=-4)
        ax.set_zlabel(axis_names[2], labelpad=-4)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=22, azim=-60)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("#eef1f7")
            pane.pane.set_edgecolor("#c9ced8")
    else:
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_aspect("equal", adjustable="datalim")


def style_ax(ax, title, subtitle=None, highlight=False):
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", pad=12)
    else:
        ax.set_title(title, pad=12)
    if highlight:
        ax.title.set_bbox(dict(
            boxstyle="round,pad=0.35",
            fc="#fff7d6", ec="#f0c419", lw=0.9,
        ))
    ax.legend(markerscale=1.6, fontsize=7, loc="best")


def make_fig(ncols, nrows=1, width=6.0, height=4.0):
    return plt.figure(figsize=(width * ncols, height * nrows))


def save_show(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    X, y_raw, y_labels = load_data(DATA_FILE)
    print(f"Загружено: {X.shape[0]} объектов, {X.shape[1]} признаков.")
    print(f"Классы: {dict(zip(*np.unique(y_raw, return_counts=True)))}\n")

    ae_err = {}
    fig = make_fig(ncols=2, width=4.6, height=4.2)
    for i, k in enumerate(DIMS, 1):
        Z, err = fit_ae(X, k)
        ae_err[k] = err
        ax = fig.add_subplot(1, 2, i, projection="3d" if k == 3 else None)
        scatter_by_class(ax, Z, y_raw, [f"z{j+1}" for j in range(k)] + [""] * (3 - k))
        style_ax(ax,
                 f"Автоэнкодер · узкий слой {k}D",
                 f"MSE = {err:.4f}",
                 highlight=True)
        print(f"[AE] {k}D: MSE = {err:.6f}")
    save_show(fig, "ae.png")

    Z_pca, vals = pca_eig(X)
    pca_loss = {k: 1 - vals[:k].sum() / vals.sum() for k in DIMS}

    fig = make_fig(ncols=2, width=4.6, height=4.2)
    for i, k in enumerate(DIMS, 1):
        ax = fig.add_subplot(1, 2, i, projection="3d" if k == 3 else None)
        scatter_by_class(ax, Z_pca[:, :k], y_raw,
                         [f"PC{j+1}" for j in range(k)] + [""] * (3 - k))
        style_ax(ax,
                 f"PCA · {k} главных компонент",
                 f"потери дисперсии = {pca_loss[k]:.2%}",
                 highlight=True)
        print(f"[PCA] {k} ГК: потери = {pca_loss[k]:.4%}")
    save_show(fig, "pca.png")

    tsne_res = {}
    for k in DIMS:
        for p in PERPS:
            emb = TSNE(
                n_components=k, perplexity=p,
                init="pca", random_state=0,
                learning_rate="auto",
            ).fit_transform(X)
            sil = silhouette_score(emb, y_labels)
            tsne_res[(k, p)] = (emb, sil)
            print(f"[t-SNE] {k}D, perplexity={p}: silhouette = {sil:.4f}")

    best_p = {k: max(PERPS, key=lambda p: tsne_res[(k, p)][1]) for k in DIMS}

    fig = plt.figure(figsize=(13.5, 8.0))
    for idx, (k, p) in enumerate([(k, p) for k in DIMS for p in PERPS], 1):
        emb, sil = tsne_res[(k, p)]
        ax = fig.add_subplot(2, 3, idx, projection="3d" if k == 3 else None)
        scatter_by_class(ax, emb, y_raw,
                         [f"t{j+1}" for j in range(k)] + [""] * (3 - k))
        best = (p == best_p[k])
        style_ax(
            ax,
            f"t-SNE {k}D · perplexity={p}" + ("  ★" if best else ""),
            f"silhouette = {sil:.3f}",
            highlight=best,
        )
    fig.suptitle("t-SNE (init=PCA): сравнение perplexity и размерности",
                 fontsize=12, fontweight="bold", y=1.02)
    save_show(fig, "tsne.png")


if __name__ == "__main__":
    main()