from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "wdbc.data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURES = ([f"{f}_mean" for f in BASE_FEATURES]
            + [f"{f}_se" for f in BASE_FEATURES]
            + [f"{f}_worst" for f in BASE_FEATURES])
ID_COL = "ID"
CLASS_COL = "Diagnosis"
CLASS_NAMES = {0: "Benign (B)", 1: "Malignant (M)"}
CLASS_COLORS = {0: "tab:green", 1: "tab:red"}
CLASS_MARKERS = {0: "o", 1: "^"}


def load_data(verbose=True):
    df = pd.read_csv(DATA_FILE, header=None, names=[ID_COL, CLASS_COL] + FEATURES)

    missing = int(df[FEATURES].isna().sum().sum())
    if missing:
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    y = df[CLASS_COL].map({"B": 0, "M": 1}).to_numpy(dtype=int)
    X = df.drop(columns=[ID_COL, CLASS_COL]).to_numpy(dtype=float)
    if verbose:
        print(f"Загружено объектов: {len(df)}, признаков: {X.shape[1]}, "
              f"пропусков заменено: {missing}")
        print(f"Классы: B = {(y == 0).sum()}, M = {(y == 1).sum()}")
    return X, y


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def separability(Z, y):
    sil = silhouette_score(Z, y)
    acc = cross_val_score(LogisticRegression(max_iter=1000), Z, y, cv=5).mean()
    return sil, acc


def plot_2d(Z, y, title, filename, labels=("Компонента 1", "Компонента 2"), ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 6))
    for cls in CLASS_NAMES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                   label=CLASS_NAMES[cls], edgecolors="k", linewidths=0.3, alpha=0.75, s=28)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    if own:
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=150)


def plot_3d(Z, y, title, filename,
            labels=("Компонента 1", "Компонента 2", "Компонента 3"), ax=None):
    own = ax is None
    if own:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    for cls in CLASS_NAMES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], c=CLASS_COLORS[cls],
                   marker=CLASS_MARKERS[cls], label=CLASS_NAMES[cls],
                   edgecolors="k", linewidths=0.3, alpha=0.8, s=22, depthshade=False)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.view_init(elev=20, azim=-60)
    ax.legend()
    if own:
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=150)
