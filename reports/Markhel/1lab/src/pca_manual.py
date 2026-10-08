from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "seeds_dataset.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Area",             # площадь A
    "Perimeter",        # периметр P
    "Compactness",      # компактность C = 4*pi*A/P^2
    "Kernel_Length",    # длина зерна
    "Kernel_Width",     # ширина зерна
    "Asymmetry",        # коэффициент асимметрии
    "Groove_Length",    # длина бороздки зерна
]
CLASS_COL = "Class"
CLASS_NAMES = {1: "Kama", 2: "Rosa", 3: "Canadian"}
CLASS_COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}
CLASS_MARKERS = {1: "o", 2: "s", 3: "^"}


def load_data():

    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None,
                     names=FEATURES + [CLASS_COL], engine="python")

    missing = int(df[FEATURES].isna().sum().sum())
    if missing:
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    print(f"Загружено объектов: {len(df)}, признаков: {len(FEATURES)}, "
          f"пропусков заменено: {missing}")
    print("Распределение по классам:",
          {CLASS_NAMES[k]: int(v) for k, v in df[CLASS_COL].value_counts().sort_index().items()})

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[CLASS_COL].to_numpy(dtype=int)
    return X, y


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def pca_method(data, n_components):
    cov_matrix = np.cov(data.T)
    V, PC = np.linalg.eig(cov_matrix)
    V, PC = V.real, PC.real
    sort_index = np.argsort(-1 * V)
    V = V[sort_index]
    PC = PC[:, sort_index]
    signs = np.sign(PC[np.argmax(np.abs(PC), axis=0), range(PC.shape[1])])
    PC = PC * signs
    projected = np.dot(PC.T[0:n_components], data.T).T
    return projected, V, PC


def plot_2d(Z, y, V, title, filename):
    total = V.sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in CLASS_NAMES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                   label=CLASS_NAMES[cls], edgecolors="k", linewidths=0.4, alpha=0.8)
    ax.set_xlabel(f"PC1 ({V[0] / total * 100:.1f}% дисперсии)")
    ax.set_ylabel(f"PC2 ({V[1] / total * 100:.1f}% дисперсии)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)


def plot_3d(Z, y, V, title, filename):
    total = V.sum()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for cls in CLASS_NAMES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], c=CLASS_COLORS[cls],
                   marker=CLASS_MARKERS[cls], label=CLASS_NAMES[cls],
                   edgecolors="k", linewidths=0.3, alpha=0.85, depthshade=False)
    ax.set_xlabel(f"PC1 ({V[0] / total * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({V[1] / total * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({V[2] / total * 100:.1f}%)")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)


def main():
    X, y = load_data()
    Xs = standardize(X)

    Z2, V, PC = pca_method(Xs, 2)
    Z3, _, _ = pca_method(Xs, 3)

    print("\nСобственные значения ковариационной матрицы (numpy.linalg.eig):")
    full_info = V.sum()
    for i, v in enumerate(V, 1):
        print(f"  λ{i} = {v:.4f}  (доля дисперсии = {v / full_info * 100:6.2f}%)")

    print("\nСобственные векторы (столбцы — главные компоненты PC1..PC3):")
    print(pd.DataFrame(PC[:, :3], index=FEATURES, columns=["PC1", "PC2", "PC3"]).round(4))

    print("\nПотери информации:")
    for k in (2, 3):
        reduce_info = V[0:k].sum()
        loss = 100 - reduce_info / full_info * 100
        print(f"  [k={k}] сохранено = {reduce_info / full_info * 100:.2f}%  |  "
              f"потеря = {loss:.2f}%")

    plot_2d(Z2, y, V, "PCA (numpy.linalg.eig), 2 компоненты — Seeds", "manual_2d.png")
    plot_3d(Z3, y, V, "PCA (numpy.linalg.eig), 3 компоненты — Seeds", "manual_3d.png")
    print(f"\nГрафики сохранены в папку: {FIG_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
