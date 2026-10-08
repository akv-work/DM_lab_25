from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "seeds_dataset.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Area", "Perimeter", "Compactness", "Kernel_Length",
    "Kernel_Width", "Asymmetry", "Groove_Length",
]
CLASS_COL = "Class"
CLASS_NAMES = {1: "Kama", 2: "Rosa", 3: "Canadian"}
CLASS_COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}
CLASS_MARKERS = {1: "o", 2: "s", 3: "^"}


def load_data():
    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None,
                     names=FEATURES + [CLASS_COL], engine="python")
    X = df.drop(columns=[CLASS_COL])          # класс в PCA не участвует
    y = df[CLASS_COL].to_numpy(dtype=int)
    print(f"Загружено объектов: {len(df)}, признаков: {X.shape[1]}, "
          f"пропусков: {int(X.isna().sum().sum())}")
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y


def fixed_sign(pca, Z):
    comps = pca.components_
    signs = np.sign(comps[range(comps.shape[0]), np.argmax(np.abs(comps), axis=1)])
    return Z * signs


def plot_2d(Z, y, ratio, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in CLASS_NAMES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                   label=CLASS_NAMES[cls], edgecolors="k", linewidths=0.4, alpha=0.8)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}% дисперсии)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}% дисперсии)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)


def plot_3d(Z, y, ratio, title, filename):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for cls in CLASS_NAMES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], c=CLASS_COLORS[cls],
                   marker=CLASS_MARKERS[cls], label=CLASS_NAMES[cls],
                   edgecolors="k", linewidths=0.3, alpha=0.85, depthshade=False)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratio[2] * 100:.1f}%)")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)


def main():
    X, y = load_data()
    Xs = StandardScaler().fit_transform(X)

    pca2 = PCA(n_components=2)
    Z2 = fixed_sign(pca2, pca2.fit_transform(Xs))
    pca3 = PCA(n_components=3)
    Z3 = fixed_sign(pca3, pca3.fit_transform(Xs))

    # Полный PCA — чтобы вывести все собственные значения.
    pca_full = PCA().fit(Xs)
    V = pca_full.explained_variance_
    ratio = pca_full.explained_variance_ratio_
    print("\nСобственные значения (explained_variance_):")
    for i, (v, r) in enumerate(zip(V, ratio), 1):
        print(f"  λ{i} = {v:.4f}  (доля дисперсии = {r * 100:6.2f}%)")

    print("\nГлавные компоненты (components_):")
    print(pd.DataFrame(pca3.components_.T, index=FEATURES,
                       columns=["PC1", "PC2", "PC3"]).round(4))

    print("\nПотери информации:")
    for k, model in ((2, pca2), (3, pca3)):
        kept = model.explained_variance_ratio_.sum() * 100
        print(f"  [k={k}] сохранено = {kept:.2f}%  |  потеря = {100 - kept:.2f}%")

    plot_2d(Z2, y, ratio, "PCA (sklearn), 2 компоненты — Seeds", "sklearn_2d.png")
    plot_3d(Z3, y, ratio, "PCA (sklearn), 3 компоненты — Seeds", "sklearn_3d.png")
    print(f"\nГрафики сохранены в папку: {FIG_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
