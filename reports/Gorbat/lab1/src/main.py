from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DATA = Path(__file__).parent / "Exasens.csv"
COLS = ["Diagnosis", "ID", "ImagMin", "ImagAvg", "RealMin", "RealAvg", "Gender", "Age", "Smoking"]
FEATS = COLS[2:]

df = pd.read_csv(DATA, skiprows=3, header=None, usecols=range(9), names=COLS)
df[FEATS] = df[FEATS].apply(pd.to_numeric, errors="coerce")
df[FEATS] = df[FEATS].fillna(df[FEATS].median())
y = df["Diagnosis"]
X = df[FEATS].to_numpy(float)
X = (X - X.mean(0)) / X.std(0)

cov = np.cov(X, rowvar=False)
vals, vecs = np.linalg.eig(cov)
idx = np.argsort(vals.real)[::-1]
vals, vecs = vals.real[idx], vecs.real[:, idx]
Z_np = X @ vecs

pca = PCA().fit(X)
Z_sk = pca.transform(X)

print("Потери PCA (доля необъяснённой дисперсии):")
for k in (2, 3):
    loss_np = 1 - vals[:k].sum() / vals.sum()
    loss_sk = 1 - pca.explained_variance_ratio_[:k].sum()
    print(f"  {k} ГК: numpy={loss_np:.4%}  sklearn={loss_sk:.4%}")
print("Собственные значения:", vals)


def plot(Z, title, ax2, ax3):
    for cls in sorted(y.unique()):
        m = y == cls
        ax2.scatter(Z[m, 0], Z[m, 1], s=18, label=cls)
        ax3.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=18, label=cls)
    ax2.set(xlabel="PC1", ylabel="PC2", title=f"{title} — 2 ГК")
    ax3.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=f"{title} — 3 ГК")
    ax2.legend()
    ax3.legend()


fig = plt.figure(figsize=(12, 10))
plot(Z_np, "numpy.linalg.eig", fig.add_subplot(221), fig.add_subplot(222, projection="3d"))
plot(Z_sk, "sklearn.PCA", fig.add_subplot(223), fig.add_subplot(224, projection="3d"))
plt.tight_layout()
plt.savefig(Path(__file__).parent / "pca_exasens.png", dpi=150)
plt.show()
