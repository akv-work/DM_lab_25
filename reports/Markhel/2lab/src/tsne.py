"""
Лабораторная работа №2. t-SNE (sklearn.manifold.TSNE) для выборки WDBC, 2 и 3 компоненты.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from common import BASE_DIR, FIG_DIR, load_data, standardize, separability, plot_2d, plot_3d

SEED = 42
PERPLEXITIES = [20, 30, 40, 50, 60]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_tsne(Xs, n_components, perplexity):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=SEED)
    Z = tsne.fit_transform(Xs)
    return Z, tsne.kl_divergence_


def main():
    X, y = load_data()
    Xs = standardize(X)

    rows, embeddings = [], {}
    for k in (2, 3):
        for p in PERPLEXITIES:
            Z, kl = run_tsne(Xs, k, p)
            sil, acc = separability(Z, y)
            embeddings[(k, p)] = Z
            rows.append({"k": k, "Perplexity": p, "KL": round(kl, 4),
                         "Silhouette": round(sil, 3), "Accuracy, %": round(acc * 100, 2)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "tsne_perplexity.csv", index=False, encoding="utf-8-sig")
    print("\nПеребор perplexity (init = 'pca'):")
    print(table.to_string(index=False))

    best = {}
    for k in (2, 3):
        sub = table[table["k"] == k]
        best[k] = int(sub.loc[sub["Silhouette"].idxmax(), "Perplexity"])
    print(f"\nНаиболее презентативная perplexity (максимум silhouette): "
          f"2D — {best[2]}, 3D — {best[3]}")

    fig, axes = plt.subplots(1, len(PERPLEXITIES), figsize=(4.2 * len(PERPLEXITIES), 4.2))
    for ax, p in zip(axes, PERPLEXITIES):
        plot_2d(embeddings[(2, p)], y, f"perplexity = {p}", None, ("t-SNE 1", "t-SNE 2"), ax=ax)
    fig.suptitle("t-SNE, 2 компоненты — WDBC: разные значения perplexity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tsne_2d_grid.png", dpi=120)

    fig = plt.figure(figsize=(4.2 * len(PERPLEXITIES), 4.6))
    for i, p in enumerate(PERPLEXITIES, 1):
        ax = fig.add_subplot(1, len(PERPLEXITIES), i, projection="3d")
        plot_3d(embeddings[(3, p)], y, f"perplexity = {p}", None,
                ("t-SNE 1", "t-SNE 2", "t-SNE 3"), ax=ax)
    fig.suptitle("t-SNE, 3 компоненты — WDBC: разные значения perplexity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tsne_3d_grid.png", dpi=120)

    plot_2d(embeddings[(2, best[2])], y, f"t-SNE (perplexity = {best[2]}), 2 компоненты — WDBC",
            "tsne_2d.png", ("t-SNE 1", "t-SNE 2"))
    plot_3d(embeddings[(3, best[3])], y, f"t-SNE (perplexity = {best[3]}), 3 компоненты — WDBC",
            "tsne_3d.png", ("t-SNE 1", "t-SNE 2", "t-SNE 3"))
    print(f"\nГрафики сохранены в папку: {FIG_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
