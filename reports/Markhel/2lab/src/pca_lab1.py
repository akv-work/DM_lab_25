"""
Лабораторная работа №2. Метод PCA из ЛР №1 (numpy.linalg.eig) для выборки WDBC.
"""
import numpy as np
import matplotlib.pyplot as plt

from common import load_data, standardize, separability, plot_2d, plot_3d, FIG_DIR


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


def main():
    X, y = load_data()
    Xs = standardize(X)

    Z2, V, _ = pca_method(Xs, 2)
    Z3, _, _ = pca_method(Xs, 3)
    full_info = V.sum()
    share = V / full_info * 100

    print("\nСобственные значения (первые 10 из 30):")
    for i, v in enumerate(V[:10], 1):
        print(f"  λ{i} = {v:.4f}  (доля дисперсии = {share[i - 1]:6.2f}%)")

    print("\nПотери информации:")
    for k in (2, 3):
        kept = V[:k].sum() / full_info * 100
        print(f"  [k={k}] сохранено = {kept:.2f}%  |  потеря = {100 - kept:.2f}%")

    print("\nРазделимость классов (silhouette | точность логистической регрессии, 5-fold):")
    for k, Z in ((2, Z2), (3, Z3)):
        sil, acc = separability(Z, y)
        print(f"  PCA, {k} компоненты: silhouette = {sil:.3f} | accuracy = {acc * 100:.2f}%")

    labels = [f"PC{i + 1} ({share[i]:.1f}%)" for i in range(3)]
    plot_2d(Z2, y, "PCA (ЛР №1), 2 компоненты — WDBC", "pca_2d.png", labels[:2])
    plot_3d(Z3, y, "PCA (ЛР №1), 3 компоненты — WDBC", "pca_3d.png", labels)
    print(f"\nГрафики сохранены в папку: {FIG_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
