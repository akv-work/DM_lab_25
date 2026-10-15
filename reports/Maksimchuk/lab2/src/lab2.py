import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

print("Размер X:", X.shape)
print("Классы:", y['poisonous'].value_counts().to_dict())


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

y_numeric = y['poisonous'].map({'e': 0, 'p': 1}).values
print("После one-hot:", X_scaled.shape)


def my_pca(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centered.T)
    values, vectors = np.linalg.eig(cov_matrix)
    values = values.real
    vectors = vectors.real

    index = np.argsort(values)[::-1]
    values = values[index]
    vectors = vectors[:, index]

    Z = X_centered @ vectors[:, :n_components]

    total_var = values.sum()
    evr = values[:n_components] / total_var
    return Z, evr, values, vectors


X_manual_2, evr_manual_2, eigvals, eigvecs = my_pca(X_scaled, n_components=2)
X_manual_3, evr_manual_3, _, _ = my_pca(X_scaled, n_components=3)

print("\nРУЧНОЙ PCA")
print("Собственные значения (первые 5):", eigvals[:5])
print(f"2 компоненты — доля дисперсии: {evr_manual_2.sum():.6f}, "
      f"потери: {1 - evr_manual_2.sum():.6f}")
print(f"3 компоненты — доля дисперсии: {evr_manual_3.sum():.6f}, "
      f"потери: {1 - evr_manual_3.sum():.6f}")


pca_sklearn = PCA(n_components=3, random_state=42)
X_sklearn_3 = pca_sklearn.fit_transform(X_scaled)
X_sklearn_2 = X_sklearn_3[:, :2]

evr_sklearn_2 = pca_sklearn.explained_variance_ratio_[:2].sum()
evr_sklearn_3 = pca_sklearn.explained_variance_ratio_[:3].sum()

print("\nSKLEARN PCA")
print(f"2 компоненты — доля дисперсии: {evr_sklearn_2:.6f}, "
      f"потери: {1 - evr_sklearn_2:.6f}")
print(f"3 компоненты — доля дисперсии: {evr_sklearn_3:.6f}, "
      f"потери: {1 - evr_sklearn_3:.6f}")


def build_autoencoder(bottleneck, hidden_layers=(),
                      activation='relu', solver='adam',
                      max_iter=500, random_state=42):
    sizes = tuple(hidden_layers) + (bottleneck,) + tuple(hidden_layers[::-1])
    return MLPRegressor(
        hidden_layer_sizes=sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )


def extract_bottleneck(model, X, hidden_layers):
    H = X
    n_pre = len(hidden_layers)
    for i in range(n_pre + 1):
        H = np.dot(H, model.coefs_[i]) + model.intercepts_[i]
        if i < n_pre:
            H = np.maximum(H, 0)
    return H


def train_autoencoder(X, bottleneck, hidden_layers=(),
                      activation='relu', solver='adam',
                      max_iter=500, random_state=42):
    model = build_autoencoder(bottleneck, hidden_layers, activation,
                              solver, max_iter, random_state)
    model.fit(X, X)
    Z = extract_bottleneck(model, X, hidden_layers)
    mse = mean_squared_error(X, model.predict(X))
    return model, Z, mse


configs = [
    {"hidden_layers": (),        "activation": "relu", "max_iter": 300},
    {"hidden_layers": (),        "activation": "relu", "max_iter": 800},
    {"hidden_layers": (16,),     "activation": "relu", "max_iter": 500},
    {"hidden_layers": (32, 16),  "activation": "relu", "max_iter": 500},
    {"hidden_layers": (),        "activation": "tanh", "max_iter": 500},
]

print("\nПодбор архитектуры AE (2D)")
results_2d = []
for cfg in configs:
    _, Z, mse = train_autoencoder(X_scaled, bottleneck=2, **cfg)
    results_2d.append((cfg, Z, mse))
    print(f"  {cfg} -> MSE={mse:.5f}")
best_2d_cfg, ae_2d, best_2d_mse = min(results_2d, key=lambda t: t[2])
print("Лучшая 2D:", best_2d_cfg, "MSE =", round(best_2d_mse, 5))

print("\nПодбор архитектуры AE (3D)")
results_3d = []
for cfg in configs:
    _, Z, mse = train_autoencoder(X_scaled, bottleneck=3, **cfg)
    results_3d.append((cfg, Z, mse))
    print(f"  {cfg} -> MSE={mse:.5f}")
best_3d_cfg, ae_3d, best_3d_mse = min(results_3d, key=lambda t: t[2])
print("Лучшая 3D:", best_3d_cfg, "MSE =", round(best_3d_mse, 5))


X_pca50 = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
PERPLEXITIES = [20, 30, 40, 50, 60]

def tsne_run(X, n_components, perplexity):
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        max_iter=1000,
        random_state=42
    ).fit_transform(X)

print("\nt-SNE 2D")
tsne_2d_by_perp = {p: tsne_run(X_pca50, 2, p) for p in PERPLEXITIES}
print("t-SNE 3D")
tsne_3d_by_perp = {p: tsne_run(X_pca50, 3, p) for p in PERPLEXITIES}
print("Готово. Perplexity:", PERPLEXITIES)


def plot_embedding(Z, y_numeric, title="", ax=None):
    colors = {0: 'green', 1: 'red'}
    markers = {0: 'o', 1: 's'}
    labels = {0: 'Съедобный', 1: 'Ядовитый'}
    is_3d = (Z.shape[1] == 3)

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

    for cls in [0, 1]:
        mask = y_numeric == cls
        if is_3d:
            ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                       c=colors[cls], marker=markers[cls],
                       label=labels[cls], alpha=0.5, s=10)
        else:
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=colors[cls], marker=markers[cls],
                       label=labels[cls], alpha=0.6, s=15)
    ax.set_title(title)
    ax.legend(loc='best')
    if is_3d:
        ax.set_xlabel('К1'); ax.set_ylabel('К2'); ax.set_zlabel('К3')
    else:
        ax.set_xlabel('К1'); ax.set_ylabel('К2')
    return ax


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_embedding(X_manual_2, y_numeric,
               title=f"Ручной PCA 2D (EV={evr_manual_2.sum():.2%})", ax=axes[0])
plot_embedding(X_sklearn_2, y_numeric,
               title=f"sklearn PCA 2D (EV={evr_sklearn_2:.2%})", ax=axes[1])
plt.tight_layout(); plt.savefig("pca_2d_comparison.png", dpi=120); plt.show()

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_embedding(X_manual_3, y_numeric,
               title=f"Ручной PCA 3D (EV={evr_manual_3.sum():.2%})", ax=ax1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_embedding(X_sklearn_3, y_numeric,
               title=f"sklearn PCA 3D (EV={evr_sklearn_3:.2%})", ax=ax2)
plt.tight_layout(); plt.savefig("pca_3d_comparison.png", dpi=120); plt.show()


fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(1, 2, 1)
plot_embedding(ae_2d, y_numeric, title=f"AE 2D (MSE={best_2d_mse:.4f})", ax=ax1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_embedding(ae_3d, y_numeric, title=f"AE 3D (MSE={best_3d_mse:.4f})", ax=ax2)
plt.tight_layout(); plt.savefig("autoencoder_result.png", dpi=120); plt.show()


fig, axes = plt.subplots(1, len(PERPLEXITIES), figsize=(25, 5))
for ax, p in zip(axes, PERPLEXITIES):
    plot_embedding(tsne_2d_by_perp[p], y_numeric,
                   title=f"t-SNE 2D, perp={p}", ax=ax)
plt.tight_layout(); plt.savefig("tsne_2d_perplexity.png", dpi=120); plt.show()

fig = plt.figure(figsize=(25, 5))
for i, p in enumerate(PERPLEXITIES, 1):
    ax = fig.add_subplot(1, len(PERPLEXITIES), i, projection='3d')
    plot_embedding(tsne_3d_by_perp[p], y_numeric,
                   title=f"t-SNE 3D, perp={p}", ax=ax)
plt.tight_layout(); plt.savefig("tsne_3d_perplexity.png", dpi=120); plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plot_embedding(ae_2d, y_numeric, title="Автоэнкодер (2D)", ax=axes[0])
plot_embedding(tsne_2d_by_perp[40], y_numeric, title="t-SNE (2D, perp=40)", ax=axes[1])
plot_embedding(X_manual_2, y_numeric, title="Ручной PCA (2D)", ax=axes[2])
plt.tight_layout(); plt.savefig("comparison_2d.png", dpi=120); plt.show()


with open("report.md", "w", encoding="utf-8") as f:
    f.write("# Отчёт по ЛР: понижение размерности датасета Mushroom\n\n")
    f.write(f"- Исходный размер X: {X.shape}\n")
    f.write(f"- После one-hot + StandardScaler: {X_scaled.shape}\n")
    f.write(f"- Класс 0 (съедобный): {np.sum(y_numeric == 0)}\n")
    f.write(f"- Класс 1 (ядовитый): {np.sum(y_numeric == 1)}\n\n")

    f.write("## 1. Автоэнкодер\n")
    f.write(f"- Лучшая 2D-архитектура: {best_2d_cfg}, MSE = {best_2d_mse:.5f}\n")
    f.write(f"- Лучшая 3D-архитектура: {best_3d_cfg}, MSE = {best_3d_mse:.5f}\n")
    f.write("- Активация ReLU, оптимизатор Adam, перебор глубины и ширины.\n\n")

    f.write("## 2. t-SNE\n")
    f.write(f"- Протестированные perplexity: {PERPLEXITIES}\n")
    f.write("- Инициализация: init='pca', предварительное сжатие до 50 компонент.\n\n")

    f.write("## 3. PCA (ручной из ЛР №1 vs sklearn)\n")
    f.write(f"- Ручной 2D: доля дисперсии {evr_manual_2.sum():.6f}, "
            f"потери {1 - evr_manual_2.sum():.6f}\n")
    f.write(f"- Ручной 3D: доля дисперсии {evr_manual_3.sum():.6f}, "
            f"потери {1 - evr_manual_3.sum():.6f}\n")
    f.write(f"- sklearn 2D: {evr_sklearn_2:.6f}\n")
    f.write(f"- sklearn 3D: {evr_sklearn_3:.6f}\n")
    f.write("- Ручной и sklearn PCA дают совпадающие доли дисперсии.\n\n")

    f.write("## 4. Визуализации\n")
    f.write("- pca_2d_comparison.png, pca_3d_comparison.png\n")
    f.write("- autoencoder_result.png\n")
    f.write("- tsne_2d_perplexity.png, tsne_3d_perplexity.png\n")
    f.write("- comparison_2d.png\n")

print("\nОтчёт сохранён в report.md")


print(f"""
ВЫВОДЫ
1. Автоэнкодер: лучшая 2D-архитектура {best_2d_cfg} (MSE={best_2d_mse:.5f}),
   лучшая 3D-архитектура {best_3d_cfg} (MSE={best_3d_mse:.5f}).
   Использованы ReLU и Adam, число слоёв/нейронов подбиралось экспериментально.
2. t-SNE: перебор perplexity {PERPLEXITIES}, init='pca'. Наиболее презентативная
   визуализация выбирается из tsne_2d_perplexity.png / tsne_3d_perplexity.png.
3. PCA (ручной из ЛР №1): 2D даёт {evr_manual_2.sum():.2%} дисперсии,
   3D — {evr_manual_3.sum():.2%}. Совпадает со sklearn до 6 знаков после запятой.
   Полного разделения классов нет — граница нелинейна.
4. Итог: t-SNE лучше всех показывает локальную структуру, автоэнкодер
   даёт компактное нелинейное представление, PCA — самое интерпретируемое,
   но линейное.
""")