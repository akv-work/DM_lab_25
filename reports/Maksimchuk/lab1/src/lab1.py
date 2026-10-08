import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 1. Загрузка и подготовка данных
data = pd.read_csv("seeds_dataset.txt", sep="\s+", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Ручной PCA
X_centered = X - X.mean(axis=0)
cov_matrix = np.cov(X_centered.T)
values, vectors = np.linalg.eig(cov_matrix)

# Сортировка собственных значений и векторов по убыванию
index = np.argsort(values)[::-1]
values = values[index]
vectors = vectors[:, index]

# Проекция на 2 и 3 компоненты
X_manual_2 = X_centered @ vectors[:, :2]
X_manual_3 = X_centered @ vectors[:, :3]

# Расчет дисперсии и потерь для ручного метода
total_var_manual = values.sum()
var_manual_2 = values[:2].sum() / total_var_manual
var_manual_3 = values[:3].sum() / total_var_manual

# 3. PCA через sklearn
pca = PCA(n_components=3)
X_sklearn = pca.fit_transform(X)
X_sklearn_2 = X_sklearn[:, :2]
X_sklearn_3 = X_sklearn[:, :3]

# Расчет дисперсии и потерь для sklearn
var_sklearn_2 = pca.explained_variance_ratio_[:2].sum()
var_sklearn_3 = pca.explained_variance_ratio_[:3].sum()

# 4. Вывод метрик в консоль
print("СТАТИСТИКА PCA ВРУЧНУЮ")
print("Собственные значения:\n", values)
print("\nДоля объясненной дисперсии по компонентам:\n", values / total_var_manual)
print(f"\nРазмер после ручного PCA: 2D {X_manual_2.shape}, 3D {X_manual_3.shape}")

print("\nСРАВНЕНИЕ РУЧНОГО И SKLEARN МЕТОДОВ")
print(f"2 компоненты — Доля дисперсии: Ручной {var_manual_2:.6f} | sklearn {var_sklearn_2:.6f}")
print(f"2 компоненты — Потери:         Ручной {1 - var_manual_2:.6f} | sklearn {1 - var_sklearn_2:.6f}")

print(f"3 компоненты — Доля дисперсии: Ручной {var_manual_3:.6f} | sklearn {var_sklearn_3:.6f}")
print(f"3 компоненты — Потери:         Ручной {1 - var_manual_3:.6f} | sklearn {1 - var_sklearn_3:.6f}")

# 5. Визуализация: 2D Графики
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
datasets_2d = [X_manual_2, X_sklearn_2]
titles_2d = ["Ручной PCA - 2 компоненты", "sklearn PCA - 2 компоненты"]

for ax, data_2d, title in zip(axes, datasets_2d, titles_2d):
    for c in np.unique(y):
        ax.scatter(data_2d[y == c, 0], data_2d[y == c, 1], label=f"Класс {c}")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# 6. Визуализация: 3D Графики
fig = plt.figure(figsize=(13, 6))
datasets_3d = [X_manual_3, X_sklearn_3]
titles_3d = ["Ручной PCA - 3 компоненты", "sklearn PCA - 3 компоненты"]

for i, (data_3d, title) in enumerate(zip(datasets_3d, titles_3d), start=1):
    ax = fig.add_subplot(1, 2, i, projection="3d")
    for c in np.unique(y):
        ax.scatter(data_3d[y == c, 0], data_3d[y == c, 1], data_3d[y == c, 2], label=f"Класс {c}")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

plt.tight_layout()
plt.show()
