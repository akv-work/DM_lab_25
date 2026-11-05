import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
data = pd.read_csv("wholesale customers data.csv")

# Region — класс
y = data["Region"].values

# Все остальные столбцы — признаки
X = data.drop(columns=["Region"]).values

# PCA вручную

# Центрирование данных
X_centered = X - X.mean(axis=0)

# Ковариационная матрица
cov_matrix = np.cov(X_centered.T)

# Собственные значения и собственные векторы
values, vectors = np.linalg.eig(cov_matrix)

# Сортировка по убыванию собственных значений
index = np.argsort(values)[::-1]

values = values[index]
vectors = vectors[:, index]

# Проекция на 2 главные компоненты
X_manual_2 = X_centered @ vectors[:, :2]

# Проекция на 3 главные компоненты
X_manual_3 = X_centered @ vectors[:, :3]

# PCA с помощью sklearn

pca = PCA(n_components=3)

X_sklearn = pca.fit_transform(X)

X_sklearn_2 = X_sklearn[:, :2]
X_sklearn_3 = X_sklearn[:, :3]

# Вывод результатов

print("PCA вручную")

print("\nСобственные значения:")
print(values)

print("\nДоля объясненной дисперсии:")
print(values / values.sum())

print("\nРазмер после PCA:")
print("2 компоненты:", X_manual_2.shape)
print("3 компоненты:", X_manual_3.shape)

# Потери информации

loss_manual_2 = 1 - values[:2].sum() / values.sum()
loss_manual_3 = 1 - values[:3].sum() / values.sum()

loss_sklearn_2 = 1 - pca.explained_variance_ratio_[:2].sum()
loss_sklearn_3 = 1 - pca.explained_variance_ratio_[:3].sum()


print("\n\nРучной PCA")

print(
    "Доля объясненной дисперсии (2 ГК):",
    values[:2].sum() / values.sum()
)

print(
    "Потери (2 ГК):",
    loss_manual_2
)

print(
    "\nДоля объясненной дисперсии (3 ГК):",
    values[:3].sum() / values.sum()
)

print(
    "Потери (3 ГК):",
    loss_manual_3
)


print("\n\nsklearn PCA")

print(
    "Доля объясненной дисперсии (2 ГК):",
    pca.explained_variance_ratio_[:2].sum()
)

print(
    "Потери (2 ГК):",
    loss_sklearn_2
)

print(
    "\nДоля объясненной дисперсии (3 ГК):",
    pca.explained_variance_ratio_[:3].sum()
)

print(
    "Потери (3 ГК):",
    loss_sklearn_3
)

# График — 2 главные компоненты

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for c in np.unique(y):

    points = X_manual_2[y == c]

    axes[0].scatter(
        points[:, 0],
        points[:, 1],
        label="Region " + str(c)
    )


axes[0].set_title("Ручной PCA - 2 компоненты")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()
axes[0].grid()


for c in np.unique(y):

    points = X_sklearn_2[y == c]

    axes[1].scatter(
        points[:, 0],
        points[:, 1],
        label="Region " + str(c)
    )

axes[1].set_title("sklearn PCA - 2 компоненты")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# График — 3 главные компоненты

fig = plt.figure(figsize=(13, 6))

ax1 = fig.add_subplot(121, projection="3d")

for c in np.unique(y):

    points = X_manual_3[y == c]

    ax1.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        label="Region " + str(c)
    )


ax1.set_title("Ручной PCA - 3 компоненты")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")
ax1.legend()


ax2 = fig.add_subplot(122, projection="3d")

for c in np.unique(y):

    points = X_sklearn_3[y == c]

    ax2.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        label="Region " + str(c)
    )
    ax2.set_title("sklearn PCA - 3 компоненты")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")
ax2.legend()


plt.tight_layout()
plt.show()

# Сравнение методов

print("\n\nСравнение методов")

print(
    "2 компоненты:",
    round(values[:2].sum() / values.sum(), 6),
    "и",
    round(pca.explained_variance_ratio_[:2].sum(), 6)
)

print(
    "3 компоненты:",
    round(values[:3].sum() / values.sum(), 6),
    "и",
    round(pca.explained_variance_ratio_[:3].sum(), 6)
)