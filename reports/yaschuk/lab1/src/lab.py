import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv("exasens.csv", skiprows=[1, 2])

print("Первые строки таблицы:")
print(data.head())
print()


classes = data["Diagnosis"]


features = data.iloc[:, 2:9].copy()

features = features.apply(pd.to_numeric, errors="coerce")

features = features.fillna(features.mean())

X = features.values

print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])
print()


cov_matrix = np.cov(X.T)

print("Ковариационная матрица:")
print(cov_matrix)
print()

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)


sort_index = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sort_index]
eigenvectors = eigenvectors[:, sort_index]


print("Собственные значения:")
print(eigenvalues)
print()


components_2 = eigenvectors[:, :2]

X_pca_2 = np.dot(X, components_2)



components_3 = eigenvectors[:, :3]

X_pca_3 = np.dot(X, components_3)


print("Размер данных после PCA (2 компоненты):",
      X_pca_2.shape)

print("Размер данных после PCA (3 компоненты):",
      X_pca_3.shape)

print()


total = eigenvalues.sum()


first_two = eigenvalues[:2].sum()

loss_2 = (1 - first_two / total) * 100

explained_2 = first_two / total * 100


first_three = eigenvalues[:3].sum()

loss_3 = (1 - first_three / total) * 100

explained_3 = first_three / total * 100


print("Сохранено с двумя:",
      round(explained_2, 2), "% информации")

print("Потеряно с двумя:",
      round(loss_2, 2), "%")

print()

print("Сохранено с тремя:",
      round(explained_3, 2), "% информации")

print("Потеряно с тремя:",
      round(loss_3, 2), "%")

print()



unique_classes = classes.unique()

plt.figure(figsize=(8, 6))

for class_name in unique_classes:

    mask = classes == class_name

    plt.scatter(
        X_pca_2[mask, 0],
        X_pca_2[mask, 1],
        label=class_name
    )

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("PCA вручную: две главные компоненты")
plt.legend()
plt.grid()

plt.show()



fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection="3d")

for class_name in unique_classes:

    mask = classes == class_name

    ax.scatter(
        X_pca_3[mask, 0],
        X_pca_3[mask, 1],
        X_pca_3[mask, 2],
        label=class_name
    )

ax.set_xlabel("Главная компонента 1")
ax.set_ylabel("Главная компонента 2")
ax.set_zlabel("Главная компонента 3")

ax.set_title("PCA вручную: три главные компоненты")

ax.legend()

plt.show()


# ============================================================
# 9. PCA с помощью sklearn
# ============================================================

# Две компоненты
pca_2 = PCA(n_components=2)

X_sklearn_2 = pca_2.fit_transform(X)


# Три компоненты
pca_3 = PCA(n_components=3)

X_sklearn_3 = pca_3.fit_transform(X)


# ------------------------------------------------------------
# 10. Результаты PCA sklearn
# ------------------------------------------------------------


print("PCA с использованием sklearn")


print()

print("Собственные значения:")
print(pca_2.explained_variance_)
print()

print("Доля объяснённой дисперсии для двух компонент:")
print(pca_2.explained_variance_ratio_ * 100)
print()

print("Доля объяснённой дисперсии для трех компонент:")
print(pca_3.explained_variance_ratio_ * 100)
print()


plt.figure(figsize=(8, 6))

for class_name in unique_classes:

    mask = classes == class_name

    plt.scatter(
        X_sklearn_2[mask, 0],
        X_sklearn_2[mask, 1],
        label=class_name
    )

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("PCA sklearn: две главные компоненты")
plt.legend()
plt.grid()

plt.show()
fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection="3d")

for class_name in unique_classes:

    mask = classes == class_name

    ax.scatter(
        X_sklearn_3[mask, 0],
        X_sklearn_3[mask, 1],
        X_sklearn_3[mask, 2],
        label=class_name
    )

ax.set_xlabel("Главная компонента 1")
ax.set_ylabel("Главная компонента 2")
ax.set_zlabel("Главная компонента 3")

ax.set_title("PCA sklearn: три главные компоненты")

ax.legend()

plt.show()




print("СРАВНЕНИЕ ДВУХ СПОСОБОВ PCA")


print()

print("Собственные значения, полученные вручную:")
print(eigenvalues[:3])

print()

print("Собственные значения sklearn:")
print(pca_3.explained_variance_)

print()

print("Потери для двух компонент:",
      round(loss_2, 2), "%")

print("Потери для трех компонент:",
      round(loss_3, 2), "%")