import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FILE_NAME = "heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(FILE_NAME)

if data.isnull().sum().sum() > 0:
    data = data.fillna(data.median(numeric_only=True))

y = data["DEATH_EVENT"].to_numpy()
X = data.drop(columns=["DEATH_EVENT"]).to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)

covariance_matrix = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvalues = np.real_if_close(eigenvalues)
eigenvectors = np.real_if_close(eigenvectors)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

W2 = eigenvectors[:, :2]
W3 = eigenvectors[:, :3]

X_manual_2 = X @ W2
X_manual_3 = X @ W3

total = eigenvalues.sum()
explained_2 = eigenvalues[:2].sum() / total
explained_3 = eigenvalues[:3].sum() / total
loss_2 = (1 - explained_2)
loss_3 = (1 - explained_3) 

print("Ручной РСА:")
print(f"Объяснённая дисперсия (2 компоненты): {explained_2:.4f}")
print(f"Потери (2 компоненты): {loss_2:.4f}")
print(f"Объяснённая дисперсия (3 компоненты): {explained_3:.4f}")
print(f"Потери (3 компоненты): {loss_3:.4f}")

pca_2 = PCA(n_components=2)
pca_3 = PCA(n_components=3)
X_sklearn_2 = pca_2.fit_transform(X)
X_sklearn_3 = pca_3.fit_transform(X)
sklearn_loss_2 = (1 - pca_2.explained_variance_ratio_.sum())
sklearn_loss_3 = (1 - pca_3.explained_variance_ratio_.sum())

print("\nРСА sklearn:")
print(f"Объяснённая дисперсия (2 компоненты): {pca_2.explained_variance_ratio_.sum():.4f}")
print(f"Потери информации (2 компоненты): {sklearn_loss_2:.4f}")
print(f"Объяснённая дисперсия (3 компоненты): {pca_3.explained_variance_ratio_.sum():.4f}")
print(f"Потери информации (3 компоненты): {sklearn_loss_3:.4f}")

class_colors = {0: "gold", 1: "green"}

plt.figure(figsize=(8, 6))
for class_value in [0, 1]:
    mask = y == class_value
    plt.scatter(
        X_manual_2[mask, 0],
        X_manual_2[mask, 1],
        c=class_colors[class_value],
        alpha=0.7,
        label=f"DEATH_EVENT = {class_value}"
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Ручной PCA (2 компоненты)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
for class_value in [0, 1]:
    mask = y == class_value
    plt.scatter(
        X_sklearn_2[mask, 0],
        X_sklearn_2[mask, 1],
        c=class_colors[class_value],
        alpha=0.7,
        label=f"DEATH_EVENT = {class_value}"
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Sklearn PCA (2 компоненты)")
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
for class_value in [0, 1]:
    mask = y == class_value
    ax.scatter(
        X_manual_3[mask, 0],
        X_manual_3[mask, 1],
        X_manual_3[mask, 2],
        c=class_colors[class_value],
        alpha=0.7,
        label=f"DEATH_EVENT = {class_value}"
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Ручной PCA (3 компоненты)")
ax.legend()
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
for class_value in [0, 1]:
    mask = y == class_value
    ax.scatter(
        X_sklearn_3[mask, 0],
        X_sklearn_3[mask, 1],
        X_sklearn_3[mask, 2],
        c=class_colors[class_value],
        alpha=0.7,
        label=f"DEATH_EVENT = {class_value}"
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Sklearn PCA (3 компоненты)")
ax.legend()
plt.show()