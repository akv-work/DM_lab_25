import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'seeds_dataset.txt')

columns = [
    'Area', 'Perimeter', 'Compactness',
    'Length_of_kernel', 'Width_of_kernel',
    'Asymmetry_coefficient', 'Length_of_kernel_groove',
    'Class'
]

df = pd.read_csv(DATA_PATH, sep=r'\s+', names=columns)
df = df.fillna(df.mean(numeric_only=True))
df['Class'] = df['Class'].astype(int)

X = df.drop('Class', axis=1).values
y = df['Class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx].real
eigenvectors_sorted = eigenvectors[:, idx].real

X_pca_manual_2 = X_scaled @ eigenvectors_sorted[:, :2]
X_pca_manual_3 = X_scaled @ eigenvectors_sorted[:, :3]

pca_2 = PCA(n_components=2)
X_pca_sklearn_2 = pca_2.fit_transform(X_scaled)

pca_3 = PCA(n_components=3)
X_pca_sklearn_3 = pca_3.fit_transform(X_scaled)

colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']
class_labels = sorted(np.unique(y))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for cls in class_labels:
    mask = y == cls
    axes[0].scatter(X_pca_manual_2[mask, 0], X_pca_manual_2[mask, 1],
                    c=colors[cls - 1], marker=markers[cls - 1],
                    label=f'Класс {cls}', alpha=0.7, edgecolors='k')
axes[0].set_xlabel('Главная компонента 1')
axes[0].set_ylabel('Главная компонента 2')
axes[0].set_title('PCA вручную (numpy.linalg.eig)\n2 главные компоненты')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for cls in class_labels:
    mask = y == cls
    axes[1].scatter(X_pca_sklearn_2[mask, 0], X_pca_sklearn_2[mask, 1],
                    c=colors[cls - 1], marker=markers[cls - 1],
                    label=f'Класс {cls}', alpha=0.7, edgecolors='k')
axes[1].set_xlabel('Главная компонента 1')
axes[1].set_ylabel('Главная компонента 2')
axes[1].set_title('PCA через sklearn.decomposition.PCA\n2 главные компоненты')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'pca_2d.png'), dpi=150)
plt.close()

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
for cls in class_labels:
    mask = y == cls
    ax1.scatter(X_pca_manual_3[mask, 0], X_pca_manual_3[mask, 1], X_pca_manual_3[mask, 2],
                c=colors[cls - 1], marker=markers[cls - 1],
                label=f'Класс {cls}', alpha=0.7, edgecolors='k')
ax1.set_xlabel('ГК 1')
ax1.set_ylabel('ГК 2')
ax1.set_zlabel('ГК 3')
ax1.set_title('PCA вручную (numpy.linalg.eig)\n3 главные компоненты')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
for cls in class_labels:
    mask = y == cls
    ax2.scatter(X_pca_sklearn_3[mask, 0], X_pca_sklearn_3[mask, 1], X_pca_sklearn_3[mask, 2],
                c=colors[cls - 1], marker=markers[cls - 1],
                label=f'Класс {cls}', alpha=0.7, edgecolors='k')
ax2.set_xlabel('ГК 1')
ax2.set_ylabel('ГК 2')
ax2.set_zlabel('ГК 3')
ax2.set_title('PCA через sklearn.decomposition.PCA\n3 главные компоненты')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'pca_3d.png'), dpi=150)
plt.close()

total_variance = np.sum(eigenvalues_sorted)

print("Собственные значения:")
for i, val in enumerate(eigenvalues_sorted):
    print(f"  λ{i + 1} = {val:.6f}")

print("\nПотери при преобразовании PCA:")
print(f"{'k':>3} | {'Доля объясн. дисп.':>20} | {'Потери':>10}")
for k in range(1, len(eigenvalues_sorted) + 1):
    explained = np.sum(eigenvalues_sorted[:k])
    loss = 1 - explained / total_variance
    print(f"{k:>3} | {explained / total_variance:>20.4f} | {loss:>10.4f}")
