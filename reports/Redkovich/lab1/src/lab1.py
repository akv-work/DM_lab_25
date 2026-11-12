import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Загрузка и подготовка данных
# Определение абсолютного пути к директории, где находится текущий скрипт (.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'heart_failure_clinical_records_dataset.csv')

# Загрузка датасета по сформированному пути из текущей папки
df = pd.read_csv(file_path)

# Замещение отсутствующих данных (стратегия заполнения средним значением)
df = df.fillna(df.mean())

# Отделение целевого класса для визуализации и удаление его из данных для PCA
target_col = 'DEATH_EVENT'
y = df[target_col].values
X_raw = df.drop(target_col, axis=1).values

# Масштабирование данных
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 2. Способ 1: Реализация PCA вручную (на 2 и 3 компоненты)
def manual_pca(data, n_components):
    # Вычисляется ковариационная матрица
    cov_matrix = np.cov(data.T)
    
    # Находятся собственные векторы и значения
    V, PC = np.linalg.eig(cov_matrix)
    
    # Сортировка по убыванию собственных значений
    sort_index = np.argsort(-1 * V)
    V_sorted = V[sort_index]
    PC_sorted = PC[:, sort_index]
    
    # Матрицу из векторов используем для понижения размерности
    reduced_data = np.dot((PC_sorted.T)[0:n_components], data.T).T
    
    # Подсчет потерь в информативности
    full_info = V_sorted.sum()
    reduce_info = V_sorted[0:n_components].sum()
    loss = 100 - (reduce_info / full_info * 100)
    
    return reduced_data, loss

X_manual_2d, loss_2d = manual_pca(X, 2)
X_manual_3d, loss_3d = manual_pca(X, 3)

# 3. Способ 2: Использование sklearn.decomposition.PCA (на 2 и 3 компоненты)
pca_2d = PCA(n_components=2)
X_sklearn_2d = pca_2d.fit_transform(X)

pca_3d = PCA(n_components=3)
X_sklearn_3d = pca_3d.fit_transform(X)

# Вывод результатов подсчета потерь
print(f"Потери информативности при сжатии до 2-х компонент: {loss_2d.real:.2f}%")
print(f"Потери информативности при сжатии до 3-х компонент: {loss_3d.real:.2f}%")

# 4. Визуализация полученных главных компонент
fig = plt.figure(figsize=(16, 12))

# Цвета для классов: 0 (Выжил) - синий, 1 (Умер) - красный
colors = ['blue' if label == 0 else 'red' for label in y]

# График 1: 2D Вручную
ax1 = fig.add_subplot(221)
ax1.scatter(X_manual_2d[:, 0], X_manual_2d[:, 1], c=colors, alpha=0.7)
ax1.set_title('PCA 2D (Вручную)')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')

# График 2: 2D scikit-learn
ax2 = fig.add_subplot(222)
# sklearn может инвертировать оси по знаку, это нормально для PCA
ax2.scatter(X_sklearn_2d[:, 0], X_sklearn_2d[:, 1], c=colors, alpha=0.7) 
ax2.set_title('PCA 2D (scikit-learn)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')

# График 3: 3D Вручную
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_manual_3d[:, 0], X_manual_3d[:, 1], X_manual_3d[:, 2], c=colors, alpha=0.7)
ax3.set_title('PCA 3D (Вручную)')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_zlabel('PC 3')

# График 4: 3D scikit-learn
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_sklearn_3d[:, 0], X_sklearn_3d[:, 1], X_sklearn_3d[:, 2], c=colors, alpha=0.7)
ax4.set_title('PCA 3D (scikit-learn)')
ax4.set_xlabel('PC 1')
ax4.set_ylabel('PC 2')
ax4.set_zlabel('PC 3')

plt.tight_layout()
plt.show()