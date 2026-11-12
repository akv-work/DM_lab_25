import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#сохранение графиков
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# подготовка данных
def prepareData(path=None):
    if path is None:
        path = Path(__file__).parent / "heart_failure_clinical_records_dataset.csv"

    df = pd.read_csv(path)

    print("Размер выборки:", df.shape)
    print("\nПропуски по колонкам:")
    print(df.isna().sum())

    #- бработка пропусков (медиана/мода) 
    if df.isna().sum().sum() > 0:
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        print("Пропуски заполнены (медиана/мода).")
    else:
        print("Пропусков нет — заполнение не требуется.")

    target = df['DEATH_EVENT'].values
    data   = df.drop(columns=['DEATH_EVENT']).values
    data   = StandardScaler().fit_transform(data)

    return data, target

#ручной PCA через numpy.linalg.eig
def pca_method(data, n_components=2):
    cov_matrix = np.cov(data.T)
    V, PC = np.linalg.eig(cov_matrix)
    V  = np.real(V)
    PC = np.real(PC)

    sort_index = np.argsort(-1 * V)
    V_sorted = V[sort_index]
    PC = PC[:, sort_index]

    data_reduced = np.dot((PC.T)[0:n_components], data.T)
    return data_reduced.T, V_sorted, PC

#визуализация с сохранением
def plot_2d(data, target, title, filename):
    plt.figure(figsize=(8, 6))
    for cls, color, marker in [(0, 'tab:blue', 'o'), (1, 'tab:red', '^')]:
        mask = (target == cls)
        plt.scatter(data[mask, 0], data[mask, 1],
                    c=color, marker=marker, alpha=0.7,
                    label=f'DEATH_EVENT = {cls}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")

    plt.show()
    plt.close()


def plot_3d(data, target, title, filename):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for cls, color, marker in [(0, 'tab:blue', 'o'), (1, 'tab:red', '^')]:
        mask = (target == cls)
        ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                   c=color, marker=marker, alpha=0.7,
                   label=f'DEATH_EVENT = {cls}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")

    plt.show()
    plt.close(fig)

#основной сценарий
data, target = prepareData()

#- ручной PCA
data_2d_manual, eigvals, eigvecs = pca_method(data, n_components=2)
data_3d_manual, _, _             = pca_method(data, n_components=3)

plot_2d(data_2d_manual, target,
        'PCA вручную (numpy) — 2 главные компоненты',
        'pca_manual_2d.png')

plot_3d(data_3d_manual, target,
        'PCA вручную (numpy) — 3 главные компоненты',
        'pca_manual_3d.png')

#- PCA через sklearn 
pca2 = PCA(n_components=2)
data_2d_sklearn = pca2.fit_transform(data)

pca3 = PCA(n_components=3)
data_3d_sklearn = pca3.fit_transform(data)

plot_2d(data_2d_sklearn, target,
        'PCA через sklearn — 2 главные компоненты',
        'pca_sklearn_2d.png')

plot_3d(data_3d_sklearn, target,
        'PCA через sklearn — 3 главные компоненты',
        'pca_sklearn_3d.png')

#- потери 
total_variance = np.sum(eigvals)
print("\nСобственные значения (по убыванию):")
print(eigvals)

print("\nПотери при преобразовании PCA:")
for k in [2, 3]:
    retained = np.sum(eigvals[:k]) / total_variance
    loss = 1 - retained
    print(f"  k = {k}: сохранено {retained*100:.2f}% дисперсии, "
          f"потери = {loss*100:.2f}%")

print("\nПроверка через sklearn:")
print(f"  k = 2: сохранено {pca2.explained_variance_ratio_.sum()*100:.2f}%, "
      f"потери = {(1 - pca2.explained_variance_ratio_.sum())*100:.2f}%")
print(f"  k = 3: сохранено {pca3.explained_variance_ratio_.sum()*100:.2f}%, "
      f"потери = {(1 - pca3.explained_variance_ratio_.sum())*100:.2f}%")

#- scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(eigvals) + 1), eigvals, alpha=0.7,
        label='Собственные значения')
plt.plot(range(1, len(eigvals) + 1),
         np.cumsum(eigvals) / total_variance,
         'r-o', label='Накопленная дисперсия')
plt.xlabel('Номер главной компоненты')
plt.ylabel('Собственное значение')
plt.title('Scree plot')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

save_path = PLOTS_DIR / 'scree_plot.png'
plt.savefig(save_path, dpi=150)
print(f"Сохранено: {save_path}")

plt.show()
plt.close()