
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# Лабораторная работа № 2
# Автоэнкодеры
# Вариант 13
# Датасет Rice (Cammeo and Osmancik)
# Класс: Class
# ============================================================

SEED = 0
EPOCHS = 300
LEARNING_RATE = 0.001
PERPLEXITIES = [20, 30, 40, 50, 60]
DATA_FILE = "Rice_Cammeo_Osmancik.arff"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Загрузка данных
raw_data, metadata = arff.loadarff(DATA_FILE)
data = pd.DataFrame(raw_data)

# Преобразуем названия и значения классов в строки
classes = data["Class"].apply(
    lambda value: value.decode("utf-8")
    if isinstance(value, bytes) else str(value)
)

# Оставляем только числовые признаки.
# Class не участвует в уменьшении размерности.
features = data.drop(columns=["Class"])
features = features.apply(pd.to_numeric, errors="coerce")
features = features.fillna(features.mean())

X = features.values.astype(np.float32)

print("Первые строки данных:")
print(data.head())
print()

print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])
print("Классы:", classes.unique())
print("Признаки:", features.columns.tolist())
print()

# 2. Стандартизация признаков
# Приводим признаки к сопоставимому масштабу.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# Метки классов для раскраски графиков
unique_classes = sorted(classes.unique())
class_array = classes.values


# 3. Функция визуализации
def plot_2d(values, title):
    plt.figure(figsize=(8, 6))

    for class_name in unique_classes:
        mask = class_array == class_name
        plt.scatter(
            values[mask, 0],
            values[mask, 1],
            label=class_name,
            alpha=0.6,
            s=18
        )

    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_3d(values, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for class_name in unique_classes:
        mask = class_array == class_name
        ax.scatter(
            values[mask, 0],
            values[mask, 1],
            values[mask, 2],
            label=class_name,
            alpha=0.6,
            s=18
        )

    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_zlabel("Компонента 3")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# 4. Автоэнкодер
# Сеть сжимает 7 признаков до 2 или 3,
# а затем пытается восстановить исходные 7 признаков.

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 4),
            nn.ReLU(),
            nn.Linear(4, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 4),
            nn.ReLU(),
            nn.Linear(4, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def run_autoencoder(latent_size):
    torch.manual_seed(SEED)

    model = Autoencoder(X_scaled.shape[1], latent_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    tensor_x = torch.tensor(X_scaled)

    for epoch in range(EPOCHS):
        model.train()

        reconstructed = model(tensor_x)
        loss = loss_function(reconstructed, tensor_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"Автоэнкодер ({latent_size} компоненты), "
                f"эпоха {epoch + 1}/{EPOCHS}, "
                f"ошибка: {loss.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        encoded = model.encoder(tensor_x).numpy()

    print(
        f"Итоговая ошибка автоэнкодера "
        f"({latent_size} компоненты): {loss.item():.6f}"
    )

    return encoded


print("\n===== АВТОЭНКОДЕР =====")

encoded_2 = run_autoencoder(2)
plot_2d(encoded_2, "Автоэнкодер: 2 компоненты")

encoded_3 = run_autoencoder(3)
plot_3d(encoded_3, "Автоэнкодер: 3 компоненты")


# 5. t-SNE
# Проверяем разные значения perplexity.
# Для выбора варианта строим двумерные графики.
# После сравнения выберем один параметр для 3D-графика.

print("\n===== t-SNE: подбор perplexity =====")

tsne_results_2d = {}

for perplexity in PERPLEXITIES:
    print(f"t-SNE, 2 компоненты, perplexity={perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=SEED
    )

    result = tsne.fit_transform(X_scaled)
    tsne_results_2d[perplexity] = result

    plot_2d(
        result,
        f"t-SNE: 2 компоненты, perplexity={perplexity}"
    )

# Выбери наиболее наглядный результат после просмотра графиков.
# Начальное значение 30 можно поменять.
BEST_PERPLEXITY = 30

if BEST_PERPLEXITY not in PERPLEXITIES:
    raise ValueError("BEST_PERPLEXITY должен быть в списке PERPLEXITIES")

print(
    "\nВыбранное значение perplexity:",
    BEST_PERPLEXITY
)

print("t-SNE, 3 компоненты")

tsne_3d = TSNE(
    n_components=3,
    perplexity=BEST_PERPLEXITY,
    init="pca",
    random_state=SEED
)

tsne_result_3d = tsne_3d.fit_transform(X_scaled)

plot_3d(
    tsne_result_3d,
    f"t-SNE: 3 компоненты, perplexity={BEST_PERPLEXITY}"
)


# 6. PCA
print("\n===== PCA =====")

pca_2 = PCA(n_components=2)
pca_result_2d = pca_2.fit_transform(X_scaled)

print(
    "Доля объяснённой дисперсии PCA (2 компоненты), %",
    pca_2.explained_variance_ratio_ * 100
)
print(
    "Суммарная доля объяснённой дисперсии, %",
    pca_2.explained_variance_ratio_.sum() * 100
)

plot_2d(pca_result_2d, "PCA: 2 компоненты")

pca_3 = PCA(n_components=3)
pca_result_3d = pca_3.fit_transform(X_scaled)

print(
    "Доля объяснённой дисперсии PCA (3 компоненты), %",
    pca_3.explained_variance_ratio_ * 100
)
print(
    "Суммарная доля объяснённой дисперсии, %",
    pca_3.explained_variance_ratio_.sum() * 100
)

plot_3d(pca_result_3d, "PCA: 3 компоненты")


# 7. Сравнение результатов PCA
print("\n===== ИТОГИ PCA =====")
print(
    "Сохранено двумя компонентами:",
    round(pca_2.explained_variance_ratio_.sum() * 100, 2),
    "%"
)
print(
    "Сохранено тремя компонентами:",
    round(pca_3.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

print("\nРабота завершена.")
