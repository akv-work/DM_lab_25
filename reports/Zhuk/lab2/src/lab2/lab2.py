import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


#   НАСТРОЙКИ 
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "optdigits.tra")
TEST_FILE = os.path.join(DATA_DIR, "optdigits.tes")

RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Чтобы ускорить t-SNE на слабом компьютере, можно взять случайную подвыборку.
# None означает использовать все данные.
TSNE_MAX_SAMPLES = 3000
PERPLEXITIES = [20, 40, 60]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  ЗАГРУЗКА ДАННЫХ 
def load_data():
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            "Не найдены файлы датасета. Поместите optdigits.tra и "
            "optdigits.tes в папку data рядом с main.py."
        )

    train = pd.read_csv(TRAIN_FILE, header=None)
    test = pd.read_csv(TEST_FILE, header=None)

    data = pd.concat([train, test], ignore_index=True)

    # В датасете первые 64 столбца — пиксельные признаки,
    # последний (65-й) столбец — класс цифры.
    X = data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = data.iloc[:, -1].to_numpy(dtype=int)

    # Нормализация признаков в диапазон [0, 1].
    X = X / 16.0

    return X, y


# АВТОЭНКОДЕР  
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(X, latent_size):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    tensor_x = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor_x),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = Autoencoder(X.shape[1], latent_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        epoch_loss = total_loss / len(tensor_x)
        history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Автоэнкодер ({latent_size} компоненты), "
                f"эпоха {epoch + 1}/{EPOCHS}, "
                f"ошибка MSE = {epoch_loss:.6f}"
            )

    model.eval()
    with torch.no_grad():
        encoded = model.encoder(
            torch.tensor(X, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    return encoded, history


#   ВИЗУАЛИЗАЦИЯ  
def plot_embedding(embedding, y, title, filename):
    components = embedding.shape[1]

    # Отдельный цвет для каждого класса цифр
    colors = plt.get_cmap("tab10", 10)

    if components == 2:
        fig, ax = plt.subplots(figsize=(11, 7))

        for digit in range(10):
            mask = y == digit
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                color=colors(digit),
                label=f"Цифра {digit}",
                s=8,
                alpha=0.7
            )

        ax.set_xlabel("Компонента 1")
        ax.set_ylabel("Компонента 2")
        ax.set_title(title)
        ax.legend(title="Классы", markerscale=2)
        plt.tight_layout()
        plt.savefig(filename, dpi=160)
        plt.close(fig)

    elif components == 3:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        for digit in range(10):
            mask = y == digit
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                color=colors(digit),
                label=f"Цифра {digit}",
                s=7,
                alpha=0.7
            )

        ax.set_xlabel("Компонента 1")
        ax.set_ylabel("Компонента 2")
        ax.set_zlabel("Компонента 3")
        ax.set_title(title)
        ax.legend(title="Классы", markerscale=2)
        plt.tight_layout()
        plt.savefig(filename, dpi=160)
        plt.close(fig)


def plot_loss(history, latent_size):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title(f"Ошибка обучения автоэнкодера ({latent_size} нейрона(ов) в коде)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/autoencoder_loss_{latent_size}d.png", dpi=160)
    plt.close()


#   ОСНОВНАЯ ПРОГРАММА  
def main():
    os.makedirs("results", exist_ok=True)

    X, y = load_data()
    print(f"Размер данных: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Классы: {np.unique(y)}")

    # Масштабирование для PCA и t-SNE.
    # Autoencoder получает исходные пиксели, приведённые к [0, 1].
    X_scaled = StandardScaler().fit_transform(X)

    # 1. Автоэнкодер: 2 и 3 нейрона в среднем слое.
    for latent_size in [2, 3]:
        embedding, history = train_autoencoder(X, latent_size)
        plot_embedding(
            embedding, y,
            f"Автоэнкодер: {latent_size} компоненты",
            f"results/autoencoder_{latent_size}d.png"
        )
        plot_loss(history, latent_size)

    # 2. PCA: 2 и 3 компоненты.
    for n_components in [2, 3]:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        embedding = pca.fit_transform(X_scaled)

        print(
            f"PCA ({n_components} компоненты): "
            f"объяснённая дисперсия = "
            f"{pca.explained_variance_ratio_.sum():.4f}"
        )

        plot_embedding(
            embedding, y,
            f"PCA: {n_components} компоненты",
            f"results/pca_{n_components}d.png"
        )

    # 3. t-SNE: 2 и 3 компоненты, несколько perplexity.
    if TSNE_MAX_SAMPLES is not None and len(X_scaled) > TSNE_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = np.sort(
            rng.choice(len(X_scaled), size=TSNE_MAX_SAMPLES, replace=False)
        )
        X_tsne = X_scaled[indices]
        y_tsne = y[indices]
    else:
        X_tsne = X_scaled
        y_tsne = y

    for n_components in [2, 3]:
        for perplexity in PERPLEXITIES:
            if perplexity >= len(X_tsne):
                print(f"Пропуск perplexity={perplexity}: слишком мало объектов")
                continue

            print(
                f"Запуск t-SNE: компоненты={n_components}, "
                f"perplexity={perplexity}"
            )
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=RANDOM_STATE
            )
            embedding = tsne.fit_transform(X_tsne)

            plot_embedding(
                embedding, y_tsne,
                f"t-SNE: {n_components} компоненты, perplexity={perplexity}",
                f"results/tsne_{n_components}d_perplexity_{perplexity}.png"
            )


if __name__ == "__main__":
    main()
