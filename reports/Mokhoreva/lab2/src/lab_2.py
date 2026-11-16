# lab_2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FILE_NAME = "winequality-white.csv"
PLOTS_DIR = Path(__file__).parent / "plots_lab2"
PLOTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

data = pd.read_csv(Path(__file__).parent / FILE_NAME, sep=";")

if data.isnull().sum().sum() > 0:
    data = data.fillna(data.median(numeric_only=True))
    print("Пропуски заполнены медианой.")
else:
    print("Пропусков нет.")

y = data["quality"].to_numpy()
X_raw = data.drop(columns=["quality"]).to_numpy()

X = StandardScaler().fit_transform(X_raw)
classes = np.unique(y)

print("Размер выборки:", X.shape)
print("Распределение quality:")
print(pd.Series(y).value_counts().sort_index())

X_tensor = torch.tensor(X, dtype=torch.float32)

def pca_method(data, n_components=2):
    """
    Ручной PCA через разложение ковариационной матрицы.
    Явное центрирование — функция самодостаточна.
    Для симметричной ковариационной матрицы используем eigh:
    быстрее eig, гарантирует вещественные собственные значения.
    """
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)          
    order = np.argsort(eigvals)[::-1]                     
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    data_reduced = data_centered @ eigvecs[:, :n_components]
    return data_reduced, eigvals, eigvecs

class Autoencoder(nn.Module):
    """
    Полносвязный автоэнкодер.
    Линейный bottleneck — аналог PCA по смыслу.
    """
    def __init__(self, input_dim, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, bottleneck)               
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(bottleneck, epochs=200, lr=1e-3, batch_size=128):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(X_tensor.shape[1], bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = loss_function(output, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(X_tensor)
        history.append(epoch_loss)
        if (epoch + 1) % 40 == 0:
            print(f"  [bottleneck={bottleneck}] epoch {epoch+1:3d}/{epochs} | loss = {epoch_loss:.5f}")

    model.eval()
    with torch.no_grad():
        _, encoded = model(X_tensor.to(device))
    return encoded.cpu().numpy(), history


def plot_2d(X, title, filename):
    plt.figure(figsize=(8, 6))
    for c in classes:
        points = X[y == c]
        plt.scatter(points[:, 0], points[:, 1],
                    alpha=0.6, label="quality = " + str(c))
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.show()
    plt.close()


def plot_3d(X, title, filename):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in classes:
        points = X[y == c]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   alpha=0.6, label="quality = " + str(c))
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_zlabel("Компонента 3")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # ---------- PCA ----------
    print("\n=== PCA ===")
    X_pca_2, eigvals, _ = pca_method(X, n_components=2)
    X_pca_3, _, _       = pca_method(X, n_components=3)

    plot_2d(X_pca_2, "PCA (2 компоненты)", "pca_2d.png")
    plot_3d(X_pca_3, "PCA (3 компоненты)", "pca_3d.png")

    total_var = np.sum(eigvals)
    print("Собственные значения (по убыванию):")
    print(np.round(eigvals, 4))
    print("Дисперсия, сохранённая PCA:")
    for k in (2, 3):
        retained = np.sum(eigvals[:k]) / total_var
        print(f"  k={k}: сохранено {retained*100:.2f}%, "
              f"потери = {(1 - retained)*100:.2f}%")

    print("\n=== Автоэнкодер ===")
    X_auto_2, hist_2 = train_autoencoder(2)
    X_auto_3, hist_3 = train_autoencoder(3)

    plot_2d(X_auto_2, "Автоэнкодер (2 компоненты)", "ae_2d.png")
    plot_3d(X_auto_3, "Автоэнкодер (3 компоненты)", "ae_3d.png")

    plt.figure(figsize=(8, 5))
    plt.plot(hist_2, label="bottleneck = 2")
    plt.plot(hist_3, label="bottleneck = 3")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE loss")
    plt.title("Кривые обучения автоэнкодера")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = PLOTS_DIR / "ae_losses.png"
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.show()
    plt.close()

    print("\n=== t-SNE ===")
    for perplexity in [20, 30, 50]:
        print(f"  perplexity = {perplexity}")
        X_tsne_2 = TSNE(
            n_components=2, perplexity=perplexity,
            init="pca", random_state=SEED,
        ).fit_transform(X)
        X_tsne_3 = TSNE(
            n_components=3, perplexity=perplexity,
            init="pca", random_state=SEED,
        ).fit_transform(X)

        plot_2d(X_tsne_2,
                f"t-SNE (2 компоненты, perplexity = {perplexity})",
                f"tsne_2d_perp{perplexity}.png")
        plot_3d(X_tsne_3,
                f"t-SNE (3 компоненты, perplexity = {perplexity})",
                f"tsne_3d_perp{perplexity}.png")