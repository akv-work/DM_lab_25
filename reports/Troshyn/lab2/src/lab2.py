import random
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim

DATASET_PATH = "rice/Rice_Cammeo_Osmancik.arff"

RANDOM_SEED = 42

AE_EPOCHS = 300
AE_BATCH_SIZE = 128
AE_LEARNING_RATE = 0.001

TSNE_PERPLEXITIES = [20, 30, 40, 50, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def load_arff(filename):
    data, meta = arff.loadarff(filename)

    feature_names = [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent"        
    ]

    X = np.zeros((len(data), len(feature_names)), dtype=np.float32)

    for i, feature in enumerate(feature_names):
        X[:, i] = np.asarray(data[feature], dtype=np.float32)

    labels = []

    for value in data["Class"]:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        labels.append(value)
    labels = np.array(labels)

    return X, labels, feature_names

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_scaled = (X - mean) / std

    return X_scaled, mean, std


def pca_manual(X, n_components):
    X_centered = X - np.mean(X, axis=0)

    covariance_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    components = eigenvectors[:, :n_components]

    X_reduced = X_centered @ components

    explained_variance_ratio = (eigenvalues / np.sum(eigenvalues))

    return (X_reduced, components, eigenvalues, explained_variance_ratio)


class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8),
            nn.ReLU(),

            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, input_size)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

def train_autoencoder(X, latent_size):
    model = Autoencoder(input_size=X.shape[1], latent_size=latent_size).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=AE_LEARNING_RATE)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    losses = []

    for epoch in range(AE_EPOCHS):
        model.train()

        epoch_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(DEVICE)

            output = model(x)

            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataset)

        losses.append(epoch_loss)

        if ((epoch + 1) % 25 == 0 or epoch == 0 or epoch == AE_EPOCHS - 1):
            print(f"Epoch {epoch + 1:3d}/{AE_EPOCHS} Loss: {epoch_loss:.6f}")

    model.eval()

    with torch.no_grad():
        X_tensor = X_tensor.to(DEVICE)
        encoded = model.encode(X_tensor)
        encoded = encoded.cpu().numpy()

    return model, encoded, losses           

def plot_autoencoder_loss(losses, latent_size):
    plt.figure(figsize=(8, 5))

    plt.plot(losses)

    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")

    plt.title(
        f"Обучение автоэнкодера "
        f"({latent_size} компоненты)"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"autoencoder_loss_{latent_size}d.png",
        dpi=300
    )

    plt.show()

def plot_2d(X_reduced, labels, title, filename):
    plt.figure(figsize=(8, 6))
    classes = np.unique(labels)
    markers = ["o", "s", "^", "D", "x", "*"]

    for i, class_name in enumerate(classes):
        mask = labels == class_name

        plt.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            label=class_name,
            marker=markers[i % len(markers)],
            alpha=0.7,
            s=35
        )

    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")

    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()

def plot_3d(X_reduced, labels, title, filename):
    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(111,projection="3d")

    classes = np.unique(labels)

    markers = ["o", "s", "^", "D", "x", "*"]

    for i, class_name in enumerate(classes):
        mask = labels == class_name

        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            X_reduced[mask, 2],
            label=class_name,
            marker=markers[i % len(markers)],
            alpha=0.7,
            s=35
        )

    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_zlabel("Компонента 3")

    ax.set_title(title)

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


def run_tsne(X, labels, n_components, perplexity):
    print()
    print(
        f"t-SNE: {n_components} компоненты, "
        f"perplexity = {perplexity}"
    )

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
        max_iter=1000,
        learning_rate="auto"
    )

    X_tsne = tsne.fit_transform(X)

    print(
        f"KL divergence: "
        f"{tsne.kl_divergence_:.6f}"
    )

    return X_tsne    





def main():
    print(f"Device: {DEVICE}")

    X, labels, feature_names = load_arff(DATASET_PATH)

    print(f"количество объектов: {X.shape[0]}")
    print(f"количество признаков: {X.shape[1]}")

    print("\nПризнаки:")

    for feature in feature_names:
        print(f"  {feature}")

    print("\nКлассы:")
    classes, counts = np.unique(labels, return_counts=True)
    for class_name, count in zip(classes, counts):
        print(f"  {class_name}: {count}")

    X_scaled, mean, std = standardize(X)

    #АВТОЭНКОДЕР

    #2 компоненты
    print("\nАВТОЭНКОДЕР — 2 КОМПОНЕНТЫ")
    model_2d, encoded_2d, loss_2d = train_autoencoder(
        X_scaled,
        latent_size=2
    )

    plot_autoencoder_loss(
        loss_2d,
        latent_size=2
    )

    plot_2d(
        encoded_2d,
        labels,
        "Автоэнкодер — 2 компоненты",
        "autoencoder_2d.png"
    )


    #3 компоненты
    print("\nАВТОЭНКОДЕР — 3 КОМПОНЕНТЫ")
    model_3d, encoded_3d, loss_3d = train_autoencoder(
        X_scaled,
        latent_size=3
    )

    plot_autoencoder_loss(
        loss_3d,
        latent_size=3
    )

    plot_3d(
        encoded_3d,
        labels,
        "Автоэнкодер — 3 компоненты",
        "autoencoder_3d.png"
    )

    #TSNE

    #2 компоненты
    tsne_results_2d = {}

    for perplexity in TSNE_PERPLEXITIES:

        X_tsne = run_tsne(
            X_scaled,
            labels,
            n_components=2,
            perplexity=perplexity
        )

        tsne_results_2d[perplexity] = X_tsne

        plot_2d(
            X_tsne,
            labels,
            f"t-SNE — 2 компоненты, "
            f"perplexity={perplexity}",
            f"tsne_2d_perplexity_{perplexity}.png"
        )

    #3 компоненты
    tsne_results_3d = {}

    for perplexity in TSNE_PERPLEXITIES:

        X_tsne = run_tsne(
            X_scaled,
            labels,
            n_components=3,
            perplexity=perplexity
        )

        tsne_results_3d[perplexity] = X_tsne

        plot_3d(
            X_tsne,
            labels,
            f"t-SNE — 3 компоненты, "
            f"perplexity={perplexity}",
            f"tsne_3d_perplexity_{perplexity}.png"
        )

    #PCA

    #2 компоненты
    pca_2d, components_2d, eigenvalues, explained_variance_ratio = pca_manual(X_scaled, n_components=2)

    print("\nPCA — 2 КОМПОНЕНТЫ")

    print("Доля объяснённой дисперсии по компонентам:")
    print(
        f"PC1: "
        f"{explained_variance_ratio[0]:.6f} "
        f"({explained_variance_ratio[0] * 100:.2f}%)"
    )

    print(
        f"PC2: "
        f"{explained_variance_ratio[1]:.6f} "
        f"({explained_variance_ratio[1] * 100:.2f}%)"
    )

    print(
        f"PC1 + PC2: "
        f"{np.sum(explained_variance_ratio[:2]):.6f} "
        f"({np.sum(explained_variance_ratio[:2]) * 100:.2f}%)"
    )

    plot_2d(
        pca_2d,
        labels,
        "PCA — 2 компоненты",
        "pca_2d.png"
    )


    #3 компоненты
    pca_3d, components_3d, eigenvalues, explained_variance_ratio = pca_manual(X_scaled, n_components=3)

    print("\nPCA — 3 КОМПОНЕНТЫ")

    print(
        f"PC1: "
        f"{explained_variance_ratio[0] * 100:.2f}%"
    )

    print(
        f"PC2: "
        f"{explained_variance_ratio[1] * 100:.2f}%"
    )

    print(
        f"PC3: "
        f"{explained_variance_ratio[2] * 100:.2f}%"
    )

    print(
        f"PC1 + PC2 + PC3: "
        f"{np.sum(explained_variance_ratio[:3]) * 100:.2f}%"
    )

    plot_3d(
        pca_3d,
        labels,
        "PCA — 3 компоненты",
        "pca_3d.png"
    )

main()