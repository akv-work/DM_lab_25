import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

FILE_NAME = "winequality-white.csv"
data = pd.read_csv(FILE_NAME, sep=";")

if data.isnull().sum().sum() > 0:
    data = data.fillna(data.median(numeric_only=True))

y = data["quality"].to_numpy()
X = data.drop(columns=["quality"]).to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

torch.manual_seed(42)                                      
np.random.seed(42)  

X_tensor = torch.tensor(X, dtype=torch.float32)
classes = np.unique(y)

class Autoencoder(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, X.shape[1])
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def autoencoder(bottleneck):
    model = Autoencoder(bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=128,
        shuffle=True
    )

    history = []
    for epoch in range(150):
        epoch_loss = 0.0
        for batch in loader:
            x_batch = batch[0]

            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = loss_function(output, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        history.append(epoch_loss / len(X_tensor))

    model.eval()
    with torch.no_grad():
        _, encoded = model(X_tensor)
    return encoded.numpy(), history

X_auto_2, _ = autoencoder(2)                            
X_auto_3, _ = autoencoder(3)

def plot_2d(X, title):
    plt.figure(figsize=(8, 6))

    for c in classes:
        points = X[y == c]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            alpha=0.6,
            label="quality = " + str(c)
        )
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_3d(X, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in classes:
        points = X[y == c]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            alpha=0.6,
            label="quality = " + str(c)
        )
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_zlabel("Компонента 3")
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_2d(X_auto_2, "Автоэнкодер (2 компоненты)")
plot_3d(X_auto_3, "Автоэнкодер (3 компоненты)")

for perplexity in [20, 30, 50]:
    X_tsne_2 = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=42
    ).fit_transform(X)

    X_tsne_3 = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        random_state=42
    ).fit_transform(X)

    plot_2d(X_tsne_2, f"t-SNE (2 компоненты, perplexity = {perplexity})")
    plot_3d(X_tsne_3, f"t-SNE (3 компоненты, perplexity = {perplexity})")

covariance_matrix = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvalues = np.real_if_close(eigenvalues)
eigenvectors = np.real_if_close(eigenvectors)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

X_pca_2 = X @ eigenvectors[:, :2]
X_pca_3 = X @ eigenvectors[:, :3]

plot_2d(X_pca_2, "PCA (2 компоненты)")
plot_3d(X_pca_3, "PCA (3 компоненты)")