import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
df = pd.read_csv(url, sep=';')
X = df.drop('quality', axis=1).values
y = df['quality'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)   # линейный узкий слой
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

def train_autoencoder(bottleneck_dim, epochs=100, lr=1e-3, batch_size=256):
    model = Autoencoder(X_tensor.shape[1], bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch in loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            x_rec, _ = model(x_batch)
            loss = criterion(x_rec, x_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        _, z = model(X_tensor)
    return z.numpy()

z2_ae = train_autoencoder(2)
z3_ae = train_autoencoder(3)


plt.figure(figsize=(8,6))
sc = plt.scatter(z2_ae[:,0], z2_ae[:,1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(sc, label='Quality')
plt.title('Автоэнкодер, 2 компоненты')
plt.xlabel('Компонента 1'); plt.ylabel('Компонента 2')
plt.show()


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(z3_ae[:,0], z3_ae[:,1], z3_ae[:,2], c=y, cmap='tab10', alpha=0.6)
fig.colorbar(sc, label='Quality')
ax.set_title('Автоэнкодер, 3 компоненты')
plt.show()


perplexities = [20, 30, 40, 50, 60]
for perp in perplexities:
    tsne2 = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42)
    z_tsne2 = tsne2.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(z_tsne2[:,0], z_tsne2[:,1], c=y, cmap='tab10', alpha=0.6)
    plt.colorbar(sc, label='Quality')
    plt.title(f't-SNE 2D, perplexity={perp}')
    plt.show()


tsne2 = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
z_tsne2 = tsne2.fit_transform(X_scaled)

tsne3 = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
z_tsne3 = tsne3.fit_transform(X_scaled)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(z_tsne3[:,0], z_tsne3[:,1], z_tsne3[:,2], c=y, cmap='tab10', alpha=0.6)
fig.colorbar(sc, label='Quality')
ax.set_title('t-SNE, 3 компоненты (perplexity=30)')
plt.show()


def pca_custom(X, n_components):

    X_centered = X - np.mean(X, axis=0)

    cov = np.cov(X_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    W = eigvecs[:, :n_components]
    return X_centered @ W, eigvals

z2_pca, _ = pca_custom(X_scaled, 2)
z3_pca, _ = pca_custom(X_scaled, 3)


plt.figure(figsize=(8,6))
sc = plt.scatter(z2_pca[:,0], z2_pca[:,1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(sc, label='Quality')
plt.title('PCA, 2 компоненты')
plt.xlabel('ГК1'); plt.ylabel('ГК2')
plt.show()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(z3_pca[:,0], z3_pca[:,1], z3_pca[:,2], c=y, cmap='tab10', alpha=0.6)
fig.colorbar(sc, label='Quality')
ax.set_title('PCA, 3 компоненты')
plt.show()