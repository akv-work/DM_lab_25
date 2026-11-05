import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

pca2 = PCA(n_components=2)
X_pca_2 = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
X_pca_3 = pca3.fit_transform(X_scaled)

tsne2 = TSNE(n_components=2, perplexity=40, init='pca', random_state=42)
X_tsne_2 = tsne2.fit_transform(X_scaled)

tsne3 = TSNE(n_components=3, perplexity=40, init='pca', random_state=42)
X_tsne_3 = tsne3.fit_transform(X_scaled)

class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_latent(self, x):
        return self.encoder(x)

def train_autoencoder(model, data, epochs=250, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(data), data)
        loss.backward()
        optimizer.step()
    return model

ae2 = Autoencoder(bottleneck_dim=2)
ae2 = train_autoencoder(ae2, X_tensor)
X_ae_2 = ae2.get_latent(X_tensor).detach().numpy()

ae3 = Autoencoder(bottleneck_dim=3)
ae3 = train_autoencoder(ae3, X_tensor)
X_ae_3 = ae3.get_latent(X_tensor).detach().numpy()

fig = plt.figure(figsize=(18, 10))
cmap = 'tab10'

ax1 = fig.add_subplot(231)
ax1.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y, cmap=cmap, alpha=0.7)
ax1.set_title('PCA (2D)')

ax2 = fig.add_subplot(232)
ax2.scatter(X_tsne_2[:, 0], X_tsne_2[:, 1], c=y, cmap=cmap, alpha=0.7)
ax2.set_title('t-SNE (2D)')

ax3 = fig.add_subplot(233)
ax3.scatter(X_ae_2[:, 0], X_ae_2[:, 1], c=y, cmap=cmap, alpha=0.7)
ax3.set_title('Autoencoder (2D)')

ax4 = fig.add_subplot(234, projection='3d')
ax4.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y, cmap=cmap, alpha=0.7)
ax4.set_title('PCA (3D)')

ax5 = fig.add_subplot(235, projection='3d')
ax5.scatter(X_tsne_3[:, 0], X_tsne_3[:, 1], X_tsne_3[:, 2], c=y, cmap=cmap, alpha=0.7)
ax5.set_title('t-SNE (3D)')

ax6 = fig.add_subplot(236, projection='3d')
ax6.scatter(X_ae_3[:, 0], X_ae_3[:, 1], X_ae_3[:, 2], c=y, cmap=cmap, alpha=0.7)
ax6.set_title('Autoencoder (3D)')

plt.tight_layout()
plt.show()