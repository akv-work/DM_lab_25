import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

np.random.seed(42)
torch.manual_seed(42)

digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'Размер выборки: {X.shape[0]} объектов, {X.shape[1]} признаков')
print(f'Классы: {np.unique(y)}')

colors = plt.cm.tab10(np.linspace(0, 1, 10))
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']

def plot_2d(ax, Z, title):
    for i in range(10):
        mask = y == i
        ax.scatter(Z[mask, 0], Z[mask, 1], s=15, marker=markers[i],
                   color=colors[i], label=str(i), alpha=0.7, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

def plot_3d(ax, Z, title):
    for i in range(10):
        mask = y == i
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=15, marker=markers[i],
                   color=colors[i], label=str(i), alpha=0.7, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    ax.set_xlabel('Comp 1')
    ax.set_ylabel('Comp 2')
    ax.set_zlabel('Comp 3')

pca_2d = PCA(n_components=2)
Z_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
Z_pca_3d = pca_3d.fit_transform(X_scaled)

print("Запуск t-SNE (2D)...")

tsne_2d_p30 = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
Z_tsne_2d_p30 = tsne_2d_p30.fit_transform(X_scaled)

tsne_2d_p50 = TSNE(n_components=2, perplexity=50, init='pca', random_state=42)
Z_tsne_2d_p50 = tsne_2d_p50.fit_transform(X_scaled)

print("Запуск t-SNE (3D)...")
tsne_3d = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
Z_tsne_3d = tsne_3d.fit_transform(X_scaled)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        # Энкодер: 64 -> 32 -> 2 (или 3)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)
        )
        # Декодер: 2 (или 3) -> 32 -> 64
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(data, bottleneck_dim, epochs=50, batch_size=64):
    X_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=data.shape[1], bottleneck_dim=bottleneck_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        Z_ae = model.encoder(X_tensor).numpy()
    return Z_ae

print("Обучение автоэнкодера (2 нейрона)...")
Z_ae_2d = train_autoencoder(X_scaled, bottleneck_dim=2, epochs=100)

print("Обучение автоэнкодера (3 нейрона)...")
Z_ae_3d = train_autoencoder(X_scaled, bottleneck_dim=3, epochs=100)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plot_2d(axes[0, 0], Z_pca_2d, 'PCA (2 компоненты)')
plot_2d(axes[0, 1], Z_tsne_2d_p30, 't-SNE (2D, perplexity=30)')
plot_2d(axes[1, 0], Z_tsne_2d_p50, 't-SNE (2D, perplexity=50)')
plot_2d(axes[1, 1], Z_ae_2d, 'Autoencoder (2 нейрона)')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Цифра", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.savefig('lab2_2d_comparison.png', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
plot_3d(ax1, Z_pca_3d, 'PCA (3 компоненты)')

ax2 = fig.add_subplot(132, projection='3d')
plot_3d(ax2, Z_tsne_3d, 't-SNE (3D)')

ax3 = fig.add_subplot(133, projection='3d')
plot_3d(ax3, Z_ae_3d, 'Autoencoder (3 нейрона)')

fig.legend(handles, labels, loc='center right', title="Цифра", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.savefig('lab2_3d_comparison.png', bbox_inches='tight')
plt.show()

print("\n АНАЛИЗ РЕЗУЛЬТАТОВ ")
print("1. PCA: Линейный метод. Хорошо сохраняет глобальную структуру, но классы сильно перемешаны.")
print("2. t-SNE: Нелинейный метод. Отлично разделяет классы на локальном уровне, образуя плотные кластеры. perplexity=30 показывает более четкое разделение, чем 50.")
print("3. Автоэнкодер: Нелинейное сжатие. Позволяет выделить скрытые нелинейные зависимости. В 2D пространстве разделение лучше, чем у PCA, но уступает t-SNE, так как автоэнкодер не оптимизирует расстояние между классами напрямую (это делает t-SNE).")
