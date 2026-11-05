import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

# Загрузка данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'agaricus-lepiota.data')

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Файл не найден: {CSV_PATH}")

columns = [
    'poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises',
    'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
    'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring',
    'veil-type', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]

df = pd.read_csv(CSV_PATH, header=None, names=columns)
print("Размер данных:", df.shape)
print(df.head())

# Предобработка
target = df['poisonous'].copy()
X = df.drop(columns=['poisonous']).copy()

X = X.replace('?', np.nan)
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].mode()[0])

X_encoded = pd.get_dummies(X, drop_first=False)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded.astype(float))

y = (target == 'p').astype(int).values
print("Размер после One-Hot:", X_encoded.shape)
print("Класс 0:", (y == 0).sum(), "| Класс 1:", (y == 1).sum())


# Автоэнкодер
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(X, latent_dim, epochs=300, lr=1e-3):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model = Autoencoder(X.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, _ = model(X_tensor)
        loss = loss_fn(x_hat, X_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        _, z = model(X_tensor)
    return z.numpy(), losses


print("\nОбучение AE (2 нейрона)")
Z_ae_2, losses_2 = train_autoencoder(X_scaled, latent_dim=2)

print("\nОбучение AE (3 нейрона)")
Z_ae_3, losses_3 = train_autoencoder(X_scaled, latent_dim=3)

# График лоссов
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(losses_2, color='steelblue')
axes[0].set_title('Функция потерь AE (2 нейрона)')
axes[0].set_xlabel('Эпоха'); axes[0].set_ylabel('MSE'); axes[0].grid(alpha=0.3)
axes[1].plot(losses_3, color='darkorange')
axes[1].set_title('Функция потерь AE (3 нейрона)')
axes[1].set_xlabel('Эпоха'); axes[1].set_ylabel('MSE'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ae_losses.png', dpi=150)
plt.show()

# Визуализация AE
colors = {0: 'green', 1: 'red'}
labels = {0: 'edible', 1: 'poisonous'}

fig, ax = plt.subplots(figsize=(9, 7))
for cls in [0, 1]:
    m = (y == cls)
    ax.scatter(Z_ae_2[m, 0], Z_ae_2[m, 1], c=colors[cls], label=labels[cls],
               alpha=0.5, s=12, edgecolors='k', linewidths=0.2)
ax.set_xlabel('Латентная компонента 1')
ax.set_ylabel('Латентная компонента 2')
ax.set_title('Автоэнкодер - 2 нейрона')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ae_2d.png', dpi=150)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for cls in [0, 1]:
    m = (y == cls)
    ax.scatter(Z_ae_3[m, 0], Z_ae_3[m, 1], Z_ae_3[m, 2],
               c=colors[cls], label=labels[cls], alpha=0.5, s=12,
               edgecolors='k', linewidths=0.2)
ax.set_xlabel('Компонента 1'); ax.set_ylabel('Компонента 2'); ax.set_zlabel('Компонента 3')
ax.set_title('Автоэнкодер - 3 нейрона')
ax.legend()
plt.tight_layout()
plt.savefig('ae_3d.png', dpi=150)
plt.show()


# PCA
def pca_manual(data, n_components=2):
    data_centered = data - data.mean(axis=0)
    cov = np.cov(data_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    comps = eigvecs[:, :n_components]
    return data_centered.dot(comps), eigvals.real, eigvecs.real


X_pca_manual_2, eigenvalues, eigenvectors = pca_manual(X_scaled, 2)
X_pca_manual_3 = (X_scaled - X_scaled.mean(axis=0)).dot(eigenvectors[:, :3])

pca_sk_2 = PCA(n_components=2)
X_pca_2 = pca_sk_2.fit_transform(X_scaled)

pca_sk_3 = PCA(n_components=3)
X_pca_3 = pca_sk_3.fit_transform(X_scaled)

print("\nОбъяснённая дисперсия (2 комп.):",
      round(pca_sk_2.explained_variance_ratio_.sum(), 4))
print("Объяснённая дисперсия (3 комп.):",
      round(pca_sk_3.explained_variance_ratio_.sum(), 4))

# 2D PCA
fig, ax = plt.subplots(figsize=(9, 7))
for cls in [0, 1]:
    m = (y == cls)
    ax.scatter(X_pca_2[m, 0], X_pca_2[m, 1], c=colors[cls], label=labels[cls],
               alpha=0.5, s=12, edgecolors='k', linewidths=0.2)
ax.set_xlabel(f'PC1 ({pca_sk_2.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_sk_2.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA - 2 компоненты')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_2d.png', dpi=150)
plt.show()

# 3D PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for cls in [0, 1]:
    m = (y == cls)
    ax.scatter(X_pca_3[m, 0], X_pca_3[m, 1], X_pca_3[m, 2],
               c=colors[cls], label=labels[cls], alpha=0.5, s=12,
               edgecolors='k', linewidths=0.2)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.set_title('PCA - 3 компоненты')
ax.legend()
plt.tight_layout()
plt.savefig('pca_3d.png', dpi=150)
plt.show()

full_info = eigenvalues.sum()
loss_2 = (1 - eigenvalues[:2].sum() / full_info) * 100
loss_3 = (1 - eigenvalues[:3].sum() / full_info) * 100
print(f"Потери при 2 компонентах: {loss_2:.2f}%")
print(f"Потери при 3 компонентах: {loss_3:.2f}%")


# t-SNE
N_SAMPLE = len(X_scaled)
if N_SAMPLE > 5000:
    idx = np.random.choice(N_SAMPLE, 5000, replace=False)
    X_tsne_input = X_scaled[idx]
    y_tsne = y[idx]
else:
    X_tsne_input = X_scaled
    y_tsne = y

perplexities = [20, 30, 40, 50, 60]
fig, axes = plt.subplots(1, len(perplexities), figsize=(4 * len(perplexities), 4))

last_Z = None
for ax, perp in zip(axes, perplexities):
    print(f"t-SNE 2D, perplexity={perp}")
    tsne = TSNE(n_components=2, perplexity=perp, init='pca',
                random_state=42, max_iter=1000)
    Z = tsne.fit_transform(X_tsne_input)
    last_Z = Z
    for cls in [0, 1]:
        m = (y_tsne == cls)
        ax.scatter(Z[m, 0], Z[m, 1], c=colors[cls], label=labels[cls],
                   alpha=0.5, s=8)
    ax.set_title(f'perp={perp}')
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend()
plt.suptitle('t-SNE - 2 компоненты при разных perplexity')
plt.tight_layout()
plt.savefig('tsne_2d_perplexities.png', dpi=150)
plt.show()

print("t-SNE 3D, perplexity=40")
tsne3 = TSNE(n_components=3, perplexity=40, init='pca',
             random_state=42, max_iter=1000)
Z3 = tsne3.fit_transform(X_tsne_input)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for cls in [0, 1]:
    m = (y_tsne == cls)
    ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], c=colors[cls],
               label=labels[cls], alpha=0.5, s=8)
ax.set_title('t-SNE - 3 компоненты (perplexity=40)')
ax.legend()
plt.tight_layout()
plt.savefig('tsne_3d.png', dpi=150)
plt.show()


# Итоговое сравнение
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for cls in [0, 1]:
    m = (y == cls)
    axes[0].scatter(Z_ae_2[m, 0], Z_ae_2[m, 1], c=colors[cls],
                    label=labels[cls], alpha=0.5, s=10)
axes[0].set_title('Автоэнкодер (2 нейрона)')
axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

for cls in [0, 1]:
    m = (y == cls)
    axes[1].scatter(X_pca_2[m, 0], X_pca_2[m, 1], c=colors[cls],
                    label=labels[cls], alpha=0.5, s=10)
axes[1].set_title('PCA (2 компоненты)')
axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

for cls in [0, 1]:
    m = (y_tsne == cls)
    axes[2].scatter(last_Z[m, 0], last_Z[m, 1], c=colors[cls],
                    label=labels[cls], alpha=0.5, s=10)
axes[2].set_title('t-SNE (2D, perp=60)')
axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('comparison_2d.png', dpi=150)
plt.show()

print("\nГотово. Сохранены файлы:")
for f in ['ae_losses.png', 'ae_2d.png', 'ae_3d.png',
          'pca_2d.png', 'pca_3d.png',
          'tsne_2d_perplexities.png', 'tsne_3d.png',
          'comparison_2d.png']:
    print(" ", f)