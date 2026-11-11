# =============================================================
# Lab 2 — Autoencoders — Variant 12  (NumPy-only version)
# Dataset: Optical Recognition of Handwritten Digits
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Data
# -------------------------------------------------------------
digits = load_digits()
X = digits.data.astype(np.float32)   # (1797, 64)
y = digits.target.astype(int)
print(f"Dataset: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

# Normalize inputs to [0, 1] range for stable training
X = X / X.max()

# -------------------------------------------------------------
# 2. NumPy Autoencoder (ReLU + Adam + MSE)
# -------------------------------------------------------------
class NumpyAutoencoder:
    """Fully connected AE: 64 -> 32 -> latent -> 32 -> 64, ReLU activations."""
    def __init__(self, in_dim=64, hidden=32, latent=2, lr=1e-3, seed=SEED):
        rng = np.random.default_rng(seed)
        # He init for ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2/in_dim), (in_dim, hidden));  self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, latent));  self.b2 = np.zeros(latent)
        self.W3 = rng.normal(0, np.sqrt(2/latent), (latent, hidden));  self.b3 = np.zeros(hidden)
        self.W4 = rng.normal(0, np.sqrt(2/hidden), (hidden, in_dim));  self.b4 = np.zeros(in_dim)

        # Adam state
        self.lr = lr
        self.params = ["W1","b1","W2","b2","W3","b3","W4","b4"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    def encode(self, X):
        h1 = np.maximum(0.0, X @ self.W1 + self.b1)
        return h1 @ self.W2 + self.b2

    def forward(self, X):
        h1 = np.maximum(0.0, X @ self.W1 + self.b1)
        z  = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z  @ self.W3 + self.b3)
        x_hat = h2 @ self.W4 + self.b4
        return x_hat, z, h1, h2

    def _adam_step(self, name, grad):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
        self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grad ** 2)
        m_hat = self.m[name] / (1 - beta1 ** self.t)
        v_hat = self.v[name] / (1 - beta2 ** self.t)
        param = getattr(self, name) - self.lr * m_hat / (np.sqrt(v_hat) + eps)
        setattr(self, name, param)

    def train_epoch(self, X, batch_size=64):
        idx = np.random.permutation(len(X))
        total = 0.0
        for start in range(0, len(X), batch_size):
            b = idx[start:start+batch_size]
            Xb = X[b]
            n, d = Xb.shape

            # ---- Forward ----
            h1 = np.maximum(0.0, Xb @ self.W1 + self.b1)
            z  = h1 @ self.W2 + self.b2
            h2 = np.maximum(0.0, z  @ self.W3 + self.b3)
            x_hat = h2 @ self.W4 + self.b4

            diff = x_hat - Xb
            loss = (diff ** 2).mean()
            total += loss * n

            # ---- Backward ----
            d_xhat = 2 * diff / (n * d)
            dW4 = h2.T @ d_xhat;  db4 = d_xhat.sum(axis=0)
            d_h2 = (d_xhat @ self.W4.T) * (h2 > 0)

            dW3 = z.T @ d_h2;     db3 = d_h2.sum(axis=0)
            d_z  = d_h2 @ self.W3.T          # linear latent, no activation
            dW2 = h1.T @ d_z;     db2 = d_z.sum(axis=0)
            d_h1 = (d_z @ self.W2.T) * (h1 > 0)

            dW1 = Xb.T @ d_h1;    db1 = d_h1.sum(axis=0)

            # ---- Adam update ----
            self.t += 1
            for name, grad in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2),
                               ("W3",dW3),("b3",db3),("W4",dW4),("b4",db4)]:
                self._adam_step(name, grad)
        return total / len(X)


def train_autoencoder(X, latent_dim=2, epochs=100, batch_size=64, lr=1e-3):
    model = NumpyAutoencoder(in_dim=X.shape[1], hidden=32, latent=latent_dim, lr=lr)
    for epoch in range(epochs):
        loss = model.train_epoch(X, batch_size=batch_size)
        if (epoch + 1) % 20 == 0:
            print(f"[AE latent={latent_dim}] epoch {epoch+1:3d} | loss={loss:.6f}")
    return model


print("\n=== Training autoencoders (NumPy) ===")
ae_2d = train_autoencoder(X, latent_dim=2, epochs=100)
ae_3d = train_autoencoder(X, latent_dim=3, epochs=100)

Z_ae_2d = ae_2d.encode(X)
Z_ae_3d = ae_3d.encode(X)

# -------------------------------------------------------------
# 3. Plot helpers
# -------------------------------------------------------------
def plot_2d(Z, y, title, savepath=None):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=10, alpha=0.85)
    plt.colorbar(sc, ticks=range(10), label="Digit")
    plt.title(title); plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight"); print(f"Saved: {savepath}")
    plt.show(); plt.close()


def plot_3d(Z, y, title, savepath=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap="tab10", s=10, alpha=0.85)
    fig.colorbar(sc, ticks=range(10), label="Digit", shrink=0.7)
    ax.set_title(title)
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.set_zlabel("Component 3")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight"); print(f"Saved: {savepath}")
    plt.show(); plt.close()


# -------------------------------------------------------------
# 4. Autoencoder plots
# -------------------------------------------------------------
print("\n=== Autoencoder visualizations ===")
plot_2d(Z_ae_2d, y, "Autoencoder — 2 latent neurons",
        os.path.join(FIG_DIR, "ae_2d.png"))
plot_3d(Z_ae_3d, y, "Autoencoder — 3 latent neurons",
        os.path.join(FIG_DIR, "ae_3d.png"))

# -------------------------------------------------------------
# 5. t-SNE
# -------------------------------------------------------------
def run_tsne(X, n_components=2, perplexities=(20, 30, 40, 50, 60), init="pca"):
    results = {}
    for perp in perplexities:
        tsne = TSNE(n_components=n_components, perplexity=perp, init=init,
                    random_state=SEED, max_iter=1000, learning_rate="auto")
        results[perp] = tsne.fit_transform(X)
        print(f"[t-SNE {n_components}D] perplexity={perp} done.")
    return results


print("\n=== Running t-SNE ===")
tsne_2d = run_tsne(X, n_components=2)
tsne_3d = run_tsne(X, n_components=3)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (perp, Z) in zip(axes.ravel(), tsne_2d.items()):
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
    ax.set_title(f"t-SNE 2D, perplexity={perp}")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tsne_2d_grid.png"), dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print(f"Saved: {os.path.join(FIG_DIR, 'tsne_2d_grid.png')}")

plot_2d(tsne_2d[30], y, "t-SNE 2D (perplexity=30)",
        os.path.join(FIG_DIR, "tsne_2d.png"))

for perp, Z in tsne_3d.items():
    plot_3d(Z, y, f"t-SNE 3D (perplexity={perp})",
            os.path.join(FIG_DIR, f"tsne_3d_perp{perp}.png"))

# -------------------------------------------------------------
# 6. PCA
# -------------------------------------------------------------
print("\n=== PCA ===")
pca_2d = PCA(n_components=2); Z_pca_2d = pca_2d.fit_transform(X)
pca_3d = PCA(n_components=3); Z_pca_3d = pca_3d.fit_transform(X)

print(f"Explained variance (2 comps): {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"Explained variance (3 comps): {pca_3d.explained_variance_ratio_.sum():.4f}")

plot_2d(Z_pca_2d, y, "PCA — 2 components",
        os.path.join(FIG_DIR, "pca_2d.png"))
plot_3d(Z_pca_3d, y, "PCA — 3 components",
        os.path.join(FIG_DIR, "pca_3d.png"))

print("\n=== Summary ===")
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.2%}")
print(f"PCA 3D explained variance: {pca_3d.explained_variance_ratio_.sum():.2%}")
print(f"All figures saved in '{FIG_DIR}/'.")
print("Done.")