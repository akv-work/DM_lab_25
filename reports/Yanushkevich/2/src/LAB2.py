import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(42)

data = load_breast_cancer()
X = data.data
y = data.target
class_names = data.target_names

X_scaled = StandardScaler().fit_transform(X)

colors = {0: "red", 1: "blue"}
labels = {0: class_names[0], 1: class_names[1]}


class DenseLayer:
    def __init__(self, in_dim, out_dim, activation="relu"):
        limit = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * limit
        self.b = np.zeros(out_dim)
        self.activation = activation

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, X):
        self.X_in = X
        self.Z = X @ self.W + self.b
        if self.activation == "relu":
            self.A = np.maximum(0, self.Z)
        else:
            self.A = self.Z
        return self.A

    def backward(self, dA):
        if self.activation == "relu":
            dZ = dA * (self.Z > 0)
        else:
            dZ = dA
        n = self.X_in.shape[0]
        self.dW = self.X_in.T @ dZ / n
        self.db = np.mean(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX

    def step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        mW_hat = self.mW / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)
        mb_hat = self.mb / (1 - beta1 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class Autoencoder:
    def __init__(self, input_dim, hidden_dim, code_dim):
        self.encoder = [
            DenseLayer(input_dim, hidden_dim, "relu"),
            DenseLayer(hidden_dim, code_dim, "linear"),
        ]
        self.decoder = [
            DenseLayer(code_dim, hidden_dim, "relu"),
            DenseLayer(hidden_dim, input_dim, "linear"),
        ]

    def encode(self, X):
        out = X
        for layer in self.encoder:
            out = layer.forward(out)
        return out

    def decode(self, code):
        out = code
        for layer in self.decoder:
            out = layer.forward(out)
        return out

    def forward(self, X):
        return self.decode(self.encode(X))

    def train(self, X, epochs=300, batch_size=32, lr=0.001, verbose_every=50):
        n = X.shape[0]
        t = 0
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                batch = X_shuffled[start:start + batch_size]
                t += 1

                X_hat = self.forward(batch)
                loss = np.mean((X_hat - batch) ** 2)
                epoch_loss += loss * batch.shape[0]

                dA = 2 * (X_hat - batch) / batch.shape[0]
                for layer in reversed(self.decoder):
                    dA = layer.backward(dA)
                for layer in reversed(self.encoder):
                    dA = layer.backward(dA)

                for layer in self.encoder + self.decoder:
                    layer.step(lr, t)

            if epoch % verbose_every == 0 or epoch == 1:
                print(f"[AE code_dim={self.encoder[-1].W.shape[1]}] "
                      f"epoch {epoch}/{epochs}, MSE = {epoch_loss / n:.4f}")


def train_autoencoder(X, code_dim, hidden_dim=16, epochs=300):
    ae = Autoencoder(X.shape[1], hidden_dim, code_dim)
    ae.train(X, epochs=epochs)
    return ae.encode(X)


ae_code_2d = train_autoencoder(X_scaled, code_dim=2)
ae_code_3d = train_autoencoder(X_scaled, code_dim=3)


def pca_manual(data, n_components):
    L = data.shape[0]
    mean_vector = np.sum(data, axis=0) / L
    X_centered = data - mean_vector
    K = (X_centered.T @ X_centered) / L

    eig_values, eig_vectors = np.linalg.eig(K)
    eig_values = eig_values.real
    eig_vectors = eig_vectors.real

    sort_index = np.argsort(-1 * eig_values)
    eig_values_sorted = eig_values[sort_index]
    eig_vectors_sorted = eig_vectors[:, sort_index]
    eig_vectors_sorted = eig_vectors_sorted / np.linalg.norm(eig_vectors_sorted, axis=0)

    W = eig_vectors_sorted[:, :n_components]
    Y = X_centered @ W
    return Y, eig_values_sorted


pca_manual_2d, eig_values_sorted = pca_manual(X_scaled, 2)
pca_manual_3d, _ = pca_manual(X_scaled, 3)

pca_sklearn_2d = PCA(n_components=2).fit_transform(X_scaled)
pca_sklearn_3d = PCA(n_components=3).fit_transform(X_scaled)


def criterion_J(eig_values, n_components):
    return eig_values[:n_components].sum() / eig_values.sum()


pca_loss_2d = 100 - criterion_J(eig_values_sorted, 2) * 100
pca_loss_3d = 100 - criterion_J(eig_values_sorted, 3) * 100


perplexities = [20, 30, 40, 50, 60]
tsne_results_2d = {}
for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42)
    tsne_results_2d[perp] = tsne.fit_transform(X_scaled)

best_perplexity = 30
tsne_3d = TSNE(n_components=3, perplexity=best_perplexity, init="pca", random_state=42).fit_transform(X_scaled)


def scatter_2d(ax, points, title):
    for cls in colors:
        mask = y == cls
        ax.scatter(points[mask, 0], points[mask, 1],
                    c=colors[cls], label=labels[cls], edgecolor="k", s=25, alpha=0.8)
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def scatter_3d(ax, points, title):
    for cls in colors:
        mask = y == cls
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                    c=colors[cls], label=labels[cls], edgecolor="k", s=25, alpha=0.8)
    ax.set_xlabel("K1")
    ax.set_ylabel("K2")
    ax.set_zlabel("K3")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


fig1 = plt.figure(figsize=(14, 10))

ax1 = fig1.add_subplot(2, 3, 1)
scatter_2d(ax1, ae_code_2d, "Автоэнкодер, 2 нейрона")

ax2 = fig1.add_subplot(2, 3, 2)
scatter_2d(ax2, pca_sklearn_2d, "PCA (sklearn), 2 компоненты")

ax3 = fig1.add_subplot(2, 3, 3)
scatter_2d(ax3, tsne_results_2d[best_perplexity], f"t-SNE, perplexity={best_perplexity}")

ax4 = fig1.add_subplot(2, 3, 4, projection="3d")
scatter_3d(ax4, ae_code_3d, "Автоэнкодер, 3 нейрона")

ax5 = fig1.add_subplot(2, 3, 5, projection="3d")
scatter_3d(ax5, pca_sklearn_3d, "PCA (sklearn), 3 компоненты")

ax6 = fig1.add_subplot(2, 3, 6, projection="3d")
scatter_3d(ax6, tsne_3d, f"t-SNE 3D, perplexity={best_perplexity}")

plt.tight_layout()
plt.savefig("lab2_comparison.png", dpi=150)
plt.show()


fig2, axes = plt.subplots(1, len(perplexities), figsize=(4 * len(perplexities), 4))
for ax, perp in zip(axes, perplexities):
    scatter_2d(ax, tsne_results_2d[perp], f"perplexity={perp}")
plt.tight_layout()
plt.savefig("lab2_tsne_perplexity.png", dpi=150)
plt.show()


print()
print("Потери информативности PCA при 2 компонентах:", round(pca_loss_2d, 2), "%")
print("Потери информативности PCA при 3 компонентах:", round(pca_loss_3d, 2), "%")