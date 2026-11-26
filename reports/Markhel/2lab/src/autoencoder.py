"""
Лабораторная работа №2. Автоэнкодер (PyTorch) для проецирования WDBC на 2 и 3 компоненты.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

from common import BASE_DIR, FIG_DIR, load_data, standardize, separability, plot_2d, plot_3d

SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ARCHITECTURES = {
    "A": {"hidden": [], "act": None},
    "B": {"hidden": [16], "act": nn.ReLU},
    "C": {"hidden": [32, 16], "act": nn.ReLU},
    "D": {"hidden": [64, 32, 16], "act": nn.ReLU},
    "E": {"hidden": [32, 16], "act": nn.Tanh},
}
FINAL_ARCH = "B"

torch.set_num_threads(1)


class Autoencoder(nn.Module):
    def __init__(self, n_in, n_latent, hidden, act):
        super().__init__()
        self.encoder = self._stack([n_in, *hidden, n_latent], act)
        self.decoder = self._stack([n_latent, *reversed(hidden), n_in], act)

    @staticmethod
    def _stack(sizes, act):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if act is not None and i < len(sizes) - 2:
                layers.append(act())
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def describe(n_in, n_latent, arch):
    sizes = [n_in, *arch["hidden"], n_latent, *reversed(arch["hidden"]), n_in]
    act = arch["act"].__name__ if arch["act"] else "линейная"
    return "-".join(map(str, sizes)), act


def train_autoencoder(X, n_latent, arch, epochs=EPOCHS):
    torch.manual_seed(SEED)
    data = torch.tensor(X, dtype=torch.float32)
    model = Autoencoder(X.shape[1], n_latent, arch["hidden"], arch["act"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(SEED)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(data), generator=gen)
        total = 0.0
        for start in range(0, len(data), BATCH_SIZE):
            batch = data[perm[start:start + BATCH_SIZE]]
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(data))
    model.eval()
    with torch.no_grad():
        recon, z = model(data)
        mse = loss_fn(recon, data).item()
    return z.numpy(), mse, history


def run_experiments(Xs, y):
    rows, runs = [], {}
    for k in (2, 3):
        for name, arch in ARCHITECTURES.items():
            Z, mse, history = train_autoencoder(Xs, k, arch)
            runs[(name, k)] = (Z, mse, history)
            sil, acc = separability(Z, y)
            layers, act = describe(Xs.shape[1], k, arch)
            rows.append({"Модель": name, "k": k, "Архитектура": layers, "Активация": act,
                         "MSE": round(mse, 4), "Silhouette": round(sil, 3),
                         "Accuracy, %": round(acc * 100, 2)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "ae_experiments.csv", index=False, encoding="utf-8-sig")
    return table, runs


def plot_history(histories):
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, h in histories.items():
        ax.plot(range(1, len(h) + 1), h, label=f"{k} нейрона в среднем слое")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE реконструкции")
    ax.set_yscale("log")
    ax.set_title(f"Обучение автоэнкодера (модель {FINAL_ARCH})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ae_loss.png", dpi=150)


def main():
    pd.set_option("display.width", 200)
    X, y = load_data()
    Xs = standardize(X)

    print(f"\nПодбор архитектуры (Adam, lr = {LR}, эпох = {EPOCHS}, batch = {BATCH_SIZE}):")
    table, runs = run_experiments(Xs, y)
    print(table.to_string(index=False))

    arch = ARCHITECTURES[FINAL_ARCH]
    layers2, act = describe(Xs.shape[1], 2, arch)
    layers3, _ = describe(Xs.shape[1], 3, arch)
    print(f"\nИтоговая модель {FINAL_ARCH}: {layers2} / {layers3}, активация {act}")

    Z2, mse2, h2 = runs[(FINAL_ARCH, 2)]
    Z3, mse3, h3 = runs[(FINAL_ARCH, 3)]
    for k, Z, mse in ((2, Z2, mse2), (3, Z3, mse3)):
        sil, acc = separability(Z, y)
        print(f"  AE, {k} нейрона: MSE = {mse:.4f} | silhouette = {sil:.3f} | "
              f"accuracy = {acc * 100:.2f}%")

    plot_history({2: h2, 3: h3})
    plot_2d(Z2, y, f"Автоэнкодер {layers2}, 2 компоненты — WDBC", "ae_2d.png",
            ("Нейрон 1", "Нейрон 2"))
    plot_3d(Z3, y, f"Автоэнкодер {layers3}, 3 компоненты — WDBC", "ae_3d.png",
            ("Нейрон 1", "Нейрон 2", "Нейрон 3"))
    print(f"\nГрафики сохранены в папку: {FIG_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
