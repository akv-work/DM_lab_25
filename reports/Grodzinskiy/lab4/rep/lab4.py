from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn

OUT = Path(__file__).parent
PRE_EPOCHS = 80
SUP_EPOCHS = 200
HIDDEN = (32, 16, 8)


def find_infrared():
    for candidate in (OUT / "infrared.csv", OUT / "data" / "infrared.csv"):
        if candidate.exists():
            return candidate

class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def new_model(dims):
    torch.manual_seed(0)
    return MLP(dims)


def pretrain_ae(model, x):
    h = x
    mse = nn.MSELoss()
    for layer in model.layers[:-1]:
        dec = nn.Linear(layer.out_features, layer.in_features)
        opt = torch.optim.Adam(list(layer.parameters()) + list(dec.parameters()), lr=1e-3)
        inp = h.detach()
        for _ in range(PRE_EPOCHS):
            opt.zero_grad()
            mse(dec(torch.relu(layer(inp))), inp).backward()
            opt.step()
        with torch.no_grad():
            h = torch.relu(layer(inp))


def pretrain_rbm(model, x, lr=0.1):
    v = x
    for layer in model.layers[:-1]:
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.zeros_(layer.bias)
        bv = torch.zeros(layer.in_features)
        with torch.no_grad():
            for _ in range(PRE_EPOCHS):
                ph = torch.sigmoid(v @ layer.weight.T + layer.bias)
                nv = torch.bernoulli(ph) @ layer.weight + bv
                nh = torch.sigmoid(nv @ layer.weight.T + layer.bias)
                layer.weight += lr * (ph.T @ v - nh.T @ nv) / v.shape[0]
                layer.bias += lr * (ph - nh).mean(0)
                bv += lr * (v - nv).mean(0)
            v = torch.sigmoid(v @ layer.weight.T + layer.bias)


def supervise(model, x, y, x_test, y_test):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_loss, test_loss = [], []
    for _ in range(SUP_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        with torch.no_grad():
            test_loss.append(loss_fn(model(x_test), y_test).item())
    return train_loss, test_loss


def evaluate(model, x, y, mu_y, sd_y):
    model.eval()
    with torch.no_grad():
        pred = model(x).detach().numpy().ravel() * sd_y + mu_y
    return mean_squared_error(y, pred), mean_absolute_error(y, pred), r2_score(y, pred)


def save_plots(tag, curves, models, x_test, y_test, mu_y, sd_y):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (train_loss, test_loss) in curves.items():
        ax.plot(train_loss, label=f"{label}, обучение")
        ax.plot(test_loss, linestyle="--", label=f"{label}, тест")
    ax.set(xlabel="эпоха", ylabel="MSE", title="Ошибка регрессии")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_loss.png", dpi=130)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (label, model) in zip(axes, models):
        model.eval()
        with torch.no_grad():
            pred = model(x_test).detach().numpy().ravel() * sd_y + mu_y
        ax.scatter(y_test, pred, s=8)
        lo = min(y_test.min(), pred.min())
        hi = max(y_test.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="red")
        ax.set(xlabel="истинное", ylabel="предсказанное", title=label)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_scatter.png", dpi=130)
    plt.show()
    plt.close(fig)


def run(X, y, title, tag):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    mu_x, sd_x = x_train.mean(0), x_train.std(0)
    sd_x[sd_x == 0] = 1
    x_train = (x_train - mu_x) / sd_x
    x_test = (x_test - mu_x) / sd_x

    mu_y, sd_y = y_train.mean(), y_train.std()
    if sd_y == 0:
        sd_y = 1
    y_train_s = (y_train - mu_y) / sd_y
    y_test_s = (y_test - mu_y) / sd_y

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32).view(-1, 1)

    dims = [X.shape[1], *HIDDEN, 1]
    plain, ae, rbm = new_model(dims), new_model(dims), new_model(dims)
    pretrain_ae(ae, x_train_t)
    pretrain_rbm(rbm, x_train_t)

    models = (("без предобучения", plain), ("автоэнкодер", ae), ("RBM", rbm))
    curves = {label: supervise(model, x_train_t, y_train_t, x_test_t, y_test_t) for label, model in models}

    print(f"\n{title}")
    print(f"слои: {' -> '.join(map(str, dims))}, предобучение {PRE_EPOCHS} эпох/слой, дообучение {SUP_EPOCHS} эпох")

    scores = {}
    for label, model in models:
        mse, mae, r2 = evaluate(model, x_test_t, y_test, mu_y, sd_y)
        scores[label] = (mse, mae, r2)
        print(f"{label}: MSE = {mse:.3f}, MAE = {mae:.3f}, R2 = {r2:.3f}")

    save_plots(tag, curves, models, x_test_t, y_test, mu_y, sd_y)
    return title, scores


def loaders():
    ir = pd.read_csv(find_infrared())
    for target in ("aveOralF", "aveOralM"):
        y = ir[target].to_numpy(float)
        X = ir.drop(columns=[c for c in ("aveOralF", "aveOralM") if c in ir.columns])
        X = pd.get_dummies(X, drop_first=True)
        X = X.select_dtypes(include=[np.number]).fillna(0).to_numpy(float)
        yield X, y, f"Infrared, {target}", f"infrared_{target}"


results = [run(*item) for item in loaders()]

print("\nВыводы:")
for title, scores in results:
    text = ", ".join(
        f"{name}: MSE {mse:.3f}, MAE {mae:.3f}, R2 {r2:.3f}"
        for name, (mse, mae, r2) in scores.items()
    )
    best = max(scores, key=lambda name: scores[name][2])
    print(f"- {title}: {text}. Лучший R2 — {best}.")