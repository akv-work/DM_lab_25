from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

OUT = Path(__file__).parent
ROOT = OUT.parent
PRE_EPOCHS = 80
SUP_EPOCHS = 200
HIDDEN = (32, 16, 8)


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
    loss_fn = nn.CrossEntropyLoss()
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


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(1).numpy()
    return f1_score(y, pred, average="macro"), confusion_matrix(y, pred)


def save_plots(tag, names, curves, cms):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (train_loss, test_loss) in curves.items():
        ax.plot(train_loss, label=f"{label}, обучение")
        ax.plot(test_loss, linestyle="--", label=f"{label}, тест")
    ax.set(xlabel="эпоха", ylabel="Cross-Entropy", title="Ошибка классификации")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_loss.png", dpi=130)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (label, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel="предсказанный", ylabel="истинный", title=label)
        limit = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > limit else "black")
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_cm.png", dpi=130)
    plt.show()
    plt.close(fig)


def run(X, y, title, tag):
    names, y_idx = np.unique(y, return_inverse=True)
    xtr, xte, ytr, yte = train_test_split(X, y_idx, test_size=0.25, random_state=0, stratify=y_idx)
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    xte_t = torch.tensor(xte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    dims = [X.shape[1], *HIDDEN, len(names)]

    plain, ae, rbm = new_model(dims), new_model(dims), new_model(dims)
    pretrain_ae(ae, xtr_t)
    pretrain_rbm(rbm, xtr_t)
    yte_t = torch.tensor(yte, dtype=torch.long)
    models = (("без предобучения", plain), ("автоэнкодер", ae), ("RBM", rbm))
    curves = {label: supervise(model, xtr_t, ytr_t, xte_t, yte_t) for label, model in models}

    print(f"\n{title}")
    print(f"слои: {' -> '.join(map(str, dims))}, предобучение {PRE_EPOCHS} эпох/слой, дообучение {SUP_EPOCHS} эпох")
    scores, cms = {}, {}
    for label, model in models:
        f1, cm = evaluate(model, xte_t, yte)
        scores[label], cms[label] = f1, cm
        print(f"{label}: F1-macro = {f1:.3f}")
        print("классы:", ", ".join(map(str, names)))
        print(cm)
    save_plots(tag, names, curves, cms)
    return title, scores


def loaders():
    maternal = pd.read_csv(ROOT / "lab3" / "maternal.csv")
    yield maternal.drop(columns="RiskLevel").to_numpy(float), maternal["RiskLevel"].str.strip().to_numpy(), "Maternal Health Risk, класс RiskLevel", "maternal"
    rice = pd.read_csv(ROOT / "lab2" / "rice.csv")
    yield rice.drop(columns="Class").to_numpy(float), rice["Class"].to_numpy(), "Rice (ЛР2), класс Class", "rice"
    wdbc = pd.read_csv(ROOT / "lab2_1" / "wdbc.data", header=None)
    yield wdbc.iloc[:, 2:].to_numpy(float), wdbc[1].to_numpy(), "WDBC (ЛР2), класс Diagnosis", "wdbc"


results = [run(*item) for item in loaders()]

print("\nВыводы:")
for title, scores in results:
    text = ", ".join(f"{name} {value:.3f}" for name, value in scores.items())
    best = max(scores, key=scores.get)
    print(f"- {title}: {text}. Выше всех — {best}.")
print("Сеть та же, что в ЛР3: без предобучения и автоэнкодер повторяют её F1. "
      "RBM — вещественный видимый слой и бинарный скрытый, один шаг CD, 80 эпох на слой. "
      "На Rice три варианта почти совпали. На WDBC RBM чуть ниже. "
      "На Maternal Health Risk RBM слабее всех: mid risk путается с low risk чаще, чем у автоэнкодера. "
      "RBM запоминает устройство признаков, а не границу классов, поэтому при полной разметке и 200 эпохах дообучения она не обгоняет случайный старт.")
