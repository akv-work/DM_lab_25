import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRE_EPOCHS = 400
FT_EPOCHS = 200
RBM_EPOCHS = 200
LR = 1e-3
RBM_LR = 0.05
RBM_DECAY = 0.012
SEED = 0
MODES = ("без предобучения", "автоэнкодер", "RBM")
LOG = []


def say(text=""):
    print(text)
    LOG.append(text)


def hidden_sizes(n_in):
    sizes, width = [], n_in
    for _ in range(3):
        width = max(width // 2, 4)
        sizes.append(width)
    return sizes


class Net(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        dims = [n_in, *hidden_sizes(n_in)]
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.ReLU()) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = nn.Linear(dims[-1], n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class AE(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, n_hid), nn.ReLU())
        self.dec = nn.Linear(n_hid, n_in)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def pretrain_ae(x, sizes):
    torch.manual_seed(SEED)
    h, states = x, []
    dims = [x.shape[1], *sizes]
    for number, (n_in, n_hid) in enumerate(zip(dims[:-1], dims[1:]), 1):
        ae = AE(n_in, n_hid)
        opt = torch.optim.Adam(ae.parameters(), lr=LR)
        loss = None
        for _ in range(PRE_EPOCHS):
            opt.zero_grad()
            rec, _ = ae(h)
            loss = nn.functional.mse_loss(rec, h)
            loss.backward()
            opt.step()
        states.append({k: v.detach().clone() for k, v in ae.enc.state_dict().items()})
        scale = h.detach().pow(2).mean().clamp_min(1e-8)
        relative = loss.detach() / scale
        with torch.no_grad():
            h = ae.enc(h)
        say(
            f"    слой {number}: {n_in} -> {n_hid}, "
            f"MSE = {loss.item():.4f}, MSE/||x||^2 = {relative.item():.4f} после {PRE_EPOCHS} эпох"
        )
    return states


def rbm_recon(v, weight, v_bias, h_bias, gaussian):
    prob = torch.sigmoid(v @ weight + h_bias)
    mean = prob @ weight.T + v_bias
    if not gaussian:
        mean = torch.sigmoid(mean)
    return prob, (v - mean).pow(2).mean()


def train_rbm(v, n_hid, gaussian):
    n_vis = v.shape[1]
    weight = torch.randn(n_vis, n_hid) * 0.01
    mean = v.mean(0)
    if gaussian:
        v_bias = mean.clone()
    else:
        visible = mean.clamp(0.05, 0.95)
        v_bias = torch.log(visible / (1 - visible))
    h_bias = torch.zeros(n_hid)
    start = rbm_recon(v, weight, v_bias, h_bias, gaussian)[1].item()
    lr, n = RBM_LR, v.shape[0]
    vel_w, vel_a, vel_b = torch.zeros_like(weight), torch.zeros_like(v_bias), torch.zeros_like(h_bias)
    for epoch in range(RBM_EPOCHS):
        momentum = 0.5 if epoch < 5 else 0.9
        prob = torch.sigmoid(v @ weight + h_bias)
        hidden = torch.bernoulli(prob)
        recon = hidden @ weight.T + v_bias
        if not gaussian:
            recon = torch.sigmoid(recon)
        prob_neg = torch.sigmoid(recon @ weight + h_bias)
        vel_w = momentum * vel_w + lr * ((v.T @ prob - recon.T @ prob_neg) / n - RBM_DECAY * weight)
        vel_a = momentum * vel_a + lr * (v - recon).mean(0)
        vel_b = momentum * vel_b + lr * (prob - prob_neg).mean(0)
        weight = weight + vel_w
        v_bias = v_bias + vel_a
        h_bias = h_bias + vel_b
    prob, final = rbm_recon(v, weight, v_bias, h_bias, gaussian)
    state = {"0.weight": weight.T.contiguous(), "0.bias": h_bias.clone()}
    return state, prob.detach(), start, final.item()


def pretrain_rbm(x, sizes):
    torch.manual_seed(SEED)
    h, states = x, []
    dims = [x.shape[1], *sizes]
    for number, (n_vis, n_hid) in enumerate(zip(dims[:-1], dims[1:]), 1):
        gaussian = number == 1
        state, h, start, final = train_rbm(h, n_hid, gaussian)
        states.append(state)
        kind = "гауссовско-бернуллиевская" if gaussian else "бернуллиевская"
        say(
            f"    слой {number}: {n_vis} -> {n_hid}, {kind} RBM, "
            f"MSE {start:.4f} -> {final:.4f} за {RBM_EPOCHS} эпох CD-1"
        )
    return states


def pretrain(mode, x, sizes):
    if mode == "автоэнкодер":
        return pretrain_ae(x, sizes)
    if mode == "RBM":
        return pretrain_rbm(x, sizes)
    return None


def fit(x_np, y_np, x_te, y_te, states=None):
    torch.manual_seed(SEED)
    net = Net(x_np.shape[1], 2)
    if states is not None:
        for layer, state in zip(net.layers, states):
            layer.load_state_dict(state)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    x_test = torch.tensor(x_te, dtype=torch.float32)
    y_test = torch.tensor(y_te, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    losses, errors = [], []
    for _ in range(FT_EPOCHS):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            wrong = (net(x_test).argmax(1) != y_test).float().mean()
        errors.append(wrong.item())
    return net, losses, errors


def metrics(net, x_np, y_np):
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(x_np, dtype=torch.float32)).argmax(1).numpy()
    return (
        accuracy_score(y_np, pred),
        f1_score(y_np, pred, average="macro", zero_division=0),
        f1_score(y_np, pred, pos_label=1, zero_division=0),
        confusion_matrix(y_np, pred),
    )


def report(net, x_np, y_np, names):
    acc, f1, f1_pos, cm = metrics(net, x_np, y_np)
    say(f"  тест: accuracy = {acc:.4f}, F1-macro = {f1:.4f}, F1({names[1]}) = {f1_pos:.4f}")
    say(f"  матрица ошибок, строки — истина ({names[0]}, {names[1]}), столбцы — прогноз:")
    say("  " + np.array2string(cm).replace("\n", "\n  "))
    return acc, f1, f1_pos, cm


def plot_errors(path, curves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, losses, errors in curves:
        axes[0].plot(losses, label=label)
        axes[1].plot(errors, label=label)
    axes[0].set_title("Ошибка на обучающей выборке")
    axes[0].set_xlabel("эпоха")
    axes[0].set_ylabel("кросс-энтропия")
    axes[1].set_title("Доля ошибок на тесте")
    axes[1].set_xlabel("эпоха")
    axes[1].set_ylabel("доля ошибок")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cm(path, names, cms):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (title, cm) in zip(axes, cms):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        ax.set_xlabel("прогноз")
        ax.set_ylabel("истина")
        ax.set_title(title)
        limit = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > limit else "black",
                )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def split_scale(x, y):
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    return scaler.fit_transform(x_tr), scaler.transform(x_te), y_tr, y_te


def load_credit():
    df = pd.read_csv(DATA / "crx.data", header=None, na_values="?")
    y = (df.iloc[:, -1] == "+").astype(int).to_numpy()
    raw = df.iloc[:, :-1].copy()
    numeric = [1, 2, 7, 10, 13, 14]
    for col in numeric:
        values = pd.to_numeric(raw[col], errors="coerce")
        raw[col] = values.fillna(values.median())
    cats = [col for col in raw.columns if col not in numeric]
    for col in cats:
        raw[col] = raw[col].fillna(raw[col].mode(dropna=True).iloc[0]).astype(str)
    x = pd.get_dummies(raw, columns=cats, dtype=float).to_numpy()
    return x, y, ("-", "+")


def load_wdbc():
    df = pd.read_csv(DATA / "wdbc.data", header=None)
    y = (df[1] == "M").astype(int).to_numpy()
    return df.iloc[:, 2:].to_numpy(float), y, ("B", "M")


def leaders(scores):
    best = max(scores.values())
    return [mode for mode in MODES if best - scores[mode] <= 0.005]


def experiment(title, slug, x, y, names):
    say(f"\n{title}")
    counts = {label: int(count) for label, count in zip(names, np.bincount(y))}
    say(f"  объектов: {len(y)}, признаков после подготовки: {x.shape[1]}, классы: {counts}")
    x_tr, x_te, y_tr, y_te = split_scale(x, y)
    sizes = hidden_sizes(x_tr.shape[1])
    say(f"  архитектура: {x_tr.shape[1]} -> {' -> '.join(map(str, sizes))} -> 2")
    x_torch = torch.tensor(x_tr, dtype=torch.float32)
    curves, cms, scores = [], [], {}
    for mode in MODES:
        say(f"  {mode}")
        states = pretrain(mode, x_torch, sizes)
        net, losses, errors = fit(x_tr, y_tr, x_te, y_te, states)
        acc, f1, f1_pos, cm = report(net, x_te, y_te, names)
        scores[mode] = (acc, f1, f1_pos)
        curves.append((mode, losses, errors))
        cms.append((mode, cm))
    plot_errors(ROOT / f"{slug}_errors.png", curves)
    plot_cm(ROOT / f"{slug}_cm.png", names, cms)
    say(f"  графики: {slug}_errors.png, {slug}_cm.png")
    text = ", ".join(f"{mode} {scores[mode][1]:.4f}" for mode in MODES)
    tied = leaders({mode: scores[mode][1] for mode in MODES})
    say(f"  F1-macro: {text}")
    say(f"  лучшее качество (порог 0.005): {', '.join(tied)}")
    return scores


def main():
    say(
        f"Автоэнкодер: {PRE_EPOCHS} эпох, Adam, lr={LR}. "
        f"RBM: CD-1, {RBM_EPOCHS} эпох, momentum 0.5 затем 0.9, "
        f"lr={RBM_LR}, weight cost={RBM_DECAY}. "
        f"Дообучение классификатора: {FT_EPOCHS} эпох, Adam, lr={LR}."
    )
    credit = experiment("1. Credit Approval — классификация +/-", "credit", *load_credit())
    wdbc = experiment("2. WDBC из ЛР2 — диагноз во 2-м столбце (M/B)", "wdbc", *load_wdbc())
    say("\nВыводы")
    for title, scores in (("Credit Approval", credit), ("WDBC", wdbc)):
        text = ", ".join(f"{mode} {scores[mode][1]:.4f}" for mode in MODES)
        tied = leaders({mode: scores[mode][1] for mode in MODES})
        say(f"{title}: F1-macro {text}. Лучшее качество: {', '.join(tied)}.")
    (ROOT / "results.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
