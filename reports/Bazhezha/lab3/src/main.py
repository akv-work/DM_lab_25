from pathlib import Path

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
LR = 1e-3
SEED = 0
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


def pretrain(x, sizes):
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


def fit(x_np, y_np, states=None):
    torch.manual_seed(SEED)
    net = Net(x_np.shape[1], 2)
    if states is not None:
        for layer, state in zip(net.layers, states):
            layer.load_state_dict(state)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(FT_EPOCHS):
        opt.zero_grad()
        nn.functional.cross_entropy(net(x), y).backward()
        opt.step()
    return net


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
    return f1


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


def verdict(f1_plain, f1_pre):
    delta = f1_pre - f1_plain
    if delta > 0.005:
        return f"предобучение выше на {delta:.4f}"
    if delta < -0.005:
        return f"без предобучения выше на {-delta:.4f}"
    return "разница меньше 0.005"


def experiment(title, x, y, names):
    say(f"\n{title}")
    counts = {label: int(count) for label, count in zip(names, np.bincount(y))}
    say(f"  объектов: {len(y)}, признаков после подготовки: {x.shape[1]}, классы: {counts}")
    x_tr, x_te, y_tr, y_te = split_scale(x, y)
    sizes = hidden_sizes(x_tr.shape[1])
    say(f"  архитектура: {x_tr.shape[1]} -> {' -> '.join(map(str, sizes))} -> 2")

    say("  обучение без предобучения")
    f1_plain = report(fit(x_tr, y_tr), x_te, y_te, names)

    say("  обучение с предобучением")
    states = pretrain(torch.tensor(x_tr, dtype=torch.float32), sizes)
    f1_pre = report(fit(x_tr, y_tr, states), x_te, y_te, names)
    say(f"  сравнение F1-macro: {verdict(f1_plain, f1_pre)}")
    return f1_plain, f1_pre


def main():
    say(f"Останов автоэнкодера: {PRE_EPOCHS} эпох. Дообучение классификатора: {FT_EPOCHS} эпох, Adam, lr={LR}.")
    credit = experiment("1. Credit Approval - классификация +/-", *load_credit())
    wdbc = experiment("2. WDBC из ЛР2 - диагноз во 2-м столбце (M/B)", *load_wdbc())
    say("\nВыводы")
    say(f"Credit Approval: F1-macro без предобучения {credit[0]:.4f}, с предобучением {credit[1]:.4f} ({verdict(*credit)}).")
    say(f"WDBC: F1-macro без предобучения {wdbc[0]:.4f}, с предобучением {wdbc[1]:.4f} ({verdict(*wdbc)}).")
    (ROOT / "results.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
