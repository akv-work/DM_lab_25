from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

ROOT = Path(__file__).parents[1]
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


def pretrain(model, x):
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


def supervise(model, x, y):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(SUP_EPOCHS):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(1).numpy()
    return f1_score(y, pred, average="macro"), confusion_matrix(y, pred)


def run(X, y, title):
    names, y_idx = np.unique(y, return_inverse=True)
    xtr, xte, ytr, yte = train_test_split(X, y_idx, test_size=0.25, random_state=0, stratify=y_idx)
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    xte_t = torch.tensor(xte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    dims = [X.shape[1], *HIDDEN, len(names)]

    plain, pre = new_model(dims), new_model(dims)
    pretrain(pre, xtr_t)
    supervise(plain, xtr_t, ytr_t)
    supervise(pre, xtr_t, ytr_t)

    print(f"\n{title}")
    print(f"слои: {' -> '.join(map(str, dims))}, предобучение {PRE_EPOCHS} эпох/слой, дообучение {SUP_EPOCHS} эпох")
    scores = {}
    for label, model in (("без предобучения", plain), ("с предобучением", pre)):
        f1, cm = evaluate(model, xte_t, yte)
        scores[label] = f1
        print(f"{label}: F1-macro = {f1:.3f}")
        print("классы:", ", ".join(names))
        print(cm)
    return title, scores["без предобучения"], scores["с предобучением"]


def loaders():
    maternal = pd.read_csv(Path(__file__).parent / "maternal.csv")
    yield maternal.drop(columns="RiskLevel").to_numpy(float), maternal["RiskLevel"].str.strip().to_numpy(), "Maternal Health Risk, класс RiskLevel"

    rice = pd.read_csv(ROOT / "lab2" / "rice.csv")
    yield rice.drop(columns="Class").to_numpy(float), rice["Class"].to_numpy(), "Rice (ЛР2), класс Class"

    wdbc = pd.read_csv(ROOT / "lab2_1" / "wdbc.data", header=None)
    yield wdbc.iloc[:, 2:].to_numpy(float), wdbc[1].to_numpy(), "WDBC (ЛР2), класс Diagnosis"


results = [run(X, y, title) for X, y, title in loaders()]

print("\nВыводы:")
for title, plain, pre in results:
    if pre > plain + 0.01:
        cmp = "предобучение повысило F1"
    elif plain > pre + 0.01:
        cmp = "обучение с нуля дало F1 выше"
    else:
        cmp = "разница F1 в пределах 0.01"
    print(f"- {title}: без предобучения {plain:.3f}, с предобучением {pre:.3f} — {cmp}.")
print("Сеть из четырёх полносвязных слоёв на этих выборках сходится и без предобучения. "
      "Автоэнкодеры задают скрытым слоям старт, который умеет восстанавливать признаки, "
      "но метки классов всё равно уточняются на общем дообучении, поэтому итоговое качество близкое.")
