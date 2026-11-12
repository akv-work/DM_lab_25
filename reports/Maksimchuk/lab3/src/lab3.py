import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, confusion_matrix
)

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_PRE, EPOCHS_HEAD, EPOCHS = 50, 20, 100
BATCH, LR = 128, 0.001


class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, n_out)
        )

    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU())
        self.decoder = nn.Linear(n_hidden, n_in)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def create_dataloader(X, y=None):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return DataLoader(TensorDataset(X_t), batch_size=BATCH, shuffle=True)

    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1) if len(y.shape) == 1 and y.dtype != int else torch.tensor(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH, shuffle=True)


def pretrain(X, sizes):
    out = X.copy()
    weights = []

    for a, b in zip(sizes[:-1], sizes[1:]):
        ae = Autoencoder(a, b).to(DEVICE)
        opt = torch.optim.Adam(ae.parameters(), lr=LR)
        loader = create_dataloader(out)

        for _ in range(EPOCHS_PRE):
            for (xb,) in loader:
                xb = xb.to(DEVICE)
                loss = nn.MSELoss()(ae(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        layer = ae.encoder[0]
        weights.append((layer.weight.detach().clone(), layer.bias.detach().clone()))

        # Получаем выход скрытого слоя для следующего шага
        ae.eval()
        with torch.no_grad():
            out = ae.encoder(torch.tensor(out, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    return weights


def make_net(n_in, n_out, weights=None):
    model = Net(n_in, n_out).to(DEVICE)
    if weights:
        linear_layers = [m for m in model.layers if isinstance(m, nn.Linear)]
        for layer, (w, b) in zip(linear_layers[:-1], weights):
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
    return model


def train_step(model, loader, opt, loss_fn, is_reg):
    model.train()
    for batch in loader:
        xb, yb = batch
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)

        if is_reg:
            yb = yb.float().view(-1, 1)
        else:
            yb = yb.long()

        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train(model, X, y, is_reg, epochs):
    loss_fn = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = create_dataloader(X, y)
    for _ in range(epochs):
        train_step(model, loader, opt, loss_fn, is_reg)


def train_pretrained(model, X, y, is_reg):
    linear_layers = [m for m in model.layers if isinstance(m, nn.Linear)]

    # 1. Замораживаем всё, кроме последнего слоя
    for p in model.parameters(): p.requires_grad = False
    for p in linear_layers[-1].parameters(): p.requires_grad = True

    train(model, X, y, is_reg, EPOCHS_HEAD)

    # 2. Размораживаем всё и дообучаем
    for p in model.parameters(): p.requires_grad = True
    train(model, X, y, is_reg, EPOCHS)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(x).cpu().numpy()


def run_regression():
    data = pd.read_csv("CASP.csv")
    X = data.iloc[:, 1:].values.astype(float)
    y = data.iloc[:, 0].values.astype(float)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

    print("\nCASP: регрессия")
    print("Размер:", X.shape)
    print("Архитектура: 9-32-16-8-1")

    model = make_net(9, 1)
    train(model, Xtr, ytr, is_reg=True, epochs=EPOCHS)
    p1 = predict(model, Xte).ravel()

    w = pretrain(Xtr, [9, 32, 16, 8])
    model = make_net(9, 1, w)
    train_pretrained(model, Xtr, ytr, is_reg=True)
    p2 = predict(model, Xte).ravel()

    for name, p in [("Без предобучения", p1), ("С предобучением", p2)]:
        mask = yte > 0
        mape = np.mean(np.abs((yte[mask] - p[mask]) / yte[mask])) * 100
        print(f"\n{name}:")
        print("MAE:", round(mean_absolute_error(yte, p), 4))
        print("RMSE:", round(mean_squared_error(yte, p) ** 0.5, 4))
        print("R2:", round(r2_score(yte, p), 4))
        print("MAPE без нулей:", round(mape, 2), "%")


def run_classification():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    data = pd.read_csv(url, header=None)
    X = data.iloc[:, 2:].values.astype(float)
    y = np.where(data.iloc[:, 1].values == "M", 1, 0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

    print("\nWDBC: классификация")
    print("Размер:", X.shape)
    print("Архитектура: 30-32-16-8-2")

    model = make_net(30, 2)
    train(model, Xtr, ytr, is_reg=False, epochs=EPOCHS)
    p1 = predict(model, Xte).argmax(1)

    w = pretrain(Xtr, [30, 32, 16, 8])
    model = make_net(30, 2, w)
    train_pretrained(model, Xtr, ytr, is_reg=False)
    p2 = predict(model, Xte).argmax(1)

    for name, p in [("Без предобучения", p1), ("С предобучением", p2)]:
        print(f"\n{name}:")
        print("F1:", round(f1_score(yte, p), 4))
        print("Accuracy:", round(accuracy_score(yte, p), 4))
        print("Матрица ошибок:\n", confusion_matrix(yte, p))


if __name__ == "__main__":
    print("Устройство:", DEVICE)
    run_regression()
    run_classification()