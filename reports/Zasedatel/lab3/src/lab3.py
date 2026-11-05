import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

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

def batches(X, y=None):
    ind = torch.randperm(len(X))
    for i in range(0, len(X), BATCH):
        j = ind[i:i+BATCH]
        xb = torch.tensor(X[j], dtype=torch.float32).to(DEVICE)
        if y is None:
            yield xb
        else:
            yield xb, torch.tensor(y[j]).to(DEVICE)

def pretrain(X, sizes):
    out, weights = X.copy(), []

    for a, b in zip(sizes[:-1], sizes[1:]):
        ae = Autoencoder(a, b).to(DEVICE)
        opt = torch.optim.Adam(ae.parameters(), lr=LR)

        for _ in range(EPOCHS_PRE):
            for xb in batches(out):
                loss = nn.MSELoss()(ae(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        layer = ae.encoder[0]
        weights.append((layer.weight.detach().clone(), layer.bias.detach().clone()))

        with torch.no_grad():
            out = ae.encoder(torch.tensor(out, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    return weights

def make_net(n_in, n_out, weights=None):
    model = Net(n_in, n_out).to(DEVICE)

    if weights:
        layers = [x for x in model.layers if isinstance(x, nn.Linear)]
        for layer, (w, b) in zip(layers[:-1], weights):
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)

    return model

def train(model, X, y, task, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss() if task == "reg" else nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in batches(X, y):
            pred = model(xb)
            if task == "reg":
                yb = yb.float().view(-1, 1)
            else:
                yb = yb.long()
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

def train_pretrained(model, X, y, task):
    layers = [x for x in model.layers if isinstance(x, nn.Linear)]

    for p in model.parameters():
        p.requires_grad = False
    for p in layers[-1].parameters():
        p.requires_grad = True

    train(model, X, y, task, EPOCHS_HEAD)

    for p in model.parameters():
        p.requires_grad = True

    train(model, X, y, task, EPOCHS)

def predict(model, X):
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
    train(model, Xtr, ytr, "reg", EPOCHS)
    p1 = predict(model, Xte).ravel()

    w = pretrain(Xtr, [9, 32, 16, 8])
    model = make_net(9, 1, w)
    train_pretrained(model, Xtr, ytr, "reg")
    p2 = predict(model, Xte).ravel()

    for name, p in [("Без предобучения", p1), ("С предобучением", p2)]:
        mask = yte > 0
        mape = np.mean(np.abs((yte[mask] - p[mask]) / yte[mask])) * 100
        print("\n" + name + ":")
        print("MAE:", round(mean_absolute_error(yte, p), 4))
        print("RMSE:", round(mean_squared_error(yte, p) ** 0.5, 4))
        print("R2:", round(r2_score(yte, p), 4))
        print("MAPE без нулей:", round(mape, 2), "%")

def run_classification():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "breast-cancer-wisconsin/wdbc.data", header=None
    )

    X = data.iloc[:, 2:].values.astype(float)
    y = np.where(data.iloc[:, 1].values == "M", 1, 0)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

    print("\nWDBC: классификация")
    print("Размер:", X.shape)
    print("Архитектура: 30-32-16-8-2")

    model = make_net(30, 2)
    train(model, Xtr, ytr, "cls", EPOCHS)
    p1 = predict(model, Xte).argmax(1)

    w = pretrain(Xtr, [30, 32, 16, 8])
    model = make_net(30, 2, w)
    train_pretrained(model, Xtr, ytr, "cls")
    p2 = predict(model, Xte).argmax(1)

    for name, p in [("Без предобучения", p1), ("С предобучением", p2)]:
        print("\n" + name + ":")
        print("F1:", round(f1_score(yte, p), 4))
        print("Accuracy:", round(accuracy_score(yte, p), 4))
        print("Матрица ошибок:")
        print(confusion_matrix(yte, p))

print("Устройство:", DEVICE)
run_regression()
run_classification()