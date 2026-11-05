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
BATCH, LR, RBM_LR = 128, 0.001, 0.01
MOM_START, MOM_FINAL, MOM_SWITCH = 0.5, 0.9, 5 

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


class RBM:
    def __init__(self, n_in, n_hidden, gaussian_visible):
        self.W = 0.01 * torch.randn(n_in, n_hidden, device=DEVICE)
        self.vb = torch.zeros(n_in, device=DEVICE)       # T_i
        self.hb = torch.zeros(n_hidden, device=DEVICE)   # T_j
        self.gauss = gaussian_visible

    def hidden(self, v):
        return torch.sigmoid(v @ self.W + self.hb)

    def visible(self, h):
        z = h @ self.W.T + self.vb
        return z if self.gauss else torch.sigmoid(z)

    @torch.no_grad()
    def train(self, X):
        dW = torch.zeros_like(self.W)
        dvb = torch.zeros_like(self.vb)
        dhb = torch.zeros_like(self.hb)
        err = 0.0

        for epoch in range(EPOCHS_PRE):
            mom = MOM_START if epoch < MOM_SWITCH else MOM_FINAL
            err, nb = 0.0, 0

            for v0 in batches(X):
                n = len(v0)

                p_h0 = self.hidden(v0)
                h0 = torch.bernoulli(p_h0)

                p_v1 = self.visible(h0)
                v1 = p_v1 if self.gauss else torch.bernoulli(p_v1)

                p_h1 = self.hidden(v1)

                dW = mom * dW + RBM_LR * (v0.T @ h0 - v1.T @ p_h1) / n
                dvb = mom * dvb + RBM_LR * (v0 - v1).mean(0)
                dhb = mom * dhb + RBM_LR * (h0 - p_h1).mean(0)

                self.W += dW
                self.vb += dvb
                self.hb += dhb

                err += ((v0 - p_v1) ** 2).mean().item()
                nb += 1

            err /= nb

        return err

def batches(X, y=None):
    ind = torch.randperm(len(X))

    for i in range(0, len(X), BATCH):
        j = ind[i:i+BATCH]
        xb = torch.tensor(X[j], dtype=torch.float32).to(DEVICE)

        if y is None:
            yield xb
        else:
            yield xb, torch.tensor(y[j]).to(DEVICE)

def pretrain_ae(X, sizes):
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
        weights.append((layer.weight.detach().clone(),
                        layer.bias.detach().clone()))

        with torch.no_grad():
            out = ae.encoder(
                torch.tensor(out, dtype=torch.float32).to(DEVICE)
            ).cpu().numpy()

    return weights

def pretrain_rbm(X, sizes):
    out, weights = X.copy(), []

    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        rbm = RBM(a, b, gaussian_visible=(k == 0))
        err = rbm.train(out)
        print(f"  RBM {a}-{b}: ошибка реконструкции = {err:.4f}")

        weights.append((rbm.W.T.clone(), rbm.hb.clone()))

        with torch.no_grad():
            out = rbm.hidden(
                torch.tensor(out, dtype=torch.float32).to(DEVICE)
            ).cpu().numpy()

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

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

    print("\nCASP: регрессия")
    print("Размер:", X.shape)
    print("Архитектура: 9-32-16-8-1")

    results = []

    model = make_net(9, 1)
    train(model, Xtr, ytr, "reg", EPOCHS)
    results.append(("Без предобучения", predict(model, Xte).ravel()))

    w = pretrain_ae(Xtr, [9, 32, 16, 8])
    model = make_net(9, 1, w)
    train_pretrained(model, Xtr, ytr, "reg")
    results.append(("AE", predict(model, Xte).ravel()))

    print("Предобучение RBM:")
    w = pretrain_rbm(Xtr, [9, 32, 16, 8])
    model = make_net(9, 1, w)
    train_pretrained(model, Xtr, ytr, "reg")
    results.append(("RBM", predict(model, Xte).ravel()))

    for name, p in results:
        mask = yte > 0
        mape = np.mean(np.abs((yte[mask] - p[mask]) / yte[mask])) * 100

        print("\n" + name + ":")
        print("MAE:", round(mean_absolute_error(yte, p), 4))
        print("RMSE:", round(mean_squared_error(yte, p) ** 0.5, 4))
        print("R2:", round(r2_score(yte, p), 4))
        print("MAPE (RMSD > 0):", round(mape, 2), "%")

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

    results = []

    model = make_net(30, 2)
    train(model, Xtr, ytr, "cls", EPOCHS)
    results.append(("Без предобучения", predict(model, Xte).argmax(1)))

    w = pretrain_ae(Xtr, [30, 32, 16, 8])
    model = make_net(30, 2, w)
    train_pretrained(model, Xtr, ytr, "cls")
    results.append(("AE", predict(model, Xte).argmax(1)))

    print("Предобучение RBM:")
    w = pretrain_rbm(Xtr, [30, 32, 16, 8])
    model = make_net(30, 2, w)
    train_pretrained(model, Xtr, ytr, "cls")
    results.append(("RBM", predict(model, Xte).argmax(1)))

    for name, p in results:
        print("\n" + name + ":")
        print("F1:", round(f1_score(yte, p), 4))
        print("Accuracy:", round(accuracy_score(yte, p), 4))
        print("Матрица ошибок:")
        print(confusion_matrix(yte, p))

print("Устройство:", DEVICE)
run_regression()
run_classification()