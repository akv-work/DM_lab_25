import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_percentage_error, mean_squared_error,
                             r2_score, f1_score, accuracy_score, confusion_matrix)

PARKINSONS_PATH = r"D:\Учёба\4 курс\ИАД\4\parkinsons_updrs.data"
MUSHROOM_PATH   = r"D:\Учёба\4 курс\ИАД\2\agaricus-lepiota.data"
OUT_DIR         = r"D:\Учёба\4 курс\ИАД\4"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
SEEDS = [42, 1, 2, 5, 7]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN = [64, 32, 16]
LR = 1e-3
BATCH = 128
EPOCHS_BASELINE = 300
EPOCHS_HEAD = 20
EPOCHS_FINETUNE = 300

EPOCHS_RBM = 50
RBM_LR = 0.01
MOM_START, MOM_FINAL, MOM_SWITCH = 0.5, 0.9, 5
RBM_INIT_STD = 0.01

EPOCHS_AE = 100
AE_LR = 1e-3

USE_SIGMOID_FOR_RBM = False


class Net(nn.Module):
    def __init__(self, n_in, hidden, n_out, act="relu"):
        super().__init__()
        act_layer = nn.ReLU if act == "relu" else nn.Sigmoid
        sizes = [n_in] + hidden
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), act_layer()]
        layers += [nn.Linear(sizes[-1], n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU())
        self.decoder = nn.Linear(n_hidden, n_in)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class RBM:
    def __init__(self, n_vis, n_hid, gaussian_visible):
        self.W = RBM_INIT_STD * torch.randn(n_vis, n_hid, device=DEVICE)
        self.vb = torch.zeros(n_vis, device=DEVICE)
        self.hb = torch.zeros(n_hid, device=DEVICE)
        self.gauss = gaussian_visible

    def hidden(self, v):
        return torch.sigmoid(v @ self.W + self.hb).clamp(1e-6, 1 - 1e-6)

    def visible(self, h):
        z = h @ self.W.T + self.vb
        return z if self.gauss else torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)

    @torch.no_grad()
    def train(self, X):
        dW = torch.zeros_like(self.W)
        dvb = torch.zeros_like(self.vb)
        dhb = torch.zeros_like(self.hb)
        err = 0.0
        for epoch in range(EPOCHS_RBM):
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


def batches(X, y=None, batch_size=BATCH):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), batch_size):
        j = idx[i:i + batch_size]
        if y is None:
            yield X[j]
        else:
            yield X[j], y[j]


def pretrain_ae(X, sizes):
    out = X.clone()
    weights = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        ae = Autoencoder(a, b).to(DEVICE)
        opt = torch.optim.Adam(ae.parameters(), lr=AE_LR)
        for _ in range(EPOCHS_AE):
            for xb in batches(out):
                loss = nn.MSELoss()(ae(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        layer = ae.encoder[0]
        weights.append((layer.weight.detach().clone(),
                        layer.bias.detach().clone()))
        with torch.no_grad():
            out = ae.encoder(out)
    return weights


def pretrain_rbm(X, sizes, gaussian_first=True):
    out = X.clone()
    weights = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        gauss = gaussian_first and k == 0
        rbm = RBM(a, b, gaussian_visible=gauss)
        err = rbm.train(out)
        print(f"    RBM {a}-{b}: recon MSE = {err:.4f}")
        weights.append((rbm.W.T.clone(), rbm.hb.clone()))
        with torch.no_grad():
            out = rbm.hidden(out)
    return weights


def make_net(n_in, hidden, n_out, weights=None, act="relu"):
    model = Net(n_in, hidden, n_out, act=act).to(DEVICE)
    if weights:
        linears = [m for m in model.net if isinstance(m, nn.Linear)]
        for layer, (w, b) in zip(linears[:-1], weights):
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
    return model


def train_supervised(model, Xtr, ytr, task, epochs, head_only=False):
    if head_only:
        for p in model.parameters():
            p.requires_grad = False
        last_linear = [m for m in model.net if isinstance(m, nn.Linear)][-1]
        for p in last_linear.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    loss_fn = nn.MSELoss() if task == "reg" else nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in batches(Xtr, ytr):
            pred = model(xb)
            if task == "reg":
                yb = yb.float().view(-1, 1)
            else:
                yb = yb.long()
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for p in model.parameters():
        p.requires_grad = True


def run_regression_parkinsons(seed, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    cols = ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
            'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
            'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    df = pd.read_csv(PARKINSONS_PATH, names=cols, skiprows=1).astype(float)
    X = df.drop(columns=['subject#', 'motor_UPDRS', 'total_UPDRS']).values
    y = df['motor_UPDRS'].values.reshape(-1, 1)
    groups = df['subject#'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, y, groups=groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    scx, scy = StandardScaler(), StandardScaler()
    Xtr = scx.fit_transform(Xtr); Xte = scx.transform(Xte)
    ytr = scy.fit_transform(ytr); yte = scy.transform(yte)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(DEVICE)
    yte_t = torch.tensor(yte, dtype=torch.float32).to(DEVICE)

    n_in = Xtr.shape[1]
    results = []

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).cpu().numpy()
        pred = scy.inverse_transform(pred)
        true = scy.inverse_transform(yte_t.cpu().numpy())
        return (mean_absolute_percentage_error(true, pred),
                float(np.sqrt(mean_squared_error(true, pred))),
                float(r2_score(true, pred)))

    torch.manual_seed(seed); np.random.seed(seed)
    m = make_net(n_in, HIDDEN, 1)
    train_supervised(m, Xtr_t, ytr_t, "reg", EPOCHS_BASELINE)
    results.append(("Без предобучения",) + evaluate(m))

    torch.manual_seed(seed); np.random.seed(seed)
    if verbose: print("  Предобучение AE:")
    w = pretrain_ae(Xtr_t, [n_in] + HIDDEN)
    m = make_net(n_in, HIDDEN, 1, w)
    train_supervised(m, Xtr_t, ytr_t, "reg", EPOCHS_HEAD, head_only=True)
    train_supervised(m, Xtr_t, ytr_t, "reg", EPOCHS_FINETUNE)
    results.append(("Предобучение AE",) + evaluate(m))

    torch.manual_seed(seed); np.random.seed(seed)
    if verbose: print("  Предобучение RBM:")
    w = pretrain_rbm(Xtr_t, [n_in] + HIDDEN, gaussian_first=True)
    act = "sigmoid" if USE_SIGMOID_FOR_RBM else "relu"
    m = make_net(n_in, HIDDEN, 1, w, act=act)
    train_supervised(m, Xtr_t, ytr_t, "reg", EPOCHS_HEAD, head_only=True)
    train_supervised(m, Xtr_t, ytr_t, "reg", EPOCHS_FINETUNE)
    results.append(("Предобучение RBM",) + evaluate(m))

    return results


def load_mushroom(path):
    first = open(path, encoding="utf-8").readline().strip()
    if first.lower().startswith("poisonous") or "cap-shape" in first.lower():
        df = pd.read_csv(path)
        y = (df["poisonous"] == "p").astype(int).values
        X_cat = df.drop(columns=["poisonous"]).astype(str)
    else:
        df = pd.read_csv(path, header=None)
        y = (df.iloc[:, 0] == "p").astype(int).values
        X_cat = df.iloc[:, 1:].astype(str)
    X = pd.get_dummies(X_cat).values.astype(np.float32)
    return X, y


def run_classification_mushroom(seed, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    X, y = load_mushroom(MUSHROOM_PATH)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long).to(DEVICE)
    yte_t = torch.tensor(yte, dtype=torch.long).to(DEVICE)

    n_in = Xtr.shape[1]
    results = []

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).argmax(1).cpu().numpy()
        return (f1_score(yte, pred),
                accuracy_score(yte, pred),
                confusion_matrix(yte, pred))

    torch.manual_seed(seed); np.random.seed(seed)
    m = make_net(n_in, HIDDEN, 2)
    train_supervised(m, Xtr_t, ytr_t, "cls", EPOCHS_BASELINE)
    results.append(("Без предобучения",) + evaluate(m))

    torch.manual_seed(seed); np.random.seed(seed)
    if verbose: print("  Предобучение AE:")
    w = pretrain_ae(Xtr_t, [n_in] + HIDDEN)
    m = make_net(n_in, HIDDEN, 2, w)
    train_supervised(m, Xtr_t, ytr_t, "cls", EPOCHS_HEAD, head_only=True)
    train_supervised(m, Xtr_t, ytr_t, "cls", EPOCHS_FINETUNE)
    results.append(("Предобучение AE",) + evaluate(m))

    torch.manual_seed(seed); np.random.seed(seed)
    if verbose: print("  Предобучение RBM:")
    w = pretrain_rbm(Xtr_t, [n_in] + HIDDEN, gaussian_first=False)
    act = "sigmoid" if USE_SIGMOID_FOR_RBM else "relu"
    m = make_net(n_in, HIDDEN, 2, w, act=act)
    train_supervised(m, Xtr_t, ytr_t, "cls", EPOCHS_HEAD, head_only=True)
    train_supervised(m, Xtr_t, ytr_t, "cls", EPOCHS_FINETUNE)
    results.append(("Предобучение RBM",) + evaluate(m))

    return results


def aggregate(results_by_seed, task):
    names = [r[0] for r in results_by_seed[0]]
    out = []
    for i, name in enumerate(names):
        if task == "reg":
            vals = np.array([[r[i][1], r[i][2], r[i][3]] for r in results_by_seed])
            mean = vals.mean(0); std = vals.std(0)
            out.append((name, mean, std))
        else:
            f1s = np.array([r[i][1] for r in results_by_seed])
            accs = np.array([r[i][2] for r in results_by_seed])
            cms = [r[i][3] for r in results_by_seed]
            cm_mean = np.mean(cms, axis=0).round(1)
            out.append((name, f1s.mean(), f1s.std(), accs.mean(), accs.std(), cm_mean))
    return out


def print_regression(title, agg):
    print(f"\n{title} (среднее ± std по {len(SEEDS)} сидам)")
    print(f"{'Метод':<22}{'MAPE':>16}{'RMSE':>16}{'R2':>16}")
    for name, m, s in agg:
        print(f"{name:<22}"
              f"{m[0]:>8.4f}±{s[0]:.4f}"
              f"{m[1]:>8.4f}±{s[1]:.4f}"
              f"{m[2]:>8.4f}±{s[2]:.4f}")


def print_classification(title, agg):
    print(f"\n{title} (среднее ± std по {len(SEEDS)} сидам)")
    print(f"{'Метод':<22}{'F1':>16}{'Accuracy':>16}")
    for name, f1m, f1s, accm, accs, cm in agg:
        print(f"{name:<22}"
              f"{f1m:>8.4f}±{f1s:.4f}"
              f"{accm:>8.4f}±{accs:.4f}")
        print(f"  confusion (mean):\n{cm}")


def save_regression(path, title, agg):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"{'Метод':<22}{'MAPE':>18}{'RMSE':>18}{'R2':>18}\n")
        for name, m, s in agg:
            f.write(f"{name:<22}"
                    f"{m[0]:>8.4f}±{s[0]:.4f}  "
                    f"{m[1]:>8.4f}±{s[1]:.4f}  "
                    f"{m[2]:>8.4f}±{s[2]:.4f}\n")


def save_classification(path, title, agg):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"{'Метод':<22}{'F1':>18}{'Accuracy':>18}\n")
        for name, f1m, f1s, accm, accs, cm in agg:
            f.write(f"{name:<22}"
                    f"{f1m:>8.4f}±{f1s:.4f}  "
                    f"{accm:>8.4f}±{accs:.4f}\n")
            f.write(f"  confusion (mean):\n{cm}\n")


def main():
    print("Устройство:", DEVICE)
    print("Сиды:", SEEDS)

    print("\nPARKINSONS (регрессия motor_UPDRS, данные ЛР3)")
    park_res = []
    for s in SEEDS:
        print(f"\nseed = {s}")
        park_res.append(run_regression_parkinsons(s))
    park_agg = aggregate(park_res, "reg")
    print_regression("PARKINSONS", park_agg)
    save_regression(os.path.join(OUT_DIR, "metrics_lab4_parkinsons.txt"),
                    "PARKINSONS (GroupShuffleSplit по subject#)", park_agg)

    print("\nMUSHROOM (классификация poisonous, данные ЛР2)")
    mush_res = []
    for s in SEEDS:
        print(f"\nseed = {s}")
        mush_res.append(run_classification_mushroom(s))
    mush_agg = aggregate(mush_res, "cls")
    print_classification("MUSHROOM", mush_agg)
    save_classification(os.path.join(OUT_DIR, "metrics_lab4_mushroom.txt"),
                        "MUSHROOM (стратифицированный split)", mush_agg)

    print("\nГотово.")
    print(f"  {os.path.join(OUT_DIR, 'metrics_lab4_parkinsons.txt')}")
    print(f"  {os.path.join(OUT_DIR, 'metrics_lab4_mushroom.txt')}")


if __name__ == "__main__":
    main()