import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

CONFIG = {
    "seed": 42,
    "epochs_stage1": 50,
    "epochs_stage2": 20,
    "epochs_stage3": 150,
    "batch_size": 64,
    "lr": 5e-4,
    "rbm_lr": 0.005,
    "rbm_mom_start": 0.5,
    "rbm_mom_final": 0.9,
    "rbm_mom_switch": 5,
    "hidden_sizes": (32, 16, 8),
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = "cpu"

class DenseNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        layers = []
        prev = n_in
        for h in CONFIG["hidden_sizes"]:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, n_hid), nn.ReLU())
        self.dec = nn.Linear(n_hid, n_in)

    def forward(self, x):
        return self.dec(self.enc(x))

class RestrictedBoltzmann:
    def __init__(self, n_vis, n_hid, gaussian_vis=False, seed=42):
        rng = np.random.RandomState(seed)
        self.W = torch.tensor(
            (0.01 * rng.normal(size=(n_vis, n_hid))).astype(np.float32),
            device=DEVICE
        )
        self.bv = torch.zeros(n_vis, device=DEVICE)
        self.bh = torch.zeros(n_hid, device=DEVICE)
        self.is_gaussian = gaussian_vis

    def prob_h(self, v):
        return torch.sigmoid(v @ self.W + self.bh)

    def prob_v(self, h):
        z = h @ self.W.T + self.bv
        return z if self.is_gaussian else torch.sigmoid(z)

    @torch.no_grad()
    def fit(self, X_data):
        epochs = CONFIG["epochs_stage1"]
        lr = CONFIG["rbm_lr"]
        bs = CONFIG["batch_size"]
        mom_s = CONFIG["rbm_mom_start"]
        mom_f = CONFIG["rbm_mom_final"]
        mom_switch = CONFIG["rbm_mom_switch"]

        dW = torch.zeros_like(self.W)
        dbv = torch.zeros_like(self.bv)
        dbh = torch.zeros_like(self.bh)
        
        X_t = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
        n = X_t.shape[0]
        last_err = 0.0

        for epoch in range(epochs):
            mom = mom_s if epoch < mom_switch else mom_f
            order = torch.randperm(n)
            err_sum, n_batches = 0.0, 0

            for start in range(0, n, bs):
                idx = order[start:start + bs]
                v0 = X_t[idx]
                m = v0.shape[0]

                ph0 = self.prob_h(v0)
                h0 = torch.bernoulli(ph0)

                pv1 = self.prob_v(h0)
                v1 = pv1 if self.is_gaussian else torch.bernoulli(pv1)
                ph1 = self.prob_h(v1)

                dW = mom * dW + lr * (v0.T @ h0 - v1.T @ ph1) / m
                dbv = mom * dbv + lr * (v0 - v1).mean(0)
                dbh = mom * dbh + lr * (h0 - ph1).mean(0)

                self.W += dW
                self.bv += dbv
                self.bh += dbh

                err_sum += ((v0 - pv1) ** 2).mean().item()
                n_batches += 1
            last_err = err_sum / max(n_batches, 1)

        return last_err

def batch_iterator(X, y=None):
    order = torch.randperm(len(X))
    for start in range(0, len(X), CONFIG["batch_size"]):
        idx = order[start:start + CONFIG["batch_size"]]
        x_b = torch.tensor(X[idx], dtype=torch.float32).to(DEVICE)
        if y is None:
            yield x_b
        else:
            yield x_b, torch.tensor(y[idx]).to(DEVICE)

def pre_train_rbm(X, layer_sizes):
    act = X.copy()
    weights = []
    for i in range(len(layer_sizes) - 1):
        rbm = RestrictedBoltzmann(
            layer_sizes[i], layer_sizes[i+1], 
            gaussian_vis=(i == 0), 
            seed=CONFIG["seed"] + i
        )
        err = rbm.fit(act)
        print(f"  RBM {layer_sizes[i]}-{layer_sizes[i+1]}: recon error = {err:.4f}")
        
        weights.append((rbm.W.T.clone(), rbm.bh.clone()))
        
        with torch.no_grad():
            x_t = torch.tensor(act, dtype=torch.float32).to(DEVICE)
            act = rbm.prob_h(x_t).cpu().numpy()
    return weights

def pre_train_ae(X, layer_sizes):
    act = X.copy()
    weights = []
    for i in range(len(layer_sizes) - 1):
        ae = DenoisingAutoEncoder(layer_sizes[i], layer_sizes[i+1]).to(DEVICE)
        opt = torch.optim.Adam(ae.parameters(), lr=CONFIG["lr"])
        
        for _ in range(CONFIG["epochs_stage1"]):
            for x_b in batch_iterator(act):
                opt.zero_grad()
                loss = nn.MSELoss()(ae(x_b), x_b)
                loss.backward()
                opt.step()

        enc_layer = ae.enc[0]
        weights.append((enc_layer.weight.detach().clone(),
                        enc_layer.bias.detach().clone()))
        
        with torch.no_grad():
            x_t = torch.tensor(act, dtype=torch.float32).to(DEVICE)
            act = ae.enc(x_t).cpu().numpy()
    return weights

def build_model(n_in, n_out, initial_weights=None):
    model = DenseNet(n_in, n_out).to(DEVICE)
    if initial_weights is None:
        return model

    linear_layers = [m for m in model.body if isinstance(m, nn.Linear)]
    for layer, (w, b) in zip(linear_layers[:-1], initial_weights):
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(b)
    return model

def fit_model(model, X, y, task, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    history = []
    
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_b, y_b in batch_iterator(X, y):
            output = model(x_b)
            if task == "reg":
                target = y_b.float().view(-1, 1)
                loss = nn.MSELoss()(output, target)
            else:
                target = y_b.long()
                loss = nn.CrossEntropyLoss()(output, target)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history.append(epoch_loss / n_batches)
    return history

def fit_with_pretrain(model, X, y, task):
    linear_layers = [m for m in model.body if isinstance(m, nn.Linear)]
    head = linear_layers[-1]
    
    for p in model.parameters(): p.requires_grad = False
    for p in head.parameters(): p.requires_grad = True
    
    h1 = fit_model(model, X, y, task, CONFIG["epochs_stage2"])
    
    for p in model.parameters(): p.requires_grad = True
    h2 = fit_model(model, X, y, task, CONFIG["epochs_stage3"])
    
    return h1 + h2

def get_predictions(model, X):
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(x_t).cpu().numpy()

def show_metrics(y_true, y_pred, title, class_names, prefix):
    print(f"\n{title}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 (macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 (weighted): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("Матрица ошибок:")
    print(confusion_matrix(y_true, y_pred))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names,
        cmap="Blues", ax=ax, colorbar=True
    )
    ax.set_title(f"Confusion matrix - {title}")
    plt.tight_layout()
    fname = f"confusion_{prefix}_{title.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_learning_curves(histories, labels, title, fname):
    plt.figure(figsize=(10, 5))
    for h, lbl in zip(histories, labels):
        plt.plot(h, label=lbl, linewidth=2)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def load_hcv():
    path = "HCV-Egy-Data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет файла {path} рядом со скриптом.")

    df = pd.read_csv(path)
    target_col = [c for c in df.columns if "staging" in c.lower()][0]
    
    df = df.fillna(df.median(numeric_only=True))
    y_raw = df[target_col].values
    X = df.drop(columns=[target_col]).values.astype(np.float32)

    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw).astype(np.int64)
    class_labels = [f"stage {c}" for c in classes]
    return X, y, class_labels

def load_mushroom():
    path = "agaricus-lepiota.data"
    cols = ["poisonous", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
            "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring",
            "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
            "ring-type", "spore-print-color", "population", "habitat"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет файла {path} рядом со скриптом.")
        
    df = pd.read_csv(path, header=None, names=cols)
    df = df.fillna("?")
    y_raw = df["poisonous"].values
    X = pd.get_dummies(df.drop(columns=["poisonous"])).to_numpy(dtype=np.float32)

    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw).astype(np.int64)
    class_labels = [f"poisonous={c}" for c in classes]
    return X, y, class_labels

def run_experiment(name, prefix, X, y, class_labels):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG["seed"], stratify=y
    )
    
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    in_dim = Xtr.shape[1]
    n_classes = len(class_labels)
    layer_sizes = [in_dim] + list(CONFIG["hidden_sizes"])

    print(f"\nДатасет: {name}")
    print(f"Размер: {X.shape}")
    print(f"Архитектура: {'-'.join(map(str, layer_sizes))}-{n_classes}")
    print(f"Классы: {class_labels}")

    print("\nМодель 1: Без предобучения")
    model_base = build_model(in_dim, n_classes)
    hist_base = fit_model(model_base, Xtr, ytr, "cls", CONFIG["epochs_stage3"])
    pred_base = get_predictions(model_base, Xte).argmax(1)

    print("Модель 2: Предобучение AE")
    weights_ae = pre_train_ae(Xtr, layer_sizes)
    model_ae = build_model(in_dim, n_classes, weights_ae)
    hist_ae = fit_with_pretrain(model_ae, Xtr, ytr, "cls")
    pred_ae = get_predictions(model_ae, Xte).argmax(1)

    print("Модель 3: Предобучение RBM")
    weights_rbm = pre_train_rbm(Xtr, layer_sizes)
    model_rbm = build_model(in_dim, n_classes, weights_rbm)
    hist_rbm = fit_with_pretrain(model_rbm, Xtr, ytr, "cls")
    pred_rbm = get_predictions(model_rbm, Xte).argmax(1)

    print("\n" + "="*40)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*40)
    
    show_metrics(yte, pred_base, "Без предобучения", class_labels, prefix)
    show_metrics(yte, pred_ae, "С предобучением AE", class_labels, prefix)
    show_metrics(yte, pred_rbm, "С предобучением RBM", class_labels, prefix)

    plot_learning_curves(
        [hist_base, hist_ae, hist_rbm],
        ["Без предобучения", "AE pretrain", "RBM pretrain"],
        f"{name} - Loss по эпохам",
        f"{prefix}_loss.png"
    )

if __name__ == "__main__":
    X_m, y_m, labels_m = load_mushroom()
    run_experiment("Mushroom", "mushroom", X_m, y_m, labels_m)

    X_h, y_h, labels_h = load_hcv()
    run_experiment("HCV Egyptian patients", "hcv", X_h, y_h, labels_h)
    
    print("\nВсе графики успешно сохранены в папку с программой.")