import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, f1_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import ucimlrepo

CFG = {
    "seed": 42,
    "epochs_pre": 50,
    "epochs_head": 20,
    "epochs_full": 100,
    "batch_size": 64,
    "lr": 5e-4,
    "hidden": (64, 32, 16),
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
DEVICE = "cpu"

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class AEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        return self.decoder(self.encoder(x))

def get_batches(X, y=None):
    order = torch.randperm(len(X))
    for start in range(0, len(X), CFG["batch_size"]):
        idx = order[start:start + CFG["batch_size"]]
        x_batch = torch.tensor(X[idx], dtype=torch.float32).to(DEVICE)
        if y is None:
            yield x_batch
        else:
            yield x_batch, torch.tensor(y[idx]).to(DEVICE)

def pretrain(X, layer_sizes):
    activations = X.copy()
    learned = []
    n_layers = len(layer_sizes) - 1

    for i in range(n_layers):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        autoenc = AEModel(in_dim, out_dim).to(DEVICE)
        optimizer = torch.optim.Adam(autoenc.parameters(), lr=CFG["lr"])

        for _ in range(CFG["epochs_pre"]):
            for x_batch in get_batches(activations):
                optimizer.zero_grad()
                reconstruction = autoenc(x_batch)
                loss = nn.MSELoss()(reconstruction, x_batch)
                loss.backward()
                optimizer.step()

        enc_layer = autoenc.encoder[0]
        learned.append((
            enc_layer.weight.detach().clone(),
            enc_layer.bias.detach().clone(),
        ))
        with torch.no_grad():
            x_tensor = torch.tensor(activations, dtype=torch.float32).to(DEVICE)
            activations = autoenc.encoder(x_tensor).cpu().numpy()
    return learned

def make_net(input_dim, output_dim, weights=None):
    model = MLPModel(input_dim, output_dim).to(DEVICE)
    if weights is None:
        return model

    linear_layers = [m for m in model.layers if isinstance(m, nn.Linear)]
    for layer, (w, b) in zip(linear_layers[:-1], weights):
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(b)
    return model

def train(model, X, y, task, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in get_batches(X, y):
            output = model(x_batch)

            if task == "reg":
                target = y_batch.float().view(-1, 1)
                loss = nn.MSELoss()(output, target)
            else:
                target = y_batch.long()
                loss = nn.CrossEntropyLoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history.append(epoch_loss / n_batches)
    return history

def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value

def train_pretrained(model, X, y, task):
    linear_layers = [m for m in model.layers if isinstance(m, nn.Linear)]
    head = linear_layers[-1]
    set_requires_grad(model, False)
    set_requires_grad(head, True)
    hist_head = train(model, X, y, task, CFG["epochs_head"])
    set_requires_grad(model, True)
    hist_full = train(model, X, y, task, CFG["epochs_full"])
    return hist_head + hist_full

def predict(model, X):
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(x).cpu().numpy()

def plot_loss(history_base, history_pre, title, fname):
    plt.figure(figsize=(10, 5))
    plt.plot(history_base, label="Без предобучения", linewidth=2)
    plt.plot(history_pre, label="С предобучением", linewidth=2)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def report_classification(yte, pred, title, labels=None):
    print(f"{title}:")
    print(f"Accuracy: {accuracy_score(yte, pred):.4f}")
    print(f"F1 (macro): {f1_score(yte, pred, average='macro', zero_division=0):.4f}")
    print(f"F1 (weight): {f1_score(yte, pred, average='weighted', zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(yte, pred))
    print()

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        yte, pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )
    ax.set_title(f"Confusion matrix - {title}")
    plt.tight_layout()
    fname = f"confusion_{title.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

def report_regression(yte, pred, title):
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)
    mask = yte != 0
    mape = (np.mean(np.abs((yte[mask] - pred[mask]) / yte[mask])) * 100
            if mask.any() else float("nan"))
    print(f"{title}:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.2f} %")
    print()

def run_quality():
    print("Датасет Wine Quality (white)")
    print("Задача: классификация")

    path = "winequality-white.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Нет файла winequality-white.csv.")

    df = pd.read_csv(path, sep=";")
    df = df.fillna(df.median(numeric_only=True))

    y_raw = df["quality"].values
    X = df.drop(columns=["quality"]).values.astype(np.float32)

    classes = np.unique(y_raw)
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in y_raw], dtype=np.int64)
    n_classes = len(classes)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=CFG["seed"], stratify=y
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    input_dim = Xtr.shape[1]
    model_base = make_net(input_dim, n_classes)
    hist_base = train(model_base, Xtr, ytr, "cls", CFG["epochs_full"])
    pred_base = predict(model_base, Xte).argmax(1)

    weights = pretrain(Xtr, [input_dim] + list(CFG["hidden"]))
    model_pre = make_net(input_dim, n_classes, weights)
    hist_pre = train_pretrained(model_pre, Xtr, ytr, "cls")
    pred_pre = predict(model_pre, Xte).argmax(1)

    print("\nРезультаты:")
    class_labels = [f"quality={c}" for c in classes]
    report_classification(yte, pred_base, "Без предобучения", labels=class_labels)
    report_classification(yte, pred_pre, "С предобучением", labels=class_labels)

    plot_loss(hist_base, hist_pre,
              "Wine Quality - Loss по эпохам",
              "wine_quality_loss.png")

def run_thermo():
    print("Датасет Infrared Thermography")
    print("Задача: регрессия")

    path = "IRThermography_Temperature.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        ds = ucimlrepo.fetch_ucirepo(id=925)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.to_csv(path, index=False)

    TARGET = "aveOralF"
    if TARGET not in df.columns:
        raise ValueError(f"Нет колонки {TARGET}. Есть: {list(df.columns)}")

    leak_cols = [c for c in df.columns
                 if c.lower() in ("subjectid", "id", "aveoralm")]
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])

    num_all = df.select_dtypes(include=[np.number]).columns
    df[num_all] = df[num_all].fillna(df[num_all].median())
    cat_all = df.select_dtypes(include=["object", "str"]).columns
    for c in cat_all:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    y = df[TARGET].values.astype(np.float32)
    X = df.drop(columns=[TARGET])

    cat_cols = [c for c in X.columns
                if X[c].dtype == object
                or c in ("Gender", "Age", "Ethnicity")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=CFG["seed"]
    )
    Xtr = preprocessor.fit_transform(Xtr).astype(np.float32)
    Xte = preprocessor.transform(Xte).astype(np.float32)

    target_scaler = StandardScaler()
    ytr_scaled = target_scaler.fit_transform(
        ytr.reshape(-1, 1)
    ).ravel().astype(np.float32)

    input_dim = Xtr.shape[1]
    model_base = make_net(input_dim, 1)
    hist_base = train(model_base, Xtr, ytr_scaled, "reg", CFG["epochs_full"])
    pred_base = predict(model_base, Xte).ravel()
    pred_base = target_scaler.inverse_transform(
        pred_base.reshape(-1, 1)
    ).ravel()

    weights = pretrain(Xtr, [input_dim] + list(CFG["hidden"]))
    model_pre = make_net(input_dim, 1, weights)
    hist_pre = train_pretrained(model_pre, Xtr, ytr_scaled, "reg")
    pred_pre = predict(model_pre, Xte).ravel()
    pred_pre = target_scaler.inverse_transform(
        pred_pre.reshape(-1, 1)
    ).ravel()

    print("\nРезультаты:")
    report_regression(yte, pred_base, "Без предобучения")
    report_regression(yte, pred_pre, "С предобучением")

    plot_loss(hist_base, hist_pre,
              "Infrared Thermography - Loss по эпохам",
              "thermo_loss.png")

if __name__ == "__main__":
    run_quality()
    run_thermo()