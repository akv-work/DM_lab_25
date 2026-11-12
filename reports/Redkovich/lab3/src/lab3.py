import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from ucimlrepo import fetch_ucirepo

# Настройка устройства (GPU, если доступно)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")

HIDDEN = [64, 32, 16]
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PRETRAIN_EPOCHS = 50
OUTPUT_LAYER_EPOCHS = 30
SEEDS = [0, 1, 2, 3, 4]


def load_raisin():
    """1. Загрузка датасета из ЛР3 (Raisin - Вариант 9) через ucimlrepo"""
    raisin = fetch_ucirepo(id=850)
    X = raisin.data.features
    y_raw = raisin.data.targets
    
    target_col = y_raw.columns[0]
    classes = y_raw[target_col].unique()
    y = (y_raw[target_col] == classes[0]).astype(int).values
    
    X = X.fillna(X.median())
    return X.values.astype(float), y


def load_wine_quality_white():
    """2. Загрузка датасета из ЛР2 (Wine Quality white - Вариант 9)"""
    try:
        # Пробуем загрузить через официальный репозиторий UCI (ID 186 - Wine Quality)
        wine = fetch_ucirepo(id=186)
        X = wine.data.features
        y_raw = wine.data.targets
        target_col = y_raw.columns[0]
        y = (y_raw[target_col] >= 6).astype(int).values
        X = X.fillna(X.median())
        return X.values.astype(float), y
    except Exception:
        # Резервный надежный вариант через CSV с автоопределением разделителя
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv"
        df = pd.read_csv(url, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
        target_col = "quality" if "quality" in df.columns else df.columns[-1]
        y = (df[target_col].astype(float) >= 6).astype(int).values
        X = df.drop(columns=[target_col]).values.astype(float)
        df_X = pd.DataFrame(X)
        df_X = df_X.fillna(df_X.median())
        return df_X.values.astype(float), y


class DensePyTorch(nn.Module):
    """Полносвязный слой с поддержкой ReLU/Linear активации"""
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu' if activation == 'relu' else 'linear')
        nn.init.zeros_(self.linear.bias)
        self.activation = activation

    def forward(self, X):
        out = self.linear(X)
        if self.activation == "relu":
            out = torch.relu(out)
        return out


def build_network(input_dim):
    """Построение архитектуры сети: вход -> скрытые слои (ReLU) -> выход (Linear)"""
    layers = []
    dims = [input_dim] + HIDDEN
    for i in range(len(HIDDEN)):
        layers.append(DensePyTorch(dims[i], dims[i+1], "relu"))
    layers.append(DensePyTorch(HIDDEN[-1], 2, "linear"))
    return nn.Sequential(*layers).to(DEVICE)


def train_supervised(model, X_train, y_train, X_test, y_test, epochs, freeze_features=False):
    """Обучение сети с учителем (Cross-Entropy Loss + Adam на GPU)"""
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        for param in model[-1].parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model[-1].parameters(), lr=LR)
    else:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    history = {"loss": [], "test_acc": []}

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            b = perm[start:start + BATCH_SIZE]
            xb, yb = X_t[b], y_t[b]

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(b)

        history["loss"].append(epoch_loss / n)
        
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_te), dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            history["test_acc"].append(acc)
        model.train()

    return history


def train_autoencoder_layer(encoder_layer, H_train, epochs):
    """Послойное предобучение автоэнкодером на GPU"""
    in_dim = encoder_layer.linear.in_features
    out_dim = encoder_layer.linear.out_features
    
    decoder = nn.Sequential(
        nn.Linear(out_dim, in_dim),
        nn.Identity()
    ).to(DEVICE)
    
    optimizer = optim.Adam(list(encoder_layer.parameters()) + list(decoder.parameters()), lr=LR)
    criterion = nn.MSELoss()
    
    H_t = torch.tensor(H_train, dtype=torch.float32, device=DEVICE)
    n = H_t.shape[0]
    
    encoder_layer.train()
    decoder.train()
    
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            b = perm[start:start + BATCH_SIZE]
            xb = H_t[b]
            
            optimizer.zero_grad()
            encoded = encoder_layer(xb)
            decoded = decoder(encoded)
            loss = criterion(decoded, xb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(b)
            
    mse = epoch_loss / H_t.numel()
    return mse


def pretrain_network(model, X_train, verbose):
    """Предобучение всех скрытых слоев сети по автоэнкодерному подходу"""
    H = X_train
    model.eval()
    for i, layer in enumerate(model[:-1]):
        mse = train_autoencoder_layer(layer, H, PRETRAIN_EPOCHS)
        if verbose:
            print(f"  Предобучение слоя {i + 1} "
                  f"({layer.linear.in_features}-{layer.linear.out_features}): MSE = {mse:.4f}")
        with torch.no_grad():
            H = layer(torch.tensor(H, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def predict(model, X_test):
    model.eval()
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        preds = torch.argmax(model(X_te), dim=1).cpu().numpy()
    return preds


def evaluate(model, X_test, y_test):
    y_pred = predict(model, X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred),
    }


def run_once(X, y, seed, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # 1. Обучение БЕЗ предобучения
    torch.manual_seed(seed)
    np.random.seed(seed)
    base_model = build_network(X.shape[1])
    hist_base = train_supervised(base_model, X_train, y_train, X_test, y_test, EPOCHS)

    # 2. Обучение С предобучением (автоэнкодерный подход)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre_model = build_network(X.shape[1])
    pretrain_network(pre_model, X_train, verbose)
    
    train_supervised(pre_model, X_train, y_train, X_test, y_test, OUTPUT_LAYER_EPOCHS, freeze_features=True)
    hist_pre = train_supervised(pre_model, X_train, y_train, X_test, y_test, EPOCHS, freeze_features=False)

    return (evaluate(base_model, X_test, y_test), evaluate(pre_model, X_test, y_test),
            hist_base, hist_pre)


def run_dataset(name, X, y, class_names, file_prefix):
    print("=" * 70)
    print(f"Датасет: {name}  |  объектов: {X.shape[0]}, признаков: {X.shape[1]}")
    print(f"Архитектура: {X.shape[1]}-" + "-".join(map(str, HIDDEN)) + "-2")
    print("=" * 70)

    all_base, all_pre = [], []
    first = None
    for seed in SEEDS:
        res_base, res_pre, hist_base, hist_pre = run_once(X, y, seed, verbose=(seed == SEEDS[0]))
        all_base.append(res_base)
        all_pre.append(res_pre)
        print(f"seed={seed}:  без предобучения F1={res_base['f1']:.4f} acc={res_base['accuracy']:.4f}"
              f"  |  с предобучением F1={res_pre['f1']:.4f} acc={res_pre['accuracy']:.4f}")
        if seed == SEEDS[0]:
            first = (res_base, res_pre, hist_base, hist_pre)

    print()
    print(f"{'Метрика':<12}{'Без предобучения':>24}{'С предобучением':>24}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        b = np.array([r[metric] for r in all_base])
        p = np.array([r[metric] for r in all_pre])
        print(f"{metric:<12}{b.mean():>15.4f} ± {b.std():.4f}{p.mean():>15.4f} ± {p.std():.4f}")
    print()

    res_base, res_pre, hist_base, hist_pre = first
    print(f"Матрица ошибок (seed={SEEDS[0]}), без предобучения:\n{res_base['cm']}")
    print(f"Матрица ошибок (seed={SEEDS[0]}), с предобучением:\n{res_pre['cm']}")
    print()

    # Построение графиков кривых обучения
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_base["loss"], label="Без предобучения")
    axes[0].plot(hist_pre["loss"], label="С предобучением")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Cross-entropy (train)")
    axes[0].set_title(f"{name}: функция потерь")
    axes[0].legend()
    axes[1].plot(hist_base["test_acc"], label="Без предобучения")
    axes[1].plot(hist_pre["test_acc"], label="С предобучением")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy (test)")
    axes[1].set_title(f"{name}: точность на тесте")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_curves.png", dpi=150)
    plt.show()

    # Построение матриц ошибок
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, title in ((axes[0], res_base, "Без предобучения"),
                           (axes[1], res_pre, "С предобучением")):
        ConfusionMatrixDisplay(res["cm"], display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}: {title}")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_confusion.png", dpi=150)
    plt.show()


# 1. Сначала запускаем датасет из ЛР3 (Raisin)
X_raisin, y_raisin = load_raisin()
run_dataset("Raisin (ЛР3 - Вариант 9)", X_raisin, y_raisin, ["Kecimen", "Besni"], "lab3_raisin")

# 2. Затем запускаем датасет из ЛР2 (Wine Quality white)
X_wine, y_wine = load_wine_quality_white()
run_dataset("Wine Quality White (ЛР2 - Вариант 9)", X_wine, y_wine, ["Обычное (<6)", "Хорошее (>=6)"], "lab3_wine")