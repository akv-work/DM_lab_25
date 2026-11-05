import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

DATA_PATH = "crx.data"

HIDDEN = [64, 32, 16]
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PRETRAIN_EPOCHS = 50
OUTPUT_LAYER_EPOCHS = 30
SEEDS = [0, 1, 2, 3, 4]


def load_credit_approval(path):
    columns = [f"A{i}" for i in range(1, 17)]
    df = pd.read_csv(path, header=None, names=columns, na_values="?")

    y = (df["A16"].str.strip() == "+").astype(int).values
    df = df.drop(columns=["A16"])

    numeric_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=categorical_cols, dtype=float)
    return df.values.astype(float), y


def load_wdbc():
    data = load_breast_cancer()
    y = (data.target == 0).astype(int)
    return data.data, y


class Dense:
    def __init__(self, in_dim, out_dim, activation, rng):
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation
        self.reset_optimizer()

    def reset_optimizer(self):
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return np.maximum(0, self.Z) if self.activation == "relu" else self.Z

    def backward(self, dA):
        dZ = dA * (self.Z > 0) if self.activation == "relu" else dA
        self.dW = self.X.T @ dZ
        self.db = dZ.sum(axis=0)
        return dZ @ self.W.T

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in ((self.W, self.dW, self.mW, self.vW),
                           (self.b, self.db, self.mb, self.vb)):
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(layers, X):
    out = X
    for layer in layers:
        out = layer.forward(out)
    return out


def predict(layers, X):
    return np.argmax(forward(layers, X), axis=1)


def train_supervised(layers, trainable, X, y, X_test, y_test, epochs, rng):
    n = X.shape[0]
    Y = np.eye(2)[y]
    history = {"loss": [], "test_acc": []}

    for _ in range(epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            P = softmax(forward(layers, X[b]))
            epoch_loss += -np.sum(Y[b] * np.log(P + 1e-12))

            grad = (P - Y[b]) / len(b)
            for layer in reversed(layers):
                grad = layer.backward(grad)
            for layer in trainable:
                layer.step(LR)

        history["loss"].append(epoch_loss / n)
        history["test_acc"].append(accuracy_score(y_test, predict(layers, X_test)))
    return history


def train_autoencoder_layer(encoder, H, epochs, rng):
    decoder = Dense(encoder.W.shape[1], encoder.W.shape[0], "linear", rng)
    n = H.shape[0]
    for epoch in range(epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            b = H[idx[start:start + BATCH_SIZE]]
            H_rec = decoder.forward(encoder.forward(b))
            epoch_loss += np.sum((H_rec - b) ** 2)

            grad = 2 * (H_rec - b) / len(b)
            grad = decoder.backward(grad)
            encoder.backward(grad)
            decoder.step(LR)
            encoder.step(LR)
    mse = epoch_loss / H.size
    return mse


def build_network(input_dim, rng):
    dims = [input_dim] + HIDDEN
    layers = [Dense(dims[i], dims[i + 1], "relu", rng) for i in range(len(HIDDEN))]
    layers.append(Dense(HIDDEN[-1], 2, "linear", rng))
    return layers


def pretrain_network(layers, X_train, rng, verbose):
    H = X_train
    for i, layer in enumerate(layers[:-1]):
        mse = train_autoencoder_layer(layer, H, PRETRAIN_EPOCHS, rng)
        if verbose:
            print(f"  Предобучение слоя {i + 1} "
                  f"({layer.W.shape[0]}-{layer.W.shape[1]}): MSE = {mse:.4f}")
        layer.reset_optimizer()
        H = layer.forward(H)


def evaluate(layers, X_test, y_test):
    y_pred = predict(layers, X_test)
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

    rng = np.random.default_rng(seed)
    base = build_network(X.shape[1], rng)
    hist_base = train_supervised(base, base, X_train, y_train,
                                 X_test, y_test, EPOCHS, rng)

    rng = np.random.default_rng(seed)
    pre = build_network(X.shape[1], rng)
    pretrain_network(pre, X_train, rng, verbose)
    train_supervised(pre, [pre[-1]], X_train, y_train,
                     X_test, y_test, OUTPUT_LAYER_EPOCHS, rng)
    for layer in pre:
        layer.reset_optimizer()
    hist_pre = train_supervised(pre, pre, X_train, y_train,
                                X_test, y_test, EPOCHS, rng)

    return (evaluate(base, X_test, y_test), evaluate(pre, X_test, y_test),
            hist_base, hist_pre)


def run_dataset(name, X, y, class_names, file_prefix):
    print("=" * 70)
    print(f"Датасет: {name}  |  объектов: {X.shape[0]}, признаков: {X.shape[1]}")
    print(f"Архитектура: {X.shape[1]}-" + "-".join(map(str, HIDDEN)) + "-2")
    print("=" * 70)

    all_base, all_pre = [], []
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, title in ((axes[0], res_base, "Без предобучения"),
                           (axes[1], res_pre, "С предобучением")):
        ConfusionMatrixDisplay(res["cm"], display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}: {title}")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_confusion.png", dpi=150)
    plt.show()


X_credit, y_credit = load_credit_approval(DATA_PATH)
run_dataset("Credit Approval", X_credit, y_credit, ["- (отказ)", "+ (одобрено)"], "lab3_credit")

X_wdbc, y_wdbc = load_wdbc()
run_dataset("WDBC (из ЛР2)", X_wdbc, y_wdbc, ["benign", "malignant"], "lab3_wdbc")