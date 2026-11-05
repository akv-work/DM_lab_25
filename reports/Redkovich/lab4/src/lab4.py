import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
import ucimlrepo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

HIDDEN = [64, 32, 16]
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PRETRAIN_EPOCHS = 50
OUTPUT_LAYER_EPOCHS = 30
SEEDS = [0, 1, 2, 3, 4]

RBM_RATE = 0.001
RBM_START_MOMENTUM = 0.5
RBM_FINAL_MOMENTUM = 0.9
RBM_WEIGHT_LOSS = 0.0002

METHODS = ["Без предобучения", "Автоэнкодер (ЛР3)", "RBM"]


def load_raisin():
    # Загрузка датасета Raisin (Вариант 9) из репозитория UCI ML Repository[cite: 2]
    dataset = ucimlrepo.fetch_ucirepo(id=850)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    y_raw = y.values
    classes = np.unique(y_raw)
    # Превращаем текстовые метки классов в бинарные (0 и 1)
    y_bin = (y_raw == classes[0]).astype(int)
    return X.values.astype(float), y_bin, [str(classes[0]), str(classes[1])]


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


ACTIVATIONS = {"relu": relu, "linear": lambda z: z, "logistic": sigmoid}


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
        return relu(self.Z) if self.activation == "relu" else self.Z

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


class RBM:
    def __init__(self, n_visible, n_hidden, vis_act, hid_act, rng):
        self.weights = 0.1 * rng.normal(size=(n_visible, n_hidden))
        self.vis_biases = np.zeros(n_visible)
        self.hid_biases = np.zeros(n_hidden)
        self.vis_act = vis_act
        self.hid_act = hid_act

    def hid_activate(self, v):
        return ACTIVATIONS[self.hid_act](v @ self.weights + self.hid_biases)

    def vis_activate(self, h):
        return ACTIVATIONS[self.vis_act](h @ self.weights.T + self.vis_biases)


def train_rbm(data, n_hidden, vis_act, hid_act, rng):
    rbm = RBM(data.shape[1], n_hidden, vis_act, hid_act, rng)
    wu_vh = np.zeros_like(rbm.weights)
    bu_v = np.zeros_like(rbm.vis_biases)
    bu_h = np.zeros_like(rbm.hid_biases)
    n = data.shape[0]

    for epoch in range(PRETRAIN_EPOCHS):
        momentum = RBM_START_MOMENTUM if epoch < 5 else RBM_FINAL_MOMENTUM
        idx = rng.permutation(n)
        err = 0.0
        for start in range(0, n, BATCH_SIZE):
            v1 = data[idx[start:start + BATCH_SIZE]]
            bs = v1.shape[0]

            h1 = rbm.hid_activate(v1)
            if hid_act == "logistic":
                h_sampled = (h1 > rng.random(h1.shape)).astype(float)
            else:
                h_sampled = h1
            v2 = rbm.vis_activate(h_sampled)
            h2 = rbm.hid_activate(v2)

            wu_vh = momentum * wu_vh + RBM_RATE * (
                (v1.T @ h1 - v2.T @ h2) / bs - RBM_WEIGHT_LOSS * rbm.weights)
            bu_v = momentum * bu_v + RBM_RATE * (v1 - v2).sum(axis=0) / bs
            bu_h = momentum * bu_h + RBM_RATE * (h1 - h2).sum(axis=0) / bs

            rbm.weights += wu_vh
            rbm.vis_biases += bu_v
            rbm.hid_biases += bu_h
            err += np.sum((v2 - v1) ** 2)

    return rbm, err / data.size


def train_autoencoder_layer(encoder, H, rng):
    decoder = Dense(encoder.W.shape[1], encoder.W.shape[0], "linear", rng)
    n = H.shape[0]
    for epoch in range(PRETRAIN_EPOCHS):
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
    return epoch_loss / H.size


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


def build_network(input_dim, rng):
    dims = [input_dim] + HIDDEN
    layers = [Dense(dims[i], dims[i + 1], "relu", rng) for i in range(len(HIDDEN))]
    layers.append(Dense(HIDDEN[-1], 2, "linear", rng))
    return layers


def pretrain_autoencoder(layers, X_train, rng, verbose):
    H = X_train
    for i, layer in enumerate(layers[:-1]):
        mse = train_autoencoder_layer(layer, H, rng)
        if verbose:
            print(f"  [AE]  слой {i + 1} ({layer.W.shape[0]}-{layer.W.shape[1]}): "
                  f"MSE реконструкции = {mse:.4f}")
        layer.reset_optimizer()
        H = layer.forward(H)


def pretrain_rbm(layers, X_train, rng, verbose):
    H = X_train
    for i, layer in enumerate(layers[:-1]):
        vis_act = "linear" if i == 0 else "relu"
        rbm, err = train_rbm(H, layer.W.shape[1], vis_act, "relu", rng)
        if verbose:
            print(f"  [RBM] слой {i + 1} ({layer.W.shape[0]}-{layer.W.shape[1]}): "
                  f"ошибка реконструкции = {err:.4f}")
        layer.W = rbm.weights.copy()
        layer.b = rbm.hid_biases.copy()
        layer.reset_optimizer()
        H = layer.forward(H)


def finish_training(layers, X_train, y_train, X_test, y_test, rng):
    train_supervised(layers, [layers[-1]], X_train, y_train,
                     X_test, y_test, OUTPUT_LAYER_EPOCHS, rng)
    for layer in layers:
        layer.reset_optimizer()
    return train_supervised(layers, layers, X_train, y_train,
                            X_test, y_test, EPOCHS, rng)


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

    results, histories = {}, {}

    rng = np.random.default_rng(seed)
    net = build_network(X.shape[1], rng)
    histories[METHODS[0]] = train_supervised(net, net, X_train, y_train,
                                             X_test, y_test, EPOCHS, rng)
    results[METHODS[0]] = evaluate(net, X_test, y_test)

    rng = np.random.default_rng(seed)
    net = build_network(X.shape[1], rng)
    pretrain_autoencoder(net, X_train, rng, verbose)
    histories[METHODS[1]] = finish_training(net, X_train, y_train, X_test, y_test, rng)
    results[METHODS[1]] = evaluate(net, X_test, y_test)

    rng = np.random.default_rng(seed)
    net = build_network(X.shape[1], rng)
    pretrain_rbm(net, X_train, rng, verbose)
    histories[METHODS[2]] = finish_training(net, X_train, y_train, X_test, y_test, rng)
    results[METHODS[2]] = evaluate(net, X_test, y_test)

    return results, histories


def run_dataset(name, X, y, class_names, file_prefix):
    print("=" * 78)
    print(f"Датасет: {name}  |  объектов: {X.shape[0]}, признаков: {X.shape[1]}")
    print(f"Архитектура: {X.shape[1]}-" + "-".join(map(str, HIDDEN)) + "-2")
    print("=" * 78)

    all_results = {m: [] for m in METHODS}
    for seed in SEEDS:
        results, histories = run_once(X, y, seed, verbose=(seed == SEEDS[0]))
        for m in METHODS:
            all_results[m].append(results[m])
        print(f"seed={seed}:  " + "  |  ".join(
            f"{m}: F1={results[m]['f1']:.4f} acc={results[m]['accuracy']:.4f}" for m in METHODS))
        if seed == SEEDS[0]:
            first_results, first_histories = results, histories

    print()
    print(f"Среднее ± ст. отклонение по {len(SEEDS)} запускам:")
    print(f"{'Метрика':<11}" + "".join(f"{m:>24}" for m in METHODS))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        row = f"{metric:<11}"
        for m in METHODS:
            vals = np.array([r[metric] for r in all_results[m]])
            row += f"{vals.mean():>15.4f} ± {vals.std():.4f}"
        print(row)
    print()
    for m in METHODS:
        print(f"Матрица ошибок (seed={SEEDS[0]}), {m}:\n{first_results[m]['cm']}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for m in METHODS:
        axes[0].plot(first_histories[m]["loss"], label=m)
        axes[1].plot(first_histories[m]["test_acc"], label=m)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Cross-entropy (train)")
    axes[0].set_title(f"{name}: функция потерь")
    axes[0].legend()
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy (test)")
    axes[1].set_title(f"{name}: точность на тесте")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_curves.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, METHODS):
        ConfusionMatrixDisplay(first_results[m]["cm"], display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}: {m}")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_confusion.png", dpi=150)
    plt.show()


# Запуск для вашего варианта 9 (Raisin)
X_raisin, y_raisin, class_names_raisin = load_raisin()
run_dataset("Raisin (Вариант 9)", X_raisin, y_raisin, class_names_raisin, "lab4_raisin")