from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

PATH = Path(__file__).parent
MODES = ("Без предобучения", "Автоэнкодер", "RBM")
torch.set_num_threads(2)


def train(model, X, y, criterion, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for _ in range(epochs):
        total = 0
        for idx in torch.randperm(len(X)).split(128):
            loss = criterion(model(X[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append(total / len(X))
    return history


def pretrain_ae(model, X):
    for layer, activation in zip(model[:-1:2], model[1::2]):
        encoder = nn.Sequential(layer, activation)
        ae = nn.Sequential(encoder, nn.Linear(layer.out_features, layer.in_features))
        train(ae, X, X, nn.MSELoss(), 40)
        with torch.no_grad():
            X = encoder(X)


@torch.no_grad()
def pretrain_rbm(model, X):
    for i, layer in enumerate(model[:-1:2]):
        W, bh = layer.weight, layer.bias
        W.normal_(0, 0.01)
        bh.zero_()
        bv = torch.zeros(layer.in_features)
        dW, dh, dv = torch.zeros_like(W), torch.zeros_like(bh), torch.zeros_like(bv)
        for epoch in range(50):
            momentum = 0.5 if epoch < 5 else 0.9
            for idx in torch.randperm(len(X)).split(64):
                v = X[idx]
                ph = torch.sigmoid(v @ W.T + bh)
                h = torch.bernoulli(ph)
                reconstruction = h @ W + bv
                v1 = reconstruction if i == 0 else torch.bernoulli(torch.sigmoid(reconstruction))
                ph1 = torch.sigmoid(v1 @ W.T + bh)
                dW.mul_(momentum).add_(0.01 * (h.T @ v - ph1.T @ v1) / len(v))
                dh.mul_(momentum).add_(0.01 * (h - ph1).mean(0))
                dv.mul_(momentum).add_(0.01 * (v - v1).mean(0))
                W.add_(dW)
                bh.add_(dh)
                bv.add_(dv)
        X = torch.sigmoid(layer(X))


def compare(X, y, title, ax, regression=False):
    outputs = 1
    if not regression:
        labels, y = np.unique(y, return_inverse=True)
        outputs = len(labels)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None if regression else y
    )
    cats = Xtr.select_dtypes(exclude="number").columns
    nums = Xtr.select_dtypes(include="number").columns
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], sparse_threshold=0)
    Xtr = torch.tensor(prep.fit_transform(Xtr), dtype=torch.float32)
    Xte = torch.tensor(prep.transform(Xte), dtype=torch.float32)
    if regression:
        mean, std = ytr.mean(), ytr.std()
        target = torch.tensor((ytr - mean) / std, dtype=torch.float32).reshape(-1, 1)
        criterion = nn.MSELoss()
    else:
        target = torch.tensor(ytr, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

    sizes = (Xtr.shape[1], 32, 16, 8)
    print(f"\n{title}: train={len(Xtr)}, test={len(Xte)}")
    if not regression:
        fig, matrices = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for i, (name, pretrain) in enumerate(zip(MODES, (None, pretrain_ae, pretrain_rbm))):
        torch.manual_seed(42)
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers.extend((nn.Linear(a, b), nn.ReLU()))
        model = nn.Sequential(*layers, nn.Linear(sizes[-1], outputs))
        if pretrain:
            pretrain(model, Xtr)
        torch.manual_seed(42)
        history = train(model, Xtr, target, criterion, 100)
        ax.plot(range(1, len(history) + 1), history, label=name)
        with torch.no_grad():
            pred = model(Xte).numpy()
        if regression:
            pred = pred.ravel() * std + mean
            mae = mean_absolute_error(yte, pred)
            mape = mean_absolute_percentage_error(yte, pred)
            rmse = root_mean_squared_error(yte, pred)
            r2 = r2_score(yte, pred)
            print(f"{name}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, R²={r2:.4f}, MAPE={mape:.4%}")
        else:
            pred = pred.argmax(1)
            f1 = f1_score(yte, pred, average="macro", zero_division=0)
            weighted = f1_score(yte, pred, average="weighted", zero_division=0)
            print(f"{name}: Accuracy={accuracy_score(yte, pred):.4f}, "
                  f"F1 macro={f1:.4f}, F1 weighted={weighted:.4f}")
            cm = ConfusionMatrixDisplay.from_predictions(
                yte, pred, labels=np.arange(outputs), display_labels=labels,
                ax=matrices[i], colorbar=False, cmap="Blues", values_format="d"
            )
            matrices[i].set(title=name, xlabel="Предсказано", ylabel="Истина")
            print(cm.confusion_matrix)
    if not regression:
        fig.savefig("lab4_matrices.png", dpi=160)
    ax.set(title=title, xlabel="Эпоха", ylabel="Ошибка обучения")
    ax.legend()


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    thermo = pd.read_csv(PATH / "thermography.csv").dropna(subset=["aveOralF"])
    temperature = thermo.pop("aveOralF")
    compare(thermo.drop(columns=["SubjectID", "aveOralM"]), temperature,
            "Infrared Thermography", axes[0], regression=True)
    wine = pd.read_csv(PATH / "winequality-white.csv", sep=";")
    quality = wine.pop("quality")
    compare(wine, quality, "Wine Quality (white)", axes[1])
    fig.tight_layout()
    fig.savefig("lab4_loss.png", dpi=160)
    plt.show()
