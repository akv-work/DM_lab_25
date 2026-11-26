from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

PATH = Path(__file__).parent
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
    for pre, name in enumerate(("Без предобучения", "С предобучением")):
        torch.manual_seed(42)
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers.extend((nn.Linear(a, b), nn.ReLU()))
        model = nn.Sequential(*layers, nn.Linear(sizes[-1], outputs))
        if pre:
            z = Xtr
            for i, a in enumerate(sizes[:-1]):
                encoder = model[2*i:2*i+2]
                ae = nn.Sequential(encoder, nn.Linear(sizes[i+1], a))
                train(ae, z, z, nn.MSELoss(), 40)
                with torch.no_grad():
                    z = encoder(z)
        torch.manual_seed(42)
        ax.plot(train(model, Xtr, target, criterion, 100), label=name)
        with torch.no_grad():
            pred = model(Xte).numpy()
        if regression:
            pred = pred.ravel() * std + mean
            mae = mean_absolute_error(yte, pred)
            mape = mean_absolute_percentage_error(yte, pred)
            print(f"{name}: MAE={mae:.4f} °C, MAPE={mape:.4%}")
        else:
            pred = pred.argmax(1)
            f1 = f1_score(yte, pred, average="macro", zero_division=0)
            print(f"{name}: Accuracy={accuracy_score(yte, pred):.4f}, F1 macro={f1:.4f}")
            print("Порядок классов:", labels)
            print(confusion_matrix(yte, pred, labels=np.arange(outputs)))
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
    plt.savefig(Path.cwd() / "lab3_loss.png", dpi=150)
    plt.show()
