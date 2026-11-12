import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ARFF = sys.argv[1] if len(sys.argv) > 1 else "Rice_Cammeo_Osmancik.arff"
SIZES = [7, 32, 32, 16, 2]
AE_EPOCHS = 100
HEAD_EPOCHS = 50
FT_EPOCHS = 200
LR, MOMENTUM, BATCH = 0.1, 0.9, 32
SEED = 0


def linear(n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def build_net():
    layers = []
    for i, (n_in, n_out) in enumerate(zip(SIZES, SIZES[1:])):
        layers.append(linear(n_in, n_out))
        if i < len(SIZES) - 2:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def linears(net):
    return [m for m in net if isinstance(m, nn.Linear)]


def loader(*tensors, gen):
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=True, generator=gen)


#Предобучение
def train_autoencoder(X, n_hidden, gen):
    enc, dec = linear(X.shape[1], n_hidden), linear(n_hidden, X.shape[1])
    opt = torch.optim.SGD([*enc.parameters(), *dec.parameters()], lr=LR, momentum=MOMENTUM)
    for _ in range(AE_EPOCHS):
        for (x,) in loader(X, gen=gen):
            recon = torch.sigmoid(dec(torch.sigmoid(enc(x))))
            loss = 0.5 * ((recon - x) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc


def pretrain(net, X, gen):
    data = X
    for layer in linears(net)[:-1]:
        enc = train_autoencoder(data, layer.out_features, gen)
        layer.load_state_dict(enc.state_dict())
        with torch.no_grad():
            data = torch.sigmoid(layer(data))


#Обучение с учителем
def set_only_last_trainable(net, only_last):
    for layer in linears(net)[:-1]:
        for p in layer.parameters():
            p.requires_grad = not only_last


def train_classifier(net, X, y, epochs, gen):
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader(X, y, gen=gen):
            loss = loss_fn(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(X))
    return history


def evaluate(net, X, y):
    with torch.no_grad():
        pred = net(X).argmax(dim=1).numpy()
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), confusion_matrix(y, pred)


def load_data():
    df = pd.DataFrame(arff.loadarff(ARFF)[0])
    df["Class"] = df["Class"].str.decode("utf-8")
    classes = ["Cammeo", "Osmancik"]
    y = df["Class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return df.drop(columns="Class").to_numpy(float), y, classes


def main():
    X, y, classes = load_data()
    print(f"Объектов: {len(X)}; классы {classes}: {np.bincount(y)}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = MinMaxScaler().fit(Xtr)
    Xtr = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    #Без предобучения
    plain = build_net()
    hist_plain = train_classifier(plain, Xtr, Ytr, FT_EPOCHS, gen)

    #С предобучением
    pre = build_net()
    pretrain(pre, Xtr, gen)
    set_only_last_trainable(pre, True)
    h1 = train_classifier(pre, Xtr, Ytr, HEAD_EPOCHS, gen)
    set_only_last_trainable(pre, False)
    h2 = train_classifier(pre, Xtr, Ytr, FT_EPOCHS, gen)

    #Сравнение
    for name, net in (("Без предобучения", plain), ("С предобучением", pre)):
        acc, f1, cm = evaluate(net, Xte, yte)
        print(f"\n{name}: accuracy = {acc:.3f}, macro-F1 = {f1:.3f}")
        print(f"Матрица ошибок (строки - истина, столбцы - прогноз: {', '.join(classes)}):\n{cm}")

    plt.plot(hist_plain, label="без предобучения")
    plt.plot(h1 + h2, label="с предобучением")
    plt.xlabel("эпоха обучения с учителем")
    plt.ylabel("CE на обучающей выборке")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("loss_curves_rice.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
