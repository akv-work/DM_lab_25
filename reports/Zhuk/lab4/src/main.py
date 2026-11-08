import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ucimlrepo import fetch_ucirepo


 
# 1. НАСТРОЙКИ
 

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS_PRE = 50
EPOCHS_HEAD = 20
EPOCHS = 100

BATCH = 128
LR = 0.001
RBM_LR = 0.01

MOM_START = 0.5
MOM_FINAL = 0.9
MOM_SWITCH = 5

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


 
# 2. НЕЙРОННАЯ СЕТЬ
 

class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, n_out)
        )

    def forward(self, x):
        return self.layers(x)


 
# 3. АВТОЭНКОДЕР
 

class Autoencoder(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU()
        )

        self.decoder = nn.Linear(n_hidden, n_in)

    def forward(self, x):
        return self.decoder(self.encoder(x))


 
# 4. ОГРАНИЧЕННАЯ МАШИНА БОЛЬЦМАНА (RBM)
 

class RBM:
    def __init__(self, n_in, n_hidden, gaussian_visible):

        self.W = 0.01 * torch.randn(
            n_in, n_hidden, device=DEVICE
        )

        self.vb = torch.zeros(n_in, device=DEVICE)
        self.hb = torch.zeros(n_hidden, device=DEVICE)

        self.gauss = gaussian_visible

    def hidden(self, v):
        return torch.sigmoid(v @ self.W + self.hb)

    def visible(self, h):

        z = h @ self.W.T + self.vb

        if self.gauss:
            return z

        return torch.sigmoid(z)

    @torch.no_grad()
    def train(self, X):

        dW = torch.zeros_like(self.W)
        dvb = torch.zeros_like(self.vb)
        dhb = torch.zeros_like(self.hb)

        err = 0.0

        for epoch in range(EPOCHS_PRE):

            mom = (
                MOM_START
                if epoch < MOM_SWITCH
                else MOM_FINAL
            )

            err = 0.0
            nb = 0

            for v0 in batches(X):

                n = len(v0)

                # Положительная фаза
                p_h0 = self.hidden(v0)
                h0 = torch.bernoulli(p_h0)

                # Отрицательная фаза
                p_v1 = self.visible(h0)

                if self.gauss:
                    v1 = p_v1
                else:
                    v1 = torch.bernoulli(p_v1)

                p_h1 = self.hidden(v1)

                # Обновление весов
                dW = (
                    mom * dW
                    + RBM_LR * (
                        v0.T @ h0 - v1.T @ p_h1
                    ) / n
                )

                dvb = (
                    mom * dvb
                    + RBM_LR * (v0 - v1).mean(0)
                )

                dhb = (
                    mom * dhb
                    + RBM_LR * (h0 - p_h1).mean(0)
                )

                self.W += dW
                self.vb += dvb
                self.hb += dhb
                err += ((v0 - p_v1) ** 2).mean().item()
                nb += 1

            err /= nb

        return err


 
# 5. ФОРМИРОВАНИЕ ПАКЕТОВ ДАННЫХ
 

def batches(X, y=None):

    ind = torch.randperm(len(X))

    for i in range(0, len(X), BATCH):

        j = ind[i:i + BATCH]

        xb = torch.tensor(
            X[j.numpy()],
            dtype=torch.float32
        ).to(DEVICE)

        if y is None:
            yield xb

        else:

            yb = torch.tensor(
                y[j.numpy()],
                dtype=torch.long
            ).to(DEVICE)

            yield xb, yb


 
# 6. ПРЕДОБУЧЕНИЕ АВТОЭНКОДЕРАМИ
 

def pretrain_ae(X, sizes):

    out = X.copy()
    weights = []

    for a, b in zip(sizes[:-1], sizes[1:]):

        print(f"  AE {a}-{b}")

        ae = Autoencoder(a, b).to(DEVICE)

        opt = torch.optim.Adam(
            ae.parameters(),
            lr=LR
        )

        loss_fn = nn.MSELoss()

        for epoch in range(EPOCHS_PRE):

            for xb in batches(out):

                pred = ae(xb)
                loss = loss_fn(pred, xb)

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Сохранение весов кодировщика
        layer = ae.encoder[0]

        weights.append((
            layer.weight.detach().clone(),
            layer.bias.detach().clone()
        ))

        # Получение признаков следующего слоя
        with torch.no_grad():

            tensor = torch.tensor(
                out,
                dtype=torch.float32
            ).to(DEVICE)

            out = ae.encoder(tensor).cpu().numpy()

    return weights


 
# 7. ПРЕДОБУЧЕНИЕ RBM
 

def pretrain_rbm(X, sizes):

    out = X.copy()
    weights = []

    for k, (a, b) in enumerate(
        zip(sizes[:-1], sizes[1:])
    ):

        print(f"  RBM {a}-{b}")

        # Первый слой использует гауссовские
        # видимые нейроны
        rbm = RBM(
            a,
            b,
            gaussian_visible=(k == 0)
        )

        err = rbm.train(out)

        print(
            f"  Ошибка реконструкции: {err:.6f}"
        )

        # Сохранение весов и смещений
        weights.append((
            rbm.W.T.clone(),
            rbm.hb.clone()
        ))

        # Передача признаков следующему RBM
        with torch.no_grad():

            tensor = torch.tensor(
                out,
                dtype=torch.float32
            ).to(DEVICE)

            out = rbm.hidden(tensor).cpu().numpy()

    return weights


 
# 8. СОЗДАНИЕ МОДЕЛИ И ПЕРЕНОС ВЕСОВ
 

def make_net(n_in, n_out, weights=None):

    model = Net(n_in, n_out).to(DEVICE)

    if weights:

        layers = [
            x for x in model.layers
            if isinstance(x, nn.Linear)
        ]

        for layer, (w, b) in zip(
            layers[:-1],
            weights
        ):

            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)

    return model


 
# 9. ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ
 

def train(model, X, y, epochs):

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    loss_fn = nn.CrossEntropyLoss()

    model.train()
    final_loss = 0.0

    for epoch in range(epochs):

        total_loss = 0.0
        nb = 0

        for xb, yb in batches(X, y):

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            nb += 1

        final_loss = total_loss / nb

    print(
        f"Эпох: {epochs}, "
        f"ошибка на конец обучения: {final_loss:.6f}"
    )

    return final_loss


 
# 10. ОБУЧЕНИЕ ПРЕДОБУЧЕННОЙ СЕТИ
 

def train_pretrained(model, X, y):

    layers = [
        x for x in model.layers
        if isinstance(x, nn.Linear)
    ]

    for p in model.parameters():
        p.requires_grad = False

    for p in layers[-1].parameters():
        p.requires_grad = True

    print("Обучение выходного слоя:")
    head_loss = train(
        model,
        X,
        y,
        EPOCHS_HEAD
    )

    for p in model.parameters():
        p.requires_grad = True

    print("Дообучение всей сети:")
    final_loss = train(
        model,
        X,
        y,
        EPOCHS
    )

    return final_loss


 
# 11. ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ
 

def predict(model, X):

    model.eval()

    with torch.no_grad():

        x = torch.tensor(
            X,
            dtype=torch.float32
        ).to(DEVICE)

        return model(x).cpu().numpy().argmax(axis=1)


 
# 12. ОЦЕНКА РЕЗУЛЬТАТОВ
 

def evaluate(model, X_test, y_test, name, dataset):

    predictions = predict(model, X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(y_test, predictions)

    print(f"\n{name}:")
    print(f"Accuracy = {accuracy * 100:.2f}%")
    print(f"Macro F1 = {f1:.4f}")

    print("Матрица ошибок:")
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{dataset} - {name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/{dataset}_{name}_confusion.png"
    )
    plt.close()

    return {
        "Датасет": dataset,
        "Метод": name,
        "Accuracy": accuracy,
        "Macro F1": f1
    }


 
# 13. ПОДГОТОВКА ДАННЫХ
 

def prepare_data(X, y):

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # Удаление некорректных значений
    X[~np.isfinite(X)] = np.nan

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Сначала выделяем тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=SEED,
        stratify=y
    )

    # Из обучающей выборки выделяем валидационную
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1765,
        random_state=SEED,
        stratify=y_train
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


 
# 14. ЗАГРУЗКА OPTICAL RECOGNITION OF HANDWRITTEN DIGITS
 

def load_digits_data():

    train_path = "data/optdigits.tra"
    test_path = "data/optdigits.tes"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Не найден файл: {train_path}"
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Не найден файл: {test_path}"
        )

    # В датасете последние столбцы содержат классы
    train_df = pd.read_csv(
        train_path,
        header=None
    )

    test_df = pd.read_csv(
        test_path,
        header=None
    )

    train_data = train_df.values
    test_data = test_df.values

    X_train = train_data[:, :-1].astype(float)
    y_train = train_data[:, -1].astype(int)

    X_test = test_data[:, :-1].astype(float)
    y_test = test_data[:, -1].astype(int)

    # Объединяем официальные обучающую и тестовую
    # выборки, чтобы одинаково сформировать
    # train/validation/test для всех экспериментов
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    return X, y


 
# 15. ЗАГРУЗКА CARDIOTOCOGRAPHY
 

def load_cardiotocography_data():

    print("Загрузка Cardiotocography...")

    dataset = fetch_ucirepo(id=193)

    X = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    # Ищем целевой признак NSP
    target_column = None

    for column in targets.columns:

        if str(column).strip().upper() == "NSP":
            target_column = column
            break

    if target_column is None:

        raise ValueError(
            "В Cardiotocography не найден целевой признак NSP"
        )

    y = pd.to_numeric(
        targets[target_column],
        errors="coerce"
    )

    # Убираем строки без целевого значения
    mask = y.notna()

    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # Переводим признаки в числовой формат
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X.values.astype(float), y.values


 
# 16. ЗАПУСК ЭКСПЕРИМЕНТОВ
 

def run_experiment(dataset_name, X, y):

    print(f"ДАТАСЕТ: {dataset_name}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_data(X, y)

    n_in = X_train.shape[1]
    n_out = len(np.unique(y_train))

    sizes = [n_in, 32, 16, 8]

    print(f"Размер выборки: {len(X)}")
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")
    print(f"Количество признаков: {n_in}")
    print(f"Количество классов: {n_out}")
    print(f"Архитектура: {'-'.join(map(str, sizes + [n_out]))}")
    print(f"Размер пакета: {BATCH}")
    print(f"Скорость обучения: {LR}")

    results = []

    # 1. БЕЗ ПРЕДОБУЧЕНИЯ

    print("ЭКСПЕРИМЕНТ 1: ОБУЧЕНИЕ БЕЗ ПРЕДОБУЧЕНИЯ")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = make_net(n_in, n_out)

    loss = train(
        model,
        X_train,
        y_train,
        EPOCHS
    )

    results.append(
        evaluate(
            model,
            X_test,
            y_test,
            "Без предобучения",
            dataset_name
        )
    )

    # 2. АВТОЭНКОДЕР

    print("ЭКСПЕРИМЕНТ 2: ПРЕДОБУЧЕНИЕ AUTOENCODER")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Предобучение слоёв:")

    weights = pretrain_ae(
        X_train,
        sizes
    )

    model = make_net(
        n_in,
        n_out,
        weights
    )

    loss = train_pretrained(
        model,
        X_train,
        y_train
    )

    results.append(
        evaluate(
            model,
            X_test,
            y_test,
            "AE",
            dataset_name
        )
    )

    # 3. RBM

    print("ЭКСПЕРИМЕНТ 3: ПРЕДОБУЧЕНИЕ RBM")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Предобучение слоёв:")

    weights = pretrain_rbm(
        X_train,
        sizes
    )

    model = make_net(
        n_in,
        n_out,
        weights
    )

    loss = train_pretrained(
        model,
        X_train,
        y_train
    )

    results.append(
        evaluate(
            model,
            X_test,
            y_test,
            "RBM",
            dataset_name
        )
    )

    # ИТОГИ ПО ДАТАСЕТУ

    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ: {dataset_name}")

    print(
        f"{'Метод':<25}"
        f"{'Accuracy':<15}"
        f"{'Macro F1':<15}"
    )

    for result in results:

        print(
            f"{result['Метод']:<25}"
            f"{result['Accuracy'] * 100:<15.2f}"
            f"{result['Macro F1']:<15.4f}"
        )

    return results


 
# 17. ГЛАВНАЯ ФУНКЦИЯ
 
def main():

    print("ПРЕДОБУЧЕНИЕ НЕЙРОННЫХ СЕТЕЙ С ПОМОЩЬЮ RBM")

    print(f"Устройство: {DEVICE}")
    print(f"Эпохи предобучения: {EPOCHS_PRE}")
    print(f"Эпохи выходного слоя: {EPOCHS_HEAD}")
    print(f"Эпохи полного обучения: {EPOCHS}")

    all_results = []

    print("\nЗагрузка Optical Digits...")

    X, y = load_digits_data()

    all_results.extend(
        run_experiment(
            "Optical Digits",
            X,
            y
        )
    )

    print("\nЗагрузка Cardiotocography...")

    X, y = load_cardiotocography_data()

    all_results.extend(
        run_experiment(
            "Cardiotocography",
            X,
            y
        )
    )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        f"{RESULTS_DIR}/comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("ОБЩИЕ ИТОГОВЫЕ РЕЗУЛЬТАТЫ")

    print(
        f"{'Датасет':<25}"
        f"{'Метод':<25}"
        f"{'Accuracy':<12}"
        f"{'Macro F1':<12}"
    )

    for _, row in results_df.iterrows():

        print(
            f"{row['Датасет']:<25}"
            f"{row['Метод']:<25}"
            f"{row['Accuracy'] * 100:<12.2f}"
            f"{row['Macro F1']:<12.4f}"
        )

    print("\nРезультаты сохранены в папке results.")


if __name__ == "__main__":
    main()