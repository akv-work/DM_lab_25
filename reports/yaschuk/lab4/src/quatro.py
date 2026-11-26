
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42
EPOCHS = 100
PRETRAIN_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RBM_LEARNING_RATE = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_rice():
    data, meta = arff.loadarff("Rice_Cammeo_Osmancik.arff")
    df = pd.DataFrame(data)

    df["Class"] = df["Class"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values

    return X, y


def load_maternal():
    df = pd.read_csv("Maternal Health Risk Data Set.csv")

    X = df.drop(columns=["RiskLevel"]).values.astype(np.float32)
    y = df["RiskLevel"].values

    return X, y


# ============================================================
# НЕЙРОННАЯ СЕТЬ
# ============================================================

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))

        return self.output(x)


# ============================================================
# ОБУЧЕНИЕ КЛАССИФИКАТОРА
# ============================================================

def train_classifier(model, X_train, y_train):
    model.to(DEVICE)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            output = model(xb)
            loss = criterion(output, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        epoch_loss = total_loss / len(dataset)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 20 == 0:
            print(
                f"Эпоха {epoch + 1}: "
                f"ошибка = {epoch_loss:.6f}"
            )

    return loss_history


# ============================================================
# ПРЕДОБУЧЕНИЕ АВТОЭНКОДЕРАМИ
# ============================================================

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)

        return decoded


def pretrain_autoencoders(model, X_train):
    data = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    layers = [
        (model.layer1, 64),
        (model.layer2, 32),
        (model.layer3, 16)
    ]

    current_data = data

    for i, (layer, hidden_size) in enumerate(layers):
        input_size = current_data.shape[1]

        autoencoder = Autoencoder(
            input_size,
            hidden_size
        ).to(DEVICE)

        optimizer = optim.Adam(
            autoencoder.parameters(),
            lr=LEARNING_RATE
        )

        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(current_data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        print(
            f"\nАвтоэнкодер {i + 1}: "
            f"{input_size} -> {hidden_size} -> {input_size}"
        )

        for epoch in range(PRETRAIN_EPOCHS):
            total_loss = 0

            for (xb,) in loader:
                optimizer.zero_grad()

                reconstructed = autoencoder(xb)
                loss = criterion(reconstructed, xb)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Эпоха {epoch + 1}: "
                    f"ошибка = {total_loss / len(dataset):.6f}"
                )

        # Перенос весов кодировщика в классификатор
        layer.weight.data.copy_(autoencoder.encoder.weight.data)
        layer.bias.data.copy_(autoencoder.encoder.bias.data)

        # Данные для следующего автоэнкодера
        with torch.no_grad():
            current_data = torch.relu(
                autoencoder.encoder(current_data)
            )

    return model


# ============================================================
# ОГРАНИЧЕННАЯ МАШИНА БОЛЬЦМАНА (RBM)
# ============================================================

class RBM:
    def __init__(self, visible_size, hidden_size):
        self.W = torch.randn(
            visible_size,
            hidden_size,
            device=DEVICE
        ) * 0.01

        self.visible_bias = torch.zeros(
            visible_size,
            device=DEVICE
        )

        self.hidden_bias = torch.zeros(
            hidden_size,
            device=DEVICE
        )

    def hidden_probabilities(self, visible):
        return torch.sigmoid(
            visible @ self.W + self.hidden_bias
        )

    def train(self, data):
        dataset = torch.utils.data.TensorDataset(data)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for epoch in range(PRETRAIN_EPOCHS):
            total_error = 0

            for (visible,) in loader:
                # Положительная фаза
                positive_hidden_prob = self.hidden_probabilities(
                    visible
                )

                positive_hidden = torch.bernoulli(
                    positive_hidden_prob
                )

                # Отрицательная фаза (CD-1)
                negative_visible = (
                    positive_hidden @ self.W.T
                    + self.visible_bias
                )

                negative_hidden_prob = self.hidden_probabilities(
                    negative_visible
                )

                # Обновление параметров
                batch_size = visible.size(0)

                self.W += RBM_LEARNING_RATE * (
                    visible.T @ positive_hidden_prob
                    - negative_visible.T @ negative_hidden_prob
                ) / batch_size

                self.visible_bias += RBM_LEARNING_RATE * torch.mean(
                    visible - negative_visible,
                    dim=0
                )

                self.hidden_bias += RBM_LEARNING_RATE * torch.mean(
                    positive_hidden_prob - negative_hidden_prob,
                    dim=0
                )

                total_error += torch.mean(
                    (visible - negative_visible) ** 2
                ).item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Эпоха {epoch + 1}: "
                    f"ошибка восстановления = "
                    f"{total_error / len(loader):.6f}"
                )

    def transform(self, data):
        with torch.no_grad():
            return self.hidden_probabilities(data)


# ============================================================
# ПРЕДОБУЧЕНИЕ СТЕКОМ RBM
# ============================================================

def pretrain_rbm(model, X_train):
    data = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE
    )

    layers = [
        (model.layer1, 64),
        (model.layer2, 32),
        (model.layer3, 16)
    ]

    current_data = data

    for i, (layer, hidden_size) in enumerate(layers):
        input_size = current_data.shape[1]

        rbm = RBM(input_size, hidden_size)

        print(
            f"\nRBM {i + 1}: "
            f"{input_size} -> {hidden_size}"
        )

        rbm.train(current_data)

        # Перенос весов RBM в классификатор
        layer.weight.data.copy_(rbm.W.T)
        layer.bias.data.copy_(rbm.hidden_bias)

        # Получение признаков для следующего RBM
        current_data = rbm.transform(current_data)

    return model


# ============================================================
# ОЦЕНКА МОДЕЛИ
# ============================================================

def evaluate_model(model, X_test, y_test):
    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        output = model(X_tensor)
        predictions = torch.argmax(output, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    matrix = confusion_matrix(y_test, predictions)

    return accuracy, f1, matrix


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================

def plot_loss(results, dataset_name):
    plt.figure(figsize=(9, 5))

    for method, result in results.items():
        plt.plot(
            result["loss"],
            label=method
        )

    plt.title(f"{dataset_name}: ошибка обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_loss.png")
    plt.close()


def plot_metrics(results, dataset_name):
    methods = list(results.keys())

    accuracy = [
        results[m]["accuracy"] for m in methods
    ]

    f1 = [
        results[m]["f1"] for m in methods
    ]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy"
    )

    plt.bar(
        x + width / 2,
        f1,
        width,
        label="F1"
    )

    plt.xticks(x, methods, rotation=10)
    plt.ylim(0, 1)
    plt.ylabel("Значение метрики")
    plt.title(f"{dataset_name}: сравнение качества")
    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_metrics.png")
    plt.close()


def plot_confusion_matrices(results, dataset_name, classes):
    methods = list(results.keys())

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(15, 4)
    )

    for ax, method in zip(axes, methods):
        matrix = results[method]["matrix"]

        image = ax.imshow(matrix, cmap="Blues")

        ax.set_title(method)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))

        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    matrix[i, j],
                    ha="center",
                    va="center"
                )

    fig.colorbar(image, ax=axes.ravel().tolist())
    plt.suptitle(f"{dataset_name}: матрицы ошибок")
    plt.tight_layout()

    plt.savefig(
        f"results/{dataset_name}_confusion.png"
    )

    plt.close()


# ============================================================
# ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТА
# ============================================================

def run_experiment(dataset_name, X, y):
    print("\n" + "=" * 60)
    print(f"ДАТАСЕТ: {dataset_name}")
    print("=" * 60)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    input_size = X_train.shape[1]
    num_classes = len(encoder.classes_)

    results = {}

    methods = [
        "Без предобучения",
        "Автоэнкодеры",
        "RBM"
    ]

    # --------------------------------------------------------
    # 1. Обучение без предобучения
    # --------------------------------------------------------

    print("\n1. ОБУЧЕНИЕ БЕЗ ПРЕДОБУЧЕНИЯ")

    set_seed()

    model = Classifier(input_size, num_classes).to(DEVICE)

    loss = train_classifier(
        model,
        X_train,
        y_train
    )

    accuracy, f1, matrix = evaluate_model(
        model,
        X_test,
        y_test
    )

    results[methods[0]] = {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "matrix": matrix
    }

    # --------------------------------------------------------
    # 2. Предобучение автоэнкодерами
    # --------------------------------------------------------

    print("\n2. ПРЕДОБУЧЕНИЕ АВТОЭНКОДЕРАМИ")

    set_seed()

    model = Classifier(input_size, num_classes).to(DEVICE)

    model = pretrain_autoencoders(
        model,
        X_train
    )

    loss = train_classifier(
        model,
        X_train,
        y_train
    )

    accuracy, f1, matrix = evaluate_model(
        model,
        X_test,
        y_test
    )

    results[methods[1]] = {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "matrix": matrix
    }

    # --------------------------------------------------------
    # 3. Предобучение стеком RBM
    # --------------------------------------------------------

    print("\n3. ПРЕДОБУЧЕНИЕ СТЕКОМ RBM")

    set_seed()

    model = Classifier(input_size, num_classes).to(DEVICE)

    model = pretrain_rbm(
        model,
        X_train
    )

    loss = train_classifier(
        model,
        X_train,
        y_train
    )

    accuracy, f1, matrix = evaluate_model(
        model,
        X_test,
        y_test
    )

    results[methods[2]] = {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "matrix": matrix
    }

    # --------------------------------------------------------
    # Вывод результатов
    # --------------------------------------------------------

    print("\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("-" * 60)

    for method, result in results.items():
        print(f"\n{method}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1:       {result['f1']:.4f}")
        print("Матрица ошибок:")
        print(result["matrix"])

    # --------------------------------------------------------
    # Графики
    # --------------------------------------------------------

    plot_loss(results, dataset_name)
    plot_metrics(results, dataset_name)

    plot_confusion_matrices(
        results,
        dataset_name,
        encoder.classes_
    )

    return results


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    print("Устройство:", DEVICE)

    rice_X, rice_y = load_rice()
    maternal_X, maternal_y = load_maternal()

    rice_results = run_experiment(
        "Rice",
        rice_X,
        rice_y
    )

    maternal_results = run_experiment(
        "Maternal",
        maternal_X,
        maternal_y
    )

    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №4 ЗАВЕРШЕНА")
    print("Графики сохранены в папке results")
    print("=" * 60)
