import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# -------------------- НАСТРОЙКИ --------------------

SEED = 42
BATCH_SIZE = 64

CLASSIFIER_EPOCHS = 100
AUTOENCODER_EPOCHS = 50

LEARNING_RATE = 0.001

HIDDEN_LAYERS = [64, 32, 16]

BASE_DIR = Path(__file__).resolve().parent

RICE_FILE = BASE_DIR / "Rice_Cammeo_Osmancik.arff"
MATERNAL_FILE = BASE_DIR / "Maternal Health Risk Data Set.csv"

RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- ПОДГОТОВКА --------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rice():
    if not RICE_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл: {RICE_FILE.name}"
        )

    data, metadata = arff.loadarff(RICE_FILE)
    df = pd.DataFrame(data)

    df["Class"] = df["Class"].apply(
        lambda value: value.decode("utf-8")
        if isinstance(value, bytes)
        else value
    )

    X = df.drop(columns=["Class"]).astype(float)
    y = df["Class"].astype(str)

    return X, y


def load_maternal():
    if not MATERNAL_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл: {MATERNAL_FILE.name}"
        )

    df = pd.read_csv(MATERNAL_FILE, encoding="utf-8-sig")

    feature_columns = [
        "Age",
        "SystolicBP",
        "DiastolicBP",
        "BS",
        "BodyTemp",
        "HeartRate"
    ]

    X = df[feature_columns].astype(float)
    y = df["RiskLevel"].astype(str).str.strip()

    return X, y


# -------------------- НЕЙРОННАЯ СЕТЬ --------------------

class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        dimensions = [input_size] + HIDDEN_LAYERS

        self.hidden_layers = nn.ModuleList([
            nn.Linear(dimensions[i], dimensions[i + 1])
            for i in range(len(dimensions) - 1)
        ])

        self.output_layer = nn.Linear(
            HIDDEN_LAYERS[-1],
            output_size
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        return self.output_layer(x)


# -------------------- АВТОЭНКОДЕР --------------------

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)

        return decoded


def pretrain_autoencoders(model, X_train):
    print("\nПослойное предобучение автоэнкодерами:")

    current_data = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    dimensions = [X_train.shape[1]] + HIDDEN_LAYERS

    for i in range(len(HIDDEN_LAYERS)):
        input_size = dimensions[i]
        hidden_size = dimensions[i + 1]

        print(
            f"Обучение автоэнкодера "
            f"{i + 1}: {input_size} -> {hidden_size} -> {input_size}"
        )

        dataset = TensorDataset(current_data)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        autoencoder = Autoencoder(
            input_size,
            hidden_size
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=LEARNING_RATE
        )

        criterion = nn.MSELoss()

        autoencoder.train()

        for epoch in range(AUTOENCODER_EPOCHS):
            total_loss = 0

            for (batch,) in loader:
                batch = batch.to(DEVICE)

                optimizer.zero_grad()

                reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.size(0)

            average_loss = total_loss / len(dataset)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Эпоха {epoch + 1}: "
                    f"ошибка = {average_loss:.6f}"
                )

        model_layer = model.hidden_layers[i]

        model_layer.weight.data.copy_(
            autoencoder.encoder.weight.data
        )
        model_layer.bias.data.copy_(
            autoencoder.encoder.bias.data
        )

        autoencoder.eval()

        with torch.no_grad():
            current_data = torch.relu(
                autoencoder.encoder(
                    current_data.to(DEVICE)
                )
            ).cpu()

    print("Послойное предобучение завершено.")


# -------------------- ОБУЧЕНИЕ КЛАССИФИКАТОРА --------------------

def train_classifier(model, X_train, y_train, title):
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # Сохранение ошибки на каждой эпохе
    loss_history = []

    print(f"\nОбучение: {title}")

    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        average_loss = total_loss / len(dataset)
        loss_history.append(average_loss)

        if (epoch + 1) % 20 == 0:
            print(
                f"Эпоха {epoch + 1}: "
                f"ошибка = {average_loss:.6f}"
            )

    return loss_history


# -------------------- ОЦЕНКА МОДЕЛИ --------------------

def evaluate_model(model, X_test, y_test, classes, title):
    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(classes))
    )

    print(f"\nРезультаты: {title}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nМатрица ошибок:")
    print("Порядок классов:", list(classes))
    print(matrix)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "matrix": matrix
    }


# -------------------- ПОСТРОЕНИЕ ГРАФИКОВ --------------------

def plot_loss(dataset_name, basic_loss, pretrained_loss):
    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(basic_loss) + 1),
        basic_loss,
        label="Без предобучения"
    )

    plt.plot(
        range(1, len(pretrained_loss) + 1),
        pretrained_loss,
        label="С предобучением"
    )

    plt.title(f"{dataset_name}: изменение функции потерь")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    filename = RESULTS_DIR / f"{dataset_name}_loss.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_metrics(dataset_name, results):
    approaches = list(results.keys())

    accuracy = [
        results[name]["accuracy"]
        for name in approaches
    ]

    f1 = [
        results[name]["f1"]
        for name in approaches
    ]

    x = np.arange(len(approaches))
    width = 0.35

    plt.figure(figsize=(9, 6))

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
        label="F1-score"
    )

    plt.xticks(
        x,
        approaches,
        rotation=10
    )

    plt.ylim(0, 1)
    plt.ylabel("Значение метрики")
    plt.title(f"{dataset_name}: сравнение качества классификации")
    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()

    filename = RESULTS_DIR / f"{dataset_name}_metrics.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_confusion_matrix(dataset_name, approach, matrix, classes):
    plt.figure(figsize=(7, 6))

    plt.imshow(matrix, interpolation="nearest", cmap="Blues")

    plt.title(f"{dataset_name}: {approach}")
    plt.colorbar()

    positions = np.arange(len(classes))

    plt.xticks(
        positions,
        classes,
        rotation=30,
        ha="right"
    )

    plt.yticks(positions, classes)

    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")

    plt.tight_layout()

    filename = (
        RESULTS_DIR /
        f"{dataset_name}_{approach}_confusion.png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()


# -------------------- ЭКСПЕРИМЕНТ --------------------

def run_experiment(dataset_name, X, y):
    print("\n" + "=" * 60)
    print(f"ДАТАСЕТ: {dataset_name}")
    print("=" * 60)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    classes = encoder.classes_
    input_size = X.shape[1]
    output_size = len(classes)

    print(f"Количество объектов: {len(X)}")
    print(f"Количество признаков: {input_size}")
    print(f"Количество классов: {output_size}")
    print(f"Классы: {list(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y_encoded,
        test_size=0.2,
        random_state=SEED,
        stratify=y_encoded
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    loss_histories = {}

    # -------- Подход 1: обучение с нуля --------

    set_seed(SEED)

    model_basic = Classifier(
        input_size,
        output_size
    ).to(DEVICE)

    loss_histories["Без предобучения"] = train_classifier(
        model_basic,
        X_train,
        y_train,
        "Обучение с нуля"
    )

    results["Без предобучения"] = evaluate_model(
        model_basic,
        X_test,
        y_test,
        classes,
        "Обучение с нуля"
    )

    # -------- Подход 2: предобучение автоэнкодерами --------

    set_seed(SEED)

    model_pretrained = Classifier(
        input_size,
        output_size
    ).to(DEVICE)

    pretrain_autoencoders(
        model_pretrained,
        X_train
    )

    loss_histories["С предобучением"] = train_classifier(
        model_pretrained,
        X_train,
        y_train,
        "Обучение после предобучения"
    )

    results["С предобучением"] = evaluate_model(
        model_pretrained,
        X_test,
        y_test,
        classes,
        "Обучение после предобучения"
    )

    # Построение и сохранение графиков
    plot_loss(
        dataset_name,
        loss_histories["Без предобучения"],
        loss_histories["С предобучением"]
    )

    plot_metrics(
        dataset_name,
        results
    )

    for approach, result in results.items():
        plot_confusion_matrix(
            dataset_name,
            approach,
            result["matrix"],
            classes
        )

    return results


# -------------------- ЗАПУСК --------------------

def main():
    set_seed(SEED)

    RESULTS_DIR.mkdir(exist_ok=True)

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

    # Итоговое сравнение
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 60)

    for dataset_name, results in [
        ("Rice Cammeo/Osmancik", rice_results),
        ("Maternal Health Risk", maternal_results)
    ]:
        print(f"\n{dataset_name}")

        print(
            f"{'Подход':<25}"
            f"{'Accuracy':<15}"
            f"{'F1-score':<15}"
        )

        for approach, result in results.items():
            print(
                f"{approach:<25}"
                f"{result['accuracy']:<15.4f}"
                f"{result['f1']:<15.4f}"
            )

    print(f"\nГрафики сохранены в папке: {RESULTS_DIR}")


if __name__ == "__main__":
    main()