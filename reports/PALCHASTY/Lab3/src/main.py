import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, models, callbacks, optimizers
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error,
                             accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from scipy.io import arff
import ucimlrepo


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def choose_dataset() -> str:

    print("Выбор датасета")
    print("1 - Forest Fires (регрессия)")
    print("2 - Rice (классификация)")

    while True:
        try:
            choice = input("Введите номер датасета (1 или 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Прерывание ввода. Выбран Rice (2).")
            return "classification"

        if choice == "1":
            print("Выбран Forest Fires")
            return "regression"
        if choice == "2":
            print("Выбран Rice")
            return "classification"

        print("Некорректный ввод. Введите 1 или 2.")


def load_forest_fires(path: str = "forestfires.csv") -> pd.DataFrame:

    if not os.path.exists(path):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "forest-fires/forestfires.csv")
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df


def load_rice(path: str = "Rice_Cammeo_Osmancik.arff") -> pd.DataFrame:
    """Загружает датасет Rice.

    Сначала пробует локальный .arff-файл. Если его нет — скачивает датасет
    через официальный пакет ucimlrepo (новая платформа UCI).
    """

    if os.path.exists(path):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
    else:
        print("Локальный arff не найден, скачиваю через ucimlrepo ...")
        dataset = ucimlrepo.fetch_ucirepo(id=545)

        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        df = X.copy()
        df["Class"] = y.values

        # На всякий случай сохраняем локально в CSV
        df.to_csv("Rice_Cammeo_Osmancik.csv", index=False)

    # Приводим bytes-поля к str (для arff-ветки)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
            )

    return df


def preprocess_regression(df: pd.DataFrame):

    y = np.log1p(df["area"].values.astype(np.float64))
    X = df.drop(columns=["area"])

    categorical_features = ["month", "day"]
    numeric_features = ["X", "Y", "FFMC", "DMC", "DC",
                        "ISI", "temp", "RH", "wind", "rain"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    X_train_prep = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_prep = preprocessor.transform(X_test).astype(np.float32)

    return (X_train_prep, X_test_prep,
            y_train.astype(np.float32),
            y_test.astype(np.float32),
            preprocessor, None)


def preprocess_classification(df: pd.DataFrame, target_col: str = "Class"):

    X = df.drop(columns=[target_col])
    y_raw = df[target_col].values

    classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    y_idx = np.array([class_to_idx[c] for c in y_raw], dtype=np.int32)
    y_onehot = tf.keras.utils.to_categorical(y_idx, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=SEED, stratify=y_idx
    )

    scaler = StandardScaler()
    X_train_prep = scaler.fit_transform(X_train).astype(np.float32)
    X_test_prep = scaler.transform(X_test).astype(np.float32)

    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    return (X_train_prep, X_test_prep,
            y_train, y_test,
            (scaler, classes),
            num_classes,
            y_train_labels, y_test_labels)


def evaluate_regression(model, X_test, y_test, name="model"):

    y_pred_log = model.predict(X_test, verbose=0).flatten()
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0

    print(name)
    print("MSE:", mse)
    print("MAE:", mae)
    print("MAPE:", mape)

    return {"mse": mse, "mae": mae, "mape": mape,
            "y_pred": y_pred, "y_true": y_true}


def evaluate_classification(model, X_test, y_test_onehot, y_test_labels,
                            class_names, name="model"):

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    f1_macro = f1_score(y_test_labels, y_pred_labels,
                        average="macro", zero_division=0)
    f1_weighted = f1_score(y_test_labels, y_pred_labels,
                           average="weighted", zero_division=0)

    print(name)
    print("Accuracy:", acc)
    print("F1 macro:", f1_macro)
    print("F1 weighted:", f1_weighted)
    print("Classification report:")
    print(classification_report(y_test_labels, y_pred_labels,
                                target_names=[str(c) for c in class_names],
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))

    return {"accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "y_pred": y_pred_labels,
            "y_true": y_test_labels}


def build_baseline_regression(input_dim: int) -> models.Model:

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", name="h1"),
        layers.Dense(32, activation="relu", name="h2"),
        layers.Dense(16, activation="relu", name="h3"),
        layers.Dense(1, name="out"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_baseline_classification(input_dim: int,
                                  num_classes: int) -> models.Model:

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", name="h1"),
        layers.Dense(32, activation="relu", name="h2"),
        layers.Dense(16, activation="relu", name="h3"),
        layers.Dense(num_classes, activation="softmax", name="out"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_baseline(X_train, y_train, input_dim, task,
                   num_classes=None, epochs=200, batch_size=32):

    if task == "regression":
        model = build_baseline_regression(input_dim)
    else:
        model = build_baseline_classification(input_dim, num_classes)

    early = callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early],
    )
    return model, history


def train_autoencoder(X: np.ndarray, hidden_dim: int,
                      epochs: int = 100, batch_size: int = 32,
                      learning_rate: float = 1e-3):
    input_dim = X.shape[1]

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(hidden_dim, activation="relu",
                           name="encoder")(input_layer)
    decoded = layers.Dense(input_dim, activation="linear",
                           name="decoder")(encoded)

    autoencoder = models.Model(input_layer, decoded,
                               name=f"ae_{input_dim}_{hidden_dim}")
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    early = callbacks.EarlyStopping(
        monitor="loss", patience=15, restore_best_weights=True
    )
    autoencoder.fit(
        X, X, epochs=epochs, batch_size=batch_size,
        verbose=0, callbacks=[early],
    )

    encoder = models.Model(input_layer, encoded,
                           name=f"enc_{input_dim}_{hidden_dim}")
    return encoder, autoencoder


def pretrain_layerwise(X_train: np.ndarray,
                       hidden_dims=(64, 32, 16),
                       epochs_per_layer: int = 100):

    encoders = []
    current = X_train
    for i, dim in enumerate(hidden_dims):
        print("Предобучение слоя", i + 1, "dim =", dim)
        encoder, ae = train_autoencoder(
            current, hidden_dim=dim, epochs=epochs_per_layer
        )
        encoders.append(encoder)
        current = encoder.predict(current, verbose=0)
        print("Выход автоэнкодера:", current.shape)

    return encoders


def build_pretrained_regression(encoders, input_dim: int) -> models.Model:

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i, enc in enumerate(encoders):
        dense_layer = enc.layers[1]
        w, b = dense_layer.get_weights()
        model.add(layers.Dense(
            w.shape[1],
            activation="relu",
            kernel_initializer=Constant(w),
            bias_initializer=Constant(b),
            name=f"h{i+1}",
        ))

    model.add(layers.Dense(1, name="out"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_pretrained_classification(encoders, input_dim: int,
                                    num_classes: int) -> models.Model:

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i, enc in enumerate(encoders):
        dense_layer = enc.layers[1]
        w, b = dense_layer.get_weights()
        model.add(layers.Dense(
            w.shape[1],
            activation="relu",
            kernel_initializer=Constant(w),
            bias_initializer=Constant(b),
            name=f"h{i+1}",
        ))

    model.add(layers.Dense(num_classes, activation="softmax", name="out"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_with_pretraining(X_train, y_train, input_dim, task,
                           num_classes=None,
                           hidden_dims=(64, 32, 16),
                           epochs_per_layer: int = 100,
                           fine_tune_epochs: int = 200,
                           batch_size: int = 32):

    encoders = pretrain_layerwise(
        X_train, hidden_dims=hidden_dims,
        epochs_per_layer=epochs_per_layer,
    )

    if task == "regression":
        model = build_pretrained_regression(encoders, input_dim)
    else:
        model = build_pretrained_classification(encoders, input_dim,
                                                num_classes)

    early = callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early],
    )
    return model, history, encoders


def plot_history(hist_no, hist_pre, task,
                 save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ylabel = "MSE" if task == "regression" else "Categorical CE"

    axes[0].plot(hist_no.history["loss"], label="train")
    axes[0].plot(hist_no.history["val_loss"], label="val")
    axes[0].set_title("Без предобучения")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel(ylabel)
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(hist_pre.history["loss"], label="train")
    axes[1].plot(hist_pre.history["val_loss"], label="val")
    axes[1].set_title("С предобучением")
    axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel(ylabel)
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Сохранено:", save_path)


def plot_confusion_matrix(res_no, res_pre, class_names,
                          save_path="confusion_matrices.png"):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, res, title in zip(
        axes,
        [res_no, res_pre],
        ["Без предобучения", "С предобучением"],
    ):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Сохранено:", save_path)


def save_summary_table_regression(res_no, res_pre,
                                  save_path="summary.csv"):
    df = pd.DataFrame({
        "Метрика": ["MSE", "MAE", "MAPE, %"],
        "Без предобучения": [res_no["mse"], res_no["mae"], res_no["mape"]],
        "С предобучением":  [res_pre["mse"], res_pre["mae"], res_pre["mape"]],
    })
    df.to_csv(save_path, index=False)
    print("Сохранено:", save_path)
    print(df.to_string(index=False))


def save_summary_table_classification(res_no, res_pre,
                                      save_path="summary.csv"):
    df = pd.DataFrame({
        "Метрика": ["Accuracy", "F1 (macro)", "F1 (weighted)"],
        "Без предобучения": [res_no["accuracy"],
                             res_no["f1_macro"],
                             res_no["f1_weighted"]],
        "С предобучением":  [res_pre["accuracy"],
                             res_pre["f1_macro"],
                             res_pre["f1_weighted"]],
    })
    df.to_csv(save_path, index=False)
    print("Сохранено:", save_path)
    print(df.to_string(index=False))


def main():

    task = choose_dataset()

    print("Задача:", task)

    if task == "regression":

        df = load_forest_fires()
        (X_train, X_test, y_train, y_test,
         preprocessor, _) = preprocess_regression(df)
        input_dim = X_train.shape[1]
        print("Датасет: Forest Fires")
        print("Обучающая:", X_train.shape)
        print("Тестовая:", X_test.shape)
        print("Размерность:", input_dim)

        print("Шаг 1. Обучение без предобучения")
        model_no, hist_no = train_baseline(
            X_train, y_train, input_dim, task="regression",
            epochs=200, batch_size=32
        )
        res_no = evaluate_regression(model_no, X_test, y_test,
                                     name="Без предобучения")

        print("Шаг 2. Обучение с предобучением")
        model_pre, hist_pre, _ = train_with_pretraining(
            X_train, y_train, input_dim, task="regression",
            hidden_dims=(64, 32, 16),
            epochs_per_layer=100,
            fine_tune_epochs=200,
            batch_size=32,
        )
        res_pre = evaluate_regression(model_pre, X_test, y_test,
                                      name="С предобучением")

        plot_history(hist_no, hist_pre, task="regression")
        save_summary_table_regression(res_no, res_pre)

        d_mae = res_no["mae"] - res_pre["mae"]
        d_mape = res_no["mape"] - res_pre["mape"]
        print("Снижение MAE:", d_mae)
        print("Снижение MAPE:", d_mape)

    else:

        df = load_rice()
        (X_train, X_test, y_train, y_test,
         aux, num_classes,
         y_train_labels, y_test_labels) = preprocess_classification(
            df, target_col="Class"
        )
        input_dim = X_train.shape[1]
        class_names = [str(c) for c in aux[1]]

        print("Датасет: Rice")
        print("Обучающая:", X_train.shape)
        print("Тестовая:", X_test.shape)
        print("Размерность:", input_dim)
        print("Классов:", num_classes, class_names)

        print("Шаг 1. Обучение без предобучения")
        model_no, hist_no = train_baseline(
            X_train, y_train, input_dim, task="classification",
            num_classes=num_classes, epochs=200, batch_size=64
        )
        res_no = evaluate_classification(
            model_no, X_test, y_test, y_test_labels,
            class_names, name="Без предобучения"
        )

        print("Шаг 2. Обучение с предобучением")
        model_pre, hist_pre, _ = train_with_pretraining(
            X_train, y_train, input_dim, task="classification",
            num_classes=num_classes,
            hidden_dims=(64, 32, 16),
            epochs_per_layer=100,
            fine_tune_epochs=200,
            batch_size=64,
        )
        res_pre = evaluate_classification(
            model_pre, X_test, y_test, y_test_labels,
            class_names, name="С предобучением"
        )

        plot_history(hist_no, hist_pre, task="classification")
        plot_confusion_matrix(res_no, res_pre, class_names)
        save_summary_table_classification(res_no, res_pre)

        d_acc = res_pre["accuracy"] - res_no["accuracy"]
        d_f1 = res_pre["f1_macro"] - res_no["f1_macro"]
        print("Прирост Accuracy:", d_acc)
        print("Прирост F1 macro:", d_f1)


if __name__ == "__main__":
    main()