from __future__ import annotations

import argparse
import json
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def load_adult_dataset(zip_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Загружает adult.data и adult.test из официального ZIP-архива UCI."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Не найден архив: {zip_path}\n"
            "Положите adult.zip рядом со скриптом или передайте путь через --zip."
        )

    with zipfile.ZipFile(zip_path) as archive:
        file_names = set(archive.namelist())
        required = {"adult.data", "adult.test"}
        if not required.issubset(file_names):
            raise ValueError(
                "В архиве должны находиться adult.data и adult.test. "
                f"Найдены файлы: {sorted(file_names)}"
            )

        with archive.open("adult.data") as file:
            train_part = pd.read_csv(
                file,
                header=None,
                names=COLUMNS,
                skipinitialspace=True,
                na_values="?",
                skip_blank_lines=True,
            )

        with archive.open("adult.test") as file:
            test_part = pd.read_csv(
                file,
                header=None,
                names=COLUMNS,
                skipinitialspace=True,
                na_values="?",
                skiprows=1,  # Первая строка adult.test — служебный заголовок
                skip_blank_lines=True,
            )

    data = pd.concat([train_part, test_part], ignore_index=True)

    # Удаляем пробелы по краям строковых значений.
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()

    # В adult.test метки заканчиваются точкой: '>50K.' и '<=50K.'.
    target_raw = data.pop("income").astype("string").str.rstrip(".")
    target = target_raw.map({"<=50K": 0, ">50K": 1})

    if target.isna().any():
        unknown = target_raw[target.isna()].dropna().unique().tolist()
        raise ValueError(f"Неизвестные значения целевого признака: {unknown}")

    return data, target.astype("int8")


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Медиана для числовых, мода + One-Hot для категориальных признаков."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def save_plots(
    results: pd.DataFrame,
    y: pd.Series,
    y_test: pd.Series,
    best_predictions: np.ndarray,
    output_dir: Path,
) -> None:
    """Строит и сохраняет графики эксперимента."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Распределение целевых классов.
    class_counts = y.map({0: "<=50K", 1: ">50K"}).value_counts().reindex(["<=50K", ">50K"])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(class_counts.index, class_counts.values)
    ax.set_title("Распределение классов дохода")
    ax.set_xlabel("Годовой доход")
    ax.set_ylabel("Количество записей")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}".replace(",", " "),
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "01_class_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Сравнение Precision.
    plot_df = results.sort_values("Precision (>50K)", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(plot_df["Модель"], plot_df["Precision (>50K)"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Precision для класса >50K")
    ax.set_title("Сравнение моделей по Precision")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, plot_df["Precision (>50K)"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "02_precision_comparison.png", dpi=160)
    plt.close(fig)

    # 3. Precision, Recall и F1.
    metrics = results.sort_values("Precision (>50K)", ascending=False)
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, column, label in [
        (-1, "Precision (>50K)", "Precision"),
        (0, "Recall (>50K)", "Recall"),
        (1, "F1 (>50K)", "F1-score"),
    ]:
        ax.bar(x + offset * width, metrics[column], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Модель"], rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение Precision, Recall и F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "03_metrics_comparison.png", dpi=160)
    plt.close(fig)

    # 4. Матрица ошибок для модели с максимальной Precision.
    cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["<=50K", ">50K"],
    ).plot(ax=ax, cmap="Blues", values_format=",d", colorbar=False)
    ax.set_title("Матрица ошибок лучшей модели по Precision")
    fig.tight_layout()
    fig.savefig(output_dir / "04_best_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 5. Время обучения.
    time_df = results.sort_values("Время обучения, сек", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(time_df["Модель"], time_df["Время обучения, сек"])
    ax.set_xlabel("Время обучения, сек")
    ax.set_title("Сравнение времени обучения моделей")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, time_df["Время обучения, сек"]):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "05_training_time_comparison.png", dpi=160)
    plt.close(fig)

    # 6. Компромисс Precision и времени обучения.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(results["Время обучения, сек"], results["Precision (>50K)"])
    for _, row in results.iterrows():
        ax.annotate(
            row["Модель"],
            (row["Время обучения, сек"], row["Precision (>50K)"]),
            xytext=(5, 5), textcoords="offset points",
        )
    ax.set_xlabel("Время обучения, сек")
    ax.set_ylabel("Precision для класса >50K")
    ax.set_ylim(0, 1)
    ax.set_title("Precision и время обучения")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "06_precision_time_tradeoff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult Census Income — вариант 10")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        default="adult.zip",
        help="Путь к архиву adult.zip (по умолчанию: adult.zip)",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default="adult_results",
        help="Каталог для сохранения результатов (по умолчанию: adult_results)",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Adult Census Income — лабораторная работа, вариант 10")
    print("=" * 72)

    X, y = load_adult_dataset(zip_path)
    print(f"Размер датасета: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print("Распределение классов:")
    print(y.map({0: "<=50K", 1: ">50K"}).value_counts())
    print(f"Количество пропущенных значений: {int(X.isna().sum().sum())}")

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

    # Stratify сохраняет доли классов в train и test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nОбучающая выборка: {len(X_train)} объектов")
    print(f"Тестовая выборка:  {len(X_test)} объектов")

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    # Единый набор параметров для воспроизведения эксперимента.
    models = {
        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=2,
            )),
        ]),
        "AdaBoost": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=1, random_state=RANDOM_STATE
                ),
                n_estimators=120,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=4,
            )),
        ]),
    }

    # CatBoost получает категориальные признаки без One-Hot Encoding.
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for column in categorical_columns:
        X_train_cb[column] = (
            X_train_cb[column]
            .astype(object)
            .where(X_train_cb[column].notna(), "__MISSING__")
            .astype(str)
        )
        X_test_cb[column] = (
            X_test_cb[column]
            .astype(object)
            .where(X_test_cb[column].notna(), "__MISSING__")
            .astype(str)
        )

    catboost_model = CatBoostClassifier(
        iterations=150,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )

    trained_models = {}
    training_times = {}

    for name, model in models.items():
        print(f"\nОбучение модели: {name}")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_times[name] = time.perf_counter() - start
        trained_models[name] = model
        print(f"Готово. Время обучения: {training_times[name]:.2f} сек.")

    print("\nОбучение модели: CatBoost")
    start = time.perf_counter()
    catboost_model.fit(
        X_train_cb,
        y_train,
        cat_features=categorical_columns,
    )
    training_times["CatBoost"] = time.perf_counter() - start
    trained_models["CatBoost"] = catboost_model
    print(f"Готово. Время обучения: {training_times['CatBoost']:.2f} сек.")

    # Оценка всех моделей на одной тестовой выборке.
    result_rows = []
    predictions: dict[str, np.ndarray] = {}

    for name, model in trained_models.items():
        if name == "CatBoost":
            predicted = np.asarray(model.predict(X_test_cb)).reshape(-1).astype(int)
        else:
            predicted = np.asarray(model.predict(X_test)).reshape(-1).astype(int)

        predictions[name] = predicted
        result_rows.append({
            "Модель": name,
            "Precision (>50K)": precision_score(
                y_test, predicted, pos_label=1, zero_division=0
            ),
            "Recall (>50K)": recall_score(
                y_test, predicted, pos_label=1, zero_division=0
            ),
            "F1 (>50K)": f1_score(
                y_test, predicted, pos_label=1, zero_division=0
            ),
            "Accuracy": accuracy_score(y_test, predicted),
            "Время обучения, сек": training_times[name],
        })

    results = pd.DataFrame(result_rows).sort_values(
        "Precision (>50K)", ascending=False
    ).reset_index(drop=True)

    print("\nРЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(results.to_string(index=False, formatters={
        "Precision (>50K)": "{:.4f}".format,
        "Recall (>50K)": "{:.4f}".format,
        "F1 (>50K)": "{:.4f}".format,
        "Accuracy": "{:.4f}".format,
        "Время обучения, сек": "{:.2f}".format,
    }))

    # Сохранение таблиц и отчёта лучшей модели.
    results.to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")

    best_name = str(results.loc[0, "Модель"])
    best_predictions = predictions[best_name]
    cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Истинный <=50K", "Истинный >50K"],
        columns=["Предсказан <=50K", "Предсказан >50K"],
    ).to_csv(output_dir / "confusion_matrix_best_model.csv", encoding="utf-8-sig")

    report_text = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        zero_division=0,
    )
    (output_dir / "classification_report_best_model.txt").write_text(
        f"Лучшая модель по Precision: {best_name}\n\n{report_text}",
        encoding="utf-8",
    )

    best_row = results.iloc[0]
    best_recall_row = results.loc[results["Recall (>50K)"].idxmax()]
    best_f1_row = results.loc[results["F1 (>50K)"].idxmax()]
    best_accuracy_row = results.loc[results["Accuracy"].idxmax()]

    summary = {
        "dataset": "Adult Census Income",
        "n_objects": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "positive_class": ">50K",
        "random_state": RANDOM_STATE,
        "test_size_fraction": 0.20,
        "best_by_precision": str(best_row["Модель"]),
        "best_precision": float(best_row["Precision (>50K)"]),
        "best_by_recall": str(best_recall_row["Модель"]),
        "best_recall": float(best_recall_row["Recall (>50K)"]),
        "best_by_f1": str(best_f1_row["Модель"]),
        "best_f1": float(best_f1_row["F1 (>50K)"]),
        "best_by_accuracy": str(best_accuracy_row["Модель"]),
        "best_accuracy": float(best_accuracy_row["Accuracy"]),
        "models": results.to_dict(orient="records"),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_plots(results, y, y_test, best_predictions, output_dir)

    print(f"\nЛучшая модель по Precision: {best_name}")
    print(f"Precision (>50K): {best_row['Precision (>50K)']:.4f}")
    print(f"Все таблицы, выводы и графики сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
