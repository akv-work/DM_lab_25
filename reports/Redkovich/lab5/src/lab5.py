import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             confusion_matrix, log_loss)

# Проверка дополнительных библиотек бустинга (при наличии)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

SEED = 42
np.random.seed(SEED)


def load_glass_data(file_path="glass.csv"):
    if not os.path.exists(file_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        cols = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
        df = pd.read_csv(url, header=None, names=cols).drop(columns=["ID"])
    else:
        df = pd.read_csv(file_path)
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
    
    X = df.drop(columns=["Type"]).values
    y_raw = df["Type"].values
    return X, y_raw


def plot_confusion_matrices(results, class_names, save_path="confusion_matrices_glass.png"):
    """Построение и сохранение матриц ошибок для всех моделей"""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Предсказанный класс", fontsize=8)
        ax.set_ylabel("Истинный класс", fontsize=8)
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

        # Подпись значений внутри матрицы
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Матрицы ошибок сохранены в файл: {save_path}")


def plot_training_curves(training_histories, save_path="training_curves_glass.png"):
    """Построение графиков изменения функции потерь (Loss) в зависимости от числа деревьев (как в ЛР 4)"""
    fig, axes = plt.subplots(1, len(training_histories), figsize=(4 * len(training_histories), 4))
    if len(training_histories) == 1:
        axes = [axes]

    for ax, (name, hist) in zip(axes, training_histories.items()):
        if "train_loss" in hist and len(hist["train_loss"]) > 0:
            ax.plot(hist["train_loss"], label="train loss")
            ax.plot(hist["test_loss"], label="test loss")
            ax.set_title(f"{name} (Loss)")
            ax.set_xlabel("Количество деревьев (n_estimators)")
            ax.set_ylabel("Log Loss")
            ax.legend(fontsize=8)
            ax.grid(True)
        else:
            # Для DecisionTree, у которого нет этапов обучения по деревьям
            ax.text(0.5, 0.5, "Одиночное дерево\n(без итераций)", ha="center", va="center", fontsize=10)
            ax.set_title(name)
            ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Графики обучения сохранены в файл: {save_path}")


def main():
    # 1. Загрузка данных (Вариант 9 - Glass Identification)
    X, y_raw = load_glass_data()

    # Кодирование классов в диапазон [0, n_classes-1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [f"Тип {c}" for c in label_encoder.classes_]

    print("=" * 80)
    print(f"Лабораторная работа №5 | Датасет: Glass Identification (Вариант 9)")
    print(f"Размер выборки: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Уникальные классы стекла: {label_encoder.classes_}")
    print("=" * 80)

    # 2. Разделение на обучающую и тестовую выборки (80/20 со стратификацией)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Стандартизация признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Инициализация моделей по заданию ЛР 5
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=SEED, algorithm="SAMME"),
    }

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(n_estimators=100, random_state=SEED, verbose=0)
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, random_state=SEED)

    # 4. Обучение моделей, сбор метрик и кривых обучения
    results = {}
    training_histories = {}
    summary_data = []

    for name, model in models.items():
        # Сбор историй обучения для построения графиков
        train_loss_hist = []
        test_loss_hist = []

        if name == "DecisionTree":
            model.fit(X_train_scaled, y_train)
        elif name == "RandomForest":
            # Имитация пошагового обучения по числу деревьев
            rf_temp = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=SEED)
            for n in range(1, 101):
                rf_temp.n_estimators = n
                rf_temp.fit(X_train_scaled, y_train)
                train_loss_hist.append(log_loss(y_train, rf_temp.predict_proba(X_train_scaled)))
                test_loss_hist.append(log_loss(y_test, rf_temp.predict_proba(X_test_scaled)))
            model = rf_temp
        elif name in ["AdaBoost", "GradientBoosting", "CatBoost"]:
            model.fit(X_train_scaled, y_train)
            # Извлечение поэтапных предсказаний (staged_predict_proba)
            try:
                for y_pred_train in model.staged_predict_proba(X_train_scaled):
                    train_loss_hist.append(log_loss(y_train, y_pred_train))
                for y_pred_test in model.staged_predict_proba(X_test_scaled):
                    test_loss_hist.append(log_loss(y_test, y_pred_test))
            except Exception:
                pass

        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n==================== Модель: {name} ====================")
        print(f"Accuracy: {acc:.4f} | F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_weighted:.4f}\n")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm
        }

        training_histories[name] = {
            "train_loss": train_loss_hist,
            "test_loss": test_loss_hist
        }

        summary_data.append({
            "Модель": name,
            "Accuracy": round(acc, 4),
            "F1 (macro)": round(f1_macro, 4),
            "F1 (weighted)": round(f1_weighted, 4)
        })

    # 5. Сохранение сводной таблицы в CSV
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("summary_glass_lr5.csv", index=False)

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА МЕТРИК (ЛР 5)")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("\n[INFO] Сводная таблица сохранена в файл: summary_glass_lr5.csv")

    # 6. Построение и сохранение графиков
    plot_confusion_matrices(results, class_names, save_path="confusion_matrices_glass.png")
    plot_training_curves(training_histories, save_path="training_curves_glass.png")


if __name__ == "__main__":
    main()