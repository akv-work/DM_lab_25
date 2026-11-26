import pandas as pd
import numpy as np

from urllib.request import urlretrieve

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. ЗАГРУЗКА ДАТАСЕТА

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# Резервный источник, если основной недоступен
backup_url = "https://huggingface.co/datasets/mstz/mushroom/resolve/main/agaricus-lepiota.data"

try:
    print("\nЗагрузка датасета из UCI...")
    urlretrieve(url, "mushrooms.csv")
except Exception:
    print("Основной источник недоступен. Используем резервный...")
    urlretrieve(backup_url, "mushrooms.csv")

columns = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

data = pd.read_csv(
    "mushrooms.csv",
    header=None,
    names=columns,
    na_values="?"
)

print("Датасет успешно загружен!")
print("Размер датасета:", data.shape)

print("\nПервые 5 строк:")
print(data.head())

print("\nКоличество пропущенных значений:")
print(data.isnull().sum().sum())

# 2. ПРЕДОБРАБОТКА ДАННЫХ

print("ПРЕДОБРАБОТКА ДАННЫХ")

# Заполняем пропущенные значения
data = data.fillna("unknown")

# Целевой признак:
# p - ядовитый гриб (1)
# e - съедобный гриб (0)

y = (data["class"] == "p").astype(int)

# Удаляем целевой столбец из признаков
X = data.drop("class", axis=1)

# Преобразуем категориальные признаки в числовые
X = pd.get_dummies(X, dtype=int)

print("Преобразование категориальных признаков выполнено.")
print("Количество признаков после кодирования:", X.shape[1])

print("\nРаспределение классов:")
print("Съедобные грибы:", (y == 0).sum())
print("Ядовитые грибы:", (y == 1).sum())

# 3. РАЗДЕЛЕНИЕ ДАННЫХ

print("РАЗДЕЛЕНИЕ ДАННЫХ")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)

# 4. СОЗДАНИЕ МОДЕЛЕЙ

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1
    )
}

# 5. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ

print("ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ МОДЕЛЕЙ")

results = []

for name, model in models.items():

    print("\nМодель:", name)

    # Обучение
    model.fit(X_train, y_train)

    # Предсказание
    y_pred = model.predict(X_test)

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # Сохраняем результаты
    results.append({
        "Модель": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, y_pred))

# 6. СРАВНЕНИЕ РЕЗУЛЬТАТОВ

print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-score": "{:.4f}".format
        }
    )
)

# 7. ОПРЕДЕЛЕНИЕ ЛУЧШЕЙ МОДЕЛИ

print("ИТОГИ")

best_model = results_df.loc[results_df["Recall"].idxmax()]

print("Лучшая модель по полноте обнаружения ядовитых грибов:")
print(best_model["Модель"])

print(f"Accuracy:  {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall:    {best_model['Recall']:.4f}")
print(f"F1-score:  {best_model['F1-score']:.4f}")
