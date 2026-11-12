import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

DATA_PATH = r"D:\\Учёба\\4 курс\\ИАД\\5\\breast_cancer.csv"

df = pd.read_csv(DATA_PATH)

print("Первые 5 строк датасета:")
print(df.head())
print("\nИнформация о датасете до обработки:")
print(df.info())

cols_all_nan = df.columns[df.isna().all()].tolist()
if cols_all_nan:
    print(f"\nУдаляем полностью пустые колонки: {cols_all_nan}")
    df = df.drop(columns=cols_all_nan)

if 'id' in df.columns:
    print("\nУдаляем колонку id")
    df = df.drop(columns=['id'])

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols:
    if df[col].isna().any():
        median_val = df[col].median()
        print(f"Заполняем пропуски в числовой колонке '{col}' медианой: {median_val}")
        df[col] = df[col].fillna(median_val)

for col in categorical_cols:
    if df[col].isna().any():
        mode_val = df[col].mode()[0]
        print(f"Заполняем пропуски в категориальной колонке '{col}' модой: {mode_val}")
        df[col] = df[col].fillna(mode_val)

print("\nИнформация о датасете после обработки:")
print(df.info())

if df.isna().any().any():
    print("\nВнимание: в данных всё ещё остались пропуски!")
    print(df.isna().sum()[df.isna().sum() > 0])
else:
    print("\nПропусков в данных нет.")

target_column = 'diagnosis'
if target_column not in df.columns:
    print(f"\nКолонка '{target_column}' не найдена. Доступные колонки: {df.columns.tolist()}")
    target_column = df.columns[-1]
    print(f"Используем последнюю колонку как целевую: '{target_column}'")

if not pd.api.types.is_numeric_dtype(df[target_column]):
    unique_vals = sorted(df[target_column].dropna().unique().tolist())
    print(f"\nУникальные значения целевой переменной: {unique_vals}")
    if set(unique_vals) == {'B', 'M'}:
        df[target_column] = df[target_column].replace({'M': 1, 'B': 0})
        print("Целевая переменная преобразована: M -> 1, B -> 0")
    else:
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[target_column] = df[target_column].replace(mapping)
        print(f"Целевая переменная преобразована через автоматический маппинг: {mapping}")

df[target_column] = df[target_column].astype(int)

X = df.drop(columns=[target_column])
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

results = {}

for name, model in models.items():
    print(f"\nОбучение модели: {name}")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Время обучения: {train_time:.2f} сек")
    print(f"Accuracy: {acc:.4f}")

    print("\nМатрица ошибок:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nОтчет по метрикам:")
    print(classification_report(y_test, y_pred, target_names=['Доброкачественная (0)', 'Злокачественная (1)']))

print("\nСравнение моделей по Accuracy:")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\nЛучшая модель по Accuracy: {best_model} ({results[best_model]:.4f})")

print("\nАнализ Recall для класса Злокачественная:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall_1 = report['1']['recall']
    print(f"{name}: Recall = {recall_1:.4f}")