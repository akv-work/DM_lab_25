import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА ДАТАСЕТА
# ============================================================
mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets

print(mushroom.metadata)
print(mushroom.variables)

# Целевая переменная: p (ядовитый) -> 1, e (съедобный) -> 0
y = y.iloc[:, 0].map({'p': 1, 'e': 0})

print("\nДанные:", X.shape)
print("Ядовитые:", (y == 1).sum(), f"({(y == 1).mean()*100:.1f}%)")
print("Съедобные:", (y == 0).sum())

# ============================================================
# 2. РАЗБИЕНИЕ И ONE-HOT ENCODING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns)
    ]
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

print(f"\nОбучающая: {X_train_enc.shape}, тестовая: {X_test_enc.shape}")
print(f"Ядовитых в тесте: {y_test.sum()}, съедобных: {(y_test == 0).sum()}")

# ============================================================
# 3. МОДЕЛИ
# ============================================================
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        eval_metric='logloss', random_state=42, n_jobs=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100, random_seed=42, verbose=0
    ),
}

# ============================================================
# 4. ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================
results = []
preds = {}

for name, model in models.items():
    model.fit(X_train_enc, y_train)
    y_pred = model.predict(X_test_enc)
    preds[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)

    results.append({
        'Модель': name,
        'Accuracy': acc,
        'Precision (ядовитый)': prec,
        'Recall (ядовитый)': rec,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Recall (ядовитый)', ascending=False)
print("\nМетрики:")
print(results_df.to_string(index=False))

print("\nМатрицы ошибок [[TN, FP], [FN, TP]] (TN — съедобные верно, TP — ядовитые верно):")
for name, y_pred in preds.items():
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"{name:<15} TN={cm[0,0]:4d}  FP={cm[0,1]:3d}  "
          f"FN={cm[1,0]:3d}  TP={cm[1,1]:4d}")

print("\nПодробный отчёт по классам для лучшей по Recall модели:")
best_name = results_df.iloc[0]['Модель']
print(f"Лучшая по Recall: {best_name}")
print(classification_report(y_test, preds[best_name],
                            target_names=['съедобный (0)', 'ядовитый (1)']))