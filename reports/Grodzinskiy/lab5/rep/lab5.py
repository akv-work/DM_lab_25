from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

digits = load_digits()
X = digits.data
y = digits.target

print("Объектов:", len(X), "| признаков:", X.shape[1])
print("Классы:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nРазбиение 70/30, stratify, random_state=42")
print(f"Обучающая: {X_train.shape}, тестовая: {X_test.shape}")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
}

rows, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    preds[name] = pred
    acc = accuracy_score(y_test, pred)
    rows.append((name, acc))
    print(f"\nОтчёт ({name}):")
    print(classification_report(y_test, pred, zero_division=0))

best_name, best_acc = max(rows, key=lambda row: row[1])
print(f"\nЛучшая модель по общей точности: {best_name} (Accuracy={best_acc:.4f})")

names = [row[0] for row in rows]
accuracies = [row[1] for row in rows]

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Сравнение моделей классификации цифр", fontsize=16, fontweight="bold")

x = np.arange(len(names))
bars = ax1.bar(x, accuracies, color="#4C72B0", edgecolor="white")
for bar, v in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
             ha="center", va="bottom", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Accuracy")
ax1.set_title("Общая точность по моделям")

ConfusionMatrixDisplay.from_estimator(
    models[best_name], X_test, y_test, ax=ax2, cmap="Blues", colorbar=False
)
ax2.set_title(f"Матрица ошибок: {best_name}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = Path(__file__).parent / "digits_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nГрафики сохранены в файл: {output_path}")
plt.show()