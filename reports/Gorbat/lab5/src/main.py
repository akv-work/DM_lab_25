from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA = Path(__file__).parent / "dataset" / "winequality-white.csv"
NAMES = ("обычное", "хорошее")

df = pd.read_csv(DATA, sep=";")
features = [column for column in df.columns if column != "quality"]
y = (df["quality"] >= 7).astype(int)
X = df[features]

print("Файл:", DATA.name)
print("Объектов:", len(df), "| признаков:", len(features))
print("Признаки:", ", ".join(features))
print("Пропусков:", int(df.isna().sum().sum()))
print("Оценки quality:", df["quality"].value_counts().sort_index().to_dict())
print(
    "Классы: обычное (quality < 7) —",
    int((y == 0).sum()),
    "| хорошее (quality >= 7) —",
    int((y == 1).sum()),
    f"| доля хорошего: {y.mean():.3f}",
)
print("Полных дубликатов:", int(df.duplicated().sum()))
print("Первые 5 объектов:")
print(pd.concat([X.head(), df["quality"].head(), y.head().rename("класс")], axis=1).to_string(index=False))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"\nРазбиение 70/30, stratify, random_state=42")
print(f"Обучающая: {X_train.shape}, тестовая: {X_test.shape}")
print("В тесте: обычное —", int((y_test == 0).sum()), "| хорошее —", int((y_test == 1).sum()))

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
}

rows, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    preds[name] = pred
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average=None, labels=[1], zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    rows.append((name, accuracy_score(y_test, pred), precision[0], recall[0], f1[0], f1_macro))

print("\nPrecision, recall и F1 — для класса «хорошее»:")
print(f"{'Модель':<16}{'Accuracy':>10}{'Precision':>11}{'Recall':>10}{'F1':>10}{'F1 macro':>10}")
for name, accuracy, precision, recall, f1, f1_macro in rows:
    print(f"{name:<16}{accuracy:10.4f}{precision:11.4f}{recall:10.4f}{f1:10.4f}{f1_macro:10.4f}")

print("\nМатрицы ошибок (строка — истина, столбцы — обычное / хорошее):")
for name, pred in preds.items():
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    print(f"{name}: TN={matrix[0, 0]} FP={matrix[0, 1]} FN={matrix[1, 0]} TP={matrix[1, 1]}")

best = max(rows, key=lambda row: (row[4], -abs(row[2] - row[3])))
name, accuracy, precision, recall, f1, f1_macro = best
print(
    f"\nЛучший баланс точности и полноты: {name}. "
    f"F1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}, "
    f"|precision-recall|={abs(precision - recall):.4f}, F1 macro={f1_macro:.4f}."
)
print(f"\nОтчёт ({name}):")
print(classification_report(y_test, preds[name], target_names=NAMES, digits=3, zero_division=0))
closest = min(rows, key=lambda row: (abs(row[2] - row[3]), -row[4]))
weak = min(rows, key=lambda row: row[4])
print("Вывод: доля класса «хорошее»", f"{y.mean():.1%}.",
      "Accuracy завышена классом «обычное», поэтому модели сравниваются по F1.")
print(f"Лучший баланс точности и полноты — {name}: максимальный F1 класса «хорошее».")
if closest[0] != name:
    print(f"Ближе всего precision и recall у {closest[0]} "
          f"({closest[2]:.4f} и {closest[3]:.4f}), но F1 ниже ({closest[4]:.4f}).")
print(f"Ниже всех F1 у {weak[0]} ({weak[4]:.4f}), recall «хорошее» = {weak[3]:.4f}.")
print(f"Для этого набора предпочтителен {name}.")

names = [row[0] for row in rows]
accuracies = [row[1] for row in rows]
precisions = [row[2] for row in rows]
recalls = [row[3] for row in rows]
f1s = [row[4] for row in rows]
f1_macros = [row[5] for row in rows]

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 15))
fig.suptitle("Сравнение моделей классификации вин (quality >= 7)", fontsize=16, fontweight="bold")
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0:2])
x = np.arange(len(names))
width = 0.2
metrics = [
    ("Accuracy", accuracies, "#4C72B0"),
    ("Precision (хорошее)", precisions, "#DD8452"),
    ("Recall (хорошее)", recalls, "#55A868"),
    ("F1 (хорошее)", f1s, "#C44E52"),
]
for i, (label, values, color) in enumerate(metrics):
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, values, width, label=label, color=color, edgecolor="white")
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7, rotation=90)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Значение метрики")
ax1.set_title("Метрики качества по моделям")
ax1.legend(fontsize=8, loc="lower right")

# 2. F1 macro
ax2 = fig.add_subplot(gs[0, 2])
bars = ax2.bar(names, f1_macros, color="#8172B3", edgecolor="white")
for bar, v in zip(bars, f1_macros):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
             ha="center", va="bottom", fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("F1 macro")
ax2.set_title("F1 macro — устойчивость к дисбалансу")
ax2.tick_params(axis="x", rotation=15)

positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
for (r, c), (model_name, pred) in zip(positions, preds.items()):
    ax = fig.add_subplot(gs[r, c])
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(NAMES)
    ax.set_yticklabels(NAMES)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(f"Матрица ошибок: {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = Path(__file__).parent / "models_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nГрафики сохранены в файл: {output_path}")
plt.show()