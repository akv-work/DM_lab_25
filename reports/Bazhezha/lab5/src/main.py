from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df = pd.read_csv(Path(__file__).with_name("iris.csv"))
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
X = df[features].to_numpy()
labels = LabelEncoder()
y = labels.fit_transform(df["variety"])
names = labels.classes_

print("Признаки:", ", ".join(features))
print("Классы:", ", ".join(names))
print("Данные:", X.shape)
print("Пропусков:", int(df[features].isna().sum().sum()))
print("Распределение классов:", {name: int((df["variety"] == name).sum()) for name in names})
print("Первые 5 строк:")
for row, label in zip(X[:5], y[:5]):
    print(" ", row, "->", names[label])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nОбучающая: {X_train.shape}, тестовая: {X_test.shape}")
print("В тесте:", {names[i]: int((y_test == i).sum()) for i in range(len(names))})

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
}

print("\nAccuracy:")
scores, preds = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds[name] = np.ravel(model.predict(X_test))
    scores[name] = accuracy_score(y_test, preds[name])
    print(f"  {name}: {scores[name]:.6f}")

print("\nМатрицы ошибок (строки — истинный класс, столбцы —", ", ".join(names) + "):")
for name, pred in preds.items():
    print(name)
    print(confusion_matrix(y_test, pred))

best_score = max(scores.values())
best = [name for name, score in scores.items() if score == best_score]
print(f"\nЛучшая accuracy {best_score:.6f}: {', '.join(best)}")
print(f"\nОтчёт по классам ({best[0]}):")
print(classification_report(y_test, preds[best[0]], target_names=names, digits=3))

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
graph_dir = Path(__file__).with_name("graphs")
graph_dir.mkdir(exist_ok=True)
palette = ["#4C78A8", "#F58518", "#54A24B"]

fig, ax = plt.subplots(figsize=(7.2, 5))
for i, name in enumerate(names):
    mask = y == i
    ax.scatter(X[mask, 2], X[mask, 3], label=name, c=palette[i], s=36, edgecolors="white", linewidths=0.4)
ax.set_xlabel("petal.length, см")
ax.set_ylabel("petal.width, см")
ax.set_title("Виды ириса по длине и ширине лепестка")
ax.legend()
fig.tight_layout()
fig.savefig(graph_dir / "classes.png", dpi=160)
plt.close()

order = list(scores)
vals = [scores[name] for name in order]
errors = [int((preds[name] != y_test).sum()) for name in order]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
bar_colors = ["#4C78A8" if value == best_score else "#B8B8B8" for value in vals]
bars = axes[0].bar(order, vals, color=bar_colors)
axes[0].set_ylim(0, 1.12)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Точность на тестовой выборке")
axes[0].tick_params(axis="x", rotation=20)
for bar, value in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
bars = axes[1].bar(order, errors, color=bar_colors)
axes[1].set_ylabel("Число ошибок")
axes[1].set_title("Ошибки на 45 объектах теста")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_ylim(0, 7)
for bar, value in zip(bars, errors):
    axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.15, str(value), ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(graph_dir / "accuracy.png", dpi=160)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
for ax, (name, pred) in zip(axes.ravel(), preds.items()):
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=15)
    ax.set_title(f"{name}\naccuracy {scores[name]:.3f}", fontsize=11)
    ax.set_xticks(range(3), names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(3), names, fontsize=8)
    ax.set_xlabel("Предсказание", fontsize=8)
    ax.set_ylabel("Истинный класс", fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > 8 else "black")
axes.ravel()[-1].axis("off")
fig.suptitle("Матрицы ошибок на тестовой выборке")
fig.tight_layout()
fig.savefig(graph_dir / "confusion.png", dpi=160)
plt.close()

print("\nГрафики:", graph_dir)
