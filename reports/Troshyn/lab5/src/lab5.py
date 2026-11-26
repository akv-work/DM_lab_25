import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
data = pd.read_csv("car_evaluation.csv", header=None, names=columns)

le = LabelEncoder()
y = le.fit_transform(data["class"])
class_names = list(le.classes_)

categories = [
    ["low", "med", "high", "vhigh"],
    ["low", "med", "high", "vhigh"],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"],
]
X = data.drop(columns="class").astype(str)
X = pd.DataFrame(
    OrdinalEncoder(categories=categories).fit_transform(X),
    columns=X.columns,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {len(X_train)} записей")
print(f"Тестовая выборка: {len(X_test)} записей")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200,
        random_state=42,
    ),
    "CatBoost": CatBoostClassifier(verbose=0, random_seed=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42),
}

results = []
matrices = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).ravel()
    y_proba = model.predict_proba(X_test)

    results.append(
        {
            "Модель": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall": recall_score(y_test, y_pred, average="macro"),
            "F1": f1_score(y_test, y_pred, average="macro"),
            "ROC AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        }
    )
    matrices[name] = confusion_matrix(y_test, y_pred)

print("\nМетрики моделей (macro-усреднение):")
print(pd.DataFrame(results).round(4).to_string(index=False))

for name, cm in matrices.items():
    print(f"\nМатрица ошибок: {name}")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

importances = pd.Series(
    models["Decision Tree"].feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\nВажность признаков (Decision Tree):")
print(importances.round(4).to_string())

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()
for ax, (name, cm) in zip(axes, matrices.items()):
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)
axes[-1].axis("off")
plt.tight_layout()
plt.show()
