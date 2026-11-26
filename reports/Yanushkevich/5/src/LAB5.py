import time

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

df = pd.read_csv("bank.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df["deposit"].value_counts())
print(df["deposit"].value_counts(normalize=True))

y = (df["deposit"] == "yes").astype(int)
X = df.drop(columns=["deposit"])

categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
print(categorical_columns)

X = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=int)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(X_train.shape, X_test.shape)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
        n_estimators=300,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

rows = []
predictions = {}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    rows.append(
        {
            "Модель": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "Время обучения, с": fit_time,
        }
    )

    print("=" * 60)
    print(name)
    print(
        classification_report(
            y_test, y_pred, target_names=["no", "yes"], digits=4
        )
    )

results = pd.DataFrame(rows).set_index("Модель")
results_sorted = results.sort_values("F1-score", ascending=False)
print(results_sorted.round(4).to_string())

best_model_name = results_sorted.index[0]
print(
    f"Лучшая модель по F1-score: {best_model_name} "
    f"(F1 = {results_sorted.loc[best_model_name, 'F1-score']:.4f})"
)

results_sorted.round(4).to_csv("lab5_results.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

results[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", ax=axes[0], rot=20
)
axes[0].set_title("Метрики моделей на тестовой выборке")
axes[0].set_ylim(0.6, 1.0)
axes[0].set_ylabel("Значение")
axes[0].grid(axis="y", alpha=0.3)

results["Время обучения, с"].plot(kind="bar", ax=axes[1], rot=20, color="tab:gray")
axes[1].set_title("Время обучения")
axes[1].set_ylabel("Секунды")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("lab5_comparison.png", dpi=150)
plt.show()

importances = pd.Series(
    models["Decision Tree"].feature_importances_, index=X.columns
).sort_values(ascending=False)
print(importances.head(10).round(4))

best_model = models[best_model_name]
best_proba = best_model.predict_proba(X_test)[:, 1]

labels = {0: "не подпишется", 1: "подпишется"}

predictions_table = df.loc[X_test.index].copy()
predictions_table["Факт"] = y_test.map(labels)
for name, y_pred in predictions.items():
    predictions_table[f"Прогноз {name}"] = pd.Series(y_pred, index=X_test.index).map(labels)
predictions_table["Вероятность подписки (лучшая модель)"] = best_proba.round(4)
predictions_table["Верно (лучшая модель)"] = (
    predictions_table["Факт"] == predictions_table[f"Прогноз {best_model_name}"]
)

show_columns = [
    "age",
    "job",
    "balance",
    "duration",
    "Факт",
    f"Прогноз {best_model_name}",
    "Вероятность подписки (лучшая модель)",
    "Верно (лучшая модель)",
]
print(predictions_table[show_columns].head(20).to_string())

print(predictions_table[f"Прогноз {best_model_name}"].value_counts())

predictions_table.to_csv("lab5_predictions.csv", encoding="utf-8-sig", index=True)

potential_clients = predictions_table[
    predictions_table[f"Прогноз {best_model_name}"] == "подпишется"
].sort_values("Вероятность подписки (лучшая модель)", ascending=False)
print(potential_clients[show_columns].head(10).to_string())

new_client = X_test.iloc[[0]]
new_proba = best_model.predict_proba(new_client)[0, 1]
new_label = labels[int(best_model.predict(new_client)[0])]
print(f"Клиент {new_client.index[0]}: {new_label}, вероятность подписки {new_proba:.4f}")