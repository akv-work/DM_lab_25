import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_REPEATS = 10
TEST_SIZE = 0.2

def load():
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(int)
    return X, y

def models(seed):
    return {
        "Дерево решений": lambda: DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=2, random_state=seed),
        "Случайный лес": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, random_state=seed, n_jobs=-1),
        "AdaBoost": lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=200, learning_rate=0.5, random_state=seed),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softmax", num_class=10,
            eval_metric="mlogloss", random_state=seed, n_jobs=-1),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.1,
            loss_function="MultiClass", random_seed=seed,
            verbose=0, thread_count=-1),
    }

def run_single(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"Обучающая выборка: {len(y_train)}")
    print(f"Тестовая выборка: {len(y_test)}")
    print()
    scores = {}
    for name, build in models(SEED).items():
        model = build().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name}:")
        print(classification_report(y_test, y_pred, digits=3))
        scores[name] = accuracy_score(y_test, y_pred)

    print("Сравнение точности:")
    for name, acc in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{name:16s}: {acc:.4f}")

def run_many(X, y):
    results = {}
    for i in range(N_REPEATS):
        seed = 100 + i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        for name, build in models(seed).items():
            model = build().fit(X_train, y_train)
            results.setdefault(name, []).append(
                accuracy_score(y_test, model.predict(X_test)))

    print(f"\nСреднее ± std accuracy по {N_REPEATS} разбиениям:")
    for name, acc_list in results.items():
        print(f"{name:16s}: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")

def main():
    X, y = load()
    print(f"Данные: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    run_single(X, y)
    run_many(X, y)

if __name__ == "__main__":
    main()