from catboost import CatBoostClassifier
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Дерево решений": DecisionTreeClassifier(random_state=42),
        "Случайный лес": RandomForestClassifier(random_state=42, n_jobs=2),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, verbose=False, random_seed=42, thread_count=2, allow_writing_files=False
        ),
        "XGBoost": XGBClassifier(random_state=42, n_jobs=2),
    }

    print(f"Данные: {len(y)} изображений 8×8, {len(set(y))} классов")
    print(f"Обучающая выборка: {len(y_train)}, тестовая: {len(y_test)}")
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test).ravel()
        print(f"\n{name}")
        print(classification_report(y_test, prediction, digits=3, zero_division=0))
        scores[name] = accuracy_score(y_test, prediction)

    print("Сравнение точности")
    for name in sorted(scores, key=scores.get, reverse=True):
        print(f"{name}: {scores[name]:.4f}")
