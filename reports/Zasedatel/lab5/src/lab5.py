import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA = "pima-indians-diabetes.csv"
SEED = 42
N_REPEATS = 20            
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]

def load():
    data = pd.read_csv(DATA, comment="#", header=None, names=COLUMNS)
    X = data.drop(columns="Outcome").values.astype(float)
    y = data["Outcome"].values.astype(int)
    return X, y

def make_models(seed, balanced):
    neg_pos = None  
    return neg_pos, {
        "Дерево решений": lambda: DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=seed,
            class_weight="balanced" if balanced else None),
        "Случайный лес": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1,
            class_weight="balanced" if balanced else None),
        "AdaBoost": lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1,
                class_weight="balanced" if balanced else None),
            n_estimators=200, learning_rate=0.5, random_state=seed),
    }

def build_all(seed, balanced, y_train):
    _, models = make_models(seed, balanced)
    spw = (y_train == 0).sum() / (y_train == 1).sum() if balanced else 1.0
    models["XGBoost"] = lambda: XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
        random_state=seed, n_jobs=-1)
    models["CatBoost"] = lambda: CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05, random_seed=seed,
        verbose=0, thread_count=-1,
        auto_class_weights="Balanced" if balanced else None)
    return models

def split_scale(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(X_tr)          
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te

def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "Recall": recall_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred),
        "Accuracy": accuracy_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }, confusion_matrix(y_te, pred)

def single_split(X, y, balanced):
    X_tr, X_te, y_tr, y_te = split_scale(X, y, SEED)
    print(f"\nОбучающая: {len(y_tr)} (диабет: {y_tr.sum()}), "
          f"тестовая: {len(y_te)} (диабет: {y_te.sum()})")

    rows, cms = {}, {}
    for name, factory in build_all(SEED, balanced, y_tr).items():
        model = factory().fit(X_tr, y_tr)
        rows[name], cms[name] = evaluate(model, X_te, y_te)

    print(pd.DataFrame(rows).T.round(4).to_string())
    print("\nМатрицы ошибок [[TN, FP], [FN, TP]]:")
    for name, cm in cms.items():
        print(f"{name:16s} TN={cm[0,0]:3d} FP={cm[0,1]:3d} "
              f"FN={cm[1,0]:3d} TP={cm[1,1]:3d}")

def repeated(X, y, balanced):
    res = {}
    for r in range(N_REPEATS):
        seed = 1000 + r
        X_tr, X_te, y_tr, y_te = split_scale(X, y, seed)
        for name, factory in build_all(seed, balanced, y_tr).items():
            model = factory().fit(X_tr, y_tr)
            m, _ = evaluate(model, X_te, y_te)
            res.setdefault(name, []).append(m)

    table = {}
    for name, runs in res.items():
        df = pd.DataFrame(runs)
        table[name] = {c: f"{df[c].mean():.3f} ± {df[c].std():.3f}"
                       for c in df.columns}
    print(f"\nСреднее ± std по {N_REPEATS} случайным разбиениям 70/30:")
    print(pd.DataFrame(table).T.to_string())

def main():
    X, y = load()
    print(f"Данные: {X.shape}, диабет: {y.sum()} ({y.mean():.1%}), "
          f"нет диабета: {(y == 0).sum()}")

    for balanced in (False, True):
        title = ("С учётом дисбаланса классов (balanced)" if balanced
                 else "Стандартные настройки (без весов классов)")
        print(title)
        print(f"\nОдно разбиение 70/30 (random_state={SEED})")
        single_split(X, y, balanced)
        print("\nПроверка устойчивости")
        repeated(X, y, balanced)

if __name__ == "__main__":
    main()