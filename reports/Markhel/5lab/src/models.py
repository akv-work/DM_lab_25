import time

from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

NAMES = {
    "tree": "Дерево (без ограничений)",
    "tree_cv": "Дерево (подбор по CV)",
    "rf": "Случайный лес",
    "ada": "AdaBoost",
    "cat": "CatBoost",
    "xgb": "XGBoost",
}


class BalancedAdaBoost(AdaBoostClassifier):
    """AdaBoost с начальными весами объектов, уравновешивающими классы."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def make_models(seed=42, balanced=False, y_train=None, only=None):
    """Возвращает {ключ: необученная модель}. balanced=True — взвешивание классов (для recall).
    Для XGBoost вес положительного класса scale_pos_weight = N_neg / N_pos считается по y_train."""
    spw = 1.0
    if balanced:
        spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    cw = "balanced" if balanced else None

    tree_grid = {"max_depth": list(range(2, 9)), "min_samples_leaf": [1, 5, 10, 20],
                 "criterion": ["gini", "entropy"]}
    models = {
        "tree": DecisionTreeClassifier(random_state=seed, class_weight=cw),
        "tree_cv": GridSearchCV(DecisionTreeClassifier(random_state=seed, class_weight=cw), tree_grid,
                                scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                     class_weight="balanced_subsample" if balanced else None),
        "ada": (BalancedAdaBoost if balanced else AdaBoostClassifier)(
            n_estimators=200, learning_rate=0.5, random_state=seed),
        "cat": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, random_seed=seed,
                                  verbose=0, allow_writing_files=False,
                                  auto_class_weights="Balanced" if balanced else None),
        "xgb": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, eval_metric="logloss",
                             scale_pos_weight=spw, n_jobs=-1),
    }
    if only is not None:
        models = {k: v for k, v in models.items() if k in only}
    return models


def fit_timed(model, X, y):
    t = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t


def oof_proba(model, X, y, seed=42, folds=5):
    """Вероятности положительного класса по 5-кратной кросс-валидации на обучающей выборке."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
