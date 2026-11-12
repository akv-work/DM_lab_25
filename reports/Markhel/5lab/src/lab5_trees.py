import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

import common
from common import (FEATURES, TARGET, FIG_DIR, RESULTS_DIR, load_pima, zeros_to_nan, metrics,
                    plot_confusions)
from models import NAMES, make_models, fit_timed, oof_proba

SEED = 42
TARGET_RECALL = 0.80   
KEYS = list(NAMES)
ENSEMBLES = ["rf", "ada", "cat", "xgb"]
COLORS = {"tree": "tab:gray", "tree_cv": "tab:brown", "rf": "tab:green", "ada": "tab:orange",
          "cat": "tab:blue", "xgb": "tab:red"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="мало повторений (проверка)")
    ap.add_argument("--no-show", action="store_true", help="не вызывать plt.show()")
    ap.add_argument("--reps", type=int, default=30, help="число повторных разбиений 70/30")
    return ap.parse_args()


def split_scale(df, seed):
    """По заданию: сначала стандартизация всех данных (StandardScaler), затем разбиение 70/30
    со стратификацией по классу. Для деревьев масштаб признаков не влияет на результат (см. проверку)."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    Xs = StandardScaler().fit_transform(X)
    return train_test_split(Xs, y, test_size=0.3, stratify=y, random_state=seed)


def fit_eval(model, Xtr, ytr, Xte, yte, threshold=0.5):
    t = fit_timed(model, Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    res = metrics(yte, pred, proba)
    res["Fit, c"] = t
    return res, proba, pred


def threshold_for_recall(y, proba, target):
    """Наибольший порог из сетки 0.01..0.99, при котором recall на (OOF-)выборке >= target."""
    best = 0.01
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        if metrics(y, (proba >= t).astype(int))["Recall"] >= target:
            best = float(t)
    return best


def summarize(df, cols, by="Модель", order=None):
    g = df.groupby(by, sort=False)[cols]
    mean, std = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c].fillna(0))]
    return out.loc[order] if order else out


def paired_p(df, col, a, b, by="Модель"):
    x = df[df[by] == a].sort_values("seed")[col].to_numpy()
    y = df[df[by] == b].sort_values("seed")[col].to_numpy()
    if len(x) < 2 or np.allclose(x, y):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.ttest_rel(x, y).pvalue)


def main():
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = 3 if args.quick else args.reps
    df = load_pima()
    Xtr, Xte, ytr, yte = split_scale(df, SEED)
    print(f"\nСтандартизация (StandardScaler) и разбиение 70/30 со стратификацией (random_state = {SEED}): "
          f"train = {len(ytr)} (диабет: {int(ytr.sum())}), test = {len(yte)} (диабет: {int(yte.sum())})")

    models = make_models(SEED)
    rows, probas, cms = [], {}, {}
    for key, model in models.items():
        res, proba, pred = fit_eval(model, Xtr, ytr, Xte, yte)
        tr_recall = metrics(ytr, model.predict(Xtr))["Recall"]
        res.update({"Модель": NAMES[key], "Recall (train)": tr_recall})
        rows.append(res)
        probas[key], cms[NAMES[key]] = proba, confusion_matrix(yte, pred, labels=[0, 1])
    main_df = pd.DataFrame(rows).set_index("Модель")
    cols = ["Recall", "Precision", "F1", "Accuracy", "ROC-AUC", "FN", "FP", "Recall (train)", "Fit, c"]
    print("\nРезультаты на тестовой выборке (порог 0,5; основная метрика — Recall класса «диабет»):")
    print(main_df[cols].round(3).to_string())
    main_df[cols].round(4).to_csv(RESULTS_DIR / "main_split.csv", encoding="utf-8-sig")

    best = models["tree_cv"].best_params_
    print(f"\nДерево, подбор по 5-fold CV на обучающей выборке (scoring = F1): {best}")
    rf = models["rf"]
    rf_oob = RandomForestOOB(Xtr, ytr)
    print(f"Случайный лес: OOB-оценка accuracy (300 деревьев) = {rf_oob:.3f}")

    plot_confusions(cms, "Матрицы ошибок на тесте (порог 0,5)", "confusion_main.png")

    Xr = df[FEATURES].to_numpy(float)
    yr = df[TARGET].to_numpy(int)
    Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(Xr, yr, test_size=0.3, stratify=yr, random_state=SEED)
    inv = []
    for key in ("tree", "rf", "ada", "cat", "xgb"):
        a = make_models(SEED, only=[key])[key].fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        b = make_models(SEED, only=[key])[key].fit(Xtr_raw, ytr_raw).predict_proba(Xte_raw)[:, 1]
        inv.append({"Модель": NAMES[key], "max |ΔP|": float(np.max(np.abs(a - b))),
                    "Различных прогнозов": int(((a >= 0.5) != (b >= 0.5)).sum()),
                    "Recall (стандарт.)": metrics(yte, (a >= 0.5).astype(int))["Recall"],
                    "Recall (исходные)": metrics(yte_raw, (b >= 0.5).astype(int))["Recall"]})
    inv = pd.DataFrame(inv).set_index("Модель")
    print("\nПроверка инвариантности к масштабу (обучение на стандартизованных и на исходных признаках):")
    print(inv.round(4).to_string())
    inv.round(4).to_csv(RESULTS_DIR / "scale_invariance.csv", encoding="utf-8-sig")

    imp = {}
    imp["Дерево (без ограничений)"] = models["tree"].feature_importances_
    imp["Случайный лес"] = models["rf"].feature_importances_
    imp["AdaBoost"] = models["ada"].feature_importances_
    c = models["cat"].get_feature_importance()
    imp["CatBoost"] = c / c.sum()
    imp["XGBoost"] = models["xgb"].feature_importances_
    imp_df = pd.DataFrame(imp, index=FEATURES)
    print("\nВажность признаков (нормирована, сумма = 1):")
    print(imp_df.round(3).to_string())
    imp_df.round(4).to_csv(RESULTS_DIR / "feature_importance.csv", encoding="utf-8-sig")
    ax = imp_df.plot.bar(figsize=(11, 5), width=0.8)
    ax.set_ylabel("Важность")
    ax.set_title("Важность признаков (основное разбиение)")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(models["tree_cv"].best_estimator_, max_depth=3, feature_names=FEATURES,
              class_names=["нет", "диабет"], filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Дерево после подбора гиперпараметров (показаны первые 3 уровня)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tree_cv.png", dpi=150)

    # ------------------------------------------------------------------ 4. ROC и PR-кривые
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key in KEYS:
        fpr, tpr, _ = roc_curve(yte, probas[key])
        axes[0].plot(fpr, tpr, color=COLORS[key], label=f"{NAMES[key]} (AUC = {main_df.loc[NAMES[key], 'ROC-AUC']:.3f})")
        p, r, _ = precision_recall_curve(yte, probas[key])
        axes[1].plot(r, p, color=COLORS[key], label=f"{NAMES[key]} (AP = {average_precision_score(yte, probas[key]):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="FPR (ложные тревоги)", ylabel="TPR = Recall", title="ROC-кривые (тест)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall (тест)")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr.png", dpi=150)

    # ------------------------------------------------------------------ 5. Повторные разбиения 70/30
    print(f"\nПовторные разбиения 70/30: {reps} разных random_state (модели с порогом 0,5)")
    rep_rows = []
    for seed in range(reps):
        a, b, c_, d = split_scale(df, seed)
        for key, m in make_models(seed).items():
            res, _, _ = fit_eval(m, a, c_, b, d)
            res.update({"seed": seed, "Модель": NAMES[key]})
            rep_rows.append(res)
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(RESULTS_DIR / "repeated_runs.csv", index=False, encoding="utf-8-sig")
    order = [NAMES[k] for k in KEYS]
    rcols = ["Recall", "Precision", "F1", "Accuracy", "ROC-AUC", "FN", "FP", "Fit, c"]
    tab = summarize(rep, rcols, order=order)
    print(tab.to_string())
    tab.to_csv(RESULTS_DIR / "repeated_summary.csv", encoding="utf-8-sig")
    print("\nПарный t-тест по Recall (p-value), модель A vs B:")
    pairs = [("tree", "rf"), ("tree", "ada"), ("tree", "cat"), ("tree", "xgb"), ("tree_cv", "rf"),
             ("rf", "ada"), ("rf", "cat"), ("rf", "xgb"), ("ada", "cat"), ("cat", "xgb")]
    pv = []
    for a_, b_ in pairs:
        p = paired_p(rep, "Recall", NAMES[a_], NAMES[b_])
        pv.append({"A": NAMES[a_], "B": NAMES[b_], "p-value (Recall)": round(p, 4),
                   "p-value (ROC-AUC)": round(paired_p(rep, "ROC-AUC", NAMES[a_], NAMES[b_]), 4)})
        print(f"  {NAMES[a_]:26s} vs {NAMES[b_]:26s}: Recall p = {p:.4f}, ROC-AUC p = {pv[-1]['p-value (ROC-AUC)']:.4f}")
    pd.DataFrame(pv).to_csv(RESULTS_DIR / "repeated_pvalues.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(order, [main_df.loc[n, "Recall"] for n in order], color=[COLORS[k] for k in KEYS])
    axes[0].set(ylabel="Recall (диабет)", title=f"Основное разбиение (random_state = {SEED})", ylim=(0, 1))
    data = [rep[rep["Модель"] == n]["Recall"].to_numpy() for n in order]
    bp = axes[1].boxplot(data, patch_artist=True, tick_labels=order)
    for patch, k in zip(bp["boxes"], KEYS):
        patch.set_facecolor(COLORS[k])
        patch.set_alpha(0.6)
    axes[1].set(ylabel="Recall (диабет)", title=f"{reps} повторных разбиений 70/30", ylim=(0, 1))
    for a in axes:
        a.grid(alpha=0.3, axis="y")
        a.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recall_comparison.png", dpi=150)

\    print(f"\nВзвешивание классов (class_weight / scale_pos_weight / auto_class_weights), {reps} разбиений:")
    bal_rows = []
    for seed in range(reps):
        a, b, c_, d = split_scale(df, seed)
        for key, m in make_models(seed, balanced=True, y_train=c_).items():
            res, _, _ = fit_eval(m, a, c_, b, d)
            res.update({"seed": seed, "Модель": NAMES[key]})
            bal_rows.append(res)
    bal = pd.DataFrame(bal_rows)
    bal.to_csv(RESULTS_DIR / "balanced_runs.csv", index=False, encoding="utf-8-sig")
    btab = summarize(bal, ["Recall", "Precision", "F1", "Accuracy", "ROC-AUC", "FN", "FP"], order=order)
    print(btab.to_string())
    btab.to_csv(RESULTS_DIR / "balanced_summary.csv", encoding="utf-8-sig")

    print(f"\nПорог классификации: выбирается по 5-fold OOF-вероятностям обучающей выборки — наибольший "
          f"порог, при котором OOF-recall >= {TARGET_RECALL} (минимум ложных тревог при заданной "
          f"чувствительности), {reps} разбиений:")
    thr_rows = []
    for seed in range(reps):
        a, b, c_, d = split_scale(df, seed)
        for key, m in make_models(seed, only=ENSEMBLES).items():
            oof = oof_proba(m, a, c_, seed)
            t_star = threshold_for_recall(c_, oof, TARGET_RECALL)
            res, _, _ = fit_eval(m, a, c_, b, d, threshold=t_star)
            res.update({"seed": seed, "Модель": NAMES[key], "Порог": t_star})
            thr_rows.append(res)
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(RESULTS_DIR / "threshold_runs.csv", index=False, encoding="utf-8-sig")
    ttab = summarize(thr, ["Порог", "Recall", "Precision", "F1", "Accuracy", "FN", "FP"],
                     order=[NAMES[k] for k in ENSEMBLES])
    print(ttab.to_string())
    ttab.to_csv(RESULTS_DIR / "threshold_summary.csv", encoding="utf-8-sig")

    comb = []
    for name, d_ in (("Порог 0,5", rep), ("Веса классов", bal), ("Порог по OOF (recall ≥ 0,8)", thr)):
        sub = d_[d_["Модель"].isin([NAMES[k] for k in ENSEMBLES])]
        sub = sub.assign(Режим=name)
        comb.append(sub)
    comb = pd.concat(comb, ignore_index=True)
    ctab = comb.groupby(["Модель", "Режим"], sort=False)[["Recall", "Precision", "F1", "FN", "FP"]].mean().round(3)
    print("\nСводка (среднее по разбиениям): Recall / Precision / F1 / FN / FP:")
    print(ctab.to_string())
    ctab.to_csv(RESULTS_DIR / "strategies_summary.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    marks = {"Порог 0,5": "o", "Веса классов": "s", "Порог по OOF (recall ≥ 0,8)": "^"}
    for key in ENSEMBLES:
        for mode, mk in marks.items():
            s = comb[(comb["Модель"] == NAMES[key]) & (comb["Режим"] == mode)]
            ax.scatter(s["Recall"].mean(), s["Precision"].mean(), color=COLORS[key], marker=mk, s=90,
                       edgecolors="k", label=f"{NAMES[key]}, {mode}")
    ax.set_xlabel("Recall (диабет)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Баланс recall / precision (среднее по {reps} разбиениям)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recall_precision_tradeoff.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    grid = np.linspace(0.05, 0.95, 91)
    for ax, key in zip(axes, ["rf", "cat", "xgb"]):
        pr = [metrics(yte, (probas[key] >= t).astype(int)) for t in grid]
        ax.plot(grid, [r["Recall"] for r in pr], label="Recall", color="tab:red")
        ax.plot(grid, [r["Precision"] for r in pr], label="Precision", color="tab:blue")
        ax.plot(grid, [r["F1"] for r in pr], label="F1", color="tab:green", ls="--")
        ax.axvline(0.5, color="k", lw=0.8, ls=":")
        ax.set(xlabel="Порог вероятности", title=NAMES[key])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Recall и Precision в зависимости от порога (тест, основное разбиение)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_curves.png", dpi=150)

    print(f"\nНули в Glucose, BloodPressure, SkinThickness, Insulin, BMI заменены на NaN, {reps} разбиений "
          f"(деревья, лес, XGBoost, CatBoost — штатная обработка NaN; AdaBoost — медианная импутация):")
    dfn = zeros_to_nan(df)
    nan_rows = []
    for seed in range(reps):
        X = dfn[FEATURES].to_numpy(float)
        y = dfn[TARGET].to_numpy(int)
        Xs = StandardScaler().fit_transform(X)       # NaN при стандартизации игнорируются
        a, b, c_, d = train_test_split(Xs, y, test_size=0.3, stratify=y, random_state=seed)
        for key, m in make_models(seed).items():
            if key == "ada":
                m = make_pipeline(SimpleImputer(strategy="median"), m)
            res, _, _ = fit_eval(m, a, c_, b, d)
            res.update({"seed": seed, "Модель": NAMES[key]})
            nan_rows.append(res)
    nan_df = pd.DataFrame(nan_rows)
    nan_df.to_csv(RESULTS_DIR / "nan_runs.csv", index=False, encoding="utf-8-sig")
    ntab = summarize(nan_df, ["Recall", "Precision", "F1", "Accuracy", "ROC-AUC"], order=order)
    print(ntab.to_string())
    ntab.to_csv(RESULTS_DIR / "nan_summary.csv", encoding="utf-8-sig")

    print(f"\nТаблицы сохранены в {RESULTS_DIR}, графики — в {FIG_DIR}")
    if not args.no_show:
        plt.show()


def RandomForestOOB(Xtr, ytr):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, oob_score=True).fit(Xtr, ytr)
    return m.oob_score_


if __name__ == "__main__":
    main()
