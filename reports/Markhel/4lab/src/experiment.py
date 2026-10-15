"""
Лабораторная работа №4. Сравнение трёх режимов обучения одной и той же глубокой сети:
  plain — без предобучения (ЛР №3);
  ae    — предобучение стеком автоэнкодеров (ЛР №3);
  rbm   — предобучение стеком ограниченных машин Больцмана (ЛР №4).
"""
import copy
import warnings

import pandas as pd
from scipy import stats

from common import set_seed
from deep_net import build_mlp, fit, finish, pretrain_layers
from rbm import pretrain_rbm

MODES = ("plain", "ae", "rbm")
MODE_NAMES = {"plain": "Без предобучения", "ae": "Автоэнкодеры", "rbm": "RBM"}


def run_modes(split, task, sizes, act_name, seed, cfg):
    """Три модели стартуют из одних и тех же случайных весов; одинаковые разбиение, оптимизатор,
    порядок мини-батчей и число эпох тонкой настройки. Различается только предобучение."""
    set_seed(seed)
    base = build_mlp(sizes, act_name)

    plain = copy.deepcopy(base)
    h = fit(plain, split, task, cfg, seed, pretrained=False)
    out = {"plain": finish(plain, h, split, task, cfg)}

    ae = copy.deepcopy(base)
    curves = pretrain_layers(ae, split["X_tr"], act_name, cfg, seed)      # только train, без меток
    h = fit(ae, split, task, cfg, seed, pretrained=True)
    out["ae"] = finish(ae, h, split, task, cfg, curves)

    rbm = copy.deepcopy(base)
    curves = pretrain_rbm(rbm, split["X_tr"], act_name, cfg, seed)                  # только train, без меток
    h = fit(rbm, split, task, cfg, seed, pretrained=True)
    out["rbm"] = finish(rbm, h, split, task, cfg, curves)
    return out


def runs_to_frame(results, extra=None):
    """results: {seed: run_modes(...)} -> таблица «по одной строке на (seed, режим)»."""
    rows = []
    for seed, res in results.items():
        for mode in MODES:
            r = res[mode]
            row = dict(extra or {})
            row.update({"seed": seed, "Режим": MODE_NAMES[mode]})
            row.update(r["test_metrics"])
            row.update({"Val loss (эпоха 0)": r["init_val_loss"],
                        "Лучшая эпоха": r["best_epoch"],
                        "Колебательность, %": r["oscillation"],
                        "Шум val-кривой, %": r["noise"],
                        "Доля эпох с ростом val": r["up_share"] * 100})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, cols, group=("Режим",)):
    """mean ± std по seed для каждого режима."""
    g = df.groupby(list(group), sort=False)[cols]
    mean, std = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c].fillna(0.0))]
    return out


def paired_p(df, col, a, b):
    """p-value парного t-теста между режимами a и b (один seed = то же разбиение и те же
    начальные веса выходного слоя)."""
    x = df[df["Режим"] == MODE_NAMES[a]].sort_values("seed")[col].to_numpy()
    y = df[df["Режим"] == MODE_NAMES[b]].sort_values("seed")[col].to_numpy()
    if len(x) < 2 or (x == y).all():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.ttest_rel(x, y).pvalue)


def convergence_report(results, thr, title):
    """Минимум val loss и число эпох тонкой настройки до достижения val loss <= thr."""
    import numpy as np
    lines = [f"{title}: минимум val loss и число эпох до val loss <= {thr}"]
    for mode in MODES:
        curves = [np.array(results[k][mode]["history"]["val_loss"]) for k in results]
        mins = [c.min() for c in curves]
        reached = [int(np.argmax(c <= thr)) for c in curves if (c <= thr).any()]
        ep = f"{np.mean(reached):.1f}" if reached else "—"
        lines.append(f"  {MODE_NAMES[mode]:18s}: min val loss = {np.mean(mins):.4f} ± {np.std(mins, ddof=1):.4f}; "
                     f"эпох до порога = {ep} (достигли порога {len(reached)} из {len(curves)} запусков)")
    text = "\n".join(lines)
    print(text)
    return text


PAIRS = (("ae", "plain"), ("rbm", "plain"), ("rbm", "ae"))


def print_pvalues(df, cols):
    """p-value парных тестов: АЭ vs без предобучения, RBM vs без предобучения, RBM vs АЭ."""
    print("  " + " " * 16 + "".join(f"{MODE_NAMES[a]} / {MODE_NAMES[b]}".center(30) for a, b in PAIRS))
    for c in cols:
        print(f"  {c:16s}" + "".join(f"{paired_p(df, c, a, b):.4f}".center(30) for a, b in PAIRS))
