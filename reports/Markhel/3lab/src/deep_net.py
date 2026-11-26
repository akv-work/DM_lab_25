"""
Лабораторная работа №3. Глубокая полносвязная сеть и жадное послойное предобучение
автоэнкодерами (алгоритм из лекции «Обучение глубоких НС»).

Алгоритм предобучения:
  1. Для первого скрытого слоя строится автоассоциативная сеть  X -> Y1 -> X
     (кодер = слой исходной сети, декодер — вспомогательный линейный слой), обучается
     по критерию MSE реконструкции (без учителя), веса кодера W1 фиксируются.
  2. Выход Y1 подаётся на вход второго автоэнкодера  Y1 -> Y2 -> Y1, получаем W2.
  3. Процесс продолжается до последнего скрытого слоя; декодеры отбрасываются.
  4. Выходной слой обучается с учителем (скрытые слои заморожены).
  5. Вся сеть дообучается (fine-tuning) методом обратного распространения ошибки.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from common import (set_seed, regression_metrics, classification_metrics)

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
ACT_LABELS = {"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Logistic (sigmoid)"}


@dataclass
class Config:
    ae_epochs: int = 30       # эпох обучения каждого автоэнкодера (лекция: не более 100)
    ae_lr: float = 1e-3
    head_epochs: int = 10     # эпох обучения выходного слоя с учителем
    epochs: int = 100         # эпох тонкой настройки всей сети
    lr: float = 1e-3
    smooth_from: int = 10     # с какой эпохи считается колебательность val-кривой


def batch_size_for(n):
    return int(max(16, min(256, n // 8)))


def build_mlp(sizes, act_name):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[act_name]())
    return nn.Sequential(*layers)


def linear_layers(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def _batches(n, batch, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        yield perm[i:i + batch]


# ----------------------------------------------------------------------------
# Предобучение
# ----------------------------------------------------------------------------
def pretrain_layers(model, X, act_name, cfg, seed):
    """Жадное послойное предобучение скрытых слоёв автоэнкодерами. Меняет веса model.
    Возвращает кривые MSE реконструкции для каждого автоэнкодера."""
    torch.manual_seed(seed + 1000)
    gen = torch.Generator().manual_seed(seed + 1000)
    batch = batch_size_for(len(X))
    mse = nn.MSELoss()
    H = X
    curves = []
    for enc_lin in linear_layers(model)[:-1]:
        act = ACTIVATIONS[act_name]()
        dec = nn.Linear(enc_lin.out_features, enc_lin.in_features)  # реконструирующий слой
        opt = torch.optim.Adam(list(enc_lin.parameters()) + list(dec.parameters()), lr=cfg.ae_lr)
        curve = []
        for _ in range(cfg.ae_epochs):
            total = 0.0
            for idx in _batches(len(H), batch, gen):
                h = H[idx]
                loss = mse(dec(act(enc_lin(h))), h)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            curve.append(total / len(H))
        curves.append(curve)
        with torch.no_grad():                 # выход слоя -> вход следующего автоэнкодера
            H = act(enc_lin(H))
    return curves


# ----------------------------------------------------------------------------
# Обучение с учителем
# ----------------------------------------------------------------------------
def make_loss(task):
    return nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, X, y, split, task):
    """Возвращает loss и метрики; регрессия оценивается в исходных единицах RMSD."""
    model.eval()
    out = model(X)
    loss = make_loss(task)(out, y).item()
    if task == "regression":
        sc = split["y_scaler"]
        y_pred = sc.inverse_transform(out.numpy()).ravel()
        y_true = sc.inverse_transform(y.numpy()).ravel()
        metrics = regression_metrics(y_true, y_pred)
    else:
        y_pred = out.argmax(dim=1).numpy()
        y_true = y.numpy()
        metrics = classification_metrics(y_true, y_pred)
    return loss, metrics, y_true, y_pred


def _train(model, params, split, task, epochs, lr, seed, record=None):
    """Эпохи обучения Adam по MSE / CE. record — история (val_loss пишется после каждой эпохи)."""
    X, y = split["X_tr"], split["y_tr"]
    gen = torch.Generator().manual_seed(seed + 2000)
    batch = batch_size_for(len(X))
    loss_fn = make_loss(task)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in _batches(len(X), batch, gen):
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if record is not None:
            record["train_loss"].append(total / len(X))
            record["val_loss"].append(evaluate(model, split["X_va"], split["y_va"], split, task)[0])
            if record["val_loss"][-1] < record["best_val"]:
                record["best_val"] = record["val_loss"][-1]
                record["best_epoch"] = len(record["val_loss"]) - 1
                record["best_state"] = copy.deepcopy(model.state_dict())


def fit(model, split, task, cfg, seed, pretrained):
    """Обучение с учителем. При pretrained=True сначала обучается только выходной слой
    (скрытые слои заморожены), затем настраивается вся сеть. Для сети без предобучения
    выходной слой отдельно не обучается — тонкая настройка начинается со случайных весов.
    history['val_loss'][0] — ошибка на валидации перед тонкой настройкой."""
    if pretrained and cfg.head_epochs > 0:
        head = linear_layers(model)[-1]
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        _train(model, list(head.parameters()), split, task, cfg.head_epochs, cfg.lr, seed)
        for p in model.parameters():
            p.requires_grad = True

    v0 = evaluate(model, split["X_va"], split["y_va"], split, task)[0]
    hist = {"train_loss": [], "val_loss": [v0], "best_val": v0, "best_epoch": 0,
            "best_state": copy.deepcopy(model.state_dict())}
    _train(model, list(model.parameters()), split, task, cfg.epochs, cfg.lr, seed, hist)
    hist["final_state"] = copy.deepcopy(model.state_dict())
    model.load_state_dict(hist.pop("best_state"))  # модель с минимальной ошибкой на валидации
    return hist


def oscillation(val_loss, start):
    """Колебательность val-кривой: средний |L_t − L_{t−1}| при t > start, в % от средней ошибки
    на этом участке; а также доля эпох, на которых ошибка выросла."""
    arr = np.asarray(val_loss[start:])
    diff = np.diff(arr)
    return float(np.mean(np.abs(diff)) / np.mean(arr) * 100), float(np.mean(diff > 0))


def noise_level(val_loss, start, window=5):
    """Шум val-кривой без учёта тренда: СКО отклонений val loss от центрированного скользящего
    среднего (окно 5 эпох) при t > start, в % от средней ошибки на этом участке."""
    arr = np.asarray(val_loss[start:])
    ma = np.convolve(arr, np.ones(window) / window, mode="valid")
    resid = arr[window // 2: len(arr) - window // 2] - ma
    return float(np.std(resid) / np.mean(arr) * 100)


# ----------------------------------------------------------------------------
# Один эксперимент «без предобучения» vs «с предобучением»
# ----------------------------------------------------------------------------
def _finish(model, hist, split, task, cfg, ae_curves=None):
    _, val_m, _, _ = evaluate(model, split["X_va"], split["y_va"], split, task)
    t_loss, test_m, y_true, y_pred = evaluate(model, split["X_te"], split["y_te"], split, task)
    osc, up = oscillation(hist["val_loss"], cfg.smooth_from)
    return {"history": hist, "val_metrics": val_m, "test_metrics": test_m, "test_loss": t_loss,
            "y_true": y_true, "y_pred": y_pred, "best_epoch": hist["best_epoch"],
            "init_val_loss": hist["val_loss"][0], "oscillation": osc, "up_share": up,
            "noise": noise_level(hist["val_loss"], cfg.smooth_from),
            "ae_curves": ae_curves}


def run_pair(split, task, sizes, act_name, seed, cfg):
    """Обе модели стартуют из одних и тех же случайных весов, одинаковые разбиение,
    оптимизатор, порядок мини-батчей и число эпох тонкой настройки."""
    set_seed(seed)
    base = build_mlp(sizes, act_name)

    plain = copy.deepcopy(base)
    h_plain = fit(plain, split, task, cfg, seed, pretrained=False)

    pre = copy.deepcopy(base)
    curves = pretrain_layers(pre, split["X_tr"], act_name, cfg, seed)  # только train, без меток
    h_pre = fit(pre, split, task, cfg, seed, pretrained=True)

    return {"plain": _finish(plain, h_plain, split, task, cfg),
            "pre": _finish(pre, h_pre, split, task, cfg, curves)}


# ----------------------------------------------------------------------------
# Сводные таблицы по нескольким запускам (seed)
# ----------------------------------------------------------------------------
MODE_NAMES = {"plain": "Без предобучения", "pre": "С предобучением"}


def runs_to_frame(results, extra=None):
    """results: {seed: run_pair(...)} -> таблица «по одной строке на (seed, режим)»."""
    import pandas as pd
    rows = []
    for seed, res in results.items():
        for mode in ("plain", "pre"):
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
    import pandas as pd
    g = df.groupby(list(group), sort=False)[cols]
    mean, std = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c].fillna(0.0))]
    return out


def paired_p(df, col):
    """p-value парного t-теста (один и тот же seed = то же разбиение и те же стартовые веса)."""
    from scipy import stats
    a = df[df["Режим"] == MODE_NAMES["plain"]].sort_values("seed")[col].to_numpy()
    b = df[df["Режим"] == MODE_NAMES["pre"]].sort_values("seed")[col].to_numpy()
    if len(a) < 2:
        return float("nan")
    return float(stats.ttest_rel(a, b).pvalue)
