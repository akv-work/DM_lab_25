"""
Лабораторная работа №4 (основа — ЛР №3). Глубокая полносвязная сеть, обучение с учителем
(fine-tuning) и жадное послойное предобучение автоэнкодерами (алгоритм из лекции).
Предобучение стеком RBM находится в модуле rbm.py, сравнение режимов — в experiment.py.

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
from dataclasses import dataclass, field

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
    # --- ЛР №4: предобучение стеком RBM (параметры как в классе Pretrain_params из лекции) ---
    rbm_epochs: int = 30      # max_epochs: эпох обучения каждой RBM
    # скорость обучения RBM по слоям (rates); подобрана по val для каждой активации (seed 0, 3 значения)
    rbm_rates: dict = field(default_factory=lambda: {"relu": (0.001,) * 4, "tanh": (0.001,) * 4,
                                                     "sigmoid": (0.01, 0.05, 0.05, 0.05)})
    rbm_start_momentum: float = 0.5   # smomentum (первые 5 эпох)
    rbm_final_momentum: float = 0.9   # fmomentum
    rbm_weight_decay: float = 0.0     # weight_loss (в лекции — without weight-loss)
    rbm_min_error: float = 0.0        # min_error: досрочный останов по ошибке реконструкции


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
# Итоги одного обучения
# ----------------------------------------------------------------------------
def finish(model, hist, split, task, cfg, pretrain_curves=None):
    """Метрики лучшей (по val loss) модели + показатели динамики val-кривой."""
    _, val_m, _, _ = evaluate(model, split["X_va"], split["y_va"], split, task)
    t_loss, test_m, y_true, y_pred = evaluate(model, split["X_te"], split["y_te"], split, task)
    osc, up = oscillation(hist["val_loss"], cfg.smooth_from)
    return {"history": hist, "val_metrics": val_m, "test_metrics": test_m, "test_loss": t_loss,
            "y_true": y_true, "y_pred": y_pred, "best_epoch": hist["best_epoch"],
            "init_val_loss": hist["val_loss"][0], "oscillation": osc, "up_share": up,
            "noise": noise_level(hist["val_loss"], cfg.smooth_from),
            "pretrain_curves": pretrain_curves}
