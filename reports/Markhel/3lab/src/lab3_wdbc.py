"""
Лабораторная работа №3. Выборка из ЛР №2: Wisconsin Diagnostic Breast Cancer (wdbc.data), классификация.
Сравнение обучения глубокой сети без предобучения и с послойным автоэнкодерным предобучением.

Запуск:  python lab3_wdbc.py            (полный эксперимент)
         python lab3_wdbc.py --quick    (быстрая проверка работоспособности)
         python lab3_wdbc.py --no-show  (не открывать окна matplotlib)
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib

import common
from common import load_wdbc, make_split, subsample_train, conf_matrix, RESULTS_DIR, FIG_DIR, CLASS_NAMES
from deep_net import (Config, run_pair, runs_to_frame, summarize, paired_p,
                      ACT_LABELS, MODE_NAMES)

HIDDEN = [64, 32, 16, 8]            # 4 скрытых слоя + выходной = 5 слоёв с весами
TASK = "classification"
METRICS = ["Efficiency, %", "F1", "Precision", "Recall"]
DYNAMICS = ["Val loss (эпоха 0)", "Лучшая эпоха", "Колебательность, %", "Шум val-кривой, %",
            "Доля эпох с ростом val"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="мало эпох и запусков (проверка)")
    ap.add_argument("--no-show", action="store_true", help="не вызывать plt.show()")
    ap.add_argument("--seeds", type=int, default=10, help="число запусков основного эксперимента")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config(ae_epochs=100, head_epochs=20, epochs=150)
    n_seeds, n_seeds_extra = args.seeds, 5
    sizes_small = [30, 60, 120]
    if args.quick:
        cfg = Config(ae_epochs=5, head_epochs=2, epochs=15, smooth_from=5)
        n_seeds, n_seeds_extra, sizes_small = 2, 2, [30, 60]

    X, y = load_wdbc()
    sizes = [X.shape[1]] + HIDDEN + [2]
    print(f"\nАрхитектура: {'-'.join(map(str, sizes))}, активация скрытых слоёв и автоэнкодеров — ReLU, "
          f"выход — 2 логита (Cross-Entropy)")
    print(f"Предобучение: {cfg.ae_epochs} эпох на автоэнкодер, затем {cfg.head_epochs} эпох выходной слой, "
          f"тонкая настройка {cfg.epochs} эпох (Adam, lr = {cfg.lr}), выбор модели по минимуму val loss")

    # ---------------- Основной эксперимент (ReLU) ----------------
    splits, results = {}, {}
    for seed in range(n_seeds):
        splits[seed] = make_split(X, y, seed, TASK)
        results[seed] = run_pair(splits[seed], TASK, sizes, "relu", seed, cfg)
        r = results[seed]
        print(f"  seed {seed}: Efficiency без / с предобучением = "
              f"{r['plain']['test_metrics']['Efficiency, %']:.2f} / {r['pre']['test_metrics']['Efficiency, %']:.2f}")
    df = runs_to_frame(results)
    df.to_csv(RESULTS_DIR / "wdbc_main_runs.csv", index=False, encoding="utf-8-sig")

    s = splits[0]
    print(f"\nРазбиение: train = {len(s['X_tr'])}, val = {len(s['X_va'])}, test = {len(s['X_te'])}")
    print("\nКачество на тестовой выборке (среднее ± СКО по запускам; положительный класс — M):")
    tab = summarize(df, METRICS)
    print(tab.to_string())
    tab.to_csv(RESULTS_DIR / "wdbc_main_summary.csv", encoding="utf-8-sig")
    print("\nПарный t-тест (p-value) по запускам:")
    for m in METRICS:
        print(f"  {m:14s}: p = {paired_p(df, m):.4f}")

    cms = {}
    for m in ("plain", "pre"):
        cms[m] = sum(conf_matrix(results[k][m]["y_true"], results[k][m]["y_pred"]) for k in results)
        print(f"\nМатрица ошибок, {MODE_NAMES[m]} (сумма по {n_seeds} тестовым выборкам; строки — истина, "
              f"столбцы — прогноз, порядок B, M):\n{cms[m]}")
        pd.DataFrame(cms[m], index=["B (истина)", "M (истина)"], columns=["B (прогноз)", "M (прогноз)"]
                     ).to_csv(RESULTS_DIR / f"wdbc_confusion_{m}.csv", encoding="utf-8-sig")

    print("\nДинамика обучения (Колебательность — средний модуль изменения val loss между эпохами > "
          f"{cfg.smooth_from}, Шум — СКО отклонений от скользящего среднего; оба в % от средней val loss):")
    dyn = summarize(df, DYNAMICS)
    print(dyn.to_string())
    dyn.to_csv(RESULTS_DIR / "wdbc_dynamics.csv", encoding="utf-8-sig")
    for m in DYNAMICS[2:]:
        print(f"  {m:24s}: p = {paired_p(df, m):.4f}")

    # ---------------- графики ----------------
    np.savez(RESULTS_DIR / "wdbc_val_curves.npz",
             **{f"{m}_seed{k}": np.array(results[k][m]["history"]["val_loss"]) for k in results for m in ("plain", "pre")})
    hist = {MODE_NAMES[m]: [results[k][m]["history"] for k in results] for m in ("plain", "pre")}
    common.plot_val_curves(hist, "WDBC: функция потерь на валидации при тонкой настройке",
                           "wdbc_val_curves.png", "Cross-Entropy")
    common.plot_pretrain_losses(results[0]["pre"]["ae_curves"],
                                "WDBC: предобучение автоэнкодерами по слоям (seed 0)", "wdbc_ae_losses.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, m in zip(axes, ("plain", "pre")):
        ax.imshow(cms[m], cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cms[m][i, j], ha="center", va="center", fontsize=14,
                        color="white" if cms[m][i, j] > cms[m].max() / 2 else "black")
        ax.set_xticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.set_yticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Истина")
        ax.set_title(MODE_NAMES[m])
    fig.suptitle(f"WDBC: матрицы ошибок (сумма по {n_seeds} тестовым выборкам)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wdbc_confusion.png", dpi=150)

    # ---------------- Влияние функции активации ----------------
    print(f"\nВлияние функции активации ({n_seeds_extra} запусков):")
    act_frames = [df[df["seed"] < n_seeds_extra].assign(Активация=ACT_LABELS["relu"])]
    for act in ("tanh", "sigmoid"):
        res = {k: run_pair(splits[k], TASK, sizes, act, k, cfg) for k in range(n_seeds_extra)}
        act_frames.append(runs_to_frame(res, {"Активация": ACT_LABELS[act]}))
    adf = pd.concat(act_frames, ignore_index=True)
    atab = summarize(adf, ["Efficiency, %", "F1", "Колебательность, %", "Шум val-кривой, %"], group=("Активация", "Режим"))
    print(atab.to_string())
    atab.to_csv(RESULTS_DIR / "wdbc_activation.csv", encoding="utf-8-sig")

    # ---------------- Малые обучающие выборки ----------------
    print(f"\nМалые обучающие выборки (ReLU, {n_seeds} запусков; val и test те же):")
    small_frames = []
    for n in sizes_small:
        res = {k: run_pair(subsample_train(splits[k], n, k), TASK, sizes, "relu", k, cfg)
               for k in range(n_seeds)}
        small_frames.append(runs_to_frame(res, {"Объектов в train": n}))
    full = df.assign(**{"Объектов в train": len(s["X_tr"])})
    sdf = pd.concat(small_frames + [full], ignore_index=True)
    stab = summarize(sdf, ["Efficiency, %", "F1"], group=("Объектов в train", "Режим"))
    print(stab.to_string())
    stab.to_csv(RESULTS_DIR / "wdbc_small_samples.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mode, c in (("Без предобучения", "tab:red"), ("С предобучением", "tab:blue")):
        g = sdf[sdf["Режим"] == mode].groupby("Объектов в train")["Efficiency, %"]
        m, sd = g.mean(), g.std(ddof=1).fillna(0)
        ax.errorbar(m.index, m.values, yerr=sd.values, marker="o", color=c, capsize=3, label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("Объектов в обучающей выборке")
    ax.set_ylabel("Efficiency на тесте, %")
    ax.set_title("WDBC: зависимость качества от размера обучающей выборки")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wdbc_small_samples.png", dpi=150)

    print(f"\nТаблицы сохранены в {RESULTS_DIR}, графики — в {FIG_DIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
