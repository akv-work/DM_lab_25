"""
Лабораторная работа №4. Выборка из ЛР №2: Wisconsin Diagnostic Breast Cancer (wdbc.data), классификация.
Сравнение: без предобучения (ЛР3) / предобучение автоэнкодерами (ЛР3) / предобучение стеком RBM (ЛР4).

Запуск:  python lab4_wdbc.py            (полный эксперимент)
         python lab4_wdbc.py --quick    (быстрая проверка работоспособности)
         python lab4_wdbc.py --no-show  (не открывать окна matplotlib)
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib

import common
from common import load_wdbc, make_split, subsample_train, conf_matrix, RESULTS_DIR, FIG_DIR, CLASS_NAMES
from deep_net import Config, ACT_LABELS
from experiment import (MODES, MODE_NAMES, run_modes, runs_to_frame, summarize, print_pvalues,
                        convergence_report)

HIDDEN = [64, 32, 16, 8]            # 4 скрытых слоя + выходной = 5 слоёв с весами (как в ЛР №3)
TASK = "classification"
METRICS = ["Efficiency, %", "F1", "Precision", "Recall"]
DYNAMICS = ["Val loss (эпоха 0)", "Лучшая эпоха", "Колебательность, %", "Шум val-кривой, %",
            "Доля эпох с ростом val"]
COLORS = dict(zip(MODES, common.MODE_COLORS))


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

    cfg = Config(ae_epochs=100, rbm_epochs=100, head_epochs=20, epochs=150)
    n_seeds, n_seeds_extra = args.seeds, 5
    sizes_small = [30, 60, 120]
    if args.quick:
        cfg = Config(ae_epochs=5, rbm_epochs=5, head_epochs=2, epochs=15, smooth_from=5)
        n_seeds, n_seeds_extra, sizes_small = 2, 2, [30, 60]

    X, y = load_wdbc()
    sizes = [X.shape[1]] + HIDDEN + [2]
    print(f"\nАрхитектура: {'-'.join(map(str, sizes))}, ReLU в скрытых слоях, выход — 2 логита (Cross-Entropy)")
    print(f"Предобучение автоэнкодерами: {cfg.ae_epochs} эпох на слой (Adam, lr = {cfg.ae_lr}); "
          f"предобучение RBM: {cfg.rbm_epochs} эпох на слой (CD-1, скорости {cfg.rbm_rates['relu'][0]} (ReLU) / {cfg.rbm_rates['sigmoid']} (Logistic), "
          f"моментум {cfg.rbm_start_momentum} -> {cfg.rbm_final_momentum})")
    print(f"Затем {cfg.head_epochs} эпох выходной слой, тонкая настройка {cfg.epochs} эпох "
          f"(Adam, lr = {cfg.lr}), выбор модели по минимуму val loss")

    # ---------------- Основной эксперимент (ReLU) ----------------
    splits, results = {}, {}
    for seed in range(n_seeds):
        splits[seed] = make_split(X, y, seed, TASK)
        results[seed] = run_modes(splits[seed], TASK, sizes, "relu", seed, cfg)
        print(f"  seed {seed}: Efficiency " + " / ".join(
            f"{MODE_NAMES[m]} {results[seed][m]['test_metrics']['Efficiency, %']:.2f}" for m in MODES))
    df = runs_to_frame(results)
    df.to_csv(RESULTS_DIR / "wdbc_main_runs.csv", index=False, encoding="utf-8-sig")
    s = splits[0]
    print(f"\nРазбиение: train = {len(s['X_tr'])}, val = {len(s['X_va'])}, test = {len(s['X_te'])}")

    print("\nКачество на тестовой выборке (среднее ± СКО по запускам; положительный класс — M):")
    tab = summarize(df, METRICS)
    print(tab.to_string())
    tab.to_csv(RESULTS_DIR / "wdbc_main_summary.csv", encoding="utf-8-sig")
    print("\nПарный t-тест, p-value:")
    print_pvalues(df, METRICS)

    cms = {}
    for m in MODES:
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
    print_pvalues(df, DYNAMICS[2:])

    text = convergence_report(results, 0.20, "WDBC (Cross-Entropy)")
    (RESULTS_DIR / "wdbc_convergence.txt").write_text(text, encoding="utf-8")

    # ---------------- графики ----------------
    np.savez(RESULTS_DIR / "wdbc_val_curves.npz",
             **{f"{m}_seed{k}": np.array(results[k][m]["history"]["val_loss"]) for k in results for m in MODES})
    hist = {MODE_NAMES[m]: [results[k][m]["history"] for k in results] for m in MODES}
    common.plot_val_curves(hist, "WDBC: функция потерь на валидации при тонкой настройке",
                           "wdbc_val_curves.png", "Cross-Entropy")
    common.plot_pretrain_losses(results[0]["ae"]["pretrain_curves"],
                                "WDBC: предобучение автоэнкодерами (seed 0)", "wdbc_ae_losses.png",
                                "Автоэнкодер слоя")
    common.plot_pretrain_losses(results[0]["rbm"]["pretrain_curves"],
                                "WDBC: предобучение стеком RBM (seed 0)", "wdbc_rbm_losses.png", "RBM слоя")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, m in zip(axes, MODES):
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

    # ---------------- Логистическая активация ----------------
    print(f"\nЛогистическая активация скрытых слоёв (как в лекции; {n_seeds_extra} запусков):")
    act_frames = [df[df["seed"] < n_seeds_extra].assign(Активация=ACT_LABELS["relu"])]
    res = {k: run_modes(splits[k], TASK, sizes, "sigmoid", k, cfg) for k in range(n_seeds_extra)}
    act_frames.append(runs_to_frame(res, {"Активация": ACT_LABELS["sigmoid"]}))
    adf = pd.concat(act_frames, ignore_index=True)
    atab = summarize(adf, ["Efficiency, %", "F1", "Колебательность, %", "Шум val-кривой, %"],
                     group=("Активация", "Режим"))
    print(atab.to_string())
    atab.to_csv(RESULTS_DIR / "wdbc_activation.csv", encoding="utf-8-sig")

    # ---------------- Малые обучающие выборки ----------------
    print(f"\nМалые обучающие выборки (ReLU, {n_seeds} запусков; val и test те же):")
    small_frames = []
    for n in sizes_small:
        res = {k: run_modes(subsample_train(splits[k], n, k), TASK, sizes, "relu", k, cfg)
               for k in range(n_seeds)}
        small_frames.append(runs_to_frame(res, {"Объектов в train": n}))
    full = df.assign(**{"Объектов в train": len(s["X_tr"])})
    sdf = pd.concat(small_frames + [full], ignore_index=True)
    sdf.to_csv(RESULTS_DIR / "wdbc_small_samples_runs.csv", index=False, encoding="utf-8-sig")
    stab = summarize(sdf, ["Efficiency, %", "F1"], group=("Объектов в train", "Режим"))
    print(stab.to_string())
    stab.to_csv(RESULTS_DIR / "wdbc_small_samples.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in MODES:
        g = sdf[sdf["Режим"] == MODE_NAMES[m]].groupby("Объектов в train")["Efficiency, %"]
        mean, sd = g.mean(), g.std(ddof=1).fillna(0)
        ax.errorbar(mean.index, mean.values, yerr=sd.values, marker="o", color=COLORS[m],
                    capsize=3, label=MODE_NAMES[m])
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
