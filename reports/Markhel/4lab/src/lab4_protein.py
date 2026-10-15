"""
Лабораторная работа №4, вариант 6.
Physicochemical Properties of Protein Tertiary Structure (CASP.csv), регрессия, целевая переменная RMSD.
Сравнение: без предобучения (ЛР3) / предобучение автоэнкодерами (ЛР3) / предобучение стеком RBM (ЛР4).

Запуск:  python lab4_protein.py            (полный эксперимент)
         python lab4_protein.py --quick    (быстрая проверка работоспособности)
         python lab4_protein.py --no-show  (не открывать окна matplotlib)
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib

import common
from common import load_protein, make_split, subsample_train, RESULTS_DIR, FIG_DIR
from deep_net import Config, ACT_LABELS
from experiment import (MODES, MODE_NAMES, run_modes, runs_to_frame, summarize, print_pvalues,
                        convergence_report)

HIDDEN = [128, 64, 32, 16]          # 4 скрытых слоя + выходной = 5 слоёв с весами (как в ЛР №3)
TASK = "regression"
METRICS = ["RMSE", "MAE", "R2", "MAPE, %", "WAPE, %"]
DYNAMICS = ["Val loss (эпоха 0)", "Лучшая эпоха", "Колебательность, %", "Шум val-кривой, %",
            "Доля эпох с ростом val"]
COLORS = dict(zip(MODES, common.MODE_COLORS))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="мало эпох и запусков (проверка)")
    ap.add_argument("--no-show", action="store_true", help="не вызывать plt.show()")
    ap.add_argument("--seeds", type=int, default=5, help="число запусков основного эксперимента")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config()
    n_seeds, n_seeds_extra = args.seeds, 3
    sizes_small = [250, 1000, 5000]
    if args.quick:
        cfg = Config(ae_epochs=3, rbm_epochs=3, head_epochs=2, epochs=10, smooth_from=5)
        n_seeds, n_seeds_extra, sizes_small = 2, 2, [250, 1000]

    X, y, features = load_protein()
    sizes = [X.shape[1]] + HIDDEN + [1]
    print(f"\nАрхитектура: {'-'.join(map(str, sizes))}, ReLU в скрытых слоях, выход линейный, MSE")
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
        print(f"  seed {seed}: RMSE " + " / ".join(
            f"{MODE_NAMES[m]} {results[seed][m]['test_metrics']['RMSE']:.4f}" for m in MODES))
    df = runs_to_frame(results)
    df.to_csv(RESULTS_DIR / "protein_main_runs.csv", index=False, encoding="utf-8-sig")
    s = splits[0]
    print(f"\nРазбиение: train = {len(s['X_tr'])}, val = {len(s['X_va'])}, test = {len(s['X_te'])}")

    print("\nКачество на тестовой выборке (среднее ± СКО по запускам; MAPE — по объектам с RMSD >= 1 Å):")
    tab = summarize(df, METRICS)
    print(tab.to_string())
    tab.to_csv(RESULTS_DIR / "protein_main_summary.csv", encoding="utf-8-sig")
    print("\nПарный t-тест, p-value:")
    print_pvalues(df, METRICS)

    print("\nДинамика обучения (Колебательность — средний модуль изменения val loss между эпохами > "
          f"{cfg.smooth_from}, Шум — СКО отклонений от скользящего среднего; оба в % от средней val loss):")
    dyn = summarize(df, DYNAMICS)
    print(dyn.to_string())
    dyn.to_csv(RESULTS_DIR / "protein_dynamics.csv", encoding="utf-8-sig")
    print_pvalues(df, DYNAMICS[2:])

    text = convergence_report(results, 0.55, "Protein (MSE, стандартизованный RMSD)")
    (RESULTS_DIR / "protein_convergence.txt").write_text(text, encoding="utf-8")

    # ---------------- графики основного эксперимента ----------------
    np.savez(RESULTS_DIR / "protein_val_curves.npz",
             **{f"{m}_seed{k}": np.array(results[k][m]["history"]["val_loss"]) for k in results for m in MODES})
    hist = {MODE_NAMES[m]: [results[k][m]["history"] for k in results] for m in MODES}
    common.plot_val_curves(hist, "Protein (RMSD): функция потерь на валидации при тонкой настройке",
                           "protein_val_curves.png", "MSE (стандартизованный RMSD)")
    common.plot_pretrain_losses(results[0]["ae"]["pretrain_curves"],
                                "Protein: предобучение автоэнкодерами (seed 0)",
                                "protein_ae_losses.png", "Автоэнкодер слоя")
    common.plot_pretrain_losses(results[0]["rbm"]["pretrain_curves"],
                                "Protein: предобучение стеком RBM (seed 0)",
                                "protein_rbm_losses.png", "RBM слоя")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, m in zip(axes, MODES):
        r = results[0][m]
        ax.scatter(r["y_true"], r["y_pred"], s=3, alpha=0.3, color=COLORS[m])
        lim = [0, max(r["y_true"].max(), r["y_pred"].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{MODE_NAMES[m]}: R² = {r['test_metrics']['R2']:.3f}")
        ax.set_xlabel("Истинный RMSD")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Предсказанный RMSD")
    fig.suptitle("Protein: тест, seed 0")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "protein_pred_vs_true.png", dpi=150)

    # ---------------- Логистическая активация (RBM согласована с сетью) ----------------
    print(f"\nЛогистическая активация скрытых слоёв (как в лекции; {n_seeds_extra} запуска):")
    act_frames = [df[df["seed"] < n_seeds_extra].assign(Активация=ACT_LABELS["relu"])]
    res = {k: run_modes(splits[k], TASK, sizes, "sigmoid", k, cfg) for k in range(n_seeds_extra)}
    act_frames.append(runs_to_frame(res, {"Активация": ACT_LABELS["sigmoid"]}))
    adf = pd.concat(act_frames, ignore_index=True)
    atab = summarize(adf, ["RMSE", "R2", "MAPE, %", "Колебательность, %", "Шум val-кривой, %"],
                     group=("Активация", "Режим"))
    print(atab.to_string())
    atab.to_csv(RESULTS_DIR / "protein_activation.csv", encoding="utf-8-sig")
    print("p-value (только Logistic):")
    print_pvalues(adf[adf["Активация"] == ACT_LABELS["sigmoid"]], ["RMSE", "R2"])

    # ---------------- Малые обучающие выборки ----------------
    print(f"\nМалые обучающие выборки (ReLU, {n_seeds} запусков; val и test те же):")
    small_frames = []
    for n in sizes_small:
        res = {k: run_modes(subsample_train(splits[k], n, k), TASK, sizes, "relu", k, cfg)
               for k in range(n_seeds)}
        small_frames.append(runs_to_frame(res, {"Объектов в train": n}))
    full = df.assign(**{"Объектов в train": len(s["X_tr"])})
    sdf = pd.concat(small_frames + [full], ignore_index=True)
    stab = summarize(sdf, ["RMSE", "R2", "MAPE, %"], group=("Объектов в train", "Режим"))
    print(stab.to_string())
    stab.to_csv(RESULTS_DIR / "protein_small_samples.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in MODES:
        g = sdf[sdf["Режим"] == MODE_NAMES[m]].groupby("Объектов в train")["RMSE"]
        mean, sd = g.mean(), g.std(ddof=1).fillna(0)
        ax.errorbar(mean.index, mean.values, yerr=sd.values, marker="o", color=COLORS[m],
                    capsize=3, label=MODE_NAMES[m])
    ax.set_xscale("log")
    ax.set_xlabel("Объектов в обучающей выборке")
    ax.set_ylabel("RMSE на тесте (RMSD)")
    ax.set_title("Protein: зависимость качества от размера обучающей выборки")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "protein_small_samples.png", dpi=150)

    print(f"\nТаблицы сохранены в {RESULTS_DIR}, графики — в {FIG_DIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
