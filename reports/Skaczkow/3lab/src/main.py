import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score)

warnings.filterwarnings("ignore")


HIDDEN = (128, 64, 32, 16)
PRE_EPOCHS = 60
MAX_EPOCHS = 200
PATIENCE = 20
BATCH = 128
LR = 1e-3
ALPHA = 1e-4


def load_energy(path, split):
    df = pd.read_csv(path)
    t = pd.to_datetime(df["date"])
    hour = t.dt.hour + t.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    y = df["Appliances"].values.astype(float)
    X = df.drop(columns=["date", "Appliances", "rv1", "rv2"]).values.astype(float)

    n = len(df)
    if split == "time":
        i1, i2 = int(n * 0.70), int(n * 0.85)
        return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])
    idx = np.random.RandomState(0).permutation(n)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    a, b, c = idx[:i1], idx[i1:i2], idx[i2:]
    return (X[a], y[a]), (X[b], y[b]), (X[c], y[c])


BITMAP = re.compile(r"^[01]{32}$")


def load_digits(path):
    bitmaps, labels, rows = [], [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if BITMAP.match(s):
                rows.append([int(ch) for ch in s])
            elif len(rows) == 32 and s.isdigit():
                bitmaps.append(rows)
                labels.append(int(s))
                rows = []
    B = np.array(bitmaps, dtype=np.uint8)
    X = B.reshape(len(B), 8, 4, 8, 4).sum(axis=(2, 4)).reshape(len(B), 64) / 16.0
    y = np.array(labels)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=0)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=0)
    return (Xtr, ytr), (Xva, yva), (Xte, yte)


def pretrain_layers(X, hidden, epochs, seed):
    layers, errors, H = [], [], X
    for k, h in enumerate(hidden):
        ae = MLPRegressor(hidden_layer_sizes=(h,), activation="relu", solver="adam",
                          batch_size=BATCH, learning_rate_init=LR, alpha=ALPHA,
                          max_iter=epochs, tol=0.0, n_iter_no_change=epochs + 1,
                          random_state=seed * 100 + k)
        ae.fit(H, H)
        errors.append(mean_squared_error(H, ae.predict(H)))
        W, b = ae.coefs_[0].copy(), ae.intercepts_[0].copy()
        layers.append((W, b))
        H = np.maximum(0.0, H @ W + b)
    return layers, errors


def make_model(task, seed, X0, y0, classes=None, pretrained=None):
    cls = MLPRegressor if task == "reg" else MLPClassifier
    m = cls(hidden_layer_sizes=HIDDEN, activation="relu", solver="adam", batch_size=BATCH,
            learning_rate_init=LR, alpha=ALPHA, random_state=seed)
    if task == "reg":
        m.partial_fit(X0[:1], y0[:1])
    else:
        m.partial_fit(X0[:1], y0[:1], classes=classes)
    if pretrained is not None:
        for i, (W, b) in enumerate(pretrained):
            m.coefs_[i][...] = W
            m.intercepts_[i][...] = b
    return m


def train(m, task, Xtr, ytr, Xva, yva, classes=None):
    best, best_w, wait, curve = np.inf, None, 0, []
    for ep in range(MAX_EPOCHS):
        m.partial_fit(Xtr, ytr) if task == "reg" else m.partial_fit(Xtr, ytr, classes=classes)
        if task == "reg":
            v = mean_squared_error(yva, m.predict(Xva))
        else:
            v = log_loss(yva, m.predict_proba(Xva), labels=classes)
        curve.append(v)
        if v < best - 1e-6:
            best, wait = v, 0
            best_w = ([c.copy() for c in m.coefs_], [b.copy() for b in m.intercepts_])
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    for i in range(len(m.coefs_)):
        m.coefs_[i][...] = best_w[0][i]
        m.intercepts_[i][...] = best_w[1][i]
    return curve


def reg_metrics(y, p):
    return {"MAPE, %": float(np.mean(np.abs(y - p) / y) * 100),
            "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def clf_metrics(y, p):
    return {"Accuracy": accuracy_score(y, p),
            "F1 macro": f1_score(y, p, average="macro")}


def run_experiment(name, task, data, seeds, outdir):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    sc = StandardScaler().fit(Xtr)
    if task == "digits":
        Xtr_s, Xva_s, Xte_s = Xtr, Xva, Xte
    else:
        Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    t = "reg" if task == "reg" else "clf"
    classes = None
    if t == "reg":
        mu, sd = ytr.mean(), ytr.std()
        ytr_t, yva_t = (ytr - mu) / sd, (yva - mu) / sd
    else:
        classes = np.unique(ytr)
        ytr_t, yva_t = ytr, yva

    print(f"\n=== {name}: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}, "
          f"признаков {Xtr.shape[1]} ===")
    rows, curves, preds, pre_err = [], {"без": [], "с": []}, {}, []
    for seed in range(seeds):

        mA = make_model(t, seed, Xtr_s, ytr_t, classes)
        cA = train(mA, t, Xtr_s, ytr_t, Xva_s, yva_t, classes)

        layers, err = pretrain_layers(Xtr_s, HIDDEN, PRE_EPOCHS, seed)
        pre_err.append(err)
        mB = make_model(t, seed, Xtr_s, ytr_t, classes, pretrained=layers)
        cB = train(mB, t, Xtr_s, ytr_t, Xva_s, yva_t, classes)
        curves["без"].append(cA)
        curves["с"].append(cB)
        for tag, m, c in (("без предобучения", mA, cA), ("с предобучением", mB, cB)):
            p = m.predict(Xte_s)
            if t == "reg":
                p = p * sd + mu
                met = reg_metrics(yte, p)
            else:
                met = clf_metrics(yte, p)
            met.update({"variant": tag, "seed": seed, "epochs": len(c),
                        "best_epoch": int(np.argmin(c)) + 1,
                        "val_loss_best": float(np.min(c))})
            rows.append(met)
            if seed == 0:
                preds[tag] = p
        print(f"  seed {seed}: эпох без/с = {len(cA)}/{len(cB)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"results_{task}.csv"), index=False)
    metric_cols = (["MAPE, %", "MAE", "RMSE", "R2"] if t == "reg"
                   else ["Accuracy", "F1 macro"]) + ["best_epoch", "val_loss_best"]
    g = df.groupby("variant")[metric_cols]
    mean, std = g.mean(), g.std()
    table = pd.DataFrame({c: [f"{mean.loc[v, c]:.3f} +- {std.loc[v, c]:.3f}"
                              for v in mean.index] for c in metric_cols}, index=mean.index)
    print("\nТест, среднее +- std по seed:\n" + table.to_string())

    key = "MAPE, %" if t == "reg" else "F1 macro"
    a = df[df.variant == "без предобучения"].sort_values("seed")[key].values
    b = df[df.variant == "с предобучением"].sort_values("seed")[key].values
    better = (b < a) if t == "reg" else (b > a)
    print(f"\nПредобучение лучше по {key} на {int(better.sum())} из {len(a)} seed "
          f"(среднее различие {np.mean(b - a):+.3f})")
    print("Ошибка реконструкции автоэнкодеров по слоям (MSE, seed 0):",
          np.round(pre_err[0], 4))


    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, col, lab in (("без", "tab:red", "без предобучения"), ("с", "tab:green", "с предобучением")):
        L = max(len(c) for c in curves[tag])
        M = np.full((len(curves[tag]), L), np.nan)
        for i, c in enumerate(curves[tag]):
            M[i, :len(c)] = c
            ax.plot(range(1, len(c) + 1), c, color=col, alpha=0.2, lw=0.8)
        ax.plot(range(1, L + 1), np.nanmean(M, axis=0), color=col, lw=2, label=lab)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Ошибка на валидации (MSE)" if t == "reg" else "Ошибка на валидации (log-loss)")
    ax.set_title(f"{name}: кривые обучения")
    if t == "reg":
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"curves_{task}.png"), dpi=140)
    plt.close(fig)

    if t == "reg":
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
        lim = np.percentile(yte, 99.5)
        for ax, tag in zip(axs, ("без предобучения", "с предобучением")):
            ax.scatter(yte, preds[tag], s=5, alpha=0.3)
            ax.plot([0, lim], [0, lim], "k--", lw=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Факт, Вт*ч")
            ax.set_title(tag)
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("Прогноз, Вт*ч")
        fig.suptitle(f"{name}: прогноз и факт на тесте (seed 0)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"pred_vs_true_{task}.png"), dpi=140)
        plt.close(fig)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, tag in zip(axs, ("без предобучения", "с предобучением")):
            cm = confusion_matrix(yte, preds[tag], labels=classes)
            ax.imshow(cm, cmap="Blues")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xlabel("Предсказано")
            ax.set_ylabel("Истинно")
            ax.set_title(f"{tag}\nF1 macro = {f1_score(yte, preds[tag], average='macro'):.3f}")
        fig.suptitle(f"{name}: матрица ошибок на тесте (seed 0)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"confusion_{task}.png"), dpi=140)
        plt.close(fig)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="energydata_complete.csv")
    ap.add_argument("--digits", default="optdigits-orig.cv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", choices=["random", "time"], default="random",
                    help="разбиение Appliances: случайное или хронологическое")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(args.energy):
        run_experiment(f"Appliances (регрессия, разбиение: {args.split})", "reg",
                       load_energy(args.energy, args.split), args.seeds, args.out)
    if os.path.exists(args.digits):
        run_experiment("optdigits (классификация)", "digits",
                       load_digits(args.digits), args.seeds, args.out)