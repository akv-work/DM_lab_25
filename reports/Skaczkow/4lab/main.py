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

RBM_EPOCHS = 50
RBM_LR_GAUSS = 5e-3
RBM_LR_BIN = 5e-2
RBM_MOMENTUM = (0.5, 0.9)
RBM_DECAY = 2e-4
RBM_BATCH = 32

VARIANTS = ("без предобучения", "AE", "RBM")
COLORS = {"без предобучения": "tab:red", "AE": "tab:green", "RBM": "tab:blue"}

ACT = "relu"


def act(z):
    if ACT == "relu":
        return np.maximum(0.0, z)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


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
    i1, i2 = int(n * 0.70), int(n * 0.85)
    if split == "time":
        return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])
    idx = np.random.RandomState(0).permutation(n)
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


def pretrain_ae(X, hidden, epochs, seed):
    layers, errors, H = [], [], X
    for k, h in enumerate(hidden):
        ae = MLPRegressor(hidden_layer_sizes=(h,), activation="relu" if ACT == "relu" else "logistic",
                          solver="adam", batch_size=BATCH, learning_rate_init=LR, alpha=ALPHA,
                          max_iter=epochs, tol=0.0, n_iter_no_change=epochs + 1,
                          random_state=seed * 100 + k)
        ae.fit(H, H)
        errors.append(mean_squared_error(H, ae.predict(H)))
        W, b = ae.coefs_[0].copy(), ae.intercepts_[0].copy()
        layers.append((W, b))
        H = act(H @ W + b)
    return layers, errors


class RBM:
    def __init__(self, n_vis, n_hid, gaussian, rng):
        self.W = rng.normal(0, 0.01, (n_vis, n_hid))
        self.b = np.zeros(n_hid)
        self.c = np.zeros(n_vis)
        self.gaussian = gaussian
        self.rng = rng

    def h_prob(self, v):
        return sigmoid(v @ self.W + self.b)

    def v_mean(self, h):
        z = h @ self.W.T + self.c
        return z if self.gaussian else sigmoid(z)

    def fit(self, V, epochs):
        lr = RBM_LR_GAUSS if self.gaussian else RBM_LR_BIN
        if not self.gaussian:
            p = np.clip(V.mean(axis=0), 1e-3, 1 - 1e-3)
            self.c = np.log(p / (1 - p))
        dW, db, dc = np.zeros_like(self.W), np.zeros_like(self.b), np.zeros_like(self.c)
        history = []
        for ep in range(epochs):
            mom = RBM_MOMENTUM[0] if ep < 5 else RBM_MOMENTUM[1]
            idx = self.rng.permutation(len(V))
            for s in range(0, len(V), RBM_BATCH):
                v0 = V[idx[s:s + RBM_BATCH]]
                n = len(v0)
                h0 = self.h_prob(v0)
                h0s = (self.rng.random_sample(h0.shape) < h0).astype(float)
                v1 = self.v_mean(h0s)
                h1 = self.h_prob(v1)
                gW = (v0.T @ h0 - v1.T @ h1) / n - RBM_DECAY * self.W
                gb = (h0 - h1).mean(axis=0)
                gc = (v0 - v1).mean(axis=0)
                dW = mom * dW + lr * gW
                db = mom * db + lr * gb
                dc = mom * dc + lr * gc
                self.W += dW
                self.b += db
                self.c += dc
            history.append(self.recon_error(V))
        return history

    def recon_error(self, V):
        return float(np.mean((V - self.v_mean(self.h_prob(V))) ** 2))


def pretrain_rbm(X, hidden, epochs, seed, gaussian_input):
    rng = np.random.RandomState(1000 + seed)
    layers, histories, H = [], [], X
    for k, h in enumerate(hidden):
        rbm = RBM(H.shape[1], h, gaussian=(gaussian_input and k == 0), rng=rng)
        histories.append(rbm.fit(H, epochs))
        layers.append((rbm.W.copy(), rbm.b.copy()))
        H = rbm.h_prob(H)
    return layers, histories


def make_model(task, seed, X0, y0, classes=None, pretrained=None):
    cls = MLPRegressor if task == "reg" else MLPClassifier
    m = cls(hidden_layer_sizes=HIDDEN, activation="relu" if ACT == "relu" else "logistic",
            solver="adam", batch_size=BATCH, learning_rate_init=LR, alpha=ALPHA, random_state=seed)
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
    t = "reg" if task == "reg" else "clf"
    if t == "reg":
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        mu, sd = ytr.mean(), ytr.std()
        ytr_t, yva_t = (ytr - mu) / sd, (yva - mu) / sd
        classes = None
    else:
        Xtr_s, Xva_s, Xte_s = Xtr, Xva, Xte
        ytr_t, yva_t = ytr, yva
        classes = np.unique(ytr)

    print(f"\n=== {name}: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}, "
          f"признаков {Xtr.shape[1]}, активация {ACT} ===")
    rows, preds = [], {}
    curves = {v: [] for v in VARIANTS}
    ae_err, rbm_hist = [], []
    for seed in range(seeds):
        inits = {"без предобучения": None}
        inits["AE"], e = pretrain_ae(Xtr_s, HIDDEN, PRE_EPOCHS, seed)
        ae_err.append(e)
        inits["RBM"], hst = pretrain_rbm(Xtr_s, HIDDEN, RBM_EPOCHS, seed, gaussian_input=(t == "reg"))
        rbm_hist.append(hst)

        ep_info = []
        for v in VARIANTS:
            m = make_model(t, seed, Xtr_s, ytr_t, classes, pretrained=inits[v])
            c = train(m, t, Xtr_s, ytr_t, Xva_s, yva_t, classes)
            curves[v].append(c)
            p = m.predict(Xte_s)
            if t == "reg":
                p = p * sd + mu
                met = reg_metrics(yte, p)
            else:
                met = clf_metrics(yte, p)
            met.update({"variant": v, "seed": seed, "epochs": len(c),
                        "best_epoch": int(np.argmin(c)) + 1,
                        "val_loss_start": float(c[0]),
                        "val_loss_best": float(np.min(c))})
            rows.append(met)
            ep_info.append(f"{v}: {len(c)}")
            if seed == 0:
                preds[v] = p
        print(f"  seed {seed}: эпох — " + ", ".join(ep_info))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"results_{task}_{ACT}.csv"), index=False)
    metric_cols = (["MAPE, %", "MAE", "RMSE", "R2"] if t == "reg"
                   else ["Accuracy", "F1 macro"]) + ["val_loss_start", "best_epoch", "val_loss_best"]
    g = df.groupby("variant")[metric_cols]
    mean, std = g.mean().loc[list(VARIANTS)], g.std().loc[list(VARIANTS)]
    table = pd.DataFrame({c: [f"{mean.loc[v, c]:.3f} +- {std.loc[v, c]:.3f}" for v in VARIANTS]
                          for c in metric_cols}, index=list(VARIANTS))
    lines = [f"{name}, активация {ACT}", "Тест, среднее +- std по seed:", table.to_string(), ""]

    key = "MAPE, %" if t == "reg" else "F1 macro"
    base = df[df.variant == VARIANTS[0]].sort_values("seed")[key].values
    for v in VARIANTS[1:]:
        x = df[df.variant == v].sort_values("seed")[key].values
        better = (x < base) if t == "reg" else (x > base)
        lines.append(f"{v} лучше варианта без предобучения по {key} на {int(better.sum())} из "
                     f"{len(base)} seed (среднее различие {np.mean(x - base):+.3f})")
    lines.append("Ошибка реконструкции AE по слоям (MSE, seed 0): "
                 + str(np.round(ae_err[0], 4).tolist()))
    lines.append("Ошибка реконструкции RBM по слоям (MSE, seed 0, первая -> последняя эпоха): "
                 + ", ".join(f"{h[0]:.4f}->{h[-1]:.4f}" for h in rbm_hist[0]))
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(outdir, f"summary_{task}_{ACT}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for v in VARIANTS:
        L = max(len(c) for c in curves[v])
        M = np.full((len(curves[v]), L), np.nan)
        for i, c in enumerate(curves[v]):
            M[i, :len(c)] = c
            ax.plot(range(1, len(c) + 1), c, color=COLORS[v], alpha=0.15, lw=0.8)
        ax.plot(range(1, L + 1), np.nanmean(M, axis=0), color=COLORS[v], lw=2, label=v)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Ошибка на валидации (MSE)" if t == "reg" else "Ошибка на валидации (log-loss)")
    ax.set_title(f"{name}: кривые обучения ({ACT})")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"curves_{task}_{ACT}.png"), dpi=140)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(HIDDEN), figsize=(14, 3.4))
    for k, ax in enumerate(axs):
        ax.plot(range(1, RBM_EPOCHS + 1), rbm_hist[0][k], marker=".")
        ax.set_title(f"RBM {k + 1}: {([Xtr.shape[1]] + list(HIDDEN))[k]} -> {HIDDEN[k]}")
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("MSE реконструкции")
    fig.suptitle(f"{name}: обучение RBM (CD-1), seed 0")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"rbm_recon_{task}_{ACT}.png"), dpi=140)
    plt.close(fig)

    if t == "reg":
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        lim = np.percentile(yte, 99.5)
        for ax, v in zip(axs, VARIANTS):
            ax.scatter(yte, preds[v], s=5, alpha=0.3, color=COLORS[v])
            ax.plot([0, lim], [0, lim], "k--", lw=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Факт, Вт*ч")
            ax.set_title(v)
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("Прогноз, Вт*ч")
        fig.suptitle(f"{name}: прогноз и факт на тесте (seed 0, {ACT})")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"pred_vs_true_{task}_{ACT}.png"), dpi=140)
        plt.close(fig)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
        for ax, v in zip(axs, VARIANTS):
            cm = confusion_matrix(yte, preds[v], labels=classes)
            ax.imshow(cm, cmap="Blues")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xlabel("Предсказано")
            ax.set_ylabel("Истинно")
            ax.set_title(f"{v}\nF1 macro = {f1_score(yte, preds[v], average='macro'):.3f}")
        fig.suptitle(f"{name}: матрица ошибок на тесте (seed 0, {ACT})")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"confusion_{task}_{ACT}.png"), dpi=140)
        plt.close(fig)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="energydata_complete.csv")
    ap.add_argument("--digits", default="optdigits-orig.cv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--act", choices=["relu", "logistic"], default="relu")
    ap.add_argument("--only", choices=["all", "reg", "digits"], default="all")
    args = ap.parse_args()
    ACT = args.act
    os.makedirs(args.out, exist_ok=True)

    if args.only in ("all", "reg") and os.path.exists(args.energy):
        run_experiment("Appliances (регрессия)", "reg",
                       load_energy(args.energy, args.split), args.seeds, args.out)
    if args.only in ("all", "digits") and os.path.exists(args.digits):
        run_experiment("optdigits (классификация)", "digits",
                       load_digits(args.digits), args.seeds, args.out)
