"""
Лабораторная работа №3. Общие функции: загрузка выборок, разбиение, метрики, графики.
"""
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

for _stream in (sys.stdout, sys.stderr):  # русский текст в консоли Windows (cp1251) без ошибок
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PROTEIN_FILE = BASE_DIR / "CASP.csv"
PROTEIN_URL = ("https://archive.ics.uci.edu/static/public/265/"
               "physicochemical+properties+of+protein+tertiary+structure.zip")
PROTEIN_TARGET = "RMSD"

WDBC_FILE = BASE_DIR / "wdbc.data"
BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
WDBC_FEATURES = ([f"{f}_mean" for f in BASE_FEATURES]
                 + [f"{f}_se" for f in BASE_FEATURES]
                 + [f"{f}_worst" for f in BASE_FEATURES])
CLASS_NAMES = {0: "Benign (B)", 1: "Malignant (M)"}

SPLIT = (0.6, 0.2, 0.2)  # train / val / test
MAPE_MIN_RMSD = 1.0      # MAPE не определён при RMSD = 0 и неустойчив при RMSD ~ 0: считаем по RMSD >= 1 Å


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------------
def load_protein(verbose=True):
    """Physicochemical Properties of Protein Tertiary Structure (вариант 6), регрессия, y = RMSD."""
    if not PROTEIN_FILE.exists():
        print("CASP.csv не найден — загрузка с UCI Machine Learning Repository...")
        zip_path = BASE_DIR / "protein.zip"
        urllib.request.urlretrieve(PROTEIN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extract("CASP.csv", BASE_DIR)
        zip_path.unlink()
    df = pd.read_csv(PROTEIN_FILE)
    features = [c for c in df.columns if c != PROTEIN_TARGET]

    missing = int(df.isna().sum().sum())
    if missing:
        df = df.fillna(df.median())
    X = df[features].to_numpy(dtype=np.float64)
    y = df[PROTEIN_TARGET].to_numpy(dtype=np.float64)
    if verbose:
        print(f"[Protein] объектов: {len(df)}, признаков: {X.shape[1]}, "
              f"пропусков заменено: {missing}")
        print(f"[Protein] RMSD: min = {y.min():.3f}, среднее = {y.mean():.3f}, "
              f"max = {y.max():.3f}, нулевых значений: {(y == 0).sum()}")
    return X, y, features


def load_wdbc(verbose=True):
    """WDBC из ЛР №2, классификация: B = 0, M = 1."""
    df = pd.read_csv(WDBC_FILE, header=None, names=["ID", "Diagnosis"] + WDBC_FEATURES)
    missing = int(df[WDBC_FEATURES].isna().sum().sum())
    if missing:
        df[WDBC_FEATURES] = df[WDBC_FEATURES].fillna(df[WDBC_FEATURES].median())
    y = df["Diagnosis"].map({"B": 0, "M": 1}).to_numpy(dtype=np.int64)
    X = df[WDBC_FEATURES].to_numpy(dtype=np.float64)
    if verbose:
        print(f"[WDBC] объектов: {len(df)}, признаков: {X.shape[1]}, "
              f"пропусков заменено: {missing}, классы: B = {(y == 0).sum()}, M = {(y == 1).sum()}")
    return X, y


# ----------------------------------------------------------------------------
# Разбиение и нормировка (параметры нормировки считаются только по train)
# ----------------------------------------------------------------------------
def make_split(X, y, seed, task):
    stratify = y if task == "classification" else None
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=1 - SPLIT[0], random_state=seed, stratify=stratify)
    stratify = y_rest if task == "classification" else None
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
        random_state=seed, stratify=stratify)

    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = None
    if task == "regression":
        y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))

    def tx(a):
        return torch.tensor(x_scaler.transform(a), dtype=torch.float32)

    def ty(a):
        if task == "regression":
            return torch.tensor(y_scaler.transform(a.reshape(-1, 1)), dtype=torch.float32)
        return torch.tensor(a, dtype=torch.long)

    return {"X_tr": tx(X_tr), "y_tr": ty(y_tr),
            "X_va": tx(X_va), "y_va": ty(y_va),
            "X_te": tx(X_te), "y_te": ty(y_te),
            "y_scaler": y_scaler}


def subsample_train(split, n, seed):
    """Уменьшенная обучающая выборка (val и test остаются прежними)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(split["X_tr"]), generator=g)[:n]
    out = dict(split)
    out["X_tr"], out["y_tr"] = split["X_tr"][idx], split["y_tr"][idx]
    return out


# ----------------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """Метрики в исходных единицах RMSD. MAPE считается по объектам с RMSD >= MAPE_MIN_RMSD
    (в выборке 272 объекта с RMSD = 0, где MAPE не определён); WAPE = sum|e| / sum|y| учитывает все объекты."""
    ok = y_true >= MAPE_MIN_RMSD
    mape = float(np.mean(np.abs((y_true[ok] - y_pred[ok]) / y_true[ok])) * 100)
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE, %": mape,
            "WAPE, %": wape}


def classification_metrics(y_true, y_pred):
    return {"Efficiency, %": float(accuracy_score(y_true, y_pred) * 100),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0))}


def conf_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def plot_val_curves(histories, title, filename, ylabel):
    """histories: {'Без предобучения': [hist, ...], 'С предобучением': [...]}.
    Слева — кривая валидации одного запуска (видны колебания), справа — среднее по запускам."""
    colors = {"Без предобучения": "tab:red", "С предобучением": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, runs in histories.items():
        c = colors[name]
        axes[0].plot(runs[0]["val_loss"], color=c, label=name, lw=1.4)
        arr = np.array([h["val_loss"] for h in runs])
        m, s = arr.mean(axis=0), arr.std(axis=0)
        axes[1].plot(m, color=c, label=name, lw=1.6)
        axes[1].fill_between(range(len(m)), m - s, m + s, color=c, alpha=0.2)
    axes[0].set_title("Валидация, один запуск (seed 0)")
    axes[1].set_title(f"Валидация, среднее ± СКО по {len(histories[name])} запускам")
    for ax in axes:
        ax.set_xlabel("Эпоха тонкой настройки")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    return fig


def plot_pretrain_losses(ae_losses, title, filename):
    """ae_losses: список кривых MSE реконструкции по слоям (один запуск)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, curve in enumerate(ae_losses, 1):
        ax.plot(curve, label=f"Автоэнкодер слоя {i}")
    ax.set_yscale("log")
    ax.set_xlabel("Эпоха предобучения")
    ax.set_ylabel("MSE реконструкции")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    return fig
