
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

for _stream in (sys.stdout, sys.stderr): 
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "pima-indians-diabetes.csv"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
CLASS_NAMES = {0: "Нет диабета", 1: "Диабет"}


def load_pima(verbose=True):
    """Файл начинается со строк-комментариев (#) с описанием признаков, заголовка нет."""
    df = pd.read_csv(DATA_FILE, comment="#", header=None, names=FEATURES + [TARGET])
    if verbose:
        vc = df[TARGET].value_counts().sort_index()
        print(f"Загружено объектов: {len(df)}, признаков: {len(FEATURES)}, "
              f"пропусков (NaN): {int(df.isna().sum().sum())}")
        print(f"Классы: нет диабета = {vc[0]}, диабет = {vc[1]} ({vc[1] / len(df) * 100:.1f} %)")
        zeros = {c: int((df[c] == 0).sum()) for c in ZERO_AS_MISSING}
        print(f"Нулевых значений (физиологически невозможных): {zeros}")
    return df


def zeros_to_nan(df):
    out = df.copy()
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].replace(0, np.nan)
    return out


def metrics(y_true, y_pred, proba=None):
    """Recall положительного класса (диабет) — основная метрика работы."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    res = {"Recall": recall_score(y_true, y_pred, zero_division=0),
           "Precision": precision_score(y_true, y_pred, zero_division=0),
           "F1": f1_score(y_true, y_pred, zero_division=0),
           "Accuracy": accuracy_score(y_true, y_pred),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    if proba is not None:
        res["ROC-AUC"] = roc_auc_score(y_true, proba)
    return res


def plot_confusions(cms, title, filename, ncols=3):
    n = len(cms)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Нет диабета", "Диабет"])
        ax.set_yticks([0, 1], ["Нет диабета", "Диабет"])
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Истина")
        ax.set_title(name, fontsize=11)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    return fig
