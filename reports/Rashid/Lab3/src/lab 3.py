"""
Laboratory work 3. Variant 12.
Dataset: Parkinsons Telemonitoring.
Task: regression.
Target: motor_UPDRS.

Compares:
1) MLP trained from scratch;
2) MLP with greedy layer-wise autoencoder pretraining + fine-tuning.

Generates 9 specific diagrams and prints a comparison table.
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

import matplotlib
matplotlib.use("Agg")  # Headless-safe
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 1. Load data
# ============================================================
DATA_PATH = "parkinsons_updrs.data"

columns = [
    "subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"
]

df = pd.read_csv(DATA_PATH, names=columns, skiprows=1)

feature_cols = [
    c for c in df.columns
    if c not in ["subject#", "motor_UPDRS", "total_UPDRS"]
]

X = df[feature_cols].values.astype(np.float32)
y = df["motor_UPDRS"].values.astype(np.float32).reshape(-1, 1)
groups = df["subject#"].values

# ============================================================
# 2. GROUP train/test split by patient + scaling
# ============================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train patients: {len(np.unique(groups[train_idx]))}")
print(f"Test patients : {len(np.unique(groups[test_idx]))}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# ============================================================
# 3. Helper functions
# ============================================================
def make_loader(X_data, y_data, batch_size=64, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X_data), torch.from_numpy(y_data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_supervised(model, train_loader, epochs=300, lr=1e-3, log_every=30):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        history.append(avg_loss)
        if epoch % log_every == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, loss={avg_loss:.4f}")
    return model, history

def evaluate(model, X_data, y_data):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_data).to(device)).cpu().numpy()
    y_true = y_data.ravel()
    y_pred = pred.ravel()
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return mape, rmse, r2, y_pred

# ============================================================
# 4. Baseline MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16)):
        super().__init__()
        layers, prev_dim = [], input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

print("\n--- Baseline MLP ---")
baseline_model = MLP(input_dim=X_train.shape[1], hidden_dims=(64, 32, 16))
baseline_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)
baseline_model, base_history = train_supervised(baseline_model, baseline_loader)
base_mape, base_rmse, base_r2, base_pred = evaluate(baseline_model, X_test, y_test)

# ============================================================
# 5. Autoencoder pretraining
# ============================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)

def train_autoencoder(ae, X_data, epochs=100, lr=1e-3):
    ae.to(device)
    loader = make_loader(X_data, X_data, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        ae.train()
        total_loss = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ae(xb), xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        history.append(total_loss / len(loader.dataset))
    return ae, history

print("\n--- Pretraining Autoencoders ---")
layer_dims = [64, 32, 16]
autoencoders, ae_histories = [], []
current_train, current_test = X_train.copy(), X_test.copy()

for layer_idx, hidden_dim in enumerate(layer_dims, start=1):
    input_dim = current_train.shape[1]
    ae = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    ae, ae_hist = train_autoencoder(ae, current_train, epochs=100)
    ae.eval()
    with torch.no_grad():
        current_train = ae.encode(torch.from_numpy(current_train).to(device)).cpu().numpy()
        current_test = ae.encode(torch.from_numpy(current_test).to(device)).cpu().numpy()
    autoencoders.append(ae)
    ae_histories.append(ae_hist)

# ============================================================
# 6. Fine-tuning
# ============================================================
class PretrainedMLP(nn.Module):
    def __init__(self, autoencoders, last_hidden_dim):
        super().__init__()
        self.encoders = nn.ModuleList([ae.encoder for ae in autoencoders])
        self.head = nn.Linear(last_hidden_dim, 1)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return self.head(x)

print("\n--- Fine-tuning ---")
pretrained_model = PretrainedMLP(autoencoders, layer_dims[-1])
fine_tune_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)
pretrained_model, pre_history = train_supervised(pretrained_model, fine_tune_loader)
pre_mape, pre_rmse, pre_r2, pre_pred = evaluate(pretrained_model, X_test, y_test)

# ============================================================
# 7. Comparison Table Output & File Saving
# ============================================================
# Print to console exactly as requested
print("\nСравнение (group split)")
print("=" * 60)
print(f"{'Метод':<25}{'MAPE':>10}{'RMSE':>10}{'R2':>10}")
print(f"{'Без предобучения':<25}{base_mape:>10.4f}{base_rmse:>10.4f}{base_r2:>10.4f}")
print(f"{'С предобучением':<25}{pre_mape:>10.4f}{pre_rmse:>10.4f}{pre_r2:>10.4f}")
print("=" * 60)

# Save to text file
with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Метод':<25}{'MAPE':>10}{'RMSE':>10}{'R2':>10}\n")
    f.write(f"{'Без предобучения':<25}{base_mape:>10.4f}{base_rmse:>10.4f}{base_r2:>10.4f}\n")
    f.write(f"{'С предобучением':<25}{pre_mape:>10.4f}{pre_rmse:>10.4f}{pre_r2:>10.4f}\n")

# ============================================================
# 8. GENERATE THE 9 DIAGRAMS
# ============================================================
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.4})
y_true = y_test.ravel()

# --- Fig 1: Autoencoder Losses ---
plt.figure(figsize=(10, 5))
for i, hist in enumerate(ae_histories):
    in_d = 19 if i == 0 else layer_dims[i-1]
    out_d = layer_dims[i]
    plt.plot(hist, label=f"AE {in_d}→{out_d}", linewidth=2)
plt.title("Обучение автоэнкодеров")
plt.xlabel("Эпоха")
plt.ylabel("MSE loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/1_ae_loss.png", dpi=150)
plt.close()

# --- Fig 2: Metrics Bar Charts ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_data = [
    ("MAPE", [base_mape, pre_mape], "MAPE (меньше — лучше)"),
    ("RMSE", [base_rmse, pre_rmse], "RMSE (меньше — лучше)"),
    ("R²", [base_r2, pre_r2], "R² (больше — лучше)")
]
labels = ["Без предоб.\n(baseline)", "С предоб.\n(pretrain)"]
colors = ["#4C72B0", "#DD8452"]

for ax, (name, vals, title) in zip(axes, metrics_data):
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', alpha=0.9)
    ax.set_title(title)
    for b, v in zip(bars, vals):
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va=va)
plt.tight_layout()
plt.savefig(f"{fig_dir}/2_metrics_bars.png", dpi=150)
plt.close()

# --- Fig 3: Error Histogram (Pretrain only) ---
resid_pre = y_true - pre_pred
plt.figure(figsize=(10, 5))
plt.hist(resid_pre, bins=40, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.title(f"Гистограмма ошибок\nсреднее={resid_pre.mean():.2f}, std={resid_pre.std():.2f}")
plt.xlabel("Ошибка (истина - предсказание)")
plt.ylabel("Частота")
plt.tight_layout()
plt.savefig(f"{fig_dir}/3_error_hist.png", dpi=150)
plt.close()

# --- Fig 4: Scatter (Pretrain only) ---
plt.figure(figsize=(8, 6))
plt.scatter(y_true, pre_pred, alpha=0.4, edgecolor='k')
lo, hi = y_true.min(), y_true.max()
plt.plot([lo, hi], [lo, hi], 'r--', label='Идеал', linewidth=2)
plt.title("Предсказание vs Истина (с предобучением)")
plt.xlabel("Истинное motor_UPDRS")
plt.ylabel("Предсказание")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/4_scatter_pre.png", dpi=150)
plt.close()

# --- Fig 5: Two Histograms (Baseline vs Pretrain) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
resid_base = y_true - base_pred
for ax, resid, title in zip(axes, [resid_base, resid_pre], ["Без предобучения", "С предобучением"]):
    ax.hist(resid, bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f"{title}\nсреднее={resid.mean():.2f}, std={resid.std():.2f}")
    ax.set_xlabel("Ошибка (истина - предсказание)")
    ax.set_ylabel("Частота")
plt.tight_layout()
plt.savefig(f"{fig_dir}/5_hist_compare.png", dpi=150)
plt.close()

# --- Fig 6: Residuals Scatter (Pretrain only) ---
plt.figure(figsize=(8, 6))
plt.scatter(pre_pred, resid_pre, alpha=0.4, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title("Остатки (с предобучением)")
plt.xlabel("Предсказание")
plt.ylabel("Остаток (истина - предсказание)")
plt.tight_layout()
plt.savefig(f"{fig_dir}/6_residuals.png", dpi=150)
plt.close()

# --- Fig 7: Two Scatters (Baseline vs Pretrain) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pred, title in zip(axes, [base_pred, pre_pred], ["Без предобучения", "С предобучением"]):
    ax.scatter(y_true, pred, alpha=0.4, edgecolor='k')
    ax.plot([lo, hi], [lo, hi], 'r--', label='Идеал', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Истинное motor_UPDRS")
    ax.set_ylabel("Предсказание")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/7_scatter_compare.png", dpi=150)
plt.close()

# --- Fig 8: Fine-tuning Loss ---
plt.figure(figsize=(10, 5))
plt.plot(pre_history, color='orange', linewidth=2, label="Fine-tuning")
plt.title("Fine-tuning предобученной сети")
plt.xlabel("Эпоха")
plt.ylabel("MSE loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/8_finetune_loss.png", dpi=150)
plt.close()

# --- Fig 9: Loss Curves (Baseline vs Pretrain) ---
plt.figure(figsize=(10, 5))
plt.plot(base_history, label="Без предобучения", linewidth=2)
plt.plot(pre_history, label="С предобучением", linewidth=2)
plt.title("Кривые обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/9_loss_curves.png", dpi=150)
plt.close()

print(f"\nВсе 9 диаграмм успешно сохранены в папку '{fig_dir}/'.")