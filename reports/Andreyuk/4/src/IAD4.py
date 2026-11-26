import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.neural_network import BernoulliRBM
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

print("Загрузка датасета Credit Approval (вариант 1: классификация +/-)...")
credit_approval = fetch_ucirepo(id=27)
X = credit_approval.data.features
y = credit_approval.data.targets

df = pd.concat([X, y], axis=1)
df.columns = [f'A{i+1}' for i in range(15)] + ['Target']
df = df.astype(object)

df = df.replace('?', np.nan)

numeric_cols = []
categorical_cols = []
for col in df.columns:
    non_na = df[col].dropna()
    if len(non_na) == 0:
        categorical_cols.append(col)
        continue
    converted = pd.to_numeric(non_na, errors='coerce')
    if converted.notna().sum() > len(non_na) * 0.9:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        numeric_cols.append(col)
    else:
        df[col] = df[col].astype(str)
        categorical_cols.append(col)

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    mode_val = df[col].mode()
    df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'unknown')

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

X_processed = df.drop('Target', axis=1).values.astype(np.float32)
y_processed = df['Target'].values.astype(np.float32)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
)

scaler_standard = StandardScaler()
X_train_std = scaler_standard.fit_transform(X_train_raw).astype(np.float32)
X_test_std = scaler_standard.transform(X_test_raw).astype(np.float32)

scaler_minmax = MinMaxScaler()
X_train_mm = scaler_minmax.fit_transform(X_train_raw).astype(np.float32)
X_test_mm = scaler_minmax.transform(X_test_raw).astype(np.float32)

input_dim = X_train_std.shape[1]
HIDDEN_LAYERS = [32, 16, 8]
print(f"Размерность входа: {input_dim}, Обучающих примеров: {X_train_std.shape[0]}")
print(f"Тестовых примеров: {X_test_std.shape[0]}")

X_train_t = torch.tensor(X_train_std, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_t = torch.tensor(X_test_std, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, act_type="relu"):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_layers:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.out = nn.Linear(prev, 1)
        if act_type == "relu":
            self.act = nn.ReLU()
        elif act_type == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(act_type)

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return torch.sigmoid(self.out(x))


def train_classifier(model, X_t, y_t, epochs=200, patience=15, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    wait = 0
    history = {'loss': [], 'val_loss': []}

    val_size = int(0.1 * len(X_t))
    X_val, y_val = X_t[:val_size], y_t[:val_size]
    X_tr, y_tr = X_t[val_size:], y_t[val_size:]
    tr_ds = TensorDataset(X_tr, y_tr)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return history


RBM_N_ITER = 200
RBM_LR = 0.1
RBM_BATCH = 16

print("\nОбучение без предобучения...")
model_no_pretrain = Classifier(input_dim, HIDDEN_LAYERS, act_type="relu").to(device)
history_no_pretrain = train_classifier(model_no_pretrain, X_train_t, y_train_t)
model_no_pretrain.eval()
with torch.no_grad():
    y_pred_no_pre = (model_no_pretrain(X_test_t) > 0.5).cpu().numpy().astype(int).flatten()
f1_no_pre = f1_score(y_test, y_pred_no_pre)
print(f"F1-мера (без предобучения): {f1_no_pre:.4f}")

print("\nПредобучение слоев автоэнкодерами...")
pretrained_weights_ae = []
current_input_ae = X_train_std.copy()

for i, layer_size in enumerate(HIDDEN_LAYERS):
    in_dim = current_input_ae.shape[1]
    encoder = nn.Linear(in_dim, layer_size)
    decoder = nn.Linear(layer_size, in_dim)
    ae = nn.Sequential(encoder, nn.ReLU(), decoder, nn.Sigmoid()).to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.001)
    ae_criterion = nn.MSELoss()

    curr_t = torch.tensor(current_input_ae, dtype=torch.float32).to(device)
    ae_ds = TensorDataset(curr_t, curr_t)
    ae_loader = DataLoader(ae_ds, batch_size=32, shuffle=True)

    for epoch in range(50):
        ae.train()
        for xb, yb in ae_loader:
            ae_optimizer.zero_grad()
            out = ae(xb)
            loss = ae_criterion(out, yb)
            loss.backward()
            ae_optimizer.step()

    pretrained_weights_ae.append({
        'weight': encoder.weight.detach().clone(),
        'bias': encoder.bias.detach().clone()
    })
    ae.eval()
    with torch.no_grad():
        hidden = torch.relu(encoder(curr_t)).cpu().numpy()
    current_input_ae = hidden

print("Предобучение автоэнкодерами завершено.")

print("\nОбучение с предобучением автоэнкодерами...")
model_pretrained_ae = Classifier(input_dim, HIDDEN_LAYERS, act_type="relu").to(device)
with torch.no_grad():
    for i, w in enumerate(pretrained_weights_ae):
        model_pretrained_ae.layers[i].weight.copy_(w['weight'])
        model_pretrained_ae.layers[i].bias.copy_(w['bias'])

history_pretrained_ae = train_classifier(model_pretrained_ae, X_train_t, y_train_t)
model_pretrained_ae.eval()
with torch.no_grad():
    y_pred_ae = (model_pretrained_ae(X_test_t) > 0.5).cpu().numpy().astype(int).flatten()
f1_ae = f1_score(y_test, y_pred_ae)
print(f"F1-мера (с предобучением автоэнкодерами): {f1_ae:.4f}")

print("\nПредобучение слоев с помощью RBM...")
print(f"Параметры RBM: n_iter={RBM_N_ITER}, learning_rate={RBM_LR}, batch_size={RBM_BATCH}")
current_input_rbm = X_train_mm.copy()
pretrained_weights_rbm = []

for i, layer_size in enumerate(HIDDEN_LAYERS):
    print(f"Предобучение RBM слоя {i+1} (размер {layer_size})...")
    rbm = BernoulliRBM(
        n_components=layer_size,
        learning_rate=RBM_LR,
        batch_size=RBM_BATCH,
        n_iter=RBM_N_ITER,
        random_state=42,
        verbose=0
    )
    rbm.fit(current_input_rbm)

    pretrained_weights_rbm.append({
        'weight': torch.tensor(rbm.components_, dtype=torch.float32),
        'bias': torch.tensor(rbm.intercept_hidden_, dtype=torch.float32)
    })
    current_input_rbm = rbm.transform(current_input_rbm).astype(np.float32)

print("Предобучение RBM завершено.")

print("\nОбучение с предобучением RBM...")
model_pretrained_rbm = Classifier(input_dim, HIDDEN_LAYERS, act_type="sigmoid").to(device)
with torch.no_grad():
    for i, w in enumerate(pretrained_weights_rbm):
        model_pretrained_rbm.layers[i].weight.copy_(w['weight'])
        model_pretrained_rbm.layers[i].bias.copy_(w['bias'])

history_pretrained_rbm = train_classifier(
    model_pretrained_rbm, X_train_t, y_train_t, epochs=300, patience=25, lr=0.0005
)
model_pretrained_rbm.eval()
with torch.no_grad():
    y_pred_rbm = (model_pretrained_rbm(X_test_t) > 0.5).cpu().numpy().astype(int).flatten()
f1_rbm = f1_score(y_test, y_pred_rbm)
print(f"F1-мера (с предобучением RBM): {f1_rbm:.4f}")

print("\nИтоговые результаты:")
print(f"1. Без предобучения:              F1 = {f1_no_pre:.4f}")
print(f"2. С предобучением автоэнкодерами: F1 = {f1_ae:.4f}")
print(f"3. С предобучением RBM:            F1 = {f1_rbm:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(history_no_pretrain['loss'], label='Без предобучения, тренировочная')
plt.plot(history_no_pretrain['val_loss'], label='Без предобучения, валидационная', linestyle='--')
plt.plot(history_pretrained_ae['loss'], label='Автоэнкодеры, тренировочная', linestyle='-.')
plt.plot(history_pretrained_ae['val_loss'], label='Автоэнкодеры, валидационная', linestyle=':')
plt.plot(history_pretrained_rbm['loss'], label='RBM, тренировочная')
plt.plot(history_pretrained_rbm['val_loss'], label='RBM, валидационная', linestyle='--')
plt.title('Сравнение кривых обучения: без предобучения vs автоэнкодеры vs RBM')
plt.xlabel('Эпоха')
plt.ylabel('Потеря (BCE)')
plt.legend()
plt.grid(True)
plt.show()