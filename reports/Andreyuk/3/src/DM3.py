import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

print("Загрузка датасета Credit Approval...")
credit_approval = fetch_ucirepo(id=27)
X = credit_approval.data.features
y = credit_approval.data.targets

df = pd.concat([X, y], axis=1)
df.columns = [f'A{i+1}' for i in range(15)] + ['Target']

df = df.astype(object)

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
    if len(mode_val) > 0:
        df[col] = df[col].fillna(mode_val[0])
    else:
        df[col] = df[col].fillna('unknown')

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

if 'Target' in numeric_cols:
    df['Target'] = df['Target'].astype(int)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

X_processed = df.drop('Target', axis=1).values.astype(np.float32)
y_processed = df['Target'].values.astype(np.float32)

scaler = StandardScaler()
X_processed = scaler.fit_transform(X_processed)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
)

input_dim = X_train.shape[1]
HIDDEN_LAYERS = [32, 16, 8]
print(f"Размерность входа: {input_dim}, Обучающих примеров: {X_train.shape[0]}")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_layers:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.out = nn.Linear(prev, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return torch.sigmoid(self.out(x))


def train_classifier(model, epochs=200, patience=15):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    wait = 0
    history = {'loss': [], 'val_loss': []}

    val_size = int(0.1 * len(X_train_t))
    X_val, y_val = X_train_t[:val_size], y_train_t[:val_size]
    X_tr, y_tr = X_train_t[val_size:], y_train_t[val_size:]
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
                print(f"  Ранняя остановка на эпохе {epoch+1}")
                break

    model.load_state_dict(best_state)
    return history


print("\nОбучение без предобучения...")
model_baseline = Classifier(input_dim, HIDDEN_LAYERS).to(device)
history_baseline = train_classifier(model_baseline)

model_baseline.eval()
with torch.no_grad():
    y_pred_base = (model_baseline(X_test_t) > 0.5).cpu().numpy().astype(int).flatten()

f1_base = f1_score(y_test, y_pred_base)
print(f"F1-мера (без предобучения): {f1_base:.4f}")

print("\nПредобучение слоев автоэнкодерами...")

pretrained_weights = []
current_input = X_train.copy()

for i, layer_size in enumerate(HIDDEN_LAYERS):
    print(f"Предобучение слоя {i+1} (размер {layer_size})...")
    in_dim = current_input.shape[1]

    encoder = nn.Linear(in_dim, layer_size)
    decoder = nn.Linear(layer_size, in_dim)
    ae = nn.Sequential(encoder, nn.ReLU(), decoder, nn.Sigmoid()).to(device)

    ae_optimizer = optim.Adam(ae.parameters(), lr=0.001)
    ae_criterion = nn.MSELoss()

    curr_t = torch.tensor(current_input, dtype=torch.float32).to(device)
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

    pretrained_weights.append({
        'weight': encoder.weight.detach().clone(),
        'bias': encoder.bias.detach().clone()
    })

    ae.eval()
    with torch.no_grad():
        hidden = torch.relu(encoder(curr_t)).cpu().numpy()
    current_input = hidden

print("Предобучение завершено.")

print("\nОбучение с предобучением...")
model_pretrained = Classifier(input_dim, HIDDEN_LAYERS).to(device)

with torch.no_grad():
    for i, w in enumerate(pretrained_weights):
        model_pretrained.layers[i].weight.copy_(w['weight'])
        model_pretrained.layers[i].bias.copy_(w['bias'])

history_pretrained = train_classifier(model_pretrained)

model_pretrained.eval()
with torch.no_grad():
    y_pred_pre = (model_pretrained(X_test_t) > 0.5).cpu().numpy().astype(int).flatten()

f1_pre = f1_score(y_test, y_pred_pre)
print(f"F1-мера (с предобучением): {f1_pre:.4f}")

print("\nИтоговое сравнение:")
print(f"F1-мера без предобучения: {f1_base:.4f}")
print(f"F1-мера с предобучением:  {f1_pre:.4f}")

if f1_pre > f1_base:
    diff = (f1_pre - f1_base) / f1_base * 100
    print(f"\nВывод: предобучение автоэнкодерами улучшило F1-меру на {diff:.2f}% по сравнению с обучением с нуля.")
    print("Послойная инициализация весов позволила модели стартовать из более удачной области пространства параметров,")
    print("что положительно сказалось на качестве классификации на тестовой выборке.")
elif f1_pre < f1_base:
    diff = (f1_base - f1_pre) / f1_base * 100
    print(f"\nВывод: предобучение автоэнкодерами снизило F1-меру на {diff:.2f}% по сравнению с обучением с нуля.")
    print("Для данного датасета предобучение не дало выигрыша: выборка достаточно мала, а признаки хорошо разделимы,")
    print("поэтому случайная инициализация в сочетании с Adam и ранней остановкой оказалась эффективнее.")
else:
    print("\nВывод: результаты обучения с предобучением и без него совпали.")
    print("Предобучение автоэнкодерами не изменило итоговое качество модели на тестовой выборке.")

plt.figure(figsize=(10, 5))
plt.plot(history_baseline['loss'], label='Без предобучения, тренировочная')
plt.plot(history_baseline['val_loss'], label='Без предобучения, валидационная', linestyle='--')
plt.plot(history_pretrained['loss'], label='С предобучением, тренировочная')
plt.plot(history_pretrained['val_loss'], label='С предобучением, валидационная', linestyle='--')
plt.title('Сравнение кривых обучения')
plt.xlabel('Эпоха')
plt.ylabel('Потеря (BCE)')
plt.legend()
plt.grid(True)
plt.show()