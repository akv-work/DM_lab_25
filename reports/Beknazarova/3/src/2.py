import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

PARKINSONS_PATH = r'D:\\Учёба\\4 курс\\ИАД\\3\\parkinsons_updrs.data'

cols = ['subject#','age','sex','test_time','motor_UPDRS','total_UPDRS',
        'Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP',
        'Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11',
        'Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']

df = pd.read_csv(PARKINSONS_PATH, names=cols, skiprows=1).astype(float)

X = df.drop(columns=['subject#', 'motor_UPDRS', 'total_UPDRS']).values
y = df['motor_UPDRS'].values.reshape(-1, 1)
groups = df['subject#'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train пациентов:", df.iloc[train_idx]['subject#'].nunique())
print("Test пациентов: ", df.iloc[test_idx]['subject#'].nunique())

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test  = scaler_y.transform(y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test,  dtype=torch.float32).to(device)

class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        x_rec = self.decoder(h)
        return x_rec, h

def train_autoencoder(data, in_dim, hidden_dim, epochs=100, lr=1e-3):
    ae = Autoencoder(in_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        ae.train()
        opt.zero_grad()
        x_rec, _ = ae(data)
        loss = loss_fn(x_rec, data)
        loss.backward()
        opt.step()
        if (epoch+1) % 25 == 0:
            print(f"  AE {in_dim}->{hidden_dim}, epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")
    return ae.encoder

input_dim = X_train.shape[1]
print(f"\nПредобучение слоёв (вход: {input_dim})")

enc1 = train_autoencoder(X_train, in_dim=input_dim, hidden_dim=64, epochs=100)
with torch.no_grad():
    h1_train = enc1(X_train)

enc2 = train_autoencoder(h1_train, in_dim=64, hidden_dim=32, epochs=100)
with torch.no_grad():
    h2_train = enc2(h1_train)

enc3 = train_autoencoder(h2_train, in_dim=32, hidden_dim=16, epochs=100)

class NetPretrained(nn.Module):
    def __init__(self, enc1, enc2, enc3):
        super().__init__()
        self.fc1 = enc1[0]
        self.fc2 = enc2[0]
        self.fc3 = enc3[0]
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model_pt = NetPretrained(enc1, enc2, enc3).to(device)

print("\nFine-tuning")
opt = torch.optim.Adam(model_pt.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 300
for epoch in range(epochs):
    model_pt.train()
    opt.zero_grad()
    pred = model_pt(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()
    if (epoch+1) % 30 == 0:
        print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

model_pt.eval()
with torch.no_grad():
    pred = model_pt(X_test).cpu().numpy()
y_true = y_test.cpu().numpy()

pred = scaler_y.inverse_transform(pred)
y_true = scaler_y.inverse_transform(y_true)

mape = mean_absolute_percentage_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2 = r2_score(y_true, pred)

print(f"\nС предобучением (group split)")
print(f"MAPE = {mape:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")

print("\nСравнение (group split)")
print(f"{'Метод':<20} {'MAPE':>8} {'RMSE':>8} {'R2':>8}")
print(f"{'Без предобучения':<20} {'0.4396':>8} {'8.9307':>8} {'-0.3679':>8}")
print(f"{'С предобучением':<20} {mape:>8.4f} {rmse:>8.4f} {r2:>8.4f}")