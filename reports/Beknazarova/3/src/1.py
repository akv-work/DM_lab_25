import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

PARKINSONS_PATH = r'D:\Учёба\4 курс\ИАД\3\parkinsons_updrs.data'

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

class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = Net(X_train.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 300
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()
    if (epoch+1) % 30 == 0:
        print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    pred = model(X_test).cpu().numpy()
y_true = y_test.cpu().numpy()

pred = scaler_y.inverse_transform(pred)
y_true = scaler_y.inverse_transform(y_true)

mape = mean_absolute_percentage_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2 = r2_score(y_true, pred)

print(f"\nБез предобучения (group split)")
print(f"MAPE = {mape:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")