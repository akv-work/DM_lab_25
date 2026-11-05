import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, f1_score, confusion_matrix, ConfusionMatrixDisplay

class Autoencoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.enc = nn.Linear(in_dim, out_dim)
        self.dec = nn.Linear(out_dim, in_dim)
        self.act = nn.ReLU()
        
    def forward(self, x):
        return self.dec(self.act(self.enc(x)))
    
    def get_features(self, x):
        return self.act(self.enc(x))

class DeepNet(nn.Module):
    def __init__(self, in_dim, h1, h2, h3, out_dim):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_dim)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.enc1(x))
        x = self.act(self.enc2(x))
        x = self.act(self.enc3(x))
        return self.out(x)

def train_ae(ae, data, epochs=100, lr=0.01):
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ae(data), data)
        loss.backward()
        optimizer.step()
    return ae

def train_net(net, X, y, epochs=150, lr=0.005, is_cls=False):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        out = net(X)
        if not is_cls: out = out.squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return net, history

print("--- ЧАСТЬ 1: Регрессия (Appliances Energy) ---")
url_energy = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
df_energy = pd.read_csv(url_energy).drop(columns=['date'])

X_eng = df_energy.drop(columns=['Appliances']).values
y_eng = df_energy['Appliances'].values

scaler_X = MinMaxScaler()
X_eng = scaler_X.fit_transform(X_eng)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_eng, y_eng, test_size=0.2, random_state=42)

Xt_e = torch.tensor(X_train_e, dtype=torch.float32)
yt_e = torch.tensor(y_train_e, dtype=torch.float32)
Xt_test_e = torch.tensor(X_test_e, dtype=torch.float32)

net_no_pt_e = DeepNet(27, 128, 64, 32, 1)
net_no_pt_e, hist_no_pt_e = train_net(net_no_pt_e, Xt_e, yt_e, is_cls=False)

ae1_e = train_ae(Autoencoder(27, 128), Xt_e)
h1_e = ae1_e.get_features(Xt_e).detach()

ae2_e = train_ae(Autoencoder(128, 64), h1_e)
h2_e = ae2_e.get_features(h1_e).detach()

ae3_e = train_ae(Autoencoder(64, 32), h2_e)

net_pt_e = DeepNet(27, 128, 64, 32, 1)
net_pt_e.enc1.weight.data, net_pt_e.enc1.bias.data = ae1_e.enc.weight.data.clone(), ae1_e.enc.bias.data.clone()
net_pt_e.enc2.weight.data, net_pt_e.enc2.bias.data = ae2_e.enc.weight.data.clone(), ae2_e.enc.bias.data.clone()
net_pt_e.enc3.weight.data, net_pt_e.enc3.bias.data = ae3_e.enc.weight.data.clone(), ae3_e.enc.bias.data.clone()

net_pt_e, hist_pt_e = train_net(net_pt_e, Xt_e, yt_e, is_cls=False)

preds_no_pt_e = net_no_pt_e(Xt_test_e).detach().numpy().squeeze()
preds_pt_e = net_pt_e(Xt_test_e).detach().numpy().squeeze()

mape_no_pt = mean_absolute_percentage_error(y_test_e, preds_no_pt_e)
mape_pt = mean_absolute_percentage_error(y_test_e, preds_pt_e)

print(f"MAPE без предобучения: {mape_no_pt:.4f}")
print(f"MAPE с предобучением: {mape_pt:.4f}")

print("\n--- ЧАСТЬ 2: Классификация (Optdigits) ---")
url_digits = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
df_digits = pd.read_csv(url_digits, header=None)

X_dig = df_digits.iloc[:, :-1].values
y_dig = df_digits.iloc[:, -1].values
X_dig = MinMaxScaler().fit_transform(X_dig)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dig, y_dig, test_size=0.2, random_state=42)

Xt_d = torch.tensor(X_train_d, dtype=torch.float32)
yt_d = torch.tensor(y_train_d, dtype=torch.long)
Xt_test_d = torch.tensor(X_test_d, dtype=torch.float32)

net_no_pt_d = DeepNet(64, 128, 64, 32, 10)
net_no_pt_d, hist_no_pt_d = train_net(net_no_pt_d, Xt_d, yt_d, epochs=200, is_cls=True)

ae1_d = train_ae(Autoencoder(64, 128), Xt_d)
h1_d = ae1_d.get_features(Xt_d).detach()

ae2_d = train_ae(Autoencoder(128, 64), h1_d)
h2_d = ae2_d.get_features(h1_d).detach()

ae3_d = train_ae(Autoencoder(64, 32), h2_d)

net_pt_d = DeepNet(64, 128, 64, 32, 10)
net_pt_d.enc1.weight.data, net_pt_d.enc1.bias.data = ae1_d.enc.weight.data.clone(), ae1_d.enc.bias.data.clone()
net_pt_d.enc2.weight.data, net_pt_d.enc2.bias.data = ae2_d.enc.weight.data.clone(), ae2_d.enc.bias.data.clone()
net_pt_d.enc3.weight.data, net_pt_d.enc3.bias.data = ae3_d.enc.weight.data.clone(), ae3_d.enc.bias.data.clone()

net_pt_d, hist_pt_d = train_net(net_pt_d, Xt_d, yt_d, epochs=200, is_cls=True)

preds_no_pt_d = torch.argmax(net_no_pt_d(Xt_test_d), dim=1).numpy()
preds_pt_d = torch.argmax(net_pt_d(Xt_test_d), dim=1).numpy()

f1_no_pt = f1_score(y_test_d, preds_no_pt_d, average='macro')
f1_pt = f1_score(y_test_d, preds_pt_d, average='macro')

print(f"F1-score без предобучения: {f1_no_pt:.4f}")
print(f"F1-score с предобучением: {f1_pt:.4f}")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(hist_no_pt_e, label='Без предобучения', color='red')
axs[0, 0].plot(hist_pt_e, label='С предобучением (Autoencoders)', color='green')
axs[0, 0].set_title(f'Регрессия: Функция потерь (MAPE: PT={mape_pt:.2f} vs NoPT={mape_no_pt:.2f})')
axs[0, 0].set_xlabel('Эпохи')
axs[0, 0].set_ylabel('MSE Loss')
axs[0, 0].legend()

axs[0, 1].plot(hist_no_pt_d, label='Без предобучения', color='red')
axs[0, 1].plot(hist_pt_d, label='С предобучением (Autoencoders)', color='green')
axs[0, 1].set_title(f'Классификация: Функция потерь (F1: PT={f1_pt:.2f} vs NoPT={f1_no_pt:.2f})')
axs[0, 1].set_xlabel('Эпохи')
axs[0, 1].set_ylabel('CrossEntropy Loss')
axs[0, 1].legend()

cm_no_pt = confusion_matrix(y_test_d, preds_no_pt_d)
disp_no_pt = ConfusionMatrixDisplay(confusion_matrix=cm_no_pt)
disp_no_pt.plot(ax=axs[1, 0], cmap='Blues', colorbar=False)
axs[1, 0].set_title('Матрица ошибок (Без предобучения)')

cm_pt = confusion_matrix(y_test_d, preds_pt_d)
disp_pt = ConfusionMatrixDisplay(confusion_matrix=cm_pt)
disp_pt.plot(ax=axs[1, 1], cmap='Greens', colorbar=False)
axs[1, 1].set_title('Матрица ошибок (С предобучением)')

plt.tight_layout()
plt.show()