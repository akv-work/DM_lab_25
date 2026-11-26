import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ------------------------------------------------------------
# Make sure 'parkinsons_updrs.data' is in the same folder as this script!
df = pd.read_csv('parkinsons_updrs.data')
X = df.drop(columns=['subject#', 'total_UPDRS', 'motor_UPDRS']).values
y = df['motor_UPDRS'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# ------------------------------------------------------------
# 2. RBM Definition & Training
# ------------------------------------------------------------
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return p_v, torch.bernoulli(p_v)

    def forward(self, v):
        p_h0, h0 = self.sample_h(v)
        p_v1, v1 = self.sample_v(h0)
        p_h1, _ = self.sample_h(v1)
        return p_h0, p_v1, p_h1

def train_rbm(rbm, data_loader, epochs=50, lr=0.01):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            v = batch[0]
            p_h0, p_v1, p_h1 = rbm(v)
            loss = torch.mean((v - p_v1)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"RBM Epoch {epoch+1}: recon loss = {total_loss/len(data_loader):.6f}")

# ------------------------------------------------------------
# 3. Autoencoder Definition & Training
# ------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(ae, data_loader, epochs=50, lr=0.001):
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch[0]
            recon = ae(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"AE Epoch {epoch+1}: loss = {total_loss/len(data_loader):.6f}")

# ------------------------------------------------------------
# 4. Full Regression Network & Training
# ------------------------------------------------------------
class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(RegressionNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Sigmoid())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Modified to return the loss history for the graph
def train_regressor(model, train_loader, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Regressor Epoch {epoch+1}: loss = {avg_loss:.6f}")
            
    return model, loss_history

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return mse, mae, r2

# ------------------------------------------------------------
# 5. Experiment 1: No Pretraining
# ------------------------------------------------------------
print("\n--- 1. No Pretraining ---")
baseline = RegressionNet(X_train.shape[1], [64, 32, 16])
baseline, loss_baseline = train_regressor(baseline, train_loader)
evaluate(baseline, X_test_t, y_test_t)

# ------------------------------------------------------------
# 6. Experiment 2: Autoencoder Pretraining
# ------------------------------------------------------------
print("\n--- 2. Autoencoder Pretraining ---")
ae1 = Autoencoder(X_train.shape[1], 64)
train_autoencoder(ae1, train_loader, epochs=50)
with torch.no_grad():
    encoded_1 = ae1.encoder(X_train_t)
ae2 = Autoencoder(64, 32)
train_autoencoder(ae2, DataLoader(TensorDataset(encoded_1), batch_size=64), epochs=50)
with torch.no_grad():
    encoded_2 = ae2.encoder(encoded_1)
ae3 = Autoencoder(32, 16)
train_autoencoder(ae3, DataLoader(TensorDataset(encoded_2), batch_size=64), epochs=50)

ae_net = RegressionNet(X_train.shape[1], [64, 32, 16])
ae_net.net[0].weight.data = ae1.encoder[0].weight.data.clone()
ae_net.net[0].bias.data = ae1.encoder[0].bias.data.clone()
ae_net.net[2].weight.data = ae2.encoder[0].weight.data.clone()
ae_net.net[2].bias.data = ae2.encoder[0].bias.data.clone()
ae_net.net[4].weight.data = ae3.encoder[0].weight.data.clone()
ae_net.net[4].bias.data = ae3.encoder[0].bias.data.clone()

ae_net, loss_ae = train_regressor(ae_net, train_loader)
evaluate(ae_net, X_test_t, y_test_t)

# ------------------------------------------------------------
# 7. Experiment 3: RBM Pretraining
# ------------------------------------------------------------
print("\n--- 3. RBM Pretraining ---")
rbm1 = RBM(X_train.shape[1], 64)
train_rbm(rbm1, train_loader, epochs=50)
with torch.no_grad():
    h1, _ = rbm1.sample_h(X_train_t)
rbm2 = RBM(64, 32)
train_rbm(rbm2, DataLoader(TensorDataset(h1), batch_size=64), epochs=50)
with torch.no_grad():
    h2, _ = rbm2.sample_h(h1)
rbm3 = RBM(32, 16)
train_rbm(rbm3, DataLoader(TensorDataset(h2), batch_size=64), epochs=50)

rbm_net = RegressionNet(X_train.shape[1], [64, 32, 16])
rbm_net.net[0].weight.data = rbm1.W.data.clone()
rbm_net.net[0].bias.data = rbm1.h_bias.data.clone()
rbm_net.net[2].weight.data = rbm2.W.data.clone()
rbm_net.net[2].bias.data = rbm2.h_bias.data.clone()
rbm_net.net[4].weight.data = rbm3.W.data.clone()
rbm_net.net[4].bias.data = rbm3.h_bias.data.clone()

rbm_net, loss_rbm = train_regressor(rbm_net, train_loader)
evaluate(rbm_net, X_test_t, y_test_t)

# ------------------------------------------------------------
# 8. GENERATE THE GRAPH (Loss Curve)
# ------------------------------------------------------------
print("\nГенерация графика...")
epochs = range(1, 101)

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_baseline, label='Без предобучения (MSE)')
plt.plot(epochs, loss_ae, label='Автоэнкодеры (MSE)')
plt.plot(epochs, loss_rbm, label='RBM (MSE)')

plt.title('Ошибка регрессии (MSE) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# Saves the graph in your folder as 'loss_graph.png'
plt.savefig('loss_graph.png', dpi=300, bbox_inches='tight') 
plt.show()

print("\nЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА. График сохранен как 'loss_graph.png' в папке проекта.")