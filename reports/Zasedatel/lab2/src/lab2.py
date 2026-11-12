import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

data = fetch_ucirepo(id=17)

X = data.data.features.values
y = data.data.targets.values.ravel()

print("Размер данных:", X.shape)
print("Классы:", np.unique(y))

X = StandardScaler().fit_transform(X)

X_tensor = torch.FloatTensor(X)

class Autoencoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30,16),
            nn.ReLU(),
            nn.Linear(16,n)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n,16),
            nn.ReLU(),
            nn.Linear(16,30)
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))

def autoencoder(n):

    model = Autoencoder(n)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()

    for i in range(100):
        out = model(X_tensor)
        err = loss(out,X_tensor)
        opt.zero_grad()
        err.backward()
        opt.step()

    return model.encoder(X_tensor).detach().numpy()

X_auto_2 = autoencoder(2)
X_auto_3 = autoencoder(3)

print("\nАвтоэнкодер:")
print("2 компоненты:", X_auto_2.shape)
print("3 компоненты:", X_auto_3.shape)

def plot2d(X,title):

    plt.figure(figsize=(7,5))

    for c in np.unique(y):
        p = X[y==c]
        plt.scatter(p[:,0],p[:,1],label="Класс "+str(c))

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot3d(X,title):

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111,projection="3d")

    for c in np.unique(y):
        p = X[y==c]
        ax.scatter(p[:,0],p[:,1],p[:,2],label="Класс "+str(c))

    ax.set_title(title)
    ax.legend()
    plt.show()

plot2d(X_auto_2,"Автоэнкодер - 2 компоненты")
plot3d(X_auto_3,"Автоэнкодер - 3 компоненты")

for p in [20,30,50]:

    tsne2 = TSNE(n_components=2, perplexity=p, init="pca", random_state=42)
    X_tsne2 = tsne2.fit_transform(X)

    plot2d(X_tsne2,"t-SNE 2D perplexity="+str(p))


    tsne3 = TSNE(n_components=3, perplexity=p, init="pca", random_state=42)
    X_tsne3 = tsne3.fit_transform(X)

    plot3d(X_tsne3,"t-SNE 3D perplexity="+str(p))

cov = np.cov(X.T)

values,vectors = np.linalg.eig(cov)

idx = np.argsort(values)[::-1]

values = values[idx]
vectors = vectors[:,idx]


X_manual_2 = X @ vectors[:,:2]
X_manual_3 = X @ vectors[:,:3]


print("\nPCA вручную")

print("\nСобственные значения:")
print(values)

print("\nДоля дисперсии:")
print(values/values.sum())

print("\nРазмер после PCA:")
print("2 компоненты:",X_manual_2.shape)
print("3 компоненты:",X_manual_3.shape)

pca = PCA(n_components=3)

X_sklearn = pca.fit_transform(X)

X_sklearn_2 = X_sklearn[:,:2]
X_sklearn_3 = X_sklearn[:,:3]

plot2d(X_manual_2,"Ручной PCA - 2 компоненты")
plot2d(X_sklearn_2,"sklearn PCA - 2 компоненты")

plot3d(X_manual_3,"Ручной PCA - 3 компоненты")
plot3d(X_sklearn_3,"sklearn PCA - 3 компоненты")

print("\nРучной PCA")

for n in [2,3]:
    exp = values[:n].sum()/values.sum()
    print(n,"ГК:",exp)
    print("Потери:",1-exp)

print("\nsklearn PCA")

for n in [2,3]:
    exp = pca.explained_variance_ratio_[:n].sum()
    print(n,"ГК:",exp)
    print("Потери:",1-exp)

print("\nСравнение методов")

print(
    "2 компоненты:",
    round(values[:2].sum()/values.sum(),6),
    "и",
    round(pca.explained_variance_ratio_[:2].sum(),6)
)

print(
    "3 компоненты:",
    round(values[:3].sum()/values.sum(),6),
    "и",
    round(pca.explained_variance_ratio_[:3].sum(),6)
)