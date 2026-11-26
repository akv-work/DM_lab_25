import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('iris.csv')
X, y = df.drop('variety', axis=1), df['variety']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42,
                             eval_metric='mlogloss', verbosity=0),
    'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
}

results = {}
for name, model in models.items():
    t0 = time.time()
    if name in ['XGBoost', 'CatBoost']:
        model.fit(X_train, y_train_enc)
        pred = le.inverse_transform(model.predict(X_test).astype(int))
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    results[name] = {'accuracy': accuracy_score(y_test, pred),
                     'time': time.time() - t0}

res = pd.DataFrame(results).T.sort_values('accuracy', ascending=False)
print("Точность моделей на тестовой выборке:")
print(res)

best = res.index[0]
print(f"\nВывод: лучшая модель — {best} (accuracy = {res.iloc[0]['accuracy']:.4f}).")

plt.figure(figsize=(10, 5))
sns.barplot(x=res.index, y=res['accuracy'], palette='viridis')
plt.ylim(0.85, 1.0)
plt.title('Accuracy моделей (Iris)')
plt.ylabel('Accuracy')
plt.xlabel('Модель')
for i, v in enumerate(res['accuracy']):
    plt.text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()