import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6,
                                   verbose=0, random_seed=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                             use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 (Churn)': f1_score(y_test, y_pred, pos_label=1),
        'F1 (macro)': f1_score(y_test, y_pred, average='macro'),
        'F1 (weighted)': f1_score(y_test, y_pred, average='weighted')
    }

results_df = pd.DataFrame(results).T
print(results_df.round(4))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[idx].set_title(f'{name}\nF1 (Churn): {results[name]["F1 (Churn)"]:.4f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

axes[-1].axis('off')
plt.suptitle('Матрицы ошибок для моделей (Telco Customer Churn)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['F1 (Churn)', 'F1 (macro)', 'F1 (weighted)']
x = np.arange(len(results_df.index))
width = 0.25

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i*width - width, results_df[metric], width, label=metric)

ax.set_xlabel('Модель')
ax.set_ylabel('F1-score')
ax.set_title('Сравнение моделей по F1-score (класс "отток")')
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

for i, metric in enumerate(metrics_to_plot):
    for j, v in enumerate(results_df[metric]):
        ax.text(j + i*width - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('f1_comparison.png', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rf_model = models['Random Forest']
importances_rf = rf_model.feature_importances_
feature_names = X.columns
indices_rf = np.argsort(importances_rf)[::-1][:15]

axes[0].barh(range(len(indices_rf)), importances_rf[indices_rf], color='steelblue')
axes[0].set_yticks(range(len(indices_rf)))
axes[0].set_yticklabels([feature_names[i] for i in indices_rf])
axes[0].set_xlabel('Важность')
axes[0].set_title('Random Forest: Топ-15 признаков')
axes[0].invert_yaxis()

cb_model = models['CatBoost']
importances_cb = cb_model.get_feature_importance()
indices_cb = np.argsort(importances_cb)[::-1][:15]

axes[1].barh(range(len(indices_cb)), importances_cb[indices_cb], color='darkorange')
axes[1].set_yticks(range(len(indices_cb)))
axes[1].set_yticklabels([feature_names[i] for i in indices_cb])
axes[1].set_xlabel('Важность')
axes[1].set_title('CatBoost: Топ-15 признаков')
axes[1].invert_yaxis()

plt.suptitle('Важность признаков (Feature Importance)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

best_model = results_df['F1 (Churn)'].idxmax()
best_f1 = results_df.loc[best_model, 'F1 (Churn)']
print(f"Лучшая модель по F1-score для класса 'отток': {best_model} (F1 = {best_f1:.4f})")