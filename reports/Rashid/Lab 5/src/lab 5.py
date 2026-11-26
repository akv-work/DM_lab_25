import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- Load Data ---
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
             "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
             "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
             "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
             "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# Make sure this path is correct for your PC
df = pd.read_csv(r'C:\lab_works\iad\lab 5\kddcup.data_10_percent_corrected', header=None, names=col_names)

# Binary target variable
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n")

# Initialize Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = []
predictions = {} # To store predictions for confusion matrices
model_names = list(models.keys())

# Train and evaluate
for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Attack)": recall_score(y_test, y_pred, pos_label=1),
        "F1 Macro": f1_score(y_test, y_pred, average='macro'),
        "Training Time (s)": train_time
    })

# --- 1. Display Full Results Table ---
results_df = pd.DataFrame(results).sort_values(by="Recall (Attack)", ascending=False)
print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))

# --- 2. Plot Metric Comparison Bar Chart ---
plt.figure(figsize=(12, 6))
metrics_to_plot = ["Accuracy", "Precision", "Recall (Attack)", "F1 Macro"]
results_df.set_index("Model")[metrics_to_plot].plot(kind='bar', figsize=(12, 6))
plt.title("Metrics Comparison by Model")
plt.ylabel("Score")
plt.ylim(0.8, 1.05) # Adjust based on your scores
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# --- 3. Plot Confusion Matrices ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, name in enumerate(model_names):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    axes[i].set_title(f"Confusion Matrix: {name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

# Hide the empty 6th subplot
axes[5].axis('off')
plt.tight_layout()
plt.show()

# --- 4. Classification Report for the Best Model ---
best_model_name = results_df.iloc[0]["Model"]
print(f"\n--- Classification Report for Best Model: {best_model_name} ---")
print(classification_report(y_test, predictions[best_model_name], target_names=['Normal', 'Attack']))

# --- 5. Final Conclusion Text ---
print("\n=== Conclusion ===")
print(f"Based on the Recall metric for the 'Attack' class, {best_model_name} performed the best.")
print("In intrusion detection, high Recall is critical because False Negatives (missing an attack) are much more dangerous than False Positives.")