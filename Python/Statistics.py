# Statistics.py
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Create results folder if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Load dataset
data = load_wine()
X, y = data.data, data.target

# -----------------------------
# 70:15:15 split
# -----------------------------
X_train_val_70, X_test_70, y_train_val_70, y_test_70 = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train_70, X_val_70, y_train_70, y_val_70 = train_test_split(
    X_train_val_70, y_train_val_70, test_size=0.1765, random_state=42, stratify=y_train_val_70
)

print(f"70:15:15 split - Train: {len(X_train_70)}, Validation: {len(X_val_70)}, Test: {len(X_test_70)}")

# Build pipeline
model_70 = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Train model
model_70.fit(X_train_70, y_train_70)

# Validation & Test evaluation
val_preds_70 = model_70.predict(X_val_70)
test_preds_70 = model_70.predict(X_test_70)
print(f"70:15:15 Validation Accuracy: {accuracy_score(y_val_70, val_preds_70):.4f}")
print(f"70:15:15 Test Accuracy: {accuracy_score(y_test_70, test_preds_70):.4f}")
print("\nClassification Report (70:15:15 Test Set):")
print(classification_report(y_test_70, test_preds_70, target_names=data.target_names))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model_70, X_test_70, y_test_70, display_labels=data.target_names)
plt.title("Confusion Matrix (70:15:15 split)")
plt.savefig("results/confusion_matrix_70_15_15.png")
plt.close()

# -----------------------------
# 60:20:20 split
# -----------------------------
X_train_val_60, X_test_60, y_train_val_60, y_test_60 = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train_60, X_val_60, y_train_60, y_val_60 = train_test_split(
    X_train_val_60, y_train_val_60, test_size=0.25, random_state=42, stratify=y_train_val_60
)

print(f"60:20:20 split - Train: {len(X_train_60)}, Validation: {len(X_val_60)}, Test: {len(X_test_60)}")

# Build pipeline
model_60 = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Train model
model_60.fit(X_train_60, y_train_60)

# Validation & Test evaluation
val_preds_60 = model_60.predict(X_val_60)
test_preds_60 = model_60.predict(X_test_60)
print(f"60:20:20 Validation Accuracy: {accuracy_score(y_val_60, val_preds_60):.4f}")
print(f"60:20:20 Test Accuracy: {accuracy_score(y_test_60, test_preds_60):.4f}")
print("\nClassification Report (60:20:20 Test Set):")
print(classification_report(y_test_60, test_preds_60, target_names=data.target_names))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model_60, X_test_60, y_test_60, display_labels=data.target_names)
plt.title("Confusion Matrix (60:20:20 split)")
plt.savefig("results/confusion_matrix_60_20_20.png")
plt.close()

print("\nAll results and confusion matrices saved in the 'results/' folder.")
