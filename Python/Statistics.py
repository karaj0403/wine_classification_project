# Required capstone component 5.1: Analysing how data splitting affects the model's performance

# Import Libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
data = load_wine()
X, y = data.data, data.target

# -----------------------------
# Split: 70:15:15
# -----------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

# Build pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Train model
model.fit(X_train, y_train)

# Validation performance
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test performance
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds, target_names=data.target_names))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=data.target_names)
plt.title("Confusion Matrix (70:15:15 split)")
plt.savefig("results/confusion_matrix_70_15_15.png")
plt.show()


