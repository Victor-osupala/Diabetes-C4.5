import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # For saving the model and selector

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Step 1: Load Dataset
df = pd.read_csv("diabetes.csv")  # Replace with your dataset path if different

# Step 2: Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Step 3: Correlation-based Feature Selection (simulated with ANOVA F-test)
selector = SelectKBest(score_func=f_classif, k='all')
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X_selected = df[selected_features]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step 5: C4.5-like Decision Tree (entropy-based)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 7: Evaluation
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Step 8: Save results
output_dir = "diabetes_model_output"
os.makedirs(output_dir, exist_ok=True)

# Save evaluation metrics to text
with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

# Step 9: Plot confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_heatmap.png"))
plt.close()

# Step 10: Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# Step 11: Save model and feature selector
model_path = os.path.join(output_dir, "diabetes_model.joblib")
selector_path = os.path.join(output_dir, "feature_selector.joblib")
joblib.dump(model, model_path)
joblib.dump(selector, selector_path)

print(f"✅ All outputs saved to: {output_dir}")
print(f"✅ Model saved to: {model_path}")
print(f"✅ Feature selector saved to: {selector_path}")
