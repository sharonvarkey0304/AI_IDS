import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# Load Preprocessed Dataset with robust path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(script_dir, "data", "train_processed.csv")
test_file_path = os.path.join(script_dir, "data", "test_processed.csv")

try:
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)

    print("Files loaded successfully")
    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

except FileNotFoundError:
    print(f"\nError: One or both processed CSV files not found.")
    print(f"Expected Train file at: {train_file_path}")
    print(f"Please ensure the 'data' folder exists in the same directory as this script.")
    exit()


# Identify Target Column
target_col = "label"   # Change this if your dataset uses another name
X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.drop(columns=[target_col])
y_test = test[target_col]

# Encode Labels
if y_train.dtype == 'object':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
else:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

#  Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# Train SVM
print("\nTraining Support Vector Machine...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nDetailed Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    # Return scores as a tuple
    return (acc, prec, rec, f1)

rf_scores = evaluate_model(y_test, rf_pred, "Random Forest")
svm_scores = evaluate_model(y_test, svm_pred, "SVM")

# Compare Models
models = ['Random Forest', 'SVM']
# Unpack the tuples into lists for plotting
accuracy = [rf_scores[0], svm_scores[0]]
precision = [rf_scores[1], svm_scores[1]]
recall = [rf_scores[2], svm_scores[2]]
f1 = [rf_scores[3], svm_scores[3]]

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width*1.5, accuracy, width, label='Accuracy', color='cyan')
plt.bar(x - width/2, precision, width, label='Precision', color='magenta')
plt.bar(x + width/2, recall, width, label='Recall', color='lime')
plt.bar(x + width*1.5, f1, width, label='F1 Score', color='orangered')

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Model Comparison (Random Forest vs SVM)")
plt.legend()
plt.tight_layout()
plt.show()

#  Save Models and Scaler
MODEL_DIR = os.path.join(script_dir, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print(f"\nModels and scaler saved successfully in: {MODEL_DIR}")
