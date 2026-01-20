import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Suppress warnings during training and evaluation
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Data Loading and Preprocessing
train = pd.read_csv("E:/CW/AI_IDS/data/train_processed.csv")
test = pd.read_csv("E:/CW/AI_IDS/data/test_processed.csv")
# Detect target column
possible_targets = ["label", "target", "class", "attack"]
target_col = None
for col in possible_targets:
    if col in train.columns:
        target_col = col
        break
if target_col is None:
    target_col = train.columns[-1]

X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.drop(columns=[target_col])
y_test = test[target_col]

# Encode categorical features
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le_col = LabelEncoder()
        X_train[col] = le_col.fit_transform(X_train[col])
        X_test[col] = le_col.transform(X_test[col])

# Encode target
if y_train.dtype == 'object':
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)
else:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

num_classes = len(np.unique(y_train))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Training ML Models

print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

print("Training Support Vector Machine (SVC)...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

rf_scores = evaluate(y_test, rf_pred)
svm_scores = evaluate(y_test, svm_pred)


# Training Neural Network
print("Training Neural Network...")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Use smaller subset for faster NN training (as defined in original prompt)
subset_size = min(5000, X_train_tensor.shape[0])
X_train_tensor = X_train_tensor[:subset_size]
y_train_tensor = y_train_tensor[:subset_size]

class IDSNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X_train_tensor.shape[1]
model = IDSNet(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
batch_size = 32
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    for i in range(0, X_train_tensor.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate Neural Network
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, nn_pred = torch.max(outputs, 1)
nn_pred = nn_pred.numpy()
nn_scores = evaluate(y_test, nn_pred)


# Results and Comparison
print("\n=== Model Comparison (Actual Results) ===")
print(f"Random Forest: Accuracy={rf_scores[0]:.4f}, Precision={rf_scores[1]:.4f}, Recall={rf_scores[2]:.4f}, F1={rf_scores[3]:.4f}")
print(f"SVM:           Accuracy={svm_scores[0]:.4f}, Precision={svm_scores[1]:.4f}, Recall={svm_scores[2]:.4f}, F1={svm_scores[3]:.4f}")
print(f"Neural Network:Accuracy={nn_scores[0]:.4f}, Precision={nn_scores[1]:.4f}, Recall={nn_scores[2]:.4f}, F1={nn_scores[3]:.4f}")


accuracy = [rf_scores[0], svm_scores[0], nn_scores[0]]
precision = [rf_scores[1], svm_scores[1], nn_scores[1]]
recall = [rf_scores[2], svm_scores[2], nn_scores[2]]
f1 = [rf_scores[3], svm_scores[3], nn_scores[3]]

models = ["Random Forest", "SVM", "Neural Network"]


# Bar Chart Comparison

x = np.arange(len(models))
width = 0.18
# Define custom colors
colors = ['#FF6347', '#4682B4', '#3CB371', '#FFA07A']

plt.figure(figsize=(10, 7))

# Plotting with distinct colors
plt.bar(x - width*1.5, accuracy, width, label='Accuracy', color=colors[0])
plt.bar(x - width/2, precision, width, label='Precision', color=colors[1])
plt.bar(x + width/2, recall, width, label='Recall', color=colors[2])
plt.bar(x + width*1.5, f1, width, label='F1-score', color=colors[3])

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Model Performance Comparison for Intrusion Detection")
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()