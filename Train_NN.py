import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt


# Load dataset
train = pd.read_csv("E:/CW/AI_IDS/data/Train_processed.csv")
test = pd.read_csv("E:/CW/AI_IDS/data/Test_processed.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)


#  Identify target column
possible_targets = ["label", "target", "class", "attack"]
target_col = None
for col in possible_targets:
    if col in train.columns:
        target_col = col
        break

if target_col is None:
    target_col = train.columns[-1]

print("Target column:", target_col)

X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.drop(columns=[target_col])
y_test = test[target_col]

#  Encode categorical features
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le_col = LabelEncoder()
        X_train[col] = le_col.fit_transform(X_train[col])
        X_test[col] = le_col.transform(X_test[col])

# Encode target if needed
if y_train.dtype == 'object':
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)
else:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

num_classes = len(set(y_train))
print("Number of classes:", num_classes)

#  Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


#  Define Neural Network
class IDSNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X_train_tensor.shape[1]
model = IDSNet(input_dim, num_classes)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#  Train neural network
epochs = 15
batch_size = 64
loss_history = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    total_loss = 0

    for i in range(0, X_train_tensor.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(X_train_tensor) // batch_size)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")


#  Evaluate Model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

y_true = y_test_tensor.numpy()
y_pred = predicted.numpy()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\nNeural Network Performance (Numeric Metrics Only):")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))


#  Training Loss Curve
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', linewidth=2)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


#   Performance Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [acc, prec, rec, f1]

plt.figure(figsize=(8,5))
plt.bar(metrics, values)
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
