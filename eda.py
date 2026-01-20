import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os


# Define paths relative to the current script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(script_dir, "data", "Train.csv")
test_file_path = os.path.join(script_dir, "data", "Test.csv")

# Load Datasets with error handling
try:
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    print("Training dataset shape:", train_df.shape)
    print("Testing dataset shape:", test_df.shape)

except FileNotFoundError:
    print(f"\nError: One or both CSV files not found.")
    print(f"Expected Train file at: {train_file_path}")
    print(f"Please ensure the 'data' folder exists in the same directory as this script.")
    exit()

# Summary Statistics

print("\nSummary statistics for training data:")
print(train_df.describe().T)

# Label Distribution

print("\nLabel distribution in training set:")
label_counts = train_df['label'].value_counts()
label_percent = (label_counts / len(train_df)) * 100
print(label_percent)

# Map Label Values:

label_map = {0: "Normal", 1: "Attack"}

train_df["label_name"] = train_df["label"].map(label_map)
test_df["label_name"] = test_df["label"].map(label_map)

# Normal vs Attack

plt.figure(figsize=(8, 5))
train_df['label_name'].value_counts().plot(kind='bar', color=['blue', 'yellow'])
plt.title("Training Dataset – Normal vs Attack Count")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Normal vs Attack (Test)

plt.figure(figsize=(8, 5))
test_df['label_name'].value_counts().plot(kind='bar', color=['blue', 'yellow'])
plt.title("Testing Dataset – Normal vs Attack Count")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation Heatmap

numeric_cols = train_df.select_dtypes(include=[np.number]).columns
corr_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.tight_layout()
plt.show()

# Attack Type Detection

attack_labels_numeric = [1]

# detect attack rows
train_df["is_attack"] = train_df["label"].apply(lambda x: 1 if x in attack_labels_numeric else 0)

# Filter attack rows

attack_df = train_df[train_df["is_attack"] == 1]

if "attack_type" not in train_df.columns:
    print("\nNo 'attack_type' column found — generating synthetic attack categories.")

    synthetic_types = ["DoS", "Probe", "R2L", "U2R"]

    # Assign a random synthetic type only for rows identified as attacks
    train_df["attack_type"] = train_df.apply(
        lambda row: random.choice(synthetic_types) if row["is_attack"] == 1 else "Normal",
        axis=1  # axis=1 ensures lambda iterates over rows
    )

    # Re-filter attack_df after generating the column
    attack_df = train_df[train_df["is_attack"] == 1]

# Attack Type Distribution

atk_counts = attack_df["attack_type"].value_counts()
atk_percent = (atk_counts / atk_counts.sum()) * 100

print("\nAttack Type Distribution (%):")
print(atk_percent)

# Plot
plt.figure(figsize=(10, 6))
atk_percent.plot(kind='bar', color='orange')
plt.title("Attack Type Distribution (%)")
plt.xlabel("Attack Category")
plt.ylabel("Percentage")
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully.")
