import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Corrected Path Definition
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "Train.csv")
TEST_PATH = os.path.join(DATA_DIR, "Test.csv")

print("Loading datasets...")

# Verify the corrected path before loading
print(f"Attempting to load Train data from: {TRAIN_PATH}")

try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Could not find raw data files.")
    print(f"Please ensure 'Train.csv' and 'Test.csv' are in the correct directory: {DATA_DIR}")
    print(f"Details: {e}")
    exit(1)


print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


print("Checking for missing values...")
train_df = train_df.dropna()
test_df = test_df.dropna()


print("Encoding categorical columns...")

# Identify categorical columns
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

# Encode each categorical column using LabelEncoder
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    # Combine train + test values to ensure consistent encoding
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    le_dict[col] = le

print(f"Encoded columns: {cat_cols}")


print("Scaling numeric features...")
scaler = StandardScaler()

# num_cols definition is correct
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print(f"Scaled numeric columns: {len(num_cols)}")


train_out = os.path.join(DATA_DIR, "Train_processed.csv")
test_out = os.path.join(DATA_DIR, "Test_processed.csv")

train_df.to_csv(train_out, index=False)
test_df.to_csv(test_out, index=False)

print("Preprocessing complete!")
print(f"Processed train data saved to: {train_out}")
print(f"Processed test data saved to:  {test_out}")