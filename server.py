import socket
import numpy as np
import joblib
import pickle
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
# Auto-detect project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Corrected path construction:
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

rf_model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
test_data_path = os.path.join(DATA_DIR, "Test_processed.csv")

print(f"Loading files from BASE_DIR: {BASE_DIR}")
print(f"Attempting to load model from: {rf_model_path}")
print(f"Attempting to load scaler from: {scaler_path}")
print(f"Attempting to load columns from: {test_data_path}")

try:
    # Load trained model and scaler
    rf_model = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")

    # Load column names used in training
    test_df = pd.read_csv(test_data_path)
    target_col = "label"
    # Ensure all columns except the target are used for prediction features
    columns = test_df.drop(columns=[target_col]).columns
    print(f"Loaded {len(columns)} feature columns for prediction.")

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: A required file was not found.")
    print(f"Please ensure your project structure is correct, specifically:")
    print(f"  - {MODEL_DIR} contains 'rf_model.joblib' and 'scaler.pkl'")
    print(f"  - {DATA_DIR} contains 'Test_processed.csv'")
    print(f"Details: {e}")
    exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred during loading: {e}")
    exit(1)

# Server Configuration and Main Loop

HOST = "127.0.0.1"
PORT = 9999

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"\nReal-Time IDS Server started at {HOST}:{PORT}...")
except socket.error as e:
    print(f"\nSocket error during binding/listening: {e}")
    exit(1)

while True:
    conn, addr = server.accept()
    print(f"\n--- Connected by {addr} ---")

    try:
        # Receive data
        data = conn.recv(4096)
        if not data:
            print("Received empty data. Closing connection.")
            continue

        # Deserialize sample
        sample = pickle.loads(data)

        # Check if the received sample matches the expected number of features
        if len(sample) != len(columns):
            raise ValueError(f"Received sample size ({len(sample)}) does not match expected features ({len(columns)}).")

        # Create DataFrame for scaling (needs 2D array: 1 row, N columns)
        sample_df = pd.DataFrame(np.array(sample).reshape(1, -1), columns=columns)

        # Scale features
        sample_scaled = scaler.transform(sample_df)

        # Predict
        pred = rf_model.predict(sample_scaled)[0]
        # Assuming 0 is Normal, 1 is Anomalous based on typical classification
        label = "Normal" if pred == 0 else "Anomalous"

        print(f"Received data size: {len(sample)}")
        print(f"Prediction: **{label}**")

        # Send back prediction
        conn.sendall(label.encode('utf-8'))

    except ValueError as ve:
        error_msg = f"Data Error: {ve}"
        print(error_msg)
        conn.sendall(error_msg.encode('utf-8'))
    except pickle.UnpicklingError:
        error_msg = "Error: Failed to deserialize data. Check client's pickling method."
        print(error_msg)
        conn.sendall(error_msg.encode('utf-8'))
    except Exception as e:
        error_msg = f"General Error: {e}"
        print(error_msg)
        conn.sendall(error_msg.encode('utf-8'))  # Send error back to client
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")