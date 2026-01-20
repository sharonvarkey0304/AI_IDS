import socket
import pickle
import pandas as pd
import os
import time


# Auto-detect project data path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

print(f"Attempting to load data from: {os.path.join(DATA_DIR, 'test_processed.csv')}")

# Load test dataset
try:
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
    target_col = "label"
    X_test = test_df.drop(columns=[target_col])
    print(f"Test data loaded successfully. Total samples: {len(X_test)}")
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Could not find 'test_processed.csv'. Please ensure the file is at: {DATA_DIR}")
    print(f"Details: {e}")
    exit(1)

# Client Execution

HOST = "127.0.0.1"
PORT = 9999

# Send multiple samples one by one to simulate real-time traffic
for i in range(5):  # send first 5 samples
    sample = X_test.iloc[i].tolist()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client.connect((HOST, PORT))

        # Serialize and send
        client.sendall(pickle.dumps(sample))

        # Receive prediction
        pred = client.recv(1024).decode()
        print(f"Sent data row {i + 1} (Sample {i}), Prediction = {pred}")

    except ConnectionRefusedError:
        print(f"\nERROR: Connection refused. Is the server running at {HOST}:{PORT}?")
        break
    except Exception as e:
        print(f"\nAn error occurred during communication: {e}")
        break
    finally:
        client.close()
        # Wait for 1 second between sending samples
        time.sleep(1)