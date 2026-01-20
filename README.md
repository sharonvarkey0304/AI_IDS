# AI-Based Intrusion Detection System (IDS)
[cite_start]This project focuses on building a robust AI-powered IDS using Python to classify network traffic into normal or anomalous categories based on 41 features from the KDD dataset[cite: 8, 9].

## Project Overview
- [cite_start]**Objective:** Create an intelligent system using ML/DL models to detect modern threats like APTs[cite: 12, 15].
- [cite_start]**Dataset:** KDD dataset containing 125,974 training records and 22,545 testing records[cite: 38].
- [cite_start]**Models Implemented:** Random Forest, Support Vector Machine (SVM), and a Deep Neural Network (PyTorch)[cite: 23, 24, 206].

## Key Findings
- [cite_start]**Best Performer:** The **Random Forest** model achieved a near-perfect F1-score of **0.9919**, proving most resilient to data redundancy[cite: 308, 359].
- [cite_start]**Comparison:** While the Neural Network showed high accuracy during initial testing, Random Forest provided better generalization for the final prototype[cite: 356, 359].

## Prototype Features
- [cite_start]Includes a **Real-Time IDS Prototype** using Python socket programming[cite: 315].
- [cite_start]Server-client architecture for rapid traffic prediction[cite: 317].

## How to Run
1. [cite_start]Start the server: `python server.py` 
2. [cite_start]Start the client: `python client.py` [cite: 340]