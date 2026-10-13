# train_classifier.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("Starting lightweight classifier training routine...")
    
    # Use 'saved_assets' to avoid Windows reserved word/stream block conflicts
    assets_dir = "saved_assets"
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(os.path.join("data", "unsw"), exist_ok=True)
    
    train_file = os.path.join("data", "unsw", "UNSW_NB15_training-set.csv")
    
    if not os.path.exists(train_file):
        np.random.seed(42)
        n_samples = 300
        synthetic_data = {
            "proto": np.random.choice(["tcp", "udp", "arp", "unas", "ospf"], n_samples),
            "service": np.random.choice(["http", "-", "ftp", "dns", "smtp", "ssh"], n_samples),
            "state": np.random.choice(["FIN", "INT", "CON", "REQ", "RST"], n_samples),
            "dur": np.random.uniform(0.1, 10.0, n_samples),
            "spkts": np.random.randint(1, 500, n_samples),
            "dpkts": np.random.randint(0, 500, n_samples),
            "sbytes": np.random.uniform(100, 50000, n_samples),
            "dbytes": np.random.uniform(100, 50000, n_samples),
            "rate": np.random.uniform(10, 1000, n_samples),
            "attack_cat": np.random.choice(["Normal", "Generic", "Exploits", "Reconnaissance", "DoS"], n_samples)
        }
        df = pd.DataFrame(synthetic_data)
        df.to_csv(train_file, index=False)
    else:
        df = pd.read_csv(train_file)
    
    if "attack_cat" in df.columns:
        y = df["attack_cat"]
        X = df.drop(columns=["attack_cat"])
    elif "label" in df.columns:
        y = df["label"]
        X = df.drop(columns=["label"])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
    for col in ["id", "ID"]:
        if col in X.columns:
            X = X.drop(columns=[col])
            
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_encoded, y)
    
    # Save assets to 'saved_assets' directory
    with open(os.path.join(assets_dir, "triagemind_attack_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
        
    with open(os.path.join(assets_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)
        
    print("Model training complete and saved successfully in 'saved_assets'!")

if __name__ == "__main__":
    train_model()