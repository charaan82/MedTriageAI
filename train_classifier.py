import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/unsw/UNSW_NB15_training-set.csv"
MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "triagemind_classifier.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)


# ============================================================
# FEATURES USED BY TRIAGEMIND
# ============================================================

FEATURE_COLUMNS = [
    "proto",
    "service",
    "state",
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate"
]


# ============================================================
# SEVERITY MAPPING
# ============================================================

def map_severity(attack_category):

    if pd.isna(attack_category):
        return "Low"

    category = str(attack_category).strip().lower()

    severity_map = {
        "normal": "Low",

        "analysis": "Medium",
        "fuzzers": "Medium",
        "reconnaissance": "Medium",

        "backdoor": "High",
        "dos": "High",
        "generic": "High",

        "exploits": "Critical",
        "shellcode": "Critical",
        "worms": "Critical",
    }

    return severity_map.get(category, "Medium")


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading UNSW-NB15...")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{TRAIN_FILE}"
    )

df = pd.read_csv(TRAIN_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + ["attack_cat"]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CREATE TARGET
# ============================================================

print("\nCreating severity labels...")

df["severity"] = df["attack_cat"].apply(
    map_severity
)

print("\nSeverity distribution:")
print(df["severity"].value_counts())


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nPreparing features...")

X = df[FEATURE_COLUMNS].copy()

y = df["severity"].copy()


# ============================================================
# CONVERT CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X,
    columns=["proto", "service", "state"],
    drop_first=False
)


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0)


# ============================================================
# SAVE EXACT FEATURE COLUMNS
# ============================================================

feature_columns = list(X.columns)

print(
    "\nNumber of model features:",
    len(feature_columns)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST MODEL
# ============================================================

print("\nEvaluating on internal test set...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n====================================")
print("RANDOM FOREST TRAINING RESULTS")
print("====================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    feature_columns,
    FEATURE_FILE
)


# ============================================================
# FINISHED
# ============================================================

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print("\nModel saved to:")
print(MODEL_FILE)

print("\nFeature columns saved to:")
print(FEATURE_FILE)

print("\nTriageMind classifier is ready.")