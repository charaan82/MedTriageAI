import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

EVALUATION_FILE = "data/evaluation/evaluation_50.csv"
MODEL_FILE = "models/triagemind_classifier.pkl"
FEATURE_FILE = "models/feature_columns.pkl"

RESULT_FILE = "data/evaluation/random_forest_results.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading TriageMind model...")

model = joblib.load(MODEL_FILE)
feature_columns = joblib.load(FEATURE_FILE)

print("Model loaded successfully.")
print(f"Number of model features: {len(feature_columns)}")


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

print("\nLoading evaluation dataset...")

df = pd.read_csv(EVALUATION_FILE)

print(f"Evaluation rows: {len(df)}")


# ============================================================
# PROJECT FEATURES
# ============================================================

FEATURES = [
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

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state"
]

NUMERIC_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate"
]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing = [col for col in FEATURES if col not in df.columns]

if missing:
    print("\nERROR: Missing required columns:")
    print(missing)
    raise SystemExit(1)


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nPreparing evaluation features...")

X = df[FEATURES].copy()


# Convert categorical columns to one-hot encoding
X = pd.get_dummies(
    X,
    columns=CATEGORICAL_FEATURES,
    drop_first=False
)


# Convert numerical columns to numbers
for col in NUMERIC_FEATURES:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# Replace invalid/missing values
X = X.replace(
    [float("inf"), float("-inf")],
    0
)

X = X.fillna(0)


# ============================================================
# ALIGN WITH TRAINING FEATURES
# ============================================================

print("Aligning features with trained model...")

X = X.reindex(
    columns=feature_columns,
    fill_value=0
)


print(f"Final evaluation feature count: {X.shape[1]}")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning predictions...")

predicted_labels = model.predict(X)

predicted_labels = pd.Series(
    predicted_labels
).astype(str).str.capitalize()


# ============================================================
# GROUND TRUTH
# ============================================================

# The evaluation dataset contains source_attack_cat.
# Convert the original attack category into the same
# severity categories used during Random Forest training.

SEVERITY_MAPPING = {
    "normal": "Low",

    "analysis": "Medium",
    "fuzzers": "Medium",
    "reconnaissance": "Medium",

    "backdoor": "High",
    "dos": "High",
    "generic": "High",

    "exploits": "Critical",
    "shellcode": "Critical",
    "worms": "Critical"
}


if "source_attack_cat" not in df.columns:
    print("\nERROR: source_attack_cat column not found.")
    raise SystemExit(1)


expected_labels = (
    df["source_attack_cat"]
    .astype(str)
    .str.lower()
    .map(SEVERITY_MAPPING)
)


# Check for unmapped categories
if expected_labels.isna().any():

    print("\nWARNING: Some attack categories could not be mapped.")

    unknown_categories = (
        df.loc[expected_labels.isna(), "source_attack_cat"]
        .unique()
    )

    print("Unknown categories:")
    print(unknown_categories)

    raise SystemExit(1)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    expected_labels,
    predicted_labels
)

precision = precision_score(
    expected_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    expected_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    expected_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# ============================================================
# HIGH SEVERITY RECALL
# ============================================================

# For the project metric, High severity means:
# High + Critical

high_or_critical_actual = expected_labels.isin(
    ["High", "Critical"]
)

high_or_critical_predicted = predicted_labels.isin(
    ["High", "Critical"]
)

actual_high_count = high_or_critical_actual.sum()

if actual_high_count > 0:

    correctly_detected_high = (
        high_or_critical_actual &
        high_or_critical_predicted
    ).sum()

    high_recall = (
        correctly_detected_high /
        actual_high_count
    )

else:
    high_recall = 0.0


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("====================================")
print("RANDOM FOREST EVALUATION RESULTS")
print("====================================")

print(f"Accuracy:             {accuracy * 100:.2f}%")
print(f"Weighted Precision:   {precision * 100:.2f}%")
print(f"Weighted Recall:      {recall * 100:.2f}%")
print(f"Weighted F1 Score:    {f1 * 100:.2f}%")
print(f"High Severity Recall: {high_recall * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        expected_labels,
        predicted_labels,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

cm = confusion_matrix(
    expected_labels,
    predicted_labels,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {x}" for x in labels],
    columns=[f"Predicted {x}" for x in labels]
)

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

print(cm_df)


# ============================================================
# SAVE RESULTS
# ============================================================

results = df.copy()

results["expected_severity"] = expected_labels
results["predicted_severity"] = predicted_labels
results["correct_prediction"] = (
    expected_labels == predicted_labels
)

results.to_csv(
    RESULT_FILE,
    index=False
)

print("\n====================================")
print("EVALUATION COMPLETE")
print("====================================")

print(f"Results saved to:")
print(RESULT_FILE)