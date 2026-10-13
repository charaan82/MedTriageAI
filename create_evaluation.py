import os
import pandas as pd

# Input UNSW-NB15 dataset
INPUT_FILE = "data/unsw/UNSW_NB15_testing-set.csv"

# Output evaluation file
OUTPUT_DIR = "data/evaluation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_50.csv")


# ---------------------------------------------------------
# 1. Check that the dataset exists
# ---------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: UNSW-NB15 testing dataset not found.")
    print("Expected location:", INPUT_FILE)
    raise SystemExit


# ---------------------------------------------------------
# 2. Create evaluation folder
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 3. Read the UNSW-NB15 dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Total rows available:", len(df))


# ---------------------------------------------------------
# 4. Columns required by our Random Forest
# ---------------------------------------------------------

required_columns = [
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


# Check required columns
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("ERROR: These columns are missing:")
    print(missing_columns)
    raise SystemExit


# ---------------------------------------------------------
# 5. Select 50 rows
# ---------------------------------------------------------

evaluation = df.sample(
    n=50,
    random_state=42
).copy()


# ---------------------------------------------------------
# 6. Keep the Random Forest input features
# ---------------------------------------------------------

evaluation_features = evaluation[required_columns].copy()


# ---------------------------------------------------------
# 7. Add original UNSW-NB15 information for reference
# ---------------------------------------------------------

if "attack_cat" in evaluation.columns:
    evaluation_features["source_attack_cat"] = (
        evaluation["attack_cat"].astype(str).values
    )
else:
    evaluation_features["source_attack_cat"] = ""


if "label" in evaluation.columns:
    evaluation_features["source_label"] = (
        evaluation["label"].astype(str).values
    )
else:
    evaluation_features["source_label"] = ""


# ---------------------------------------------------------
# 8. Add alert IDs
# ---------------------------------------------------------

evaluation_features.insert(
    0,
    "alert_id",
    [f"TM-{i:03d}" for i in range(1, 51)]
)


# ---------------------------------------------------------
# 9. Add columns for our manual evaluation labels
# ---------------------------------------------------------

evaluation_features["expected_severity"] = ""
evaluation_features["expected_mitre_ids"] = ""
evaluation_features["analyst_description"] = ""


# ---------------------------------------------------------
# 10. Arrange columns
# ---------------------------------------------------------

evaluation_features = evaluation_features[
    [
        "alert_id",
        "proto",
        "service",
        "state",
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "source_attack_cat",
        "source_label",
        "expected_severity",
        "expected_mitre_ids",
        "analyst_description"
    ]
]


# ---------------------------------------------------------
# 11. Save evaluation dataset
# ---------------------------------------------------------

evaluation_features.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Evaluation dataset created successfully!")
print()
print("File saved to:")
print(OUTPUT_FILE)
print()
print("Rows created:", len(evaluation_features))
print()
print("The following columns are for manual evaluation:")
print("expected_severity")
print("expected_mitre_ids")
print("analyst_description")
print()
print("source_attack_cat and source_label are included")
print("as reference information from UNSW-NB15.")