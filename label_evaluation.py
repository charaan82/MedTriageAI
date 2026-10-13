import pandas as pd

INPUT_FILE = "data/evaluation/evaluation_50.csv"
OUTPUT_FILE = "data/evaluation/evaluation_50.csv"


# Severity mapping used for this project evaluation
def assign_severity(category):
    category = str(category).strip().lower()

    if category == "normal":
        return "Low"

    elif category in ["reconnaissance", "analysis"]:
        return "Medium"

    else:
        return "High"


# Load evaluation dataset
df = pd.read_csv(INPUT_FILE)

# Make sure source_attack_cat exists
if "source_attack_cat" not in df.columns:
    print("ERROR: source_attack_cat column not found.")
    raise SystemExit

# Fill expected severity
df["expected_severity"] = df["source_attack_cat"].apply(
    assign_severity
)

# Save the updated file
df.to_csv(OUTPUT_FILE, index=False)

print("Evaluation labels created successfully!")
print()
print("Severity distribution:")
print(df["expected_severity"].value_counts())
print()
print("Updated file:")
print(OUTPUT_FILE)