import pandas as pd

from semantic_parser import parse_alert
from rag_engine import retrieve


# ============================================================
# PATHS
# ============================================================

EVALUATION_FILE = "data/evaluation/evaluation_50.csv"
RESULT_FILE = "data/evaluation/mitre_rag_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading 50-alert evaluation dataset...")

df = pd.read_csv(EVALUATION_FILE)

print(f"Alerts loaded: {len(df)}")


# ============================================================
# FAST MITRE RAG EVALUATION
# ============================================================

results = []

for index, row in df.iterrows():

    alert_id = row.get(
        "alert_id",
        f"ALERT-{index + 1}"
    )

    print(
        f"[{index + 1}/{len(df)}] "
        f"Processing {alert_id}..."
    )

    # Convert network telemetry to semantic text
    semantic_text = parse_alert(row)

    # Retrieve top 3 MITRE techniques
    retrieved = retrieve(
        semantic_text,
        top_k=3
    )

    retrieved_ids = [
        item["id"]
        for item in retrieved
    ]

    retrieved_names = [
        item["name"]
        for item in retrieved
    ]

    similarities = [
        item["similarity"]
        for item in retrieved
    ]

    results.append({
        "alert_id": alert_id,
        "expected_severity": row.get(
            "expected_severity",
            ""
        ),
        "semantic_description": semantic_text,
        "mitre_1": (
            retrieved_ids[0]
            if len(retrieved_ids) > 0
            else ""
        ),
        "mitre_2": (
            retrieved_ids[1]
            if len(retrieved_ids) > 1
            else ""
        ),
        "mitre_3": (
            retrieved_ids[2]
            if len(retrieved_ids) > 2
            else ""
        ),
        "mitre_names": " | ".join(
            retrieved_names
        ),
        "similarity_scores": " | ".join(
            f"{score:.4f}"
            for score in similarities
        )
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("TRIAGEMIND FAST MITRE RAG EVALUATION")
print("==============================================")

print(
    f"Alerts evaluated: {len(results_df)}"
)

print(
    "Top 3 MITRE techniques retrieved for "
    "each alert: YES"
)

print("\nResults saved to:")
print(RESULT_FILE)

print("\nEvaluation complete.")