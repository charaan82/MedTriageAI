import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "index/mitre.index"
METADATA_FILE = "index/mitre_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

print("Loading MITRE ATT&CK RAG components...")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

embedding_model = SentenceTransformer(MODEL_NAME)

print(f"MITRE index loaded: {index.ntotal} vectors")


# ============================================================
# RETRIEVE MITRE TECHNIQUES
# ============================================================

def retrieve(query, top_k=3):

    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx]

        results.append({
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "similarity": float(score)
        })

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "A host is making network connections and "
        "appears to be discovering network configuration."
    )

    results = retrieve(test_query)

    print("\n====================================")
    print("MITRE ATT&CK RETRIEVAL TEST")
    print("====================================")

    for i, result in enumerate(results, start=1):

        print(
            f"\n{i}. {result['id']} - {result['name']}"
        )

        print(
            f"Similarity: {result['similarity']:.4f}"
        )

        print(
            f"Description: {result['description'][:300]}"
        )