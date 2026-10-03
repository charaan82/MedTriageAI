import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MITRE_FILE = "data/mitre/enterprise-attack.json"
INDEX_FILE = "index/mitre.index"
METADATA_FILE = "index/mitre_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print("Loading MITRE ATT&CK data...")

with open(MITRE_FILE, "r", encoding="utf-8") as f:
    bundle = json.load(f)


documents = []

for obj in bundle.get("objects", []):

    if obj.get("type") != "attack-pattern":
        continue

    if obj.get("revoked", False):
        continue

    if obj.get("x_mitre_deprecated", False):
        continue

    technique_id = None

    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            technique_id = ref.get("external_id")
            break

    if not technique_id:
        continue

    name = obj.get("name", "")

    description = obj.get(
        "description",
        ""
    )

    clean_text = f"""
Technique ID: {technique_id}
Technique Name: {name}
Description: {description}
""".strip()

    documents.append({
        "id": technique_id,
        "name": name,
        "description": description,
        "text": clean_text
    })


print("Techniques loaded:", len(documents))


print("Loading Sentence-BERT...")

encoder = SentenceTransformer(MODEL_NAME)

texts = [
    item["text"]
    for item in documents
]


print("Creating embeddings...")

embeddings = encoder.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


os.makedirs("index", exist_ok=True)

faiss.write_index(
    index,
    INDEX_FILE
)

with open(METADATA_FILE, "wb") as f:
    pickle.dump(documents, f)


print("\nFAISS index created.")

print("Index:", INDEX_FILE)
print("Metadata:", METADATA_FILE)
print("Vectors:", index.ntotal)