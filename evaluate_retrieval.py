# evaluate_retrieval.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from hybrid_retrieval import HybridRetriever

def benchmark_retrieval():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = [
        "T1071.001 - Web Protocols: Adversaries may communicate using application layer protocols.",
        "T1498 - Network Denial of Service: Adversaries may perform network denial of service.",
        "T1499.001 - OS Exhaustion Flood: Adversaries may target system resources.",
        "T1059 - Command and Scripting Interpreter: Adversaries may abuse command scripts.",
        "T1595 - Active Scanning: Adversaries may execute active scanning tasks."
    ]
    
    embeddings = embedder.encode(corpus).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    retriever = HybridRetriever(index, corpus, embedder, alpha=0.7)
    
    # Test query with known ground truth technique ID
    test_queries = [
        {"query": "HTTP web traffic command control", "ground_truth": "T1071.001"},
        {"query": "Network denial of service flood", "ground_truth": "T1498"}
    ]
    
    recalls = {1: 0, 3: 0, 5: 0}
    mrr_total = 0.0
    
    for item in test_queries:
        results = retriever.search(item["query"], top_k=5)
        retrieved_ids = [res["technique_text"].split(" - ")[0] for res in results]
        
        gt = item["ground_truth"]
        # Check Recall@K
        for k in [1, 3, 5]:
            if gt in retrieved_ids[:k]:
                recalls[k] += 1
                
        # Calculate Reciprocal Rank
        if gt in retrieved_ids:
            rank = retrieved_ids.index(gt) + 1
            mrr_total += 1.0 / rank

    n = len(test_queries)
    print("--- Retrieval Benchmark Results ---")
    for k in [1, 3, 5]:
        print(f"Recall@{k}: {recalls[k] / n:.2f}")
    print(f"MRR: {mrr_total / n:.2f}")

if __name__ == "__main__":
    benchmark_retrieval()