# hybrid_retrieval.py
import numpy as np

class HybridRetriever:
    """
    Implements Hybrid Semantic-Lexical Retrieval as defined in 
    Section 19 of the TriageMind Revised Synopsis.
    S_j = alpha * S_semantic + (1 - alpha) * S_lexical
    """
    def __init__(self, faiss_index, corpus_texts, embedder, alpha=0.7):
        self.index = faiss_index
        self.corpus = corpus_texts
        self.embedder = embedder
        self.alpha = alpha  # Fusion weight parameter balancing semantic and lexical scores

    def search(self, query_text, top_k=5):
        """
        Performs hybrid retrieval combining semantic vector search and exact lexical keyword matching.
        """
        if not self.corpus:
            return []

        # 1. Semantic Search via Sentence-BERT and FAISS
        q_emb = self.embedder.encode([query_text]).astype('float32')
        # Retrieve distances/scores from FAISS
        sem_scores, sem_indices = self.index.search(q_emb, len(self.corpus))
        
        # Flatten results
        sem_scores = sem_scores[0]
        sem_indices = sem_indices[0]

        # Map scores back to original corpus indices
        unsorted_sem_scores = np.zeros(len(self.corpus))
        for idx, score in zip(sem_indices, sem_scores):
            if idx != -1:
                unsorted_sem_scores[idx] = score

        # Normalize Semantic Scores between [0, 1]
        sem_min, sem_max = np.min(unsorted_sem_scores), np.max(unsorted_sem_scores)
        if sem_max - sem_min > 1e-6:
            sem_norm = (unsorted_sem_scores - sem_min) / (sem_max - sem_min)
        else:
            sem_norm = np.zeros_like(unsorted_sem_scores)

        # 2. Lexical Search (Term overlap / BM25 style count)
        query_tokens = set(query_text.lower().split())
        lex_scores = np.array([
            sum(1 for token in query_tokens if token in doc.lower()) 
            for doc in self.corpus
        ], dtype=float)

        # Normalize Lexical Scores between [0, 1]
        lex_min, lex_max = np.min(lex_scores), np.max(lex_scores)
        if lex_max - lex_min > 1e-6:
            lex_norm = (lex_scores - lex_min) / (lex_max - lex_min)
        else:
            lex_norm = np.zeros_like(lex_scores)

        # 3. Hybrid Score Fusion Formula
        combined_scores = (self.alpha * sem_norm) + ((1.0 - self.alpha) * lex_norm)

        # Sort and get Top-K indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "technique_text": self.corpus[idx],
                "hybrid_score": float(combined_scores[idx]),
                "semantic_score": float(sem_norm[idx]),
                "lexical_score": float(lex_norm[idx])
            })

        return results