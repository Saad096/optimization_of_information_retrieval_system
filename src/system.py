import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sentence_transformers import SentenceTransformer

from .preprocessing import TextPreprocessor
from .indexer import _resolve_model_path


class RetrievalSystem:
    """
    Unified IR system:
    - TF-IDF similarity (cosine or dot)
    - BM25 scoring
    - Neural embedding similarity
    - Pseudo relevance feedback (Rocchio) on TF-IDF
    - Final ML-style fusion of scores
    - Embedding-based re-ranking
    """

    def __init__(self, config: Dict):
        processed_dir = Path("data/processed")
        self.config = config

        # Load metadata
        self.docs_metadata: List[Dict] = load(
            processed_dir / "docs_metadata.joblib"
        )

        # Preprocessors
        pre_cfg = config["preprocessing"]
        self.doc_preproc = TextPreprocessor(
            language=pre_cfg["language"],
            lowercase=pre_cfg["lowercase"],
            remove_punctuation=pre_cfg["remove_punctuation"],
            use_stemming=pre_cfg["use_stemming"],
            use_lemmatization=pre_cfg["use_lemmatization"],
            fix_spelling=False,
            expand_query=False,
            is_query=False,
        )
        self.query_preproc = TextPreprocessor(
            language=pre_cfg["language"],
            lowercase=pre_cfg["lowercase"],
            remove_punctuation=pre_cfg["remove_punctuation"],
            use_stemming=pre_cfg["use_stemming"],
            use_lemmatization=pre_cfg["use_lemmatization"],
            fix_spelling=pre_cfg["fix_spelling_queries"],
            expand_query=pre_cfg["expand_query"],
            is_query=True,
        )

        # Load TF-IDF
        self.vectorizer = load(processed_dir / "tfidf_vectorizer.joblib")
        self.doc_term_matrix = load(processed_dir / "tfidf_matrix.joblib")

        # Load BM25
        self.bm25 = load(processed_dir / "bm25_index.joblib")
        self.bm25_tokens: List[List[str]] = load(
            processed_dir / "bm25_tokens.joblib"
        )

        # Load embeddings
        emb_cfg = config["embeddings"]
        model_path = _resolve_model_path(emb_cfg["model_name"])
        self.embedder = SentenceTransformer(model_path)
        self.doc_embeddings = load(processed_dir / "doc_embeddings.joblib")

        # Fusion + PRF configs
        self.retr_cfg = config["retrieval"]
        self.fusion_weights = self.retr_cfg["fusion_weights"]
        self.sim_cfg = self.retr_cfg["similarity"]

    # ---------- low-level scorers ----------

    def _tfidf_scores(self, query_vec):
        # Ensure dense array for sklearn metrics when coming from sparse
        if hasattr(query_vec, "toarray"):
            query_vec = query_vec.toarray()
        sim_type = self.sim_cfg["tfidf"]
        if sim_type == "cosine":
            sims = cosine_similarity(query_vec, self.doc_term_matrix)[0]
        else:
            sims = linear_kernel(query_vec, self.doc_term_matrix)[0]
        return np.array(sims)

    def _bm25_scores(self, query_tokens: List[str]):
        scores = self.bm25.get_scores(query_tokens)
        return np.array(scores)

    def _embed_scores(self, query: str):
        sim_type = self.sim_cfg["embed"]
        q_emb = self.embedder.encode([query])
        sims = cosine_similarity(q_emb, self.doc_embeddings)[0]
        # (for dot product you'd use linear_kernel here)
        return np.array(sims)

    def _pseudo_relevance_feedback(self, query_vec, ranked_indices: np.ndarray):
        prf = self.retr_cfg["pseudo_relevance_feedback"]
        if not prf["enabled"]:
            return query_vec

        top_m = prf["top_m"]
        alpha = prf["alpha"]
        beta = prf["beta"]

        m = min(top_m, len(ranked_indices))
        if m == 0:
            return query_vec

        rel_docs = self.doc_term_matrix[ranked_indices[:m]]
        rel_centroid = rel_docs.mean(axis=0)

        if hasattr(query_vec, "toarray"):
            query_vec = query_vec.toarray()
        rel_centroid = np.asarray(rel_centroid)

        new_q = alpha * query_vec + beta * rel_centroid
        return new_q

    # ---------- main public method ----------

    def search(self, raw_query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        if top_k is None:
            top_k = self.retr_cfg["top_k"]

        # 1) preprocess query
        processed_query = self.query_preproc.preprocess(raw_query)
        tokens = processed_query.split()

        # 2) compute initial scores
        q_vec = self.vectorizer.transform([processed_query])
        tfidf_s = self._tfidf_scores(q_vec)
        bm25_s = self._bm25_scores(tokens)
        embed_s = self._embed_scores(raw_query)

        # 3) initial hybrid ranking for PRF
        hybrid_initial = (
            self.fusion_weights["tfidf"] * tfidf_s
            + self.fusion_weights["bm25"] * bm25_s
            + self.fusion_weights["embed"] * embed_s
        )
        initial_rank = np.argsort(hybrid_initial)[::-1]

        # 4) pseudo relevance feedback on TF-IDF
        q_vec_prf = self._pseudo_relevance_feedback(q_vec, initial_rank)
        tfidf_s_prf = self._tfidf_scores(q_vec_prf)

        # 5) final fusion after PRF
        hybrid_scores = (
            self.fusion_weights["tfidf"] * tfidf_s_prf
            + self.fusion_weights["bm25"] * bm25_s
            + self.fusion_weights["embed"] * embed_s
        )

        # 6) select top N candidates for embedding re-ranking
        top_n = max(top_k * 3, top_k)
        candidate_idx = np.argsort(hybrid_scores)[::-1][:top_n]

        # 7) embedding re-ranking among candidates
        q_emb = self.embedder.encode([raw_query])
        cand_embs = self.doc_embeddings[candidate_idx]
        emb_s_cand = cosine_similarity(q_emb, cand_embs)[0]

        # final rank within candidates by embedding similarity
        order = np.argsort(emb_s_cand)[::-1]
        final_idx = candidate_idx[order][:top_k]

        results = []
        for idx in final_idx:
            doc_meta = self.docs_metadata[int(idx)]
            score = float(hybrid_scores[int(idx)])
            results.append((doc_meta, score))

        return results
