from pathlib import Path
from typing import List, Dict, Tuple, Literal, Optional

import numpy as np
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from .preprocessing import TextPreprocessor


SimilarityType = Literal["cosine", "dot"]


class HybridRetriever:
    """
    Supports:
    - TF-IDF with cosine / dot
    - BM25
    - optional pseudo relevance feedback (Rocchio)
    - optional embedding-based re-ranking
    """

    def __init__(
        self,
        processed_dir: str,
        preproc_kwargs: Dict,
        method: Literal["tfidf", "bm25"] = "tfidf",
        similarity: SimilarityType = "cosine",
        prf_cfg: Optional[Dict] = None,
        embeddings_cfg: Optional[Dict] = None,
    ):
        self.processed_path = Path(processed_dir)
        self.method = method
        self.similarity = similarity
        self.preproc_kwargs = preproc_kwargs
        self.prf_cfg = prf_cfg or {"enabled": False}
        self.embeddings_cfg = embeddings_cfg or {"enabled": False}

        self.docs_metadata: List[Dict] = load(
            self.processed_path / "docs_metadata.joblib"
        )

        # Preprocessor for queries
        self.query_preprocessor = TextPreprocessor(
            is_query=True, **preproc_kwargs
        )

        # TF-IDF components
        if method == "tfidf":
            self.vectorizer = load(
                self.processed_path / "tfidf_vectorizer.joblib"
            )
            self.doc_term_matrix = load(
                self.processed_path / "tfidf_matrix.joblib"
            )
        else:
            self.vectorizer = None
            self.doc_term_matrix = None

        # BM25 components
        if method == "bm25":
            from rank_bm25 import BM25Okapi  # just for type info
            self.bm25 = load(self.processed_path / "bm25_index.joblib")
            self.bm25_tokens: List[List[str]] = load(
                self.processed_path / "bm25_tokens.joblib"
            )
        else:
            self.bm25 = None
            self.bm25_tokens = None

        # Embedding-based reranker (optional, local)
        self.embedder = None
        if self.embeddings_cfg.get("enabled", False):
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.embeddings_cfg.get(
                    "model_name", "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.embedder = SentenceTransformer(model_name)
                # Precompute doc embeddings (once)
                doc_texts = [d["content"] for d in self.docs_metadata]
                self.doc_embeddings = self.embedder.encode(
                    doc_texts, show_progress_bar=True
                )
            except ImportError:
                print(
                    "[WARNING] sentence-transformers not installed; "
                    "embedding reranker disabled."
                )
                self.embeddings_cfg["enabled"] = False
                self.embedder = None

    def _score_tfidf(self, query_vec, top_k: int) -> List[Tuple[int, float]]:
        if self.similarity == "cosine":
            sims = cosine_similarity(query_vec, self.doc_term_matrix)[0]
        else:
            # dot product (linear kernel)
            sims = linear_kernel(query_vec, self.doc_term_matrix)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_idx]

    def _score_bm25(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def _pseudo_relevance_feedback(
        self, query_vec, ranked_indices: List[int]
    ):
        """
        Simple Rocchio-style PRF:
            q' = alpha * q + beta * avg(d_rel)
        where d_rel are top-M docs.
        """
        if self.doc_term_matrix is None:
            return query_vec

        top_m = self.prf_cfg.get("top_m", 5)
        alpha = self.prf_cfg.get("alpha", 1.0)
        beta = self.prf_cfg.get("beta", 0.8)

        m = min(top_m, len(ranked_indices))
        if m == 0:
            return query_vec

        rel_docs = self.doc_term_matrix[ranked_indices[:m]]
        rel_centroid = rel_docs.mean(axis=0)

        # q' = alpha q + beta * rel_centroid
        new_q = alpha * query_vec + beta * rel_centroid
        return new_q

    def _rerank_with_embeddings(
        self, candidate_ids: List[int]
    ) -> List[int]:
        """
        Very simple embedding re-ranker: cosine similarity
        between query embedding and doc embeddings of
        candidates, reorder them.
        """
        if not self.embedder:
            return candidate_ids

        # Build list of candidate texts
        texts = [self.docs_metadata[i]["content"] for i in candidate_ids]
        q_emb = self.embedder.encode(texts[0:0] + [" "], show_progress_bar=False)  # dummy
        # Actually re-encode query, not docs; we need the query text
        # -> This function will be called with the latest query embedding
        # To keep code short, we'll instead encode on-the-fly in search()

        return candidate_ids  # we will handle in search() for clarity

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        processed_query = self.query_preprocessor.preprocess(query)

        if self.method == "tfidf":
            query_vec = self.vectorizer.transform([processed_query])

            # First scoring
            prelim_scores = self._score_tfidf(query_vec, top_k * 3)
            prelim_ids = [i for i, _ in prelim_scores]

            # PRF (only TF-IDF)
            if self.prf_cfg.get("enabled", False):
                query_vec = self._pseudo_relevance_feedback(
                    query_vec, prelim_ids
                )
                final_scores = self._score_tfidf(query_vec, top_k)
                ranked_ids = [i for i, _ in final_scores]
                scores = {i: s for i, s in final_scores}
            else:
                ranked_ids = [i for i, _ in prelim_scores[:top_k]]
                scores = {i: s for i, s in prelim_scores}

        else:  # BM25
            tokens = processed_query.split()
            prelim_scores = self._score_bm25(tokens, top_k * 3)
            ranked_ids = [i for i, _ in prelim_scores[:top_k]]
            scores = {i: s for i, s in prelim_scores}

        # Optional embedding re-ranker (local)
        if self.embeddings_cfg.get("enabled", False) and self.embedder:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            # Encode query and candidate docs
            q_emb = self.embedder.encode([query])
            cand_embs = self.doc_embeddings[ranked_ids]
            sims = cos_sim(q_emb, cand_embs)[0]
            order = np.argsort(sims)[::-1]
            ranked_ids = [ranked_ids[i] for i in order]

        results: List[Tuple[Dict, float]] = []
        for doc_id in ranked_ids[:top_k]:
            doc_meta = self.docs_metadata[doc_id]
            results.append((doc_meta, scores.get(doc_id, 0.0)))

        return results
