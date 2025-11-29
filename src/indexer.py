from pathlib import Path
from typing import List, Dict, Tuple

import os
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .preprocessing import TextPreprocessor


def _resolve_model_path(model_name: str) -> str:
    """
    If the model is already cached locally (HF offline), return the latest
    snapshot path; otherwise return the original hub id.
    """
    def _complete_model_dir(p: Path) -> bool:
        # SentenceTransformer expects these files to load without network
        return (p / "modules.json").exists() and (
            (p / "sentence_bert_config.json").exists()
            or (p / "0_Transformer").exists()
        )

    # Explicit local folder
    local_dir = Path(model_name)
    if local_dir.exists() and _complete_model_dir(local_dir):
        return str(local_dir)

    project_local = Path("models") / model_name.replace("/", "_")
    if project_local.exists() and _complete_model_dir(project_local):
        return str(project_local)

    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    repo_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        # pick the most recent snapshot
        snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snapshot_dirs:
            if _complete_model_dir(snap):
                return str(snap)
    return model_name


def build_indices(
    docs: List[Dict],
    processed_dir: str,
    preproc_doc_kwargs: Dict,
    max_features: int,
    ngram_range: Tuple[int, int],
    embedding_model_name: str,
):
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1) Document preprocessing
    doc_preproc = TextPreprocessor(is_query=False, **preproc_doc_kwargs)
    texts = [doc_preproc.preprocess(d["content"]) for d in docs]

    # 2) TF-IDF index
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    dump(vectorizer, processed_path / "tfidf_vectorizer.joblib")
    dump(doc_term_matrix, processed_path / "tfidf_matrix.joblib")

    # 3) BM25 index
    from rank_bm25 import BM25Okapi

    tokenized_docs = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_docs)
    dump(bm25, processed_path / "bm25_index.joblib")
    dump(tokenized_docs, processed_path / "bm25_tokens.joblib")

    # 4) Neural embeddings
    model_path = _resolve_model_path(embedding_model_name)
    embedder = SentenceTransformer(model_path)
    doc_embeddings = embedder.encode(
        [d["content"] for d in docs],
        show_progress_bar=True
    )
    dump(doc_embeddings, processed_path / "doc_embeddings.joblib")

    # 5) Save metadata
    dump(docs, processed_path / "docs_metadata.joblib")
