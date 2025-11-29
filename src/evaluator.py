from typing import Dict, List, Set

from .system import RetrievalSystem

# YOU fill this with real queries + relevant doc IDs
GROUND_TRUTH: Dict[str, Set[int]] = {
    # These IDs come from manual inspection using the dataset and search results.
    "pakistan stock market": {2612, 718, 723, 722, 2580},
    "hong kong stocks": {67, 2459, 2, 11, 566},
}


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    num_rel = sum(1 for d in retrieved_k if d in relevant)
    return num_rel / len(retrieved_k)


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    num_rel = sum(1 for d in retrieved_k if d in relevant)
    return num_rel / len(relevant)


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            num_hits += 1
            score += num_hits / i
    return score / len(relevant)


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    import math

    def dcg(scores: List[int]) -> float:
        return sum(
            (2**rel - 1) / math.log2(idx + 2)
            for idx, rel in enumerate(scores)
        )

    rel_scores = [1 if d in relevant else 0 for d in retrieved[:k]]
    ideal_scores = sorted(rel_scores, reverse=True)

    dcg_val = dcg(rel_scores)
    idcg_val = dcg(ideal_scores)

    return (dcg_val / idcg_val) if idcg_val > 0 else 0.0


def evaluate_system(system: RetrievalSystem, k: int) -> Dict[str, float]:
    precisions, recalls, maps, ndcgs = [], [], [], []

    for query, relevant_docs in GROUND_TRUTH.items():
        results = system.search(query, top_k=k)
        retrieved_ids = [doc["doc_id"] for doc, _ in results]

        precisions.append(precision_at_k(retrieved_ids, relevant_docs, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_docs, k))
        maps.append(average_precision(retrieved_ids, relevant_docs))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_docs, k))

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    return {
        f"precision@{k}": avg(precisions),
        f"recall@{k}": avg(recalls),
        "MAP": avg(maps),
        f"nDCG@{k}": avg(ndcgs),
    }
