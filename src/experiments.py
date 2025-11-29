import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .config import load_config
from .data_loader import load_articles
from .indexer import build_indices
from .system import RetrievalSystem
from .evaluator import evaluate_system
from .logging_utils import setup_logging


# Each experiment toggles some parts but all code paths exist.
EXPERIMENTS = [
    {
        "name": "baseline_tfidf_only",
        "fusion": {"tfidf": 1.0, "bm25": 0.0, "embed": 0.0},
        "prf_enabled": False,
    },
    {
        "name": "tfidf_with_prf",
        "fusion": {"tfidf": 1.0, "bm25": 0.0, "embed": 0.0},
        "prf_enabled": True,
    },
    {
        "name": "bm25_only",
        "fusion": {"tfidf": 0.0, "bm25": 1.0, "embed": 0.0},
        "prf_enabled": False,
    },
    {
        "name": "tfidf_bm25_hybrid",
        "fusion": {"tfidf": 0.5, "bm25": 0.5, "embed": 0.0},
        "prf_enabled": True,
    },
    {
        "name": "full_hybrid_with_embeddings",
        "fusion": {"tfidf": 0.4, "bm25": 0.3, "embed": 0.3},
        "prf_enabled": True,
    },
]


def run_experiments():
    logger = setup_logging()
    logger.info("Starting experiments.")

    config = load_config()
    k = config["evaluation"]["k"]

    # Load data and build indices ONCE
    docs = load_articles(config["data"]["articles_csv"])
    pre = config["preprocessing"]
    preproc_doc_kwargs = {
        "language": pre["language"],
        "lowercase": pre["lowercase"],
        "remove_punctuation": pre["remove_punctuation"],
        "use_stemming": pre["use_stemming"],
        "use_lemmatization": pre["use_lemmatization"],
        "fix_spelling": False,
        "expand_query": False,
    }
    build_indices(
        docs=docs,
        processed_dir="data/processed",
        preproc_doc_kwargs=preproc_doc_kwargs,
        max_features=config["indexing"]["max_features"],
        ngram_range=tuple(config["indexing"]["ngram_range"]),
        embedding_model_name=config["embeddings"]["model_name"],
    )

    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)

    rows = []

    for exp in EXPERIMENTS:
        logger.info(f"Running experiment: {exp['name']}")
        config["retrieval"]["fusion_weights"] = exp["fusion"]
        config["retrieval"]["pseudo_relevance_feedback"]["enabled"] = exp["prf_enabled"]

        system = RetrievalSystem(config)
        metrics = evaluate_system(system, k=k)
        logger.info(f"Metrics for {exp['name']}: {metrics}")

        row = {"experiment": exp["name"], **metrics}
        rows.append(row)

    # Save CSV
    csv_file = experiments_dir / "results.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                f"precision@{k}",
                f"recall@{k}",
                "MAP",
                f"nDCG@{k}",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    make_plots(rows, k, experiments_dir)
    logger.info("Experiments completed.")


def make_plots(rows, k: int, out_dir: Path):
    exps = [r["experiment"] for r in rows]
    p_vals = [r[f"precision@{k}"] for r in rows]
    r_vals = [r[f"recall@{k}"] for r in rows]
    map_vals = [r["MAP"] for r in rows]
    ndcg_vals = [r[f"nDCG@{k}"] for r in rows]

    def barplot(values, title, fname, ylabel):
        plt.figure(figsize=(10, 5))
        plt.bar(exps, values)
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / fname)
        plt.close()

    barplot(p_vals, f"Precision@{k} by Experiment", "precision_at_k.png", "Precision")
    barplot(r_vals, f"Recall@{k} by Experiment", "recall_at_k.png", "Recall")
    barplot(map_vals, "MAP by Experiment", "map.png", "MAP")
    barplot(ndcg_vals, f"nDCG@{k} by Experiment", "ndcg.png", "nDCG")
