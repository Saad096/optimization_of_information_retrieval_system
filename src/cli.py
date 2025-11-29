import argparse

from .config import load_config
from .logging_utils import setup_logging
from .data_loader import load_articles
from .indexer import build_indices
from .system import RetrievalSystem
from .experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="News Articles IR System (TF-IDF, BM25, embeddings, PRF)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-indices", help="Build TF-IDF, BM25, embeddings")

    search_parser = subparsers.add_parser("search", help="Search the IR system")
    search_parser.add_argument("--query", type=str, required=True)
    search_parser.add_argument("--top-k", type=int, default=10)

    subparsers.add_parser("run-experiments", help="Run all experiments and save metrics/plots")

    args = parser.parse_args()
    logger = setup_logging()
    config = load_config()

    if args.command == "build-indices":
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
        logger.info("Indices built successfully.")

    elif args.command == "search":
        system = RetrievalSystem(config)
        results = system.search(args.query, top_k=args.top_k)
        print(f'\nQuery: "{args.query}"\n')
        for rank, (doc, score) in enumerate(results, start=1):
            print(
                f"{rank:2d}. doc_id={doc['doc_id']} | "
                f"score={score:.4f} | type={doc.get('news_type','')}"
            )
            print(f"    title: {doc.get('title','')[:120]}\n")

    elif args.command == "run-experiments":
        run_experiments()


if __name__ == "__main__":
    main()
