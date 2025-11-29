from pathlib import Path
from typing import List, Dict

import pandas as pd

from .config import PROJECT_ROOT, load_config


def _pick_column(cols_lower: dict, candidates: List[str]) -> str | None:
    """
    Return the original column name that matches one of the candidate keys
    (case-insensitive). If none match, return None.
    """
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _resolve_path(path: str) -> Path:
    """
    Resolve CSV path relative to project root if it is not absolute.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_articles(csv_path: str | None = None) -> List[Dict]:
    """
    Load news articles from the Kaggle Articles.csv file.

    Returns a list of dicts:
        {
          "doc_id": int,
          "title": str,
          "content": str,    # title + body
          "news_type": str   # e.g., sports/business if present
        }

    If `csv_path` is None, it will be taken from configs.data.articles_csv.
    """
    if csv_path is None:
        cfg = load_config()
        csv_path = cfg["data"]["articles_csv"]

    path = _resolve_path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Articles CSV not found at: {path}")

    # Read CSV (adjust encoding if needed)
    df = pd.read_csv(path, encoding="latin1")

    # Try to guess column names in a robust way
    cols_lower = {c.lower(): c for c in df.columns}

    # Title column (prefer "Heading" from Articles.csv)
    title_col = _pick_column(
        cols_lower, ["heading", "title", "headline", "article_title"]
    )
    if title_col is None:
        title_col = df.columns[0]

    # Body/content column (prefer "Article" from Articles.csv)
    body_col = _pick_column(
        cols_lower, ["article", "content", "text", "body"]
    )
    if body_col is None or body_col == title_col:
        # fallback to any other column that is not the title
        for col in df.columns:
            if col != title_col:
                body_col = col
                break
        if body_col is None:
            body_col = title_col

    # Optional metadata columns
    type_col = _pick_column(cols_lower, ["newstype", "category", "section", "label"])
    date_col = _pick_column(cols_lower, ["date", "published", "publish_date"])

    docs: List[Dict] = []

    for idx, row in df.iterrows():
        title = str(row[title_col]) if not pd.isna(row[title_col]) else ""
        body = str(row[body_col]) if not pd.isna(row[body_col]) else ""
        news_type = ""
        if type_col is not None and type_col in row:
            val = row[type_col]
            news_type = "" if pd.isna(val) else str(val)
        published_date = ""
        if date_col is not None and date_col in row:
            val = row[date_col]
            published_date = "" if pd.isna(val) else str(val)

        full_text = (title + ". " + body).strip(". ").strip()

        docs.append(
            {
                "doc_id": int(idx),
                "title": title,
                "content": full_text,
                "news_type": news_type,
                "published_date": published_date,
            }
        )

    return docs
