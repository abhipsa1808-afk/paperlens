"""Embedding index and semantic search over paper abstracts."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PAPERS = Path("data/papers.parquet")
DEFAULT_INDEX = Path("data/papers.index")


def paper_texts(papers: pd.DataFrame) -> list[str]:
    """Combine title and abstract into the text that gets embedded."""
    return (papers["title"] + ". " + papers["abstract"]).tolist()


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts into L2-normalised float32 vectors (so inner product = cosine)."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an exact inner-product FAISS index from normalised vectors."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


class SearchEngine:
    """Semantic search over a table of papers and a matching FAISS index."""

    def __init__(self, papers: pd.DataFrame, index: faiss.Index, model: SentenceTransformer):
        if len(papers) != index.ntotal:
            raise ValueError("papers table and index have different sizes")
        self.papers = papers.reset_index(drop=True)
        self.index = index
        self.model = model

    @classmethod
    def load(
        cls,
        papers_path: Path = DEFAULT_PAPERS,
        index_path: Path = DEFAULT_INDEX,
        model_name: str = MODEL_NAME,
    ) -> SearchEngine:
        papers = pd.read_parquet(papers_path)
        index = faiss.read_index(str(index_path))
        return cls(papers, index, SentenceTransformer(model_name))

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k papers most similar in meaning to the query."""
        if not query.strip() or k < 1:
            return []
        vector = embed_texts(self.model, [query])
        scores, ids = self.index.search(vector, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0:
                continue
            row = self.papers.iloc[int(idx)]
            results.append(
                {
                    "arxiv_id": row["arxiv_id"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "authors": list(row["authors"]),
                    "published": row["published"],
                    "primary_category": row["primary_category"],
                    "url": row["url"],
                    "score": float(score),
                }
            )
        return results
