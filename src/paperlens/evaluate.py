"""Offline evaluation of retrieval quality.

Task: use a paper's title as the query and retrieve its own abstract from the pool
of all abstracts. The index is built from abstracts only, so the title is never part
of the indexed text (no leakage).
"""

from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

from paperlens.search import build_index, embed_texts


def score_retrieval(
    doc_vecs: np.ndarray,
    query_vecs: np.ndarray,
    categories: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute recall@k, MRR and same-category precision for aligned query/doc vectors.

    Query i is expected to retrieve document i.
    """
    n = len(doc_vecs)
    top_k = max(ks)
    depth = min(max(top_k, 6), n)
    # Single-threaded search avoids an OpenMP crash on Apple Silicon when PyTorch
    # and FAISS are loaded together.
    faiss.omp_set_num_threads(1)
    _, ids = build_index(doc_vecs).search(query_vecs, depth)

    ranks = np.full(n, np.inf)
    for i in range(n):
        found = np.flatnonzero(ids[i] == i)
        if found.size and found[0] + 1 <= top_k:
            ranks[i] = found[0] + 1

    metrics = {f"recall@{k}": float(np.mean(ranks <= k)) for k in ks}
    metrics[f"mrr@{top_k}"] = float(np.mean(1.0 / ranks))

    cats = np.asarray(categories)
    same_category = []
    for i in range(n):
        neighbours = [j for j in ids[i] if j != i and j >= 0][:5]
        if neighbours:
            same_category.append(float(np.mean(cats[neighbours] == cats[i])))
    metrics["category_precision@5"] = float(np.mean(same_category)) if same_category else 0.0
    return metrics


def evaluate_retrieval(
    model,
    papers: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10),
    show_progress: bool = False,
) -> dict[str, float]:
    """Embed abstracts (documents) and titles (queries), then score retrieval."""
    doc_vecs = embed_texts(model, papers["abstract"].tolist(), show_progress=show_progress)
    query_vecs = embed_texts(model, papers["title"].tolist(), show_progress=show_progress)
    return score_retrieval(doc_vecs, query_vecs, papers["primary_category"].tolist(), ks)
