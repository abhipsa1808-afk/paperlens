import numpy as np
import pandas as pd
import pytest

from paperlens.search import SearchEngine, build_index, paper_texts


class FakeModel:
    """Stands in for SentenceTransformer: maps known texts to fixed vectors."""

    def __init__(self, table):
        self.table = table

    def encode(self, texts, **kwargs):
        return np.array([self.table[t] for t in texts], dtype="float32")


def make_papers():
    return pd.DataFrame(
        {
            "arxiv_id": ["a", "b", "c"],
            "title": ["Title A", "Title B", "Title C"],
            "abstract": ["abs a", "abs b", "abs c"],
            "authors": [["Ann"], ["Bob"], ["Cy"]],
            "published": ["2024-01-01"] * 3,
            "primary_category": ["cs.LG"] * 3,
            "url": ["u/a", "u/b", "u/c"],
        }
    )


def test_paper_texts_combines_title_and_abstract():
    assert paper_texts(make_papers())[0] == "Title A. abs a"


def test_search_returns_nearest_paper_first():
    vectors = np.eye(3, dtype="float32")
    engine = SearchEngine(make_papers(), build_index(vectors), FakeModel({"q": vectors[1]}))
    results = engine.search("q", k=2)
    assert results[0]["arxiv_id"] == "b"
    assert len(results) == 2
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_empty_query_returns_nothing():
    vectors = np.eye(3, dtype="float32")
    engine = SearchEngine(make_papers(), build_index(vectors), FakeModel({}))
    assert engine.search("   ") == []


def test_mismatched_sizes_raise():
    vectors = np.eye(2, dtype="float32")
    with pytest.raises(ValueError):
        SearchEngine(make_papers(), build_index(vectors), FakeModel({}))
