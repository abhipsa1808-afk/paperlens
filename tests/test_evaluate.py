import numpy as np
import pytest

from paperlens.evaluate import score_retrieval


def test_perfect_retrieval():
    vecs = np.eye(4, dtype="float32")
    metrics = score_retrieval(vecs, vecs, ["a", "a", "b", "b"], ks=(1, 3))
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["mrr@3"] == pytest.approx(1.0)


def test_wrong_top_result_lowers_recall_at_1():
    docs = np.eye(3, dtype="float32")
    queries = np.roll(docs, 1, axis=0)  # query i points at document i-1
    metrics = score_retrieval(docs, queries, ["a", "b", "c"], ks=(1, 3))
    assert metrics["recall@1"] == pytest.approx(0.0)
    assert metrics["recall@3"] == pytest.approx(1.0)
