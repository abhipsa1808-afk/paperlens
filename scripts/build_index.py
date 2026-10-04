"""Embed every paper in data/papers.parquet and save a FAISS index."""

import argparse
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from paperlens.search import (
    DEFAULT_INDEX,
    DEFAULT_PAPERS,
    MODEL_NAME,
    build_index,
    embed_texts,
    paper_texts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    papers = pd.read_parquet(args.papers)
    print(f"Embedding {len(papers)} papers with {args.model} ...")
    model = SentenceTransformer(args.model)
    vectors = embed_texts(model, paper_texts(papers), show_progress=True)
    index = build_index(vectors)
    faiss.write_index(index, str(args.index))
    print(f"Saved index with {index.ntotal} vectors of dimension {vectors.shape[1]}")


if __name__ == "__main__":
    main()
