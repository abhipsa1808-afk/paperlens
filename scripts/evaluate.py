"""Evaluate retrieval quality and save results to docs/eval_results.json."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from paperlens.evaluate import evaluate_retrieval
from paperlens.search import DEFAULT_PAPERS, MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--out", type=Path, default=Path("docs/eval_results.json"))
    args = parser.parse_args()

    papers = pd.read_parquet(args.papers)
    model = SentenceTransformer(args.model)
    metrics = evaluate_retrieval(model, papers, show_progress=True)

    result = {"model": args.model, "n_papers": len(papers), "metrics": metrics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\nEvaluated {len(papers)} papers with {args.model}")
    for name, value in metrics.items():
        print(f"  {name:<24} {value:.3f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
