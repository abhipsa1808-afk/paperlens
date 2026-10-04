"""Download arXiv paper metadata into data/papers.parquet."""

import argparse
from pathlib import Path

from paperlens.data import DEFAULT_CATEGORIES, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=2000)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--out", type=Path, default=Path("data/papers.parquet"))
    args = parser.parse_args()

    df = build_dataset(args.categories, args.per_category, out_path=args.out)
    print(f"Saved {len(df)} papers to {args.out}")


if __name__ == "__main__":
    main()
