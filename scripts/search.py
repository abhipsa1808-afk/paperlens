"""Search papers from the command line: python scripts/search.py "your query" -k 5"""

import argparse
import textwrap

from paperlens.search import SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    engine = SearchEngine.load()
    for rank, hit in enumerate(engine.search(args.query, k=args.k), start=1):
        print(f"\n{rank}. {hit['title']}  (score {hit['score']:.3f})")
        print(f"   {hit['url']}  [{hit['primary_category']}]")
        print(textwrap.indent(textwrap.shorten(hit["abstract"], width=240), "   "))


if __name__ == "__main__":
    main()
