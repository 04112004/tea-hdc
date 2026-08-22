import argparse

from query_search import search_by_query


def recommend(query, limit=5):

    print("=" * 80)
    print("TEA RECOMMENDATION SYSTEM")
    print("=" * 80)

    print(f"\nQuery: {query}")
    print("\nGenerating recommendations...\n")

    results = search_by_query(
        query,
        limit
    )

    print("-" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\n{i}. {result['title']}")

        print(
            f"   Similarity : "
            f"{result['similarity']:.4f}"
        )

        print(
            f"   Class      : "
            f"{result['class']}"
        )

        print(
            f"   Country    : "
            f"{result['country']}"
        )

        print(
            f"   Region     : "
            f"{result['region']}"
        )

        print(
            f"   Oxidation  : "
            f"{result['oxidation']}"
        )

        print(
            f"   Roast      : "
            f"{result['roast']}"
        )

        print(
            f"   Elevation  : "
            f"{result['elevation_meters']} m"
        )

    print("\n" + "=" * 80)
    print("RECOMMENDATION COMPLETE")
    print("=" * 80)


def main():

    parser = argparse.ArgumentParser(
        description="Recommend teas using HDC similarity"
    )

    parser.add_argument(
        "--query",
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5
    )

    args = parser.parse_args()

    recommend(
        args.query,
        args.limit
    )


if __name__ == "__main__":
    main()