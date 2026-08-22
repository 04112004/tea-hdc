import numpy as np
import lancedb

from query_search import search_by_query


DB_PATH = "data/tea_hypervectors.lance"


# ---------------------------------------------------------
# Test queries
# ---------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "green tea",
        "expected_classes": ["green"],
    },
    {
        "query": "black tea",
        "expected_classes": ["black"],
    },
    {
        "query": "oolong tea",
        "expected_classes": ["oolong"],
    },
    {
        "query": "white tea",
        "expected_classes": ["white"],
    },
    {
        "query": "yellow tea",
        "expected_classes": ["yellow"],
    },
    {
        "query": "floral tea",
        "expected_classes": [
            "oolong",
            "white",
            "green",
        ],
    },
    {
        "query": "fruity tea",
        "expected_classes": [
            "oolong",
            "black",
            "white",
            "green",
        ],
    },
    {
        "query": "roasted tea",
        "expected_roasts": [
            "light",
            "heavy",
        ],
    },
    {
        "query": "high elevation tea",
        "expected_min_elevation": 1000,
    },
    {
        "query": "Taiwanese oolong",
        "expected_classes": ["oolong"],
        "expected_country": "Taiwan",
    },
]


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def load_database():

    db = lancedb.connect(DB_PATH)

    table = db.open_table("teas")

    return table.to_pandas()


# ---------------------------------------------------------
# Evaluate one query
# ---------------------------------------------------------

def evaluate_query(test, df):

    query = test["query"]

    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    # Get recommendations
    results = search_by_query(
        query,
        limit=5
    )

    if not results:
        print("No results returned.")
        return {
            "query": query,
            "top1": 0,
            "top3": 0,
            "top5": 0,
        }

    # -----------------------------------------------------
    # Check relevance
    # -----------------------------------------------------

    relevant = []

    for result in results:

        is_relevant = True

        # Class check
        if "expected_classes" in test:

            if result["class"] not in test["expected_classes"]:
                is_relevant = False

        # Country check
        if "expected_country" in test:

            if result.get("country") != test["expected_country"]:
                is_relevant = False

        # Roast check
        if "expected_roasts" in test:

            if result.get("roast") not in test["expected_roasts"]:
                is_relevant = False

        # Elevation check
        if "expected_min_elevation" in test:

            elevation = result.get(
                "elevation_meters"
            )

            if elevation is None:
                is_relevant = False

            elif elevation < test["expected_min_elevation"]:
                is_relevant = False

        relevant.append(is_relevant)

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    for i, (result, is_relevant) in enumerate(
        zip(results, relevant),
        start=1
    ):

        status = "✓" if is_relevant else "✗"

        print(
            f"{i}. {status} "
            f"{result['title']}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"   Class: "
            f"{result['class']}"
        )

        print(
            f"   Country: "
            f"{result.get('country')}"
        )

        print(
            f"   Roast: "
            f"{result.get('roast')}"
        )

        print(
            f"   Elevation: "
            f"{result.get('elevation_meters')} m"
        )

    # -----------------------------------------------------
    # Top-K metrics
    # -----------------------------------------------------

    top1 = int(
        any(relevant[:1])
    )

    top3 = int(
        any(relevant[:3])
    )

    top5 = int(
        any(relevant[:5])
    )

    print()
    print(
        f"Top-1 relevant: "
        f"{'YES' if top1 else 'NO'}"
    )

    print(
        f"Top-3 relevant: "
        f"{'YES' if top3 else 'NO'}"
    )

    print(
        f"Top-5 relevant: "
        f"{'YES' if top5 else 'NO'}"
    )

    return {
        "query": query,
        "top1": top1,
        "top3": top3,
        "top5": top5,
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("NATURAL LANGUAGE QUERY EVALUATION")
    print("=" * 80)

    df = load_database()

    print(
        f"Database contains {len(df)} teas"
    )

    results = []

    for test in TEST_QUERIES:

        result = evaluate_query(
            test,
            df
        )

        results.append(result)

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    top1_accuracy = np.mean([
        r["top1"]
        for r in results
    ])

    top3_accuracy = np.mean([
        r["top3"]
        for r in results
    ])

    top5_accuracy = np.mean([
        r["top5"]
        for r in results
    ])

    print()
    print("=" * 80)
    print("OVERALL QUERY EVALUATION")
    print("=" * 80)

    print(
        f"Queries tested: "
        f"{len(results)}"
    )

    print(
        f"Top-1 accuracy: "
        f"{top1_accuracy:.2%}"
    )

    print(
        f"Top-3 accuracy: "
        f"{top3_accuracy:.2%}"
    )

    print(
        f"Top-5 accuracy: "
        f"{top5_accuracy:.2%}"
    )

    print()
    print("=" * 80)
    print("QUERY EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()