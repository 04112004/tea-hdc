from pathlib import Path

import numpy as np
import lancedb
import pyarrow as pa

from data_loader import load_tea_dataset
from encoder import encode_tea


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DB_PATH = "data/tea_hypervectors.lance"


# ---------------------------------------------------------
# Build hypervectors
# ---------------------------------------------------------

def build_hypervectors():

    print("=" * 70)
    print("BUILDING TEA HYPERVECTORS")
    print("=" * 70)

    # Load dataset
    dataset = load_tea_dataset()
    teas = dataset["train"]

    print(f"\nNumber of teas: {len(teas)}")

    records = []

    print("\nEncoding teas...")

    for index, tea in enumerate(teas):

        vector = encode_tea(tea)

        record = {
            "id": str(tea["id"]),
            "title": str(tea["title"]),
            "class": str(tea["class"]),
            "country": (
                str(tea["country"])
                if tea["country"] is not None
                else ""
            ),
            "region": str(tea["region"]),
            "oxidation": str(tea["oxidation"]),
            "roast": str(tea["roast"]),
            "aroma": list(tea.get("aroma") or []),
            "taste": list(tea.get("taste") or []),
            "elevation_meters": float(
                tea["elevation_meters"]
            ),
            "elevation_confidence": float(
                tea["elevation_confidence"]
            ),
            "hypervector": vector.astype(
                np.float32
            ).tolist(),
        }

        records.append(record)

        print(
            f"\rEncoded {index + 1}/{len(teas)}",
            end=""
        )

    print("\n")

    # -----------------------------------------------------
    # Convert to Arrow table
    # -----------------------------------------------------

    print("Creating Arrow table...")

    table = pa.Table.from_pylist(records)

    # -----------------------------------------------------
    # Create LanceDB database
    # -----------------------------------------------------

    db_path = Path(DB_PATH)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Creating Lance database at:")
    print(db_path)

    db = lancedb.connect(str(db_path))

    # Remove existing table if present
    if "teas" in db.table_names():

        print("Existing table found. Replacing it...")

        db.drop_table("teas")

    # Create table
    table = db.create_table(
        "teas",
        data=table
    )

    print("\nDatabase created successfully!")

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)

    print(f"\nNumber of records: {len(records)}")
    print("Vector dimensions: 10,000")
    print(f"Database: {DB_PATH}")

    print("\nStored columns:")

    for column in table.schema.names:
        print(f"  - {column}")

    print("\n✓ All tea hypervectors built")
    print("✓ Metadata stored")
    print("✓ Lance database created")


if __name__ == "__main__":
    build_hypervectors()