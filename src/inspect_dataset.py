from datasets import load_dataset


DATASET_NAME = "prrao87/tea-hypervectors"


def inspect_dataset():
    dataset = load_dataset(DATASET_NAME)
    teas = dataset["train"]

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Number of teas: {len(teas)}")
    print(f"Columns: {teas.column_names}")

    # ---------------------------------------------------------
    # Classes
    # ---------------------------------------------------------

    classes = sorted(set(teas["class"]))

    print("\n" + "=" * 60)
    print("TEA CLASSES")
    print("=" * 60)

    for value in classes:
        print(value)

    # ---------------------------------------------------------
    # Oxidation
    # ---------------------------------------------------------

    oxidation_values = sorted(set(teas["oxidation"]))

    print("\n" + "=" * 60)
    print("OXIDATION LEVELS")
    print("=" * 60)

    for value in oxidation_values:
        print(value)

    # ---------------------------------------------------------
    # Roast
    # ---------------------------------------------------------

    roast_values = sorted(set(teas["roast"]))

    print("\n" + "=" * 60)
    print("ROAST LEVELS")
    print("=" * 60)

    for value in roast_values:
        print(value)

    # ---------------------------------------------------------
    # Elevation
    # ---------------------------------------------------------

    elevations = [
        value
        for value in teas["elevation_meters"]
        if value is not None
    ]

    print("\n" + "=" * 60)
    print("ELEVATION")
    print("=" * 60)

    print(f"Minimum elevation: {min(elevations)} m")
    print(f"Maximum elevation: {max(elevations)} m")
    print(f"Number with elevation: {len(elevations)}")

    # ---------------------------------------------------------
    # Elevation confidence
    # ---------------------------------------------------------

    confidences = [
        value
        for value in teas["elevation_confidence"]
        if value is not None
    ]

    print("\n" + "=" * 60)
    print("ELEVATION CONFIDENCE")
    print("=" * 60)

    print(f"Minimum confidence: {min(confidences)}")
    print(f"Maximum confidence: {max(confidences)}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    for column in teas.column_names:
        values = teas[column]

        missing = sum(
            value is None
            for value in values
        )

        print(f"{column}: {missing}")

    # ---------------------------------------------------------
    # Aroma statistics
    # ---------------------------------------------------------

    aroma_lists = [
        value
        for value in teas["aroma"]
        if value is not None
    ]

    aroma_count = sum(
        len(value)
        for value in aroma_lists
    )

    unique_aromas = set()

    for aroma_list in aroma_lists:
        for aroma in aroma_list:
            unique_aromas.add(aroma.strip().lower())

    print("\n" + "=" * 60)
    print("AROMA")
    print("=" * 60)

    print(f"Total aroma phrases: {aroma_count}")
    print(f"Unique aroma phrases: {len(unique_aromas)}")

    # ---------------------------------------------------------
    # Taste statistics
    # ---------------------------------------------------------

    taste_lists = [
        value
        for value in teas["taste"]
        if value is not None
    ]

    taste_count = sum(
        len(value)
        for value in taste_lists
    )

    unique_tastes = set()

    for taste_list in taste_lists:
        for taste in taste_list:
            unique_tastes.add(taste.strip().lower())

    print("\n" + "=" * 60)
    print("TASTE")
    print("=" * 60)

    print(f"Total taste phrases: {taste_count}")
    print(f"Unique taste phrases: {len(unique_tastes)}")

    print("\n" + "=" * 60)
    print("ELEVATION DETAILS")
    print("=" * 60)

    for tea in teas:
        elevation = tea["elevation_meters"]
        confidence = tea["elevation_confidence"]

        print(
            f"{tea['title']}: "
            f"{elevation} m, "
            f"confidence={confidence}"
        )


if __name__ == "__main__":
    inspect_dataset()