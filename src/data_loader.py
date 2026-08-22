from datasets import load_dataset


DATASET_NAME = "prrao87/tea-hypervectors"


def load_tea_dataset():
    """Load the public tea dataset from Hugging Face."""
    dataset = load_dataset(DATASET_NAME)

    print("Dataset loaded successfully!")
    print()

    for split_name, split in dataset.items():
        print(f"Split: {split_name}")
        print(f"Rows: {len(split)}")
        print(f"Columns: {split.column_names}")
        print()

        # Show the first tea
        first_tea = split[0]

        print("First tea record:")
        for key, value in first_tea.items():
            print(f"{key}: {value}")
            print()

    return dataset


if __name__ == "__main__":
    load_tea_dataset()