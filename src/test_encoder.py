from data_loader import load_tea_dataset
from encoder import encode_tea


print("=" * 60)
print("TEA ENCODER TEST")
print("=" * 60)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = load_tea_dataset()

teas = dataset["train"]


print("\nDataset loaded")
print("Number of teas:", len(teas))


# ---------------------------------------------------------
# Take first tea
# ---------------------------------------------------------

tea = teas[0]


print("\nTea:")
print("Title:", tea["title"])
print("Class:", tea["class"])
print("Oxidation:", tea["oxidation"])
print("Roast:", tea["roast"])
print("Elevation:", tea["elevation_meters"])
print(
    "Elevation confidence:",
    tea["elevation_confidence"],
)


# ---------------------------------------------------------
# Encode
# ---------------------------------------------------------

hypervector = encode_tea(tea)


print("\nEncoded hypervector")
print("Shape:", hypervector.shape)
print("First 20 values:")
print(hypervector[:20])


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print("\nHypervector statistics")

print(
    "Minimum:",
    hypervector.min()
)

print(
    "Maximum:",
    hypervector.max()
)

print(
    "Mean:",
    hypervector.mean()
)

print(
    "Non-zero coordinates:",
    (hypervector != 0).sum()
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert hypervector.shape == (
    10_000,
)

assert hypervector.dtype == "float32"

print("\n✓ Tea successfully encoded")
print("✓ Dimension = 10,000")
print("✓ Encoder test passed")


print("\n" + "=" * 60)
print("ALL ENCODER TESTS PASSED")
print("=" * 60)