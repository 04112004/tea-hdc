import numpy as np

from hypervectors import (
    DIMENSION,
    random_bipolar_vector,
    bind,
    bundle,
    cosine_similarity,
    level_hypervector,
)


print("=" * 60)
print("HYPERVECTOR TEST")
print("=" * 60)


# --------------------------------------------------
# 1. Random bipolar vector
# --------------------------------------------------

hv = random_bipolar_vector()

print("\nRandom hypervector")
print("Shape:", hv.shape)
print("Data type:", hv.dtype)
print("First 10:", hv[:10])

assert hv.shape == (DIMENSION,)

assert np.all(
    np.isin(hv, [-1.0, 1.0])
)

print("✓ Random bipolar vector works")


# --------------------------------------------------
# 2. Binding
# --------------------------------------------------

a = random_bipolar_vector()
b = random_bipolar_vector()

bound = bind(a, b)

print("\nBinding")
print("Shape:", bound.shape)
print("First 10:", bound[:10])

assert bound.shape == (DIMENSION,)

assert np.all(
    np.isin(bound, [-1.0, 1.0])
)

print("✓ Binding works")


# --------------------------------------------------
# 3. Bundling
# --------------------------------------------------

bundled = bundle(
    [a, b],
    weights=[0.5, 0.5],
)

print("\nBundling")
print("Shape:", bundled.shape)
print("First 10:", bundled[:10])

assert bundled.shape == (DIMENSION,)

print("✓ Bundling works")


# --------------------------------------------------
# 4. Binding reversibility
# --------------------------------------------------

recovered = bind(bound, b)

similarity = cosine_similarity(recovered, a)

print("\nBinding recovery similarity:")
print(similarity)

assert np.isclose(similarity, 1.0)

print("✓ Binding is exactly reversible for bipolar vectors")


# --------------------------------------------------
# 5. Level hypervectors
# --------------------------------------------------

low = level_hypervector(
    index=0,
    total_levels=3,
)

medium = level_hypervector(
    index=1,
    total_levels=3,
)

high = level_hypervector(
    index=2,
    total_levels=3,
)

low_medium = cosine_similarity(low, medium)
medium_high = cosine_similarity(medium, high)
low_high = cosine_similarity(low, high)

print("\nLevel hypervectors")
print("low ↔ medium :", low_medium)
print("medium ↔ high :", medium_high)
print("low ↔ high   :", low_high)

assert low_medium > low_high
assert medium_high > low_high

print("✓ Level hypervectors preserve ordering")


print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)