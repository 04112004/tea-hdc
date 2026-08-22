import numpy as np

from hypervectors import (
    DIMENSION,
    bind,
    bundle,
    random_bipolar_vector,
    level_hypervector,
    cosine_similarity,
)

from embeddings import phrases_to_hypervector


# ============================================================
# Field weights
# ============================================================

AROMA_WEIGHT = 0.25
TASTE_WEIGHT = 0.25
CLASS_WEIGHT = 0.16
OXIDATION_WEIGHT = 0.16
ROAST_WEIGHT = 0.03
ELEVATION_WEIGHT = 0.15


# ============================================================
# Fixed role hypervectors
# ============================================================

RNG = np.random.default_rng(42)


def deterministic_bipolar_vector():
    return RNG.choice(
        [-1.0, 1.0],
        size=DIMENSION,
    ).astype(np.float32)


R_AROMA = deterministic_bipolar_vector()
R_TASTE = deterministic_bipolar_vector()
R_CLASS = deterministic_bipolar_vector()
R_OXIDATION = deterministic_bipolar_vector()
R_ROAST = deterministic_bipolar_vector()
R_ELEVATION = deterministic_bipolar_vector()


# ============================================================
# Categorical hypervectors
# ============================================================

CLASS_VALUES = [
    "black",
    "green",
    "oolong",
    "white",
    "yellow",
]

CLASS_VECTORS = {
    value: deterministic_bipolar_vector()
    for value in CLASS_VALUES
}


# ============================================================
# Ordered hypervectors
# ============================================================

OXIDATION_VALUES = [
    "low",
    "medium",
    "high",
]

OXIDATION_VECTORS = {
    value: level_hypervector(
        index=index,
        total_levels=len(OXIDATION_VALUES),
    )
    for index, value in enumerate(OXIDATION_VALUES)
}


ROAST_VALUES = [
    "none",
    "light",
    "heavy",
]

ROAST_VECTORS = {
    value: level_hypervector(
        index=index,
        total_levels=len(ROAST_VALUES),
    )
    for index, value in enumerate(ROAST_VALUES)
}


# ============================================================
# Elevation
# ============================================================

ELEVATION_LEVELS = 24


ELEVATION_MIN = 0.0
ELEVATION_MAX = 2300.0


def elevation_to_index(elevation):
    """
    Convert elevation in meters into an ordered level index.
    """

    elevation = float(elevation)

    elevation = max(
        ELEVATION_MIN,
        min(ELEVATION_MAX, elevation),
    )

    position = (
        elevation - ELEVATION_MIN
    ) / (
        ELEVATION_MAX - ELEVATION_MIN
    )

    index = round(
        position * (ELEVATION_LEVELS - 1)
    )

    return int(index)


ELEVATION_BASE = deterministic_bipolar_vector()


def elevation_hypervector(elevation):
    """
    Convert elevation in meters into an ordered
    hypervector.
    """

    index = elevation_to_index(elevation)

    return level_hypervector(
        index=index,
        total_levels=ELEVATION_LEVELS,
        base_vector=ELEVATION_BASE,
    )


# ============================================================
# Encode one field
# ============================================================

def encode_field(role_vector, value_vector):
    """
    Bind a field's role with its value.
    """

    return bind(
        role_vector,
        value_vector,
    )


# ============================================================
# Encode a complete tea
# ============================================================

def encode_tea(tea):
    """
    Encode one tea record into a single
    10,000-dimensional hypervector.
    """

    vectors = []
    weights = []

    # --------------------------------------------------------
    # Aroma
    # --------------------------------------------------------

    aroma_vector = phrases_to_hypervector(
        tea.get("aroma", [])
    )

    if aroma_vector is not None:

        h_aroma = encode_field(
            R_AROMA,
            aroma_vector,
        )

        vectors.append(h_aroma)
        weights.append(AROMA_WEIGHT)

    # --------------------------------------------------------
    # Taste
    # --------------------------------------------------------

    taste_vector = phrases_to_hypervector(
        tea.get("taste", [])
    )

    if taste_vector is not None:

        h_taste = encode_field(
            R_TASTE,
            taste_vector,
        )

        vectors.append(h_taste)
        weights.append(TASTE_WEIGHT)

    # --------------------------------------------------------
    # Class
    # --------------------------------------------------------

    tea_class = tea.get("class")

    if tea_class:

        tea_class = tea_class.lower()

        if tea_class in CLASS_VECTORS:

            h_class = encode_field(
                R_CLASS,
                CLASS_VECTORS[tea_class],
            )

            vectors.append(h_class)
            weights.append(CLASS_WEIGHT)

    # --------------------------------------------------------
    # Oxidation
    # --------------------------------------------------------

    oxidation = tea.get("oxidation")

    if oxidation:

        oxidation = oxidation.lower()

        if oxidation in OXIDATION_VECTORS:

            h_oxidation = encode_field(
                R_OXIDATION,
                OXIDATION_VECTORS[oxidation],
            )

            vectors.append(h_oxidation)
            weights.append(OXIDATION_WEIGHT)

    # --------------------------------------------------------
    # Roast
    # --------------------------------------------------------

    roast = tea.get("roast")

    if roast:

        roast = roast.lower()

        if roast in ROAST_VECTORS:

            h_roast = encode_field(
                R_ROAST,
                ROAST_VECTORS[roast],
            )

            vectors.append(h_roast)
            weights.append(ROAST_WEIGHT)

    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    elevation = tea.get("elevation_meters")
    confidence = tea.get(
        "elevation_confidence"
    )

    if (
        elevation is not None
        and confidence is not None
        and float(confidence) > 0
    ):

        h_elevation_value = elevation_hypervector(
            elevation
        )

        h_elevation = encode_field(
            R_ELEVATION,
            h_elevation_value,
        )

        vectors.append(h_elevation)

        weights.append(
            ELEVATION_WEIGHT * float(confidence)
        )

    # --------------------------------------------------------
    # Final bundle
    # --------------------------------------------------------

    if not vectors:
        raise ValueError(
            "Tea contains no encodable fields."
        )

    return bundle(
        vectors,
        weights,
    )


# ============================================================
# Field contribution
# ============================================================

def field_contribution(
    query_tea,
    candidate_tea,
):
    """
    Calculate the individual contribution of each
    field to the similarity between two teas.
    """

    contributions = {}

    # --------------------------------------------------------
    # Aroma
    # --------------------------------------------------------

    query_aroma = phrases_to_hypervector(
        query_tea.get("aroma", [])
    )

    candidate_aroma = phrases_to_hypervector(
        candidate_tea.get("aroma", [])
    )

    if (
        query_aroma is not None
        and candidate_aroma is not None
    ):

        q = bind(R_AROMA, query_aroma)
        c = bind(R_AROMA, candidate_aroma)

        contributions["aroma"] = (
            AROMA_WEIGHT
            * cosine_similarity(q, c)
        )

    else:

        contributions["aroma"] = 0.0

    # --------------------------------------------------------
    # Taste
    # --------------------------------------------------------

    query_taste = phrases_to_hypervector(
        query_tea.get("taste", [])
    )

    candidate_taste = phrases_to_hypervector(
        candidate_tea.get("taste", [])
    )

    if (
        query_taste is not None
        and candidate_taste is not None
    ):

        q = bind(R_TASTE, query_taste)
        c = bind(R_TASTE, candidate_taste)

        contributions["taste"] = (
            TASTE_WEIGHT
            * cosine_similarity(q, c)
        )

    else:

        contributions["taste"] = 0.0

    return contributions