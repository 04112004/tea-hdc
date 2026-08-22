# ---------------------------------------------------------
# Query attribute parser
# ---------------------------------------------------------

TEA_CLASSES = {
    "black": "black",
    "green": "green",
    "oolong": "oolong",
    "white": "white",
    "yellow": "yellow",
}

COUNTRIES = {
    "taiwan": "Taiwan",
    "taiwanese": "Taiwan",
    "china": "China",
    "chinese": "China",
    "japan": "Japan",
    "japanese": "Japan",
    "india": "India",
    "indian": "India",
    "nepal": "Nepal",
    "nepalese": "Nepal",
    "sri lanka": "Sri Lanka",
    "ceylon": "Sri Lanka",
}

ROAST_LEVELS = {
    "roasted": "roasted",
    "roast": "roasted",
    "light roast": "light",
    "lightly roasted": "light",
    "heavy roast": "heavy",
    "heavily roasted": "heavy",
    "charcoal roasted": "heavy",
}

OXIDATION_LEVELS = {
    "low oxidation": "low",
    "light oxidation": "low",
    "medium oxidation": "medium",
    "moderate oxidation": "medium",
    "high oxidation": "high",
    "fully oxidized": "high",
}

SENSORY_TERMS = {
    "floral": "floral",
    "flower": "floral",
    "fruity": "fruity",
    "fruit": "fruity",
    "citrus": "citrus",
    "sweet": "sweet",
    "creamy": "creamy",
    "buttery": "buttery",
    "nutty": "nutty",
    "spicy": "spicy",
    "herbal": "herbal",
    "vegetal": "vegetal",
    "earthy": "earthy",
    "smoky": "smoky",
    "roasted": "roasted",
    "woody": "woody",
    "vanilla": "vanilla",
    "lavender": "lavender",
    "peach": "peach",
    "cherry": "cherry",
    "lemongrass": "lemongrass",
}


def parse_query(query):
    """
    Extract structured attributes from a natural-language
    tea query.
    """

    query = query.lower().strip()

    metadata = {
        "class": None,
        "country": None,
        "roast": None,
        "oxidation": None,
        "sensory": [],
    }

    # -----------------------------------------------------
    # Detect tea class
    # -----------------------------------------------------

    for word, tea_class in TEA_CLASSES.items():

        if word in query:
            metadata["class"] = tea_class
            break

    # -----------------------------------------------------
    # Detect country
    # -----------------------------------------------------

    for word, country in COUNTRIES.items():

        if word in query:
            metadata["country"] = country
            break

    # -----------------------------------------------------
    # Detect roast
    # -----------------------------------------------------

    for phrase, roast in ROAST_LEVELS.items():

        if phrase in query:
            metadata["roast"] = roast
            break

    # -----------------------------------------------------
    # Detect oxidation
    # -----------------------------------------------------

    for phrase, oxidation in OXIDATION_LEVELS.items():

        if phrase in query:
            metadata["oxidation"] = oxidation
            break

    # -----------------------------------------------------
    # Detect sensory characteristics
    # -----------------------------------------------------

    detected_sensory = []

    for word, sensory in SENSORY_TERMS.items():

        if word in query:
            if sensory not in detected_sensory:
                detected_sensory.append(sensory)

    metadata["sensory"] = detected_sensory

    return metadata


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        "black tea",
        "green tea",
        "Taiwanese oolong",
        "floral tea",
        "fruity tea",
        "roasted oolong",
        "high oxidation black tea",
        "floral fruity Taiwanese oolong",
    ]

    print("=" * 70)
    print("QUERY PARSER TEST")
    print("=" * 70)

    for query in test_queries:

        result = parse_query(query)

        print()
        print(f"Query: {query}")
        print(f"Parsed: {result}")

    print()
    print("=" * 70)
    print("QUERY PARSER TEST COMPLETE")
    print("=" * 70)