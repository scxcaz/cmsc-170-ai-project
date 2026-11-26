import re


# keywords used to identify different types of work and infrastructure
DESCRIPTION_PATTERNS = {
    "construction": r"\b(?:construction|const)\b",
    "improvement": r"\b(?:improvement|imp)\b",
    "repair": r"\brepairs?\b",
    "rehabilitation": r"\b(?:rehabilitation|rehab)\b",
    "completion": r"\bcompletion\b",
    "classroom": r"\bclassrooms?\b|\bcl\s*bldg\b",
    "covered_court": r"\bcovered[\s-]+court\b",
    "drainage": r"\bdrainage\b",
    "water_system": r"\bwater\s+(?:system|supply)\b",
    "revetment": r"\brevetment\b",
}


# converts a project description into binary numerical features
def encode_description(description):
    description = (description or "").lower()
    features = []

    for pattern in DESCRIPTION_PATTERNS.values():
        if re.search(pattern, description):
            features.append(1)
        else:
            features.append(0)

    return features
