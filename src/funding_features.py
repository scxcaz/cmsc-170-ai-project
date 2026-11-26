import re


# removes the year and standardizes a project's funding label
def normalize_funding(source):
    source = (source or "").strip().lower()

    # removes four-digit years already represented by infraYear
    label = re.sub(r"\b(?:19|20)\d{2}\b", "", source)

    # removes extra spaces left after removing the year
    label = re.sub(r"\s+", " ", label).strip()

    return label


# converts the funding label into binary numerical features
def encode_funding(label, common_funding):
    if label not in common_funding:
        label = "Other"

    funding_names = common_funding + ["Other"]
    features = []

    for name in funding_names:
        if label == name:
            features.append(1)
        else:
            features.append(0)

    return features
