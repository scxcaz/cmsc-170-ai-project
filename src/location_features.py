import re


# extracts and simplifies a project's DPWH location label
def normalize_location(location):
    location = location or ""

    match = re.search(r"'province':\s*([^,}]+)", location)

    if not match:
        return ""

    label = match.group(1).strip().lower()

    # removes quotation marks if present
    label = label.strip("'\"")

    # combines district offices under the same location name
    label = re.sub(
        r"\s+(?:\d+(?:st|nd|rd|th)\s+)?deo$",
        "",
        label
    )

    return label


# converts a location into numerical features
def encode_location(label, common_locations):
    if label not in common_locations:
        label = "Other"

    location_names = common_locations + ["Other"]
    features = []

    for name in location_names:
        if label == name:
            features.append(1)
        else:
            features.append(0)

    return features
