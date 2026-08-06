import json


def extract_garment_attributes(description):

    text = description.lower()

    result = {
        "type": "unknown",
        "sleeve_length": "unknown",
        "neckline": "unknown",
        "primary_color": "unknown",
        "pattern": "plain"
    }

    # --------------------------
    # Garment Type
    # --------------------------

    garments = [
        "t-shirt",
        "shirt",
        "hoodie",
        "jacket",
        "dress",
        "blouse",
        "crop top",
        "top",
        "sweater",
        "tank top"
    ]

    for garment in garments:
        if garment in text:
            result["type"] = garment
            break

    # --------------------------
    # Sleeve Length
    # --------------------------

    if any(x in text for x in [
        "short sleeve",
        "short sleeves",
        "short-sleeved"
    ]):
        result["sleeve_length"] = "short"

    elif any(x in text for x in [
        "long sleeve",
        "long sleeves",
        "long-sleeved"
    ]):
        result["sleeve_length"] = "long"

    elif any(x in text for x in [
        "sleeveless",
        "tank top",
        "thin straps",
        "spaghetti straps"
    ]):
        result["sleeve_length"] = "sleeveless"

    # --------------------------
    # Neckline
    # --------------------------

    necklines = {
        "round": [
            "round neckline",
            "round neck"
        ],
        "crew": [
            "crew neckline",
            "crew neck"
        ],
        "square": [
            "square neckline"
        ],
        "high": [
            "high neckline"
        ],
        "v-neck": [
            "v-neck"
        ],
        "scoop": [
            "scoop neckline"
        ]
    }

    for neck, keywords in necklines.items():

        for keyword in keywords:

            if keyword in text:
                result["neckline"] = neck
                break

    # --------------------------
    # Primary Color
    # --------------------------

    first_sentence = text.split(".")[0]

    colors = [
        "light purple",
        "dark blue",
        "black",
        "white",
        "yellow",
        "blue",
        "green",
        "red",
        "purple",
        "pink",
        "orange",
        "brown",
        "grey",
        "gray"
    ]

    for color in colors:

        if color in first_sentence:

            result["primary_color"] = color
            break

    # --------------------------
    # Pattern
    # --------------------------

    if "floral" in text:

        result["pattern"] = "floral"

    elif "graphic" in text:

        result["pattern"] = "graphic"

    elif "logo" in text:

        result["pattern"] = "logo"

    elif "striped" in text:

        result["pattern"] = "striped"

    elif "checked" in text:

        result["pattern"] = "checked"

    elif "printed" in text:

        result["pattern"] = "printed"

    return result


def extract_person_attributes(description):

    text = description.lower()

    result = {
        "pose": "unknown",
        "upper_body_visible": False,
        "lower_body_visible": False
    }

    # --------------------------
    # Pose
    # --------------------------

    if "side" in text or "turned to the side" in text:

        result["pose"] = "side"

    elif "back" in text:

        result["pose"] = "back"

    elif "seated" in text or "sitting" in text:

        result["pose"] = "seated"

    else:

        result["pose"] = "front"

    # --------------------------
    # Visibility
    # --------------------------

    upper_keywords = [
        "shirt",
        "t-shirt",
        "top",
        "tank top",
        "blouse",
        "hoodie",
        "shoulders"
    ]

    lower_keywords = [
        "pants",
        "jeans",
        "shorts",
        "skirt",
        "legs"
    ]

    result["upper_body_visible"] = any(
        word in text for word in upper_keywords
    )

    result["lower_body_visible"] = any(
        word in text for word in lower_keywords
    )

    return result


def save_json(data, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )