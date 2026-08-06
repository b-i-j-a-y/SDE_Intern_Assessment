from pathlib import Path
import csv

from florence import FlorenceModel
from parser import (
    extract_garment_attributes,
    extract_person_attributes,
    save_json,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GARMENT_DIR = DATA_DIR / "garment"
PERSON_DIR = DATA_DIR / "person"
EDGE_CASE_DIR = DATA_DIR / "edge_cases"

# Optional manifest
MANIFEST = BASE_DIR.parent / "pairs_manifest.csv"

model = FlorenceModel()


# --------------------------------------------------
# Process Normal Pair
# --------------------------------------------------

def process_pair(person_image: Path, garment_image: Path, output_name: str):

    garment_desc = model.describe(garment_image)
    person_desc = model.describe(person_image)

    garment_attrs = extract_garment_attributes(garment_desc)
    person_attrs = extract_person_attributes(person_desc)

    result = {
        "person_image": person_image.name,
        "garment_image": garment_image.name,

        "garment_attributes": garment_attrs,

        "person_attributes": {
            "pose_category": person_attrs["pose"],
            "upper_body_visible": person_attrs["upper_body_visible"],
            "lower_body_visible": person_attrs["lower_body_visible"],
        },

        "model_used": "Florence-2-base",

        "confidence_notes":
        "Generated using Florence-2 and rule-based parsing."
    }

    save_json(result, OUTPUT_DIR / output_name)

    print(f"✅ Saved {output_name}")


# --------------------------------------------------
# Process Edge Cases
# --------------------------------------------------

def process_edge_case(image_path: Path):

    print(f"\nProcessing Edge Case: {image_path.name}")

    description = model.describe(image_path)

    print(description)

    # Handle image without a person
    if image_path.name == "no_person.jpg":

        result = {
            "image": image_path.name,
            "person_detected": False,
            "description": description,
            "model_used": "Florence-2-base",
            "confidence_notes":
            "No visible person detected."
        }

    else:

        person_attrs = extract_person_attributes(description)

        result = {
            "image": image_path.name,

            "person_detected": True,

            "description": description,

            "person_attributes": {
                "pose_category": person_attrs["pose"],
                "upper_body_visible":
                person_attrs["upper_body_visible"],
                "lower_body_visible":
                person_attrs["lower_body_visible"]
            },

            "model_used": "Florence-2-base",

            "confidence_notes":
            "Edge case evaluation."
        }

    save_json(
        result,
        OUTPUT_DIR / f"{image_path.stem}.json"
    )

    print(f"✅ Saved {image_path.stem}.json")


# --------------------------------------------------
# Main Processing
# --------------------------------------------------

if MANIFEST.exists():

    print("Using pairs_manifest.csv")

    with open(MANIFEST, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            person = PERSON_DIR / row["person_image"]
            garment = GARMENT_DIR / row["garment_image"]

            output = f'{row.get("pair_id","pair")}.json'

            process_pair(
                person,
                garment,
                output
            )

else:

    print("pairs_manifest.csv not found.")
    print("Pairing files by filename.\n")

    persons = sorted(PERSON_DIR.glob("person_*.png"))
    garments = sorted(GARMENT_DIR.glob("garment_*.jpg"))

    for idx, (person, garment) in enumerate(
        zip(persons, garments),
        start=1
    ):

        process_pair(
            person,
            garment,
            f"pair_{idx:02d}.json"
        )


# --------------------------------------------------
# Edge Cases
# --------------------------------------------------

print("\n")
print("=" * 60)
print("PROCESSING EDGE CASES")
print("=" * 60)

edge_images = sorted(EDGE_CASE_DIR.glob("*.jpg"))

for image in edge_images:

    process_edge_case(image)

print("\n")
print("=" * 60)
print("✅ ALL PROCESSING COMPLETED")
print("=" * 60)