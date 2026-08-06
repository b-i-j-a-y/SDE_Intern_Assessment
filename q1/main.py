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

# Optional manifest (recommended by the assessment)
MANIFEST = BASE_DIR.parent / "pairs_manifest.csv"

model = FlorenceModel()


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
        "confidence_notes": "Generated using Florence-2 and rule-based parsing."
    }

    save_json(result, OUTPUT_DIR / output_name)
    print(f"Saved {output_name}")


if MANIFEST.exists():
    print("Using pairs_manifest.csv")
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person = PERSON_DIR / row["person_image"]
            garment = GARMENT_DIR / row["garment_image"]
            out_name = f'{row.get("pair_id","pair")}.json'
            process_pair(person, garment, out_name)
else:
    print("pairs_manifest.csv not found. Pairing files by number.")
    persons = sorted(PERSON_DIR.glob("person_*.png"))
    garments = sorted(GARMENT_DIR.glob("garment_*.jpg"))

    for idx, (person, garment) in enumerate(zip(persons, garments), start=1):
        process_pair(person, garment, f"pair_{idx:02d}.json")

print("Done.")
