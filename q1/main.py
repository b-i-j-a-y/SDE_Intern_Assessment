from pathlib import Path
import csv

# Updated package imports for Q5 integration
from q1.florence import FlorenceModel
from q1.parser import (
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

MANIFEST = BASE_DIR.parent / "pairs_manifest.csv"


# Load Florence model once
model = FlorenceModel()


# ==================================================
# Q5 REUSABLE FUNCTION
# ==================================================

def analyze_images(person_image: Path, garment_image: Path):
    """
    Analyze one person image and one garment image.
    Used by Q5 Streamlit app.
    """

    garment_desc = model.describe(garment_image)

    person_desc = model.describe(person_image)


    garment_attrs = extract_garment_attributes(
        garment_desc
    )

    person_attrs = extract_person_attributes(
        person_desc
    )


    result = {

        "person_image": person_image.name,

        "garment_image": garment_image.name,


        "garment_attributes": garment_attrs,


        "person_attributes": {

            "pose_category":
                person_attrs["pose"],

            "upper_body_visible":
                person_attrs["upper_body_visible"],

            "lower_body_visible":
                person_attrs["lower_body_visible"],
        },


        "model_used":
            "Florence-2-base",


        "confidence_notes":
            "Generated using Florence-2 and rule-based parsing."
    }


    return result



# ==================================================
# SAVE NORMAL PAIR
# ==================================================

def process_pair(
    person_image: Path,
    garment_image: Path,
    output_name: str
):

    result = analyze_images(
        person_image,
        garment_image
    )


    save_json(
        result,
        OUTPUT_DIR / output_name
    )


    print(
        f"✅ Saved {output_name}"
    )



# ==================================================
# EDGE CASE PROCESSING
# ==================================================

def process_edge_case(image_path: Path):

    print(
        f"\nProcessing Edge Case: {image_path.name}"
    )


    description = model.describe(
        image_path
    )


    if image_path.name == "no_person.jpg":


        result = {

            "image":
                image_path.name,


            "person_detected":
                False,


            "description":
                description,


            "model_used":
                "Florence-2-base"
        }


    else:


        person_attrs = extract_person_attributes(
            description
        )


        result = {


            "image":
                image_path.name,


            "person_detected":
                True,


            "person_attributes":
            {

                "pose_category":
                    person_attrs["pose"],


                "upper_body_visible":
                    person_attrs["upper_body_visible"],


                "lower_body_visible":
                    person_attrs["lower_body_visible"]
            },


            "model_used":
                "Florence-2-base"
        }



    save_json(
        result,
        OUTPUT_DIR /
        f"{image_path.stem}.json"
    )


    print(
        f"✅ Saved {image_path.stem}.json"
    )



# ==================================================
# ORIGINAL Q1 EXECUTION
# ==================================================

if __name__ == "__main__":


    print("=" * 60)
    print("Q1 Florence Processing")
    print("=" * 60)



    if MANIFEST.exists():


        print(
            "Using pairs_manifest.csv"
        )


        with open(
            MANIFEST,
            newline="",
            encoding="utf-8"
        ) as f:


            reader = csv.DictReader(f)


            for row in reader:


                person = (
                    PERSON_DIR /
                    row["person_image"]
                )


                garment = (
                    GARMENT_DIR /
                    row["garment_image"]
                )


                output = (
                    f'{row.get("pair_id","pair")}.json'
                )


                process_pair(
                    person,
                    garment,
                    output
                )



    else:


        print(
            "Manifest not found. "
            "Pairing by filename."
        )


        persons = sorted(
            PERSON_DIR.glob("person_*.png")
        )


        garments = sorted(
            GARMENT_DIR.glob("garment_*.jpg")
        )



        for idx, (
            person,
            garment
        ) in enumerate(
            zip(persons, garments),
            start=1
        ):


            process_pair(
                person,
                garment,
                f"pair_{idx:02d}.json"
            )



    print("\nProcessing Edge Cases")


    edge_images = sorted(
        EDGE_CASE_DIR.glob("*.jpg")
    )


    for image in edge_images:

        process_edge_case(image)



    print("=" * 60)
    print("✅ COMPLETED")
    print("=" * 60)