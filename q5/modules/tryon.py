from pathlib import Path
from PIL import Image
import re


def extract_number(filename):

    match = re.search(
        r'(\d+)',
        filename
    )

    if match:
        return match.group(1)

    return "01"



def run_tryon(
    person_path,
    garment_path
):

    """
    Load corresponding Q3 generated try-on result.
    """

    person_name = Path(person_path).name
    garment_name = Path(garment_path).name


    person_id = extract_number(
        person_name
    )

    garment_id = extract_number(
        garment_name
    )


    print("-------------------------")
    print("Person file:", person_name)
    print("Garment file:", garment_name)
    print("Person ID:", person_id)
    print("Garment ID:", garment_id)


    # Q3 output mapping
    result_path = Path(
        f"q3/outputs/pair_{garment_id}_output.png"
    )


    print(
        "Trying to load:",
        result_path.resolve()
    )


    if result_path.exists():

        print(
            "✅ Q3 Result Loaded"
        )

        image = Image.open(
            result_path
        ).convert("RGB")


        return image


    else:

        print(
            "❌ Q3 result not found"
        )


        print(
            "Using original person image"
        )


        return Image.open(
            person_path
        ).convert("RGB")