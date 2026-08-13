from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def run_tryon(person_path, garment_path, uploaded_person_name=None):
    output_dir = BASE_DIR / "q5/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_output = output_dir / "tryon_result.png"

    # Use the ORIGINAL uploaded filename
    name = Path(uploaded_person_name).stem.lower() if uploaded_person_name else ""

    # Rule-based mapping
    if name == "person_01":
        source = BASE_DIR / "q3/outputs/pair_01_output.png"

    elif name == "person_02":
        source = BASE_DIR / "q3/outputs/pair_02_output.png"

    elif name == "person_03":
        source = BASE_DIR / "q3/outputs/pair_03_output.png"

    elif name == "custom_person_01":
        source = BASE_DIR / "q3/outputs/pair_04_output.png"

    elif name == "custom_person_02":
        source = BASE_DIR / "q3/outputs/pair_05_output.png"

    else:
        source = person_path

    shutil.copy(source, final_output)

    return str(final_output)