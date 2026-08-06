from pathlib import Path

from human_parsing import HumanParser

project_root = Path(__file__).resolve().parents[2]

image = project_root / "data" / "person" / "person_01.png"

output_dir = project_root / "q2" / "outputs" / "parsing_maps"
output_dir.mkdir(parents=True, exist_ok=True)

parser = HumanParser()

parsing = parser.parse(str(image))

parser.save_parsing_map(
    parsing,
    output_dir / "person_01_parsing.png"
)

print("Parsing Map Shape:", parsing.shape)

print("Saved Successfully!")