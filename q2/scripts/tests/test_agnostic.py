from pathlib import Path
import cv2

from agnostic import AgnosticGenerator

project_root = Path(__file__).resolve().parents[2]

image = project_root / "data" / "person" / "person_01.png"

output_dir = project_root / "q2" / "outputs" / "agnostic"
output_dir.mkdir(parents=True, exist_ok=True)

generator = AgnosticGenerator()

agnostic = generator.generate(image)

save_path = output_dir / "person_01_agnostic.png"

cv2.imwrite(str(save_path), agnostic)

print("Saved!")

print(save_path)