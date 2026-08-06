from pathlib import Path
import cv2
import numpy as np

from human_parsing import HumanParser


class AgnosticGenerator:

    def __init__(self):
        self.parser = HumanParser()

    def generate(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Cannot open {image_path}")

        parsing = self.parser.parse(str(image_path))

        # Labels to remove
        REMOVE_LABELS = [
            4,   # Upper-clothes
            7,   # Dress
            8,   # Belt
        ]

        agnostic = image.copy()

        for label in REMOVE_LABELS:
            agnostic[parsing == label] = (128, 128, 128)

        return agnostic


def main():

    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "person"

    output_dir = project_root / "q2" / "outputs" / "agnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = AgnosticGenerator()

    images = sorted(input_dir.glob("*"))

    print(f"\nFound {len(images)} images\n")

    for image_path in images:

        print(f"Processing {image_path.name}")

        agnostic = generator.generate(image_path)

        save_path = output_dir / f"{image_path.stem}_agnostic.png"

        cv2.imwrite(str(save_path), agnostic)

        print(f"Saved: {save_path.name}")

    print("\n✅ All agnostic images generated.")


if __name__ == "__main__":
    main()