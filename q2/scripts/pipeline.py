from pathlib import Path
import cv2
import numpy as np

from detector import GroundingDINODetector
from segmenter import SAMSegmenter
from human_parsing import HumanParser
from create_agnostic import AgnosticGenerator


def main():

    project_root = Path(__file__).resolve().parents[2]

    person_dir = project_root / "data" / "person"

    outputs_dir = project_root / "q2" / "outputs"

    person_mask_dir = outputs_dir / "person_masks"
    parsing_dir = outputs_dir / "parsing_maps"
    agnostic_dir = outputs_dir / "agnostic"

    person_mask_dir.mkdir(parents=True, exist_ok=True)
    parsing_dir.mkdir(parents=True, exist_ok=True)
    agnostic_dir.mkdir(parents=True, exist_ok=True)

    detector = GroundingDINODetector()
    segmenter = SAMSegmenter()
    parser = HumanParser()
    agnostic_generator = AgnosticGenerator()

    images = sorted(person_dir.glob("*"))

    print("=" * 70)
    print("Q2 PIPELINE")
    print("=" * 70)
    print(f"\nFound {len(images)} person images\n")

    for image_path in images:

        print("=" * 70)
        print(f"Processing : {image_path.name}")

        # --------------------------------------------------
        # STEP 1 : PERSON DETECTION
        # --------------------------------------------------

        detection = detector.detect(
            str(image_path),
            prompt="person."
        )

        if len(detection["boxes"]) == 0:
            print("❌ Person not detected.")
            continue

        box = detection["boxes"][0].cpu().numpy()

        # --------------------------------------------------
        # STEP 2 : PERSON SEGMENTATION
        # --------------------------------------------------

        mask, score, image = segmenter.segment(
            str(image_path),
            box
        )

        mask = (mask * 255).astype(np.uint8)

        mask_path = person_mask_dir / f"{image_path.stem}_mask.png"

        cv2.imwrite(str(mask_path), mask)

        print(f"✔ Person Mask : {mask_path.name}")

        # --------------------------------------------------
        # STEP 3 : HUMAN PARSING
        # --------------------------------------------------

        parsing = parser.parse(str(image_path))

        parsing_path = parsing_dir / f"{image_path.stem}_parsing.png"

        parser.save_parsing_map(
            parsing,
            parsing_path
        )

        print(f"✔ Parsing Map : {parsing_path.name}")

        # --------------------------------------------------
        # STEP 4 : AGNOSTIC IMAGE
        # --------------------------------------------------

        agnostic = agnostic_generator.generate(
            image_path
        )

        agnostic_path = agnostic_dir / f"{image_path.stem}_agnostic.png"

        cv2.imwrite(
            str(agnostic_path),
            agnostic
        )

        print(f"✔ Agnostic    : {agnostic_path.name}")

        print(f"Segmentation Score : {score:.4f}")

    print("\n")
    print("=" * 70)
    print("🎉 Q2 PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()