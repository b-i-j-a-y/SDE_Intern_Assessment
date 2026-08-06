from pathlib import Path
import cv2
import numpy as np

from detector import GroundingDINODetector
from segmenter import SAMSegmenter


PROMPTS = [
    "shirt.",
    "t-shirt.",
    "dress.",
    "jacket.",
    "hoodie.",
    "top.",
]


def main():

    project_root = Path(__file__).resolve().parents[2]

    garment_dir = project_root / "data" / "garment"

    output_dir = project_root / "q2" / "outputs" / "garment_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = GroundingDINODetector()
    segmenter = SAMSegmenter()

    images = sorted(garment_dir.glob("*"))

    print(f"\nFound {len(images)} garments\n")

    for image_path in images:

        print("=" * 60)
        print(f"Processing {image_path.name}")

        detection = None

        for prompt in PROMPTS:

            result = detector.detect(
                str(image_path),
                prompt=prompt
            )

            if len(result["boxes"]) > 0:
                detection = result
                print(f"Detected using: {prompt}")
                break

        if detection is None:
            print("No garment detected.\n")
            continue

        box = detection["boxes"][0].cpu().numpy()

        mask, score, image = segmenter.segment(
            str(image_path),
            box
        )

        mask = (mask * 255).astype(np.uint8)

        save_path = output_dir / f"{image_path.stem}_mask.png"

        cv2.imwrite(str(save_path), mask)

        print(f"Saved: {save_path.name}")
        print(f"Score: {score:.4f}")

    print("\n✅ Finished garment segmentation.")


if __name__ == "__main__":
    main()