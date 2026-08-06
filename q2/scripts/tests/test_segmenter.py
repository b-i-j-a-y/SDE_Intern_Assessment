from pathlib import Path
import cv2
import numpy as np

from detector import GroundingDINODetector
from segmenter import SAMSegmenter

project_root = Path(__file__).resolve().parents[2]

image_path = project_root / "data" / "person" / "person_01.png"

detector = GroundingDINODetector()

segmenter = SAMSegmenter()

result = detector.detect(str(image_path))

box = result["boxes"][0].cpu().numpy()

mask, score, image = segmenter.segment(
    str(image_path),
    box
)

output_dir = project_root / "q2" / "outputs" / "person_masks"
output_dir.mkdir(parents=True, exist_ok=True)

mask_image = (mask * 255).astype(np.uint8)

cv2.imwrite(
    str(output_dir / "person_01_mask.png"),
    mask_image
)

print("Mask Score:", score)

print("Done!")