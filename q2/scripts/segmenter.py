from pathlib import Path
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


class SAMSegmenter:

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]

        checkpoint = project_root / "q2" / "models" / "sam_vit_b_01ec64.pth"

        self.sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.predictor = SamPredictor(self.sam)

    def segment(self, image_path, box):

        image = cv2.imread(str(image_path))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image_rgb)

        masks, scores, _ = self.predictor.predict(
            box=np.array(box),
            multimask_output=False
        )

        return masks[0], float(scores[0]), image