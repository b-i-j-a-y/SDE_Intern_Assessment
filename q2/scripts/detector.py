from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


class GroundingDINODetector:

    def __init__(self):

        print("Loading Grounding DINO...")

        self.processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )

        self.model.eval()

        print("✅ Grounding DINO Ready!\n")

    def detect(
        self,
        image_path,
        prompt="person.",
        box_threshold=0.20,
        text_threshold=0.20,
    ):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        return results[0]


if __name__ == "__main__":

    detector = GroundingDINODetector()

    image_path = "data/person/person_01.png"

    result = detector.detect(image_path)

    print("=" * 60)
    print("GROUNDING DINO OUTPUT")
    print("=" * 60)

    if len(result["boxes"]) == 0:
        print("No objects detected.")

    else:

        for box, score, label in zip(
            result["boxes"],
            result["scores"],
            result["text_labels"],
        ):

            print(f"Label : {label}")
            print(f"Score : {score:.4f}")
            print(f"Box   : {box.tolist()}")
            print("-" * 40)