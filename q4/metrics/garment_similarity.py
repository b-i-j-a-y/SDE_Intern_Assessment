import torch
import open_clip
from PIL import Image
from pathlib import Path


class GarmentSimilarity:

    def __init__(self):

        print("Loading OpenCLIP...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k",
            )
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"OpenCLIP Ready ({self.device})")

    # --------------------------------------------------

    def load_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.preprocess(image)

        image = image.unsqueeze(0)

        return image.to(self.device)

    # --------------------------------------------------

    def get_embedding(self, image_path):

        image = self.load_image(image_path)

        with torch.no_grad():

            embedding = self.model.encode_image(image)

            embedding /= embedding.norm(
                dim=-1,
                keepdim=True
            )

        return embedding

    # --------------------------------------------------

    def cosine_similarity(
        self,
        embedding1,
        embedding2
    ):

        similarity = (
            embedding1 @ embedding2.T
        ).item()

        return float(similarity)

    # --------------------------------------------------

    def evaluate(
        self,
        garment_image,
        generated_image
    ):

        garment_embedding = self.get_embedding(
            garment_image
        )

        generated_embedding = self.get_embedding(
            generated_image
        )

        score = self.cosine_similarity(
            garment_embedding,
            generated_embedding
        )

        return {
            "garment_score": round(score, 4),

            "garment_grade": self.grade(score)
        }

    # --------------------------------------------------

    def grade(
        self,
        score
    ):

        if score >= 0.90:
            return "Excellent"

        if score >= 0.80:
            return "Very Good"

        if score >= 0.70:
            return "Good"

        if score >= 0.60:
            return "Fair"

        return "Poor"


# ------------------------------------------------------

if __name__ == "__main__":

    model = GarmentSimilarity()

    result = model.evaluate(

        "../q2/outputs/garment_masks/garment_01_mask.png",

        "../q3/outputs/pair_01_output.png"

    )

    print(result)