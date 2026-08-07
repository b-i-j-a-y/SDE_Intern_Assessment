import cv2
import numpy as np
from insightface.app import FaceAnalysis


class IdentityScore:
    """
    Computes identity similarity between the original person
    and the generated try-on image using InsightFace.
    """

    def __init__(self):

        print("Loading InsightFace...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=0, det_size=(640, 640))

        print("InsightFace Ready!")

    def get_embedding(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        # Largest face
        face = max(
            faces,
            key=lambda f: (
                f.bbox[2] - f.bbox[0]
            ) * (
                f.bbox[3] - f.bbox[1]
            )
        )

        return face.embedding

    def cosine_similarity(self, emb1, emb2):

        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        return float(np.dot(emb1, emb2))

    def compute_score(
        self,
        original_person,
        generated_person
    ):

        emb1 = self.get_embedding(original_person)
        emb2 = self.get_embedding(generated_person)

        if emb1 is None or emb2 is None:

            return {
                "score": 0.0,
                "status": "Face not detected"
            }

        similarity = self.cosine_similarity(
            emb1,
            emb2
        )

        return {
            "score": round(similarity, 4),
            "status": "Success"
        }


if __name__ == "__main__":

    scorer = IdentityScore()

    result = scorer.compute_score(
        "../data/person/person_01.png",
        "../q3/outputs/pair_01_output.png"
    )

    print(result)