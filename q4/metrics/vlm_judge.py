import sys
from pathlib import Path

# Allow importing florence.py from q4/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from florence import FlorenceModel


class VLMJudge:
    """
    Florence-2 based VLM Judge.

    Florence generates a detailed caption of the try-on image.
    A simple rule-based rubric converts the description
    into a quality score and reasons.
    """

    def __init__(self):

        print("Loading Florence-2...")

        self.model = FlorenceModel()

        print("Florence-2 Ready!")

    # --------------------------------------------------------

    def describe_image(self, image_path):

        description = self.model.describe(
            image_path,
            "<MORE_DETAILED_CAPTION>"
        )

        return description.lower()

    # --------------------------------------------------------

    def evaluate(self, image_path):

        description = self.describe_image(image_path)

        score = 10
        reasons = []

        # ----------------------------------------------------
        # Fit Realism
        # ----------------------------------------------------

        fit_positive = [
            "wearing",
            "shirt",
            "t-shirt",
            "hoodie",
            "dress",
            "jacket",
            "standing",
            "person"
        ]

        fit_found = False

        for word in fit_positive:
            if word in description:
                fit_found = True
                break

        if fit_found:
            reasons.append("Garment appears naturally worn.")
        else:
            score -= 2
            reasons.append("Fit realism could not be confirmed.")

        # ----------------------------------------------------
        # Texture Transfer
        # ----------------------------------------------------

        texture_words = [
            "graphic",
            "printed",
            "logo",
            "pattern",
            "striped",
            "floral",
            "checkered",
            "design"
        ]

        texture_found = False

        for word in texture_words:
            if word in description:
                texture_found = True
                break

        if texture_found:
            reasons.append("Texture and garment details detected.")
        else:
            score -= 1
            reasons.append("Texture transfer not clearly visible.")

        # ----------------------------------------------------
        # Artifact Detection
        # ----------------------------------------------------

        artifact_words = [
            "blur",
            "blurry",
            "artifact",
            "distorted",
            "deformed",
            "noise",
            "unnatural",
            "broken",
            "misaligned"
        ]

        artifact_found = False

        for word in artifact_words:
            if word in description:
                artifact_found = True
                score -= 2
                reasons.append(f"Possible artifact detected ({word}).")
                break

        if not artifact_found:
            reasons.append("No obvious artifacts mentioned.")

        # ----------------------------------------------------
        # Final Score
        # ----------------------------------------------------

        score = max(1, min(score, 10))

        if score >= 9:
            verdict = "Excellent"
        elif score >= 7:
            verdict = "Good"
        elif score >= 5:
            verdict = "Acceptable"
        else:
            verdict = "Poor"

        return {
            "judge_score": score,
            "verdict": verdict,
            "description": description,
            "reasons": reasons
        }


# ------------------------------------------------------------

if __name__ == "__main__":

    judge = VLMJudge()

    result = judge.evaluate(
        "../q3/outputs/pair_01_output.png"
    )

    print("\n" + "=" * 60)
    print("FLORENCE VLM JUDGE")
    print("=" * 60)

    print(f"\nScore    : {result['judge_score']}/10")
    print(f"Verdict  : {result['verdict']}")

    print("\nReasons:")

    for reason in result["reasons"]:
        print(f" - {reason}")

    print("\nFlorence Description:\n")
    print(result["description"])