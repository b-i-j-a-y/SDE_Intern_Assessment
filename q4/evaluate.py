import csv
from pathlib import Path

from metrics.garment_similarity import GarmentSimilarity
from metrics.identity_score import IdentityScore
from metrics.vlm_judge import VLMJudge


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAIR_CSV = PROJECT_ROOT / "q4" / "pairs_manifest.csv"

PERSON_DIR = PROJECT_ROOT / "data"

GARMENT_DIR = PROJECT_ROOT / "data"

TRYON_DIR = PROJECT_ROOT / "q3" / "outputs"

GARMENT_MASK_DIR = PROJECT_ROOT / "q2" / "outputs" / "garment_masks"

OUTPUT_CSV = PROJECT_ROOT / "q4" / "outputs" / "evaluation_results.csv"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("Q4 AUTOMATED EVALUATION")
    print("=" * 60)

    garment_metric = GarmentSimilarity()

    identity_metric = IdentityScore()

    vlm_metric = VLMJudge()

    results = []

    with open(PAIR_CSV, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            pair_id = row["pair_id"]

            if not pair_id.startswith("pair_"):
                continue

            person_image = row["person_image"]

            garment_image = row["garment_image"]

            if person_image == "" or garment_image == "":
                print(f"Skipping {pair_id}")
                continue

            print(f"\nProcessing {pair_id}")

            person_path = PERSON_DIR / person_image

            garment_filename = Path(garment_image).stem

            garment_mask = (
                GARMENT_MASK_DIR /
                f"{garment_filename}_mask.png"
            )

            generated_image = (
                TRYON_DIR /
                f"{pair_id}_output.png"
            )

            if not generated_image.exists():

                alt = TRYON_DIR / f"{pair_id}_output.png .png"

                if alt.exists():
                    generated_image = alt

            # ----------------------------

            garment = garment_metric.evaluate(

                str(garment_mask),

                str(generated_image)

            )

            identity = identity_metric.compute_score(

                str(person_path),

                str(generated_image)

            )

            judge = vlm_metric.evaluate(

                str(generated_image)

            )

            results.append({

                "pair_id": pair_id,

                "garment_score": garment["garment_score"],

                "garment_grade": garment["garment_grade"],

                "identity_score": identity["score"],

                "identity_status": identity["status"],

                "judge_score": judge["judge_score"],

                "judge_verdict": judge["verdict"],

                "judge_reason": "; ".join(judge["reasons"])

            })

    with open(OUTPUT_CSV, "w", newline="") as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "pair_id",

                "garment_score",

                "garment_grade",

                "identity_score",

                "identity_status",

                "judge_score",

                "judge_verdict",

                "judge_reason"

            ]

        )

        writer.writeheader()

        writer.writerows(results)

    print("\n")

    print("=" * 60)

    print("Evaluation Complete!")

    print(f"Results saved to:\n{OUTPUT_CSV}")

    print("=" * 60)


if __name__ == "__main__":

    main()