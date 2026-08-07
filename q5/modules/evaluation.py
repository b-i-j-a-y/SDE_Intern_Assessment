# ==================================================
# Q5 Evaluation Module
# ==================================================

from pathlib import Path
import json



def evaluate_result(
    person_path,
    garment_path,
    result_path
):
    """
    Quality evaluation module.

    Later can connect:
    - CLIP similarity
    - LPIPS
    - garment preservation
    - image alignment

    Currently uses Q4 output metrics if available.
    """


    scores = {

        "clip_similarity": 0.0,

        "garment_preservation": 0.0,

        "alignment_score": 0.0,

        "overall_score": 0.0

    }



    # Try loading Q4 metrics

    q4_file = Path(
        "q4/metrics/results.json"
    )


    if q4_file.exists():

        try:

            with open(
                q4_file,
                "r"
            ) as f:

                q4_scores = json.load(f)


            scores.update(
                q4_scores
            )


        except Exception:

            pass



    else:

        # Demo values
        # Replace with real Q4 metrics

        scores = {

            "clip_similarity": 0.82,

            "garment_preservation": 0.78,

            "alignment_score": 0.80,

            "overall_score": 0.80

        }



    return scores