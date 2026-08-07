# ==================================================
# Q5 Pose Guardrail Module
# ==================================================

from pathlib import Path


def detect_pose(image_path):
    """
    Simple pose guardrail.

    Later this can be replaced with:
    - OpenPose
    - MediaPipe Pose
    - Human Parsing model

    Current version uses filename-based edge case handling
    because Q2 already generated edge-case samples.
    """

    image_name = Path(image_path).stem.lower()


    # No person case

    if "no_person" in image_name:

        return {
            "pose_category": "none",
            "warning": "No person detected. Try-on cannot continue.",
            "allow_tryon": False
        }



    # Side pose case

    if "side" in image_name:

        return {
            "pose_category": "side",
            "warning": "Side pose may reduce try-on accuracy.",
            "allow_tryon": True
        }



    # Seated case

    if "seated" in image_name:

        return {
            "pose_category": "seated",
            "warning": "Seated pose may reduce try-on quality.",
            "allow_tryon": True
        }



    # Crossed arms case

    if "crossed" in image_name:

        return {
            "pose_category": "crossed_arms",
            "warning": "Arms may affect garment generation.",
            "allow_tryon": True
        }



    # Normal case

    return {
        "pose_category": "front",
        "warning": None,
        "allow_tryon": True
    }