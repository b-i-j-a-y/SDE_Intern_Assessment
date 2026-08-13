import cv2
import numpy as np


def detect_pose(image_path):
    """
    Lightweight content-based pose guardrail.

    Heuristics:
    - wide aspect ratio + shorter body → side pose
    - upper body lower in frame → seated pose
    - otherwise front pose
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "pose_category": "unknown",
            "warning": "Could not analyze pose.",
            "allow_tryon": True,
        }

    h, w = image.shape[:2]

    aspect_ratio = w / h

    # Foreground estimate
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    ys, xs = np.where(thresh > 0)

    if len(xs) == 0:
        return {
            "pose_category": "unknown",
            "warning": "Pose could not be analyzed.",
            "allow_tryon": True,
        }

    top = ys.min()
    bottom = ys.max()

    height_ratio = (bottom - top) / h
    top_ratio = top / h

    # Side pose heuristic
    if aspect_ratio > 0.9 and height_ratio < 0.65:
        return {
            "pose_category": "side",
            "warning": "Side pose may reduce try-on accuracy.",
            "allow_tryon": True,
        }

    # Seated pose heuristic
    if top_ratio > 0.18:
        return {
            "pose_category": "seated",
            "warning": "Seated pose may reduce try-on quality.",
            "allow_tryon": True,
        }

    return {
        "pose_category": "front",
        "warning": None,
        "allow_tryon": True,
    }