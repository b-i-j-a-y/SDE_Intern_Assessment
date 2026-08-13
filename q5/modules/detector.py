import cv2
import numpy as np


def detect_person(image_path):
    """
    Content-based person detection guardrail.

    Uses image content rather than filenames.
    """

    try:
        image = cv2.imread(str(image_path))

        if image is None:
            return False, "❌ Could not read image."

        h, w = image.shape[:2]

        # Reject very small images
        if w < 100 or h < 100:
            return False, "❌ Image is too small."

        # Foreground coverage
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

        coverage = np.sum(thresh > 0) / (h * w)

        if coverage < 0.08:
            return False, "❌ No clear person foreground detected."

        # Simple skin-tone heuristic
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)

        skin = cv2.inRange(ycrcb, lower, upper)

        skin_ratio = np.sum(skin > 0) / (h * w)

        if skin_ratio < 0.01:
            return False, "❌ No visible person detected."

        return True, "✅ Person detected from image content."

    except Exception as e:
        return False, f"Detection error: {e}"