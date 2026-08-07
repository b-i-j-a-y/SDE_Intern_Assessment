from PIL import Image



def detect_person(image_path):

    """
    Person detection guardrail.

    Temporary implementation.
    Will later be replaced with GroundingDINO.
    """


    try:

        image = Image.open(
            image_path
        )


        width, height = image.size



        # Basic validation

        if width < 50 or height < 50:

            return (
                False,
                "❌ No person detected."
            )



        return (
            True,
            "✅ Person detected."
        )



    except Exception as e:


        return (
            False,
            f"Detection error: {e}"
        )