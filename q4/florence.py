from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch


class FlorenceModel:
    """
    Wrapper around Microsoft Florence-2
    """

    def __init__(self):

        self.MODEL_ID = "microsoft/Florence-2-base"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\nLoading Florence-2...\n")

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Florence Ready ({self.device})")

    def describe(
        self,
        image_path,
        prompt="<MORE_DETAILED_CAPTION>"
    ):
        """
        Generate Florence description.
        """

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                do_sample=False
            )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return output