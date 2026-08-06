from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)


class HumanParser:

    def __init__(self):

        print("Loading Human Parsing Model...")

        self.processor = AutoImageProcessor.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing"
        )

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing"
        )

        self.model.eval()

        print("✅ Human Parser Ready!\n")

    def parse(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        segmentation = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]

        return segmentation.cpu().numpy()

    def save_parsing_map(self, parsing_map, save_path):

        colors = np.array([
            [0, 0, 0],          # Background
            [255, 0, 0],        # Hat
            [255, 128, 0],      # Hair
            [255, 255, 0],      # Sunglasses
            [0, 255, 0],        # Upper Clothes
            [0, 255, 255],      # Skirt
            [0, 0, 255],        # Pants
            [255, 0, 255],      # Dress
            [128, 0, 255],      # Belt
            [128, 128, 255],    # Left Shoe
            [255, 128, 255],    # Right Shoe
            [255, 200, 150],    # Face
            [150, 255, 150],    # Left Leg
            [150, 150, 255],    # Right Leg
            [255, 150, 150],    # Left Arm
            [200, 255, 255],    # Right Arm
            [255, 255, 255],    # Bag
            [180, 180, 180],    # Scarf
        ], dtype=np.uint8)

        colored = colors[parsing_map]

        Image.fromarray(colored).save(save_path)