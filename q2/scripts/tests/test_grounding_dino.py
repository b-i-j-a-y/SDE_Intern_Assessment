from transformers import AutoProcessor
from transformers import AutoModelForZeroShotObjectDetection

print("Loading Grounding DINO Processor...")

processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

print("Loading Grounding DINO Model...")

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

print("\n✅ Grounding DINO Loaded Successfully!")