from transformers import pipeline
from PIL import Image

print("Loading Human Parsing Model...")

pipe = pipeline(
    "image-segmentation",
    model="matei-dorian/segformer-b5-finetuned-human-parsing"
)

print("✅ Model Loaded!")

image = Image.open("data/person/person_01.png").convert("RGB")

results = pipe(image)

print("\nDetected Segments:\n")

for segment in results:
    print(segment["label"])