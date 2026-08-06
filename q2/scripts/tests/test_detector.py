from detector import GroundingDINODetector

detector = GroundingDINODetector()

result = detector.detect(
    "data/person/person_01.png"
)

print(result)