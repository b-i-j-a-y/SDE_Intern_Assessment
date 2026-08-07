### Garment Fidelity

- Model: OpenCLIP ViT-B-32
- Metric: Cosine Similarity between garment mask and generated garment.

### Identity Preservation

- Model: InsightFace (Buffalo-L)
- Metric: Face embedding cosine similarity.

### VLM-as-Judge

Model:
- Microsoft Florence-2 Base

Rubric:

- Fit Realism (4 pts)
- Texture Transfer (3 pts)
- Artifact Detection (3 pts)

Final Score:
1–10

The Florence-2 model generates a detailed description of the generated image. A rule-based rubric analyzes the description to assign scores for fit realism, texture preservation, and visible artifacts.