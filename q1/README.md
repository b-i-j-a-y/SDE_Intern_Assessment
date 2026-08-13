# Q1 — Garment & Person Understanding

## Objective

Use the Florence-2 vision-language model to analyze a person image and a garment image and extract structured attributes that can be used by later stages of the virtual try-on pipeline.

---

## Model and Technologies

- **Florence-2 Base** (Vision-Language Model)
- **Transformers**
- **PyTorch**
- **Pillow**
- **NumPy**

Florence-2 is used to generate descriptive captions for the person and garment images. A lightweight rule-based parser then converts those descriptions into structured attributes.

---

## Extracted Attributes

### Person
- Pose category
- Upper-body visibility
- Lower-body visibility

### Garment
- Garment type
- Sleeve length
- Neckline
- Primary color
- Pattern

---

## Run

From the project root:

```bash
python q1/main.py
```

---

## Inputs

The script uses the images available in:

```text
data/person/
data/garment/
```

It also processes the provided edge-case images.

---

## Outputs

Generated outputs are saved to:

```text
q1/outputs/
```

Typical outputs include:

- Person analysis
- Garment analysis
- Structured JSON-style attribute files

---

## Example

Input:

- Person image
- Garment image

Output:

```json
{
  "garment_type": "t-shirt",
  "sleeve_length": "short",
  "neckline": "round",
  "primary_color": "purple",
  "pattern": "logo"
}
```

---

## Design Choice

I selected Florence-2 because it is:

- Open-source and MIT licensed
- Lightweight enough for CPU execution
- Suitable for image captioning and attribute extraction
- Easy to integrate with the later parsing and try-on stages

---

## Limitations

- Fine-grained fashion attributes may be imperfect.
- Pose estimation is derived from image descriptions rather than a dedicated pose model.
- The final structured attributes depend on the quality of the Florence-2 caption.

---

## Role in the Pipeline

This stage provides semantic information that is later used for:

- Q2 human parsing
- Q3 virtual try-on
- Q5 demo visualization and validation
