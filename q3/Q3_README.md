# Q3 – End-to-End Virtual Try-On using CatVTON

## Problem Statement

Run an open-source virtual try-on model using the preprocessing outputs from Q2 and generate final try-on results for the required person–garment pairs.

---

## Model Used

- **Model:** CatVTON
- **Purpose:** Diffusion-based virtual try-on

---

## Execution Environment

- Visual Studio Code (Local)
- Python 3.13
- PyTorch
- Diffusers
- Transformers

---

## Technologies Used

- PyTorch
- Diffusers
- Transformers
- OpenCV
- Pillow

---

## Working Flow

- Prepare the person image.
- Generate the person mask, parsing map, and agnostic image from Q2.
- Prepare the garment image.
- Load the CatVTON model.
- Run diffusion-based try-on inference.
- Generate the final virtual try-on image.
- Save the output for each pair.

---

## Pipeline Architecture

```text
Person Image
      |
      v
Q2 Preprocessing
(Mask + Parsing + Agnostic)
      |
      v
CatVTON
      |
      v
Virtual Try-On Output
```

---

## What I Completed

- Integrated the Q2 preprocessing outputs.
- Loaded and configured the CatVTON model.
- Ran the try-on inference successfully.
- Generated final outputs for all required pairs.
- Saved the outputs in the `outputs/` directory.

---

## Constraints Encountered

- Large model downloads
- Long model loading time
- High memory usage during inference

---

## Workarounds

- Switched from IDM-VTON to the lighter CatVTON model.
- Reduced input resolution where required.
- Cleared temporary files during execution.

---

## Outputs

The generated outputs are stored in the `outputs/` folder.

- `pair_01_output.png`
- `pair_02_output.png`
- `pair_03_output.png`
- `pair_04_output.png`
- `pair_05_output.png`

---

## Notes

The preprocessing outputs from Q2 (person mask, parsing map, and agnostic image) were used as inputs for the CatVTON inference stage. The final virtual try-on results for all five required pairs are included in the `outputs/` directory.
