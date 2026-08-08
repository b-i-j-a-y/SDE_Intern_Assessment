import argparse
import os

import gradio as gr
import torch

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline

from utils import (
    init_weight_dtype,
    resize_and_crop,
    resize_and_padding
)


# ==================================================
# Device
# ==================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)



# ==================================================
# Arguments
# ==================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="CatVTON Apple Silicon Demo"
    )


    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting"
    )


    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON"
    )


    parser.add_argument(
        "--width",
        type=int,
        default=768
    )


    parser.add_argument(
        "--height",
        type=int,
        default=1024
    )


    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no"
    )


    return parser.parse_args()



args = parse_args()



# ==================================================
# Download Model
# ==================================================

print("Downloading CatVTON weights...")


repo_path = snapshot_download(
    repo_id=args.resume_path
)


print(
    "Weights:",
    repo_path
)



# ==================================================
# Pipeline
# ==================================================

print("Loading CatVTON pipeline...")


pipeline = CatVTONPipeline(

    base_ckpt=args.base_model_path,

    attn_ckpt=repo_path,

    attn_ckpt_version="mix",

    weight_dtype=init_weight_dtype(
        args.mixed_precision
    ),

    use_tf32=False,

    device=DEVICE

)


print("Pipeline loaded")



# ==================================================
# Mask Processor
# ==================================================

mask_processor = VaeImageProcessor(

    vae_scale_factor=8,

    do_normalize=False,

    do_binarize=True,

    do_convert_grayscale=True

)



# ==================================================
# Masker
# ==================================================

print("Loading masker...")


automasker = AutoMasker(

    densepose_ckpt=os.path.join(
        repo_path,
        "DensePose"
    ),

    schp_ckpt=os.path.join(
        repo_path,
        "SCHP"
    ),

    device="cpu"

)


print("Masker loaded")



# ==================================================
# Try On
# ==================================================

def generate_tryon(

    person_image,

    cloth_image,

    cloth_type,

    steps,

    guidance,

    seed

):


    person_image = Image.open(
        person_image
    ).convert(
        "RGB"
    )


    cloth_image = Image.open(
        cloth_image
    ).convert(
        "RGB"
    )



    person_image = resize_and_crop(

        person_image,

        (
            args.width,
            args.height
        )

    )



    cloth_image = resize_and_padding(

        cloth_image,

        (
            args.width,
            args.height
        )

    )



    print("Creating mask...")


    mask = automasker(

        person_image,

        cloth_type

    )["mask"]



    mask = mask_processor.blur(

        mask,

        blur_factor=9

    )



    generator = None


    if seed != -1:

        generator = torch.Generator().manual_seed(
            int(seed)
        )



    print("Generating try-on...")


    result = pipeline(

        image=person_image,

        condition_image=cloth_image,

        mask=mask,

        num_inference_steps=int(steps),

        guidance_scale=float(guidance),

        generator=generator

    )[0]



    print("Done")


    return result



# ==================================================
# Gradio
# ==================================================

demo = gr.Interface(

    fn=generate_tryon,


    inputs=[

        gr.Image(
            type="filepath",
            label="Person Image"
        ),


        gr.Image(
            type="filepath",
            label="Garment Image"
        ),


        gr.Dropdown(

            choices=[
                "upper",
                "lower",
                "dress"
            ],

            value="upper",

            label="Garment Type"

        ),


        gr.Slider(

            minimum=10,

            maximum=100,

            value=50,

            label="Steps"

        ),


        gr.Slider(

            minimum=1,

            maximum=5,

            value=2.5,

            label="Guidance"

        ),


        gr.Number(

            value=42,

            label="Seed"

        )

    ],



    outputs=gr.Image(
        type="pil",
        label="Try-On Result"
    ),



    title="CatVTON Apple Silicon Demo"

)



# ==================================================
# Launch
# ==================================================

if __name__ == "__main__":


    demo.launch(

        server_name="127.0.0.1",

        server_port=7860,

        share=False,

        show_error=True

    )