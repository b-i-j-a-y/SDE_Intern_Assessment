import os
import gc

import torch
import streamlit as st

from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.image_processor import VaeImageProcessor

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
# Load Model
# ==================================================

@st.cache_resource
def load_model():

    st.info("Downloading CatVTON weights...")

    repo_path = snapshot_download(
        repo_id="zhengchong/CatVTON"
    )


    st.info("Loading CatVTON pipeline...")


    pipeline = CatVTONPipeline(

        base_ckpt=
        "booksforcharlie/stable-diffusion-inpainting",

        attn_ckpt=repo_path,

        attn_ckpt_version="mix",

        weight_dtype=
        init_weight_dtype("no"),

        use_tf32=False,

        device=DEVICE
    )


    st.success("Pipeline loaded")


    st.info("Loading DensePose masker...")


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


    st.success("Masker loaded")


    mask_processor = VaeImageProcessor(

        vae_scale_factor=8,

        do_normalize=False,

        do_binarize=True,

        do_convert_grayscale=True

    )


    return pipeline, automasker, mask_processor



pipeline, automasker, mask_processor = load_model()



# ==================================================
# Try On Function
# ==================================================

def generate_tryon(

        person,

        cloth,

        cloth_type,

        steps,

        guidance,

        seed

):


    gc.collect()


    if torch.backends.mps.is_available():

        torch.mps.empty_cache()



    # Convert

    person = person.convert(
        "RGB"
    )

    cloth = cloth.convert(
        "RGB"
    )



    # LOWER RESOLUTION FOR M2

    person = resize_and_crop(

        person,

        (
            384,
            512
        )

    )


    cloth = resize_and_padding(

        cloth,

        (
            384,
            512
        )

    )



    st.write(
        "Generating mask..."
    )


    mask = automasker(

        person,

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



    st.write(
        "Running CatVTON inference..."
    )


    with torch.no_grad():


        result = pipeline(

            image=person,

            condition_image=cloth,

            mask=mask,

            num_inference_steps=int(steps),

            guidance_scale=float(guidance),

            generator=generator

        )[0]



    gc.collect()


    if torch.backends.mps.is_available():

        torch.mps.empty_cache()



    return result



# ==================================================
# Streamlit UI
# ==================================================

st.set_page_config(
    page_title="CatVTON",
    layout="centered"
)


st.title(
    "👕 CatVTON Virtual Try-On"
)


st.write(
    "Upload a person image and garment image"
)



person_file = st.file_uploader(

    "Person Image",

    type=[
        "png",
        "jpg",
        "jpeg"
    ]

)



cloth_file = st.file_uploader(

    "Garment Image",

    type=[
        "png",
        "jpg",
        "jpeg"
    ]

)



cloth_type = st.selectbox(

    "Garment Type",

    [
        "upper",
        "lower",
        "dress"
    ]

)



steps = st.slider(

    "Inference Steps",

    10,

    50,

    20

)



guidance = st.slider(

    "Guidance Scale",

    1.0,

    5.0,

    2.0

)



seed = st.number_input(

    "Seed",

    value=42

)



if st.button(
    "Generate Try-On"
):


    if person_file and cloth_file:


        person_img = Image.open(
            person_file
        )


        cloth_img = Image.open(
            cloth_file
        )


        try:


            output = generate_tryon(

                person_img,

                cloth_img,

                cloth_type,

                steps,

                guidance,

                seed

            )


            st.image(

                output,

                caption="Generated Result",

                use_container_width=True

            )


        except Exception as e:

            st.error(
                str(e)
            )


    else:

        st.warning(
            "Please upload both images"
        )