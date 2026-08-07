import sys
sys.path.append("./")

import os
from typing import List

import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
)

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import (
    UNet2DConditionModel as UNet2DConditionModel_ref,
)
from src.unet_hacked_tryon import UNet2DConditionModel

from utils_mask import get_mask_location
import apply_net

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Loading IDM-VTON")
print("=" * 60)

BASE_PATH = "yisol/IDM-VTON"

print("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(
    BASE_PATH,
    subfolder="unet",
    torch_dtype=torch.float16,
)

print("Loading Tokenizers...")
tokenizer_one = AutoTokenizer.from_pretrained(
    BASE_PATH,
    subfolder="tokenizer",
    use_fast=False,
)

tokenizer_two = AutoTokenizer.from_pretrained(
    BASE_PATH,
    subfolder="tokenizer_2",
    use_fast=False,
)

print("Loading Scheduler...")
noise_scheduler = DDPMScheduler.from_pretrained(
    BASE_PATH,
    subfolder="scheduler",
)

print("Loading Text Encoder 1...")
text_encoder_one = CLIPTextModel.from_pretrained(
    BASE_PATH,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)

print("Loading Text Encoder 2...")
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    BASE_PATH,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)

print("Loading Image Encoder...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    BASE_PATH,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    BASE_PATH,
    subfolder="vae",
    torch_dtype=torch.float16,
)

print("Loading Garment Encoder...")
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    BASE_PATH,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

print("Loading Human Parsing...")
parsing_model = Parsing(0)

print("Loading OpenPose...")
openpose_model = OpenPose(0)

for model in [
    unet,
    UNet_Encoder,
    image_encoder,
    vae,
    text_encoder_one,
    text_encoder_two,
]:
    model.requires_grad_(False)

tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

print("Building Pipeline...")

pipe = TryonPipeline.from_pretrained(
    BASE_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)

pipe.unet_encoder = UNet_Encoder

print("✅ Pipeline Ready")
