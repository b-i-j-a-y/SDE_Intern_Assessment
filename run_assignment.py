import torch
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

MODEL_PATH = "yisol/IDM-VTON"

print("=" * 60)
print("IDM-VTON Assignment Runner")
print("=" * 60)

print("\nLoading scheduler...")
scheduler = DDPMScheduler.from_pretrained(
    MODEL_PATH,
    subfolder="scheduler"
)
print("✅ Scheduler loaded")

print("\nLoading VAE...")
vae = AutoencoderKL.from_pretrained(
    MODEL_PATH,
    subfolder="vae",
    torch_dtype=torch.float16,
)
print("✅ VAE loaded")

print("\nLoading Text Encoder 1...")
text_encoder = CLIPTextModel.from_pretrained(
    MODEL_PATH,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
print("✅ Text Encoder 1 loaded")

print("\nLoading Text Encoder 2...")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    MODEL_PATH,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
print("✅ Text Encoder 2 loaded")

print("\nLoading Image Encoder...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    MODEL_PATH,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
print("✅ Image Encoder loaded")

print("\n======================================")
print("All currently required models loaded!")
print("======================================")
