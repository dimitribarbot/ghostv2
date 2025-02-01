import torch
from diffusers import AutoPipelineForInpainting
from transformers import AutoModelForImageClassification, ViTImageProcessor


nsfw_processor = ViTImageProcessor.from_pretrained(
    "AdamCodd/vit-base-nsfw-detector"
)
nsfw_detector_pipe = AutoModelForImageClassification.from_pretrained(
    "AdamCodd/vit-base-nsfw-detector",
    use_safetensors=True,
)

sd_pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)