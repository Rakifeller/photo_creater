import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
from .config import MODEL_ID, USE_LCM

_device = "cuda" if torch.cuda.is_available() else "cpu"
_pipe = None
_image_encoder = None

def load_pipeline():
    global _pipe, _image_encoder
    if _pipe is not None:
        return _pipe

    torch.set_grad_enabled(False)
    _image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )

    _pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        image_encoder=_image_encoder
    ).to(_device)

    # IP-Adapter Plus Face (SDXL ViT-H)
    _pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
    )
    _pipe.set_ip_adapter_scale(0.7)

    # İsteğe bağlı hız: LCM LoRA + LCM scheduler
    if USE_LCM:
        try:
            _pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            _pipe.scheduler = LCMScheduler.from_config(_pipe.scheduler.config)
        except Exception:
            pass

    # Bellek optimizasyonları
    if _device == "cuda":
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return _pipe

def generate_with_images(prompt: str, ref_images, steps: int = 30, guidance: float = 5.0, seed: int | None = None, size: tuple[int, int] = (1024, 1024)):
    pipe = load_pipeline()
    images = [load_image(img) for img in ref_images]  # file-like kabul eder
    g = None if seed is None else torch.Generator(device=_device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        ip_adapter_image=images if len(images) > 1 else images[0],
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
        width=size[0],
        height=size[1],
        negative_prompt="deformed, bad anatomy, lowres, text, watermark"
    ).images[0]
    return result

def prepare_embeds(ref_images):
    pipe = load_pipeline()
    images = [load_image(img) for img in ref_images]
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=images if len(images) > 1 else images[0],
        ip_adapter_image_embeds=None,
        device=_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return embeds

def generate_with_embeds(prompt: str, embeds, steps: int = 30, guidance: float = 5.0, seed: int | None = None, size=(1024,1024)):
    pipe = load_pipeline()
    g = None if seed is None else torch.Generator(device=_device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        ip_adapter_image_embeds=embeds,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
        width=size[0],
        height=size[1],
        negative_prompt="deformed, bad anatomy, lowres, text, watermark"
    ).images[0]
    return image
