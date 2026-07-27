#!/usr/bin/env python3
"""
Model runner for H100 profiling workload characterization.
Supports vision, language, and video generation models.

Usage:
    python run_model.py --model <model_name> --batch <batch_size>

Models:
    Vision/Diffusion: unet-sd, sdxl, dit-xl, unet-3d
    Language: llama-8b-1k, llama-8b-4k, llama-8b-16k, llama-8b-64k
    Dense: resnet-50
    Video: cogvideox-16f-480p, cogvideox-49f-480p, cogvideox-81f-480p, cogvideox-49f-720p
"""

import torch
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DiTPipeline,
    CogVideoXPipeline
)
import torchvision.models as models

def run_stable_diffusion(batch_size):
    """UNet-SD: Stable Diffusion v1.5"""
    print("  Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    prompt = ["a photo of an astronaut riding a horse on mars"] * batch_size

    print(f"  Generating {batch_size} images...")
    with torch.no_grad():
        images = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images

    torch.cuda.synchronize()
    return images

def run_sdxl(batch_size):
    """SDXL: Stable Diffusion XL"""
    print("  Loading Stable Diffusion XL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    prompt = ["a professional photo of an astronaut riding a horse"] * batch_size

    print(f"  Generating {batch_size} images...")
    with torch.no_grad():
        images = pipe(prompt, num_inference_steps=20).images

    torch.cuda.synchronize()
    return images

def run_dit_xl(batch_size):
    """DiT-XL: Diffusion Transformer XL"""
    print("  Loading DiT-XL...")
    pipe = DiTPipeline.from_pretrained(
        "facebook/DiT-XL-2-256",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print(f"  Generating {batch_size} images...")
    # DiT uses class labels, not text prompts
    with torch.no_grad():
        images = pipe(
            batch_size=batch_size,
            num_inference_steps=20
        ).images

    torch.cuda.synchronize()
    return images

def run_unet_3d(batch_size):
    """UNet-3D: 3D video UNet (using small video model as proxy)"""
    print("  Loading 3D UNet (via CogVideoX-2b)...")
    # Using CogVideoX as 3D UNet example with short video
    return run_cogvideox(batch_size, num_frames=16, height=480, width=720, model_id="THUDM/CogVideoX-2b")

def run_llama(batch_size, seq_length=1024):
    """Llama-8B at various sequence lengths (prefill only)"""
    model_id = "meta-llama/Meta-Llama-3-8B"  # Change to "meta-llama/Llama-2-7b-hf" if access issues

    print(f"  Loading Llama-8B for seq_len={seq_length}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Generate input of specific sequence length
    # Use a repeated text to reach desired length
    base_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for language model profiling. "
    text = base_text * (seq_length // len(tokenizer.encode(base_text)) + 1)

    inputs = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_length
    ).to("cuda")

    actual_len = inputs.input_ids.shape[1]
    print(f"  Input shape: {inputs.input_ids.shape} (target={seq_length}, actual={actual_len})")

    # Run prefill (one forward pass, no generation)
    with torch.no_grad():
        outputs = model(**inputs)

    torch.cuda.synchronize()
    return outputs

def run_resnet50(batch_size):
    """ResNet-50: Pure dense/conv anchor (no attention)"""
    print("  Loading ResNet-50...")
    model = models.resnet50(weights="IMAGENET1K_V1").to("cuda").eval()
    model = model.to(torch.bfloat16)

    # Random input images (batch, 3, 224, 224)
    print(f"  Generating {batch_size} random 224×224 images...")
    inputs = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16).to("cuda")

    with torch.no_grad():
        outputs = model(inputs)

    torch.cuda.synchronize()
    return outputs

def run_cogvideox(batch_size, num_frames=49, height=480, width=720, model_id="THUDM/CogVideoX-2b"):
    """
    CogVideoX: Video diffusion generation

    Args:
        num_frames: Temporal sequence length (16, 49, or 81)
        height, width: Spatial resolution (480×720 or 720×1280)
    """
    print(f"  Loading CogVideoX ({model_id})...")
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    prompt = ["A cat playing piano in a cozy room with warm lighting"] * batch_size

    spatial_tokens_approx = (height * width) // 64  # Rough estimate with patch_size=8
    print(f"  Video config:")
    print(f"    Frames (temporal): {num_frames}")
    print(f"    Resolution (spatial): {height}×{width}")
    print(f"    Approx spatial tokens/frame: ~{spatial_tokens_approx}")
    print(f"    Total spatiotemporal tokens: ~{num_frames * spatial_tokens_approx}")

    print(f"  Generating {batch_size} videos...")
    with torch.no_grad():
        videos = pipe(
            prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=20
        ).frames

    torch.cuda.synchronize()
    return videos

# Model dispatch table
MODEL_RUNNERS = {
    # Vision/Diffusion
    "unet-sd": lambda b: run_stable_diffusion(b),
    "sdxl": lambda b: run_sdxl(b),
    "dit-xl": lambda b: run_dit_xl(b),
    "unet-3d": lambda b: run_unet_3d(b),

    # Language (Llama-8B at different sequence lengths)
    "llama-8b-1k": lambda b: run_llama(b, seq_length=1024),
    "llama-8b-4k": lambda b: run_llama(b, seq_length=4096),
    "llama-8b-16k": lambda b: run_llama(b, seq_length=16384),
    "llama-8b-64k": lambda b: run_llama(b, seq_length=65536),

    # Dense anchor
    "resnet-50": lambda b: run_resnet50(b),

    # Video generation (diffusion-based, fixed frame count)
    "cogvideox-16f-480p": lambda b: run_cogvideox(b, num_frames=16, height=480, width=720),
    "cogvideox-49f-480p": lambda b: run_cogvideox(b, num_frames=49, height=480, width=720),
    "cogvideox-81f-480p": lambda b: run_cogvideox(b, num_frames=81, height=480, width=720),
    "cogvideox-49f-720p": lambda b: run_cogvideox(b, num_frames=49, height=720, width=1280),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run models for H100 profiling workload characterization"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_RUNNERS.keys(),
        help="Model to run"
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Batch size"
    )
    args = parser.parse_args()

    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}")
    print("="*60)

    start = time.time()

    try:
        result = MODEL_RUNNERS[args.model](args.batch)
        torch.cuda.synchronize()
        end = time.time()

        print("="*60)
        print(f"✓ Success!")
        print(f"  Elapsed: {end-start:.2f}s")
        print("="*60)

    except torch.cuda.OutOfMemoryError as e:
        print("="*60)
        print(f"✗ CUDA Out of Memory!")
        print(f"  Model: {args.model}, Batch: {args.batch}")
        print(f"  Try reducing batch size or using a larger GPU")
        print("="*60)
        raise
    except Exception as e:
        print("="*60)
        print(f"✗ Error: {e}")
        print("="*60)
        raise
