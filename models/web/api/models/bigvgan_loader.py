"""
BigVGAN loader for inference using the official NVIDIA package.

Uses the bigvgan pip package for proper model loading.
"""

import torch
import json
import os
from typing import Tuple


def load_bigvgan(checkpoint_dir: str, device: str = "cuda") -> Tuple:
    """
    Load BigVGAN from a HuggingFace checkpoint directory using the official package.

    Args:
        checkpoint_dir: Directory containing bigvgan_generator.pt and config.json
        device: Device to load model on

    Returns:
        Tuple of (loaded BigVGAN generator model, config dict)
    """
    import bigvgan

    config_path = os.path.join(checkpoint_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"BigVGAN config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # The bigvgan package can load directly from HuggingFace model ID
    # or from a local directory that has the same structure
    # We need to determine the model name from the directory
    dir_name = os.path.basename(checkpoint_dir)

    # Map directory names to HuggingFace model IDs
    model_map = {
        "bigvgan_22khz_80band": "nvidia/bigvgan_22khz_80band",
        "bigvgan_v2_22khz_80band_256x": "nvidia/bigvgan_v2_22khz_80band_256x",
        "bigvgan_24khz_100band": "nvidia/bigvgan_24khz_100band",
        "bigvgan_v2_24khz_100band_256x": "nvidia/bigvgan_v2_24khz_100band_256x",
    }

    model_id = model_map.get(dir_name)

    if model_id:
        # Load from HuggingFace (will use cache if already downloaded)
        model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False)
    else:
        # Try loading from local directory directly
        model = bigvgan.BigVGAN.from_pretrained(checkpoint_dir, use_cuda_kernel=False)

    model = model.to(device)
    model.train(False)

    return model, config
