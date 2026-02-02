"""
Standalone Hubert feature extractor using torchaudio.

This module provides a Hubert implementation using torchaudio's bundled model,
avoiding the fairseq dependency. It includes KMeans quantization for Noro.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import joblib
import numpy as np
from einops import repeat
from typing import Tuple


class TorchaudioHubertExtractor(nn.Module):
    """
    Hubert feature extractor using torchaudio's bundled HUBERT_LARGE model.

    Provides feature extraction and KMeans quantization compatible with Noro.
    """

    def __init__(self, kmeans_model_path: str = None, device: str = "cuda"):
        """
        Initialize the Hubert extractor.

        Args:
            kmeans_model_path: Path to joblib-saved KMeans model for quantization.
                             If None, only features are returned (no quantization).
            device: Device to load model on.
        """
        super().__init__()
        self.device = device

        # Load torchaudio's Hubert model (BASE for 768-dim output matching KMeans)
        print("Loading Hubert BASE model from torchaudio...")
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()
        self.model.to(device)
        self.sample_rate = bundle.sample_rate  # 16000

        # Load KMeans quantizer if provided
        self.kmeans = None
        self.cluster_centers = None
        if kmeans_model_path is not None:
            print(f"Loading KMeans model from {kmeans_model_path}...")
            self.kmeans = joblib.load(kmeans_model_path)
            self.register_buffer(
                "cluster_centers_",
                torch.from_numpy(self.kmeans.cluster_centers_).float()
            )

    def extract_features(self, wav_input: torch.Tensor, layer: int = 9) -> torch.Tensor:
        """
        Extract Hubert features from audio.

        Args:
            wav_input: Audio tensor [B, T] at 16kHz
            layer: Which layer's features to extract (default 9 for Noro)

        Returns:
            features: Tensor [B, T', D] where T' depends on Hubert's downsampling
        """
        # Pad input for proper convolution
        wav_input = F.pad(wav_input, (40, 40), "reflect")

        with torch.no_grad():
            # Get features from specified layer
            features, _ = self.model.extract_features(wav_input)
            # features is a list of tensors from each layer
            embed = features[layer]  # [B, T', 1024]

        return embed

    def quantize(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize features using KMeans clustering.

        Args:
            features: Tensor [B, T, D] from Hubert

        Returns:
            clusters: Tensor [B, T] of cluster indices
            quantized: Tensor [B, T, D] of quantized features
        """
        if self.cluster_centers_ is None:
            raise ValueError("KMeans model not loaded. Provide kmeans_model_path.")

        # Get cluster centers on same device as features
        centers = self.cluster_centers_.to(features.device)

        # Compute distances and get cluster assignments
        batched_centers = repeat(centers, "c d -> b c d", b=features.shape[0])
        dists = -torch.cdist(features, batched_centers, p=2)
        clusters = dists.argmax(dim=-1)  # [B, T]

        # Get quantized features
        quantized = F.embedding(clusters, centers)

        return clusters, quantized

    def extract_content_features(self, wav_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and quantize content features (compatible with Noro HubertExtractor API).

        Args:
            wav_input: Audio tensor [B, T] at 16kHz

        Returns:
            clusters: Tensor [B, T'] of cluster indices
            quantized: Tensor [B, T', D] of quantized features
        """
        # Extract features
        features = self.extract_features(wav_input, layer=9)

        # Interpolate to match Noro's expected hop size
        # Hubert uses hop_size=320, Noro expects hop_size=200 (factor of 1.6)
        features = features.permute(0, 2, 1)  # [B, D, T]
        features = F.interpolate(features, scale_factor=1.6, mode="nearest")
        features = features.permute(0, 2, 1)  # [B, T, D]

        # Quantize
        clusters, quantized = self.quantize(features)

        return clusters, quantized


def load_hubert_extractor(kmeans_model_path: str, device: str = "cuda") -> TorchaudioHubertExtractor:
    """
    Load a Hubert extractor with KMeans quantization.

    Args:
        kmeans_model_path: Path to KMeans model (.bin file)
        device: Device to load on

    Returns:
        TorchaudioHubertExtractor instance
    """
    model = TorchaudioHubertExtractor(kmeans_model_path=kmeans_model_path, device=device)
    model.train(False)  # Set to inference mode
    return model
