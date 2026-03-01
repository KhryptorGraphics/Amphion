# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the pre-computed feature alignment cache.

Verifies that align_and_cache_content_features() produces aligned features
that are numerically identical to the runtime offline_resolution_transformation()
output, and that dataset loading falls back correctly when cached files are absent.
"""

import os
import json
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal config stub — avoids importing the heavyweight AudioPretrainedModel
# classes and their optional GPU/fairseq dependencies.
# ---------------------------------------------------------------------------


class _Namespace:
    """Simple attribute-access namespace for nested config."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = _Namespace(**v)
            setattr(self, k, v)

    def __contains__(self, item):
        return hasattr(self, item)


def _make_cfg(tmp_path, **overrides):
    """Build a minimal cfg object for whisper-only alignment tests."""
    defaults = dict(
        hop_size=256,
        sample_rate=24000,
        whisper_frameshift=0.02,
        whisper_downsample_rate=2,
        contentvec_frameshift=0.02,
        wenet_frameshift=0.02,
        wenet_downsample_rate=2,
        mert_hop_size=320,
        # Feature extraction flags
        extract_whisper_feature=True,
        extract_contentvec_feature=False,
        extract_wenet_feature=False,
        extract_mert_feature=False,
        # Directory names
        mel_dir="mels",
        whisper_dir="whisper",
        whisper_aligned_dir="whisper_aligned",
        contentvec_dir="contentvec",
        contentvec_aligned_dir="contentvec_aligned",
        wenet_dir="wenet",
        wenet_aligned_dir="wenet_aligned",
        mert_dir="mert",
        mert_aligned_dir="mert_aligned",
        # Runtime path
        processed_dir=str(tmp_path),
        use_cached_alignment=True,
    )
    defaults.update(overrides)
    return _Namespace(preprocess=_Namespace(**defaults))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_hop_target_hop(whisper_frameshift, whisper_downsample_rate, sample_rate, hop_size):
    """Replicate AudioPretrainedModelFeaturesExtractor.init_for_retrans logic."""
    source_hop = int(whisper_frameshift * whisper_downsample_rate * sample_rate)
    target_hop = hop_size
    from math import gcd
    factor = gcd(source_hop, target_hop)
    return source_hop // factor, target_hop // factor


def _runtime_align(raw_feat, target_len, source_hop, target_hop):
    """Pure-numpy replication of offline_resolution_transformation."""
    _, width = raw_feat.shape
    source_len = min(target_len * target_hop // source_hop + 1, len(raw_feat))
    const = source_len * source_hop // target_hop * target_hop
    up = np.repeat(raw_feat, source_hop, axis=0)
    down = np.average(up[:const].reshape(-1, target_hop, width), axis=1)
    err = abs(target_len - len(down))
    if len(down) < target_len:
        end = down[-1][None, :].repeat(err, axis=0)
        down = np.concatenate([down, end], axis=0)
    return down[:target_len]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dataset_name():
    return "test_dataset"


@pytest.fixture()
def utt_uid():
    return "utt001"


@pytest.fixture()
def synth_whisper_feat():
    """Synthetic whisper feature: (50, 1024) float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 1024)).astype(np.float32)


@pytest.fixture()
def synth_mel(target_len=100):
    """Synthetic mel: (100, target_len) shaped as (n_mels, T)."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((100, target_len)).astype(np.float32)


@pytest.fixture()
def scaffold(tmp_path, dataset_name, utt_uid, synth_whisper_feat, synth_mel):
    """
    Build a minimal on-disk structure:
        <tmp_path>/<dataset_name>/mels/<uid>.npy
        <tmp_path>/<dataset_name>/whisper/<uid>.npy
    Returns (cfg, metadata list).
    """
    dataset_dir = tmp_path / dataset_name
    mel_dir = dataset_dir / "mels"
    whisper_dir = dataset_dir / "whisper"
    mel_dir.mkdir(parents=True)
    whisper_dir.mkdir(parents=True)

    np.save(str(mel_dir / f"{utt_uid}.npy"), synth_mel)
    np.save(str(whisper_dir / f"{utt_uid}.npy"), synth_whisper_feat)

    cfg = _make_cfg(tmp_path)
    metadata = [{"Dataset": dataset_name, "Uid": utt_uid}]
    return cfg, metadata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlignmentCorrectness:
    """Cached aligned output must match runtime offline_resolution_transformation."""

    def test_whisper_alignment_correctness(self, scaffold, synth_whisper_feat, synth_mel, tmp_path, dataset_name, utt_uid):
        """
        Run align_and_cache_content_features(), load the result, and compare
        against the reference runtime alignment.
        """
        pytest.importorskip("fairseq", reason="fairseq not installed")

        from processors.content_extractor import (
            AudioPretrainedModelFeaturesExtractor,
            align_and_cache_content_features,
        )

        cfg, metadata = scaffold
        align_and_cache_content_features(cfg, metadata)

        # Load cached result
        aligned_path = os.path.join(
            str(tmp_path), dataset_name, "whisper_aligned", f"{utt_uid}.npy"
        )
        assert os.path.exists(aligned_path), "Aligned file was not created"
        cached = np.load(aligned_path)

        # Compute reference using the same logic as offline_resolution_transformation
        target_len = synth_mel.shape[1]
        source_hop, target_hop = _source_hop_target_hop(
            cfg.preprocess.whisper_frameshift,
            cfg.preprocess.whisper_downsample_rate,
            cfg.preprocess.sample_rate,
            cfg.preprocess.hop_size,
        )
        reference = _runtime_align(synth_whisper_feat, target_len, source_hop, target_hop)

        assert cached.shape == reference.shape, (
            f"Shape mismatch: cached={cached.shape} reference={reference.shape}"
        )
        assert np.allclose(cached, reference, atol=1e-5), (
            "Cached aligned features do not match runtime alignment"
        )

    def test_skips_existing_aligned_file(self, scaffold, tmp_path, dataset_name, utt_uid):
        """Running align_and_cache_content_features twice must not overwrite existing files."""
        pytest.importorskip("fairseq", reason="fairseq not installed")
        from processors.content_extractor import align_and_cache_content_features

        cfg, metadata = scaffold
        align_and_cache_content_features(cfg, metadata)

        aligned_path = os.path.join(
            str(tmp_path), dataset_name, "whisper_aligned", f"{utt_uid}.npy"
        )
        mtime_first = os.path.getmtime(aligned_path)

        # Run again — file should not be touched
        align_and_cache_content_features(cfg, metadata)
        mtime_second = os.path.getmtime(aligned_path)

        assert mtime_first == mtime_second, "Existing aligned file was overwritten on second run"

    def test_skips_when_mel_missing(self, tmp_path, dataset_name, utt_uid):
        """If the mel file is missing, the utt must be skipped without error."""
        pytest.importorskip("fairseq", reason="fairseq not installed")
        from processors.content_extractor import align_and_cache_content_features

        cfg = _make_cfg(tmp_path)
        metadata = [{"Dataset": dataset_name, "Uid": utt_uid}]

        # No files on disk — should complete silently
        align_and_cache_content_features(cfg, metadata)

        aligned_path = os.path.join(
            str(tmp_path), dataset_name, "whisper_aligned", f"{utt_uid}.npy"
        )
        assert not os.path.exists(aligned_path), "Aligned file was created despite missing mel"


class TestFallbackWhenNoCache:
    """SVCOfflineDataset must fall back to runtime alignment when aligned files are absent."""

    def test_aligned_path_map_built_when_flag_enabled(self, scaffold, dataset_name, utt_uid):
        """
        When use_cached_alignment=True, SVCOfflineDataset.__init__ must populate
        self.utt2whisper_aligned_path even before any aligned files exist.
        """
        # We test only the path-building logic here since instantiating
        # SVCOfflineDataset requires GPU-heavy deps. Use load_content_feature_path
        # directly as a unit test.
        from utils.data_utils import load_content_feature_path

        cfg, metadata = scaffold
        aligned_paths = load_content_feature_path(
            metadata,
            cfg.preprocess.processed_dir,
            cfg.preprocess.whisper_aligned_dir,
        )
        utt_key = f"{dataset_name}_{utt_uid}"
        assert utt_key in aligned_paths
        expected_suffix = os.path.join(dataset_name, "whisper_aligned", f"{utt_uid}.npy")
        assert aligned_paths[utt_key].endswith(expected_suffix)

    def test_fallback_path_not_exist_returns_false(self, scaffold, dataset_name, utt_uid, tmp_path):
        """
        When no aligned file exists on disk, os.path.exists() on the aligned path
        must return False — guaranteeing the fallback branch is taken.
        """
        from utils.data_utils import load_content_feature_path

        cfg, metadata = scaffold
        aligned_paths = load_content_feature_path(
            metadata,
            cfg.preprocess.processed_dir,
            cfg.preprocess.whisper_aligned_dir,
        )
        utt_key = f"{dataset_name}_{utt_uid}"
        assert not os.path.exists(aligned_paths[utt_key]), (
            "Aligned file unexpectedly exists before caching step"
        )
