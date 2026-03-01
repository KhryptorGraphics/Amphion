# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the FastSpeech2 TTS model forward pass.

Tests verify that:
- A FastSpeech2 model can be instantiated with a minimal CPU config
- The model forward pass executes without errors
- The output mel tensor has the correct shape [batch, T_mel, n_mel]
- Both pre-postnet and postnet outputs are produced

These tests are marked @pytest.mark.integration and require only CPU.
No GPU or pretrained weights are needed.
"""

import json
import os

import pytest
import torch

from utils.util import JsonHParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_stats_json(path: str, dataset: str) -> None:
    """Write a minimal pitch/energy statistics JSON expected by VarianceAdaptor."""
    stats = {
        f"{dataset}_{dataset}": {
            "voiced_positions": {"mean": 0.0, "std": 1.0},
            "total_positions": {"min": -3.0, "max": 3.0},
        }
    }
    with open(path, "w") as f:
        json.dump(stats, f)


def _build_fs2_cfg(processed_dir: str, n_mel: int = 20) -> JsonHParams:
    """Return a minimal FastSpeech2 JsonHParams config for testing.

    Encoder/decoder are tiny (hidden=64, 2 layers each) so the test
    runs quickly on CPU.  The statistics directories are created by the
    fs2_config fixture before this function is called.
    """
    return JsonHParams(
        dataset=["TestSet"],
        preprocess=dict(
            processed_dir=processed_dir,
            n_mel=n_mel,
            use_frame_pitch=False,
            use_frame_energy=False,
            # phoneme-level feature directories
            phone_pitch_dir="phone_pitches",
            phone_energy_dir="phone_energys",
            # frame-level feature directories (unused when use_frame_* is False)
            pitch_dir="pitches",
            energy_dir="energys",
        ),
        model=dict(
            max_seq_len=200,
            transformer=dict(
                encoder_layer=2,
                encoder_head=2,
                encoder_hidden=64,
                decoder_layer=2,
                decoder_head=2,
                decoder_hidden=64,
                conv_filter_size=128,
                conv_kernel_size=[9, 1],
                encoder_dropout=0.0,
                decoder_dropout=0.0,
            ),
            variance_predictor=dict(
                filter_size=64,
                kernel_size=3,
                dropout=0.0,
            ),
            variance_embedding=dict(
                pitch_quantization="linear",
                energy_quantization="linear",
                n_bins=16,
            ),
        ),
        train=dict(
            multi_speaker_training=False,
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs2_config(tmp_path):
    """Create stats files and return a minimal FastSpeech2 config.

    The VarianceAdaptor reads pitch/energy statistics from disk during
    __init__; this fixture writes the required JSON files to a temp
    directory so the model can be instantiated without real data.
    """
    dataset = "TestSet"

    # Energy statistics (phoneme-level)
    energy_dir = tmp_path / dataset / "phone_energys"
    energy_dir.mkdir(parents=True)
    _write_stats_json(str(energy_dir / "statistics.json"), dataset)

    # Pitch statistics (phoneme-level)
    pitch_dir = tmp_path / dataset / "phone_pitches"
    pitch_dir.mkdir(parents=True)
    _write_stats_json(str(pitch_dir / "statistics.json"), dataset)

    return _build_fs2_cfg(str(tmp_path), n_mel=20)


def _make_synthetic_batch(
    batch_size: int,
    n_phones: int,
    duration_per_phone: int,
    n_mel: int,
) -> dict:
    """Create a synthetic training-style batch for FastSpeech2.

    Args:
        batch_size: Number of samples in the batch.
        n_phones: Phoneme sequence length (same for all samples).
        duration_per_phone: How many mel frames each phone expands to.
        n_mel: Number of mel frequency bins.

    Returns:
        A dict with keys matching what FastSpeech2.forward() expects.
    """
    n_mel_frames = n_phones * duration_per_phone

    # Phone IDs — keep well within the 152-token vocabulary
    texts = torch.randint(1, 50, (batch_size, n_phones))
    src_lens = torch.full((batch_size,), n_phones, dtype=torch.long)

    # Ground-truth durations (integer, one per phone)
    durations = torch.full(
        (batch_size, n_phones), duration_per_phone, dtype=torch.long
    )

    mel_lens = torch.full((batch_size,), n_mel_frames, dtype=torch.long)
    mel_targets = torch.randn(batch_size, n_mel_frames, n_mel)

    # Phoneme-level pitch and energy targets (normalised)
    pitch_targets = torch.zeros(batch_size, n_phones)
    energy_targets = torch.zeros(batch_size, n_phones)

    return {
        "spk_id": torch.zeros(batch_size, dtype=torch.long),
        "texts": texts,
        "text_len": src_lens,
        "target_len": mel_lens,
        "mel": mel_targets,
        "pitch": pitch_targets,
        "energy": energy_targets,
        "durations": durations,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFastSpeech2ModelForwardPass:
    """Integration tests for FastSpeech2 model forward pass on CPU."""

    def test_model_instantiation(self, fs2_config):
        """FastSpeech2 should instantiate without errors given a minimal config."""
        from models.tts.fastspeech2.fs2 import FastSpeech2

        model = FastSpeech2(fs2_config)
        assert model is not None

    def test_forward_output_keys(self, fs2_config):
        """Forward pass must return a dict with the expected output keys."""
        from models.tts.fastspeech2.fs2 import FastSpeech2

        model = FastSpeech2(fs2_config)
        model.eval()

        batch = _make_synthetic_batch(
            batch_size=2,
            n_phones=6,
            duration_per_phone=4,
            n_mel=fs2_config.preprocess.n_mel,
        )

        with torch.no_grad():
            outputs = model(batch)

        expected_keys = {
            "output",
            "postnet_output",
            "p_predictions",
            "e_predictions",
            "log_d_predictions",
            "d_rounded",
            "src_masks",
            "mel_masks",
            "src_lens",
            "mel_lens",
        }
        assert expected_keys.issubset(
            outputs.keys()
        ), f"Missing keys: {expected_keys - outputs.keys()}"

    def test_forward_output_mel_shape(self, fs2_config):
        """Output mel tensor must have shape [batch, T_mel, n_mel]."""
        from models.tts.fastspeech2.fs2 import FastSpeech2

        model = FastSpeech2(fs2_config)
        model.eval()

        batch_size = 2
        n_phones = 5
        duration_per_phone = 3
        n_mel = fs2_config.preprocess.n_mel
        n_mel_frames = n_phones * duration_per_phone

        batch = _make_synthetic_batch(
            batch_size=batch_size,
            n_phones=n_phones,
            duration_per_phone=duration_per_phone,
            n_mel=n_mel,
        )

        with torch.no_grad():
            outputs = model(batch)

        output = outputs["output"]

        assert output.ndim == 3, (
            f"Expected 3-D mel tensor [batch, T, n_mel], got {output.ndim}-D"
        )
        assert output.shape[0] == batch_size, (
            f"Expected batch dimension {batch_size}, got {output.shape[0]}"
        )
        assert output.shape[1] == n_mel_frames, (
            f"Expected time dimension {n_mel_frames}, got {output.shape[1]}"
        )
        assert output.shape[2] == n_mel, (
            f"Expected n_mel={n_mel}, got {output.shape[2]}"
        )

    def test_postnet_output_matches_output_shape(self, fs2_config):
        """PostNet output must have the same shape as the pre-PostNet mel output."""
        from models.tts.fastspeech2.fs2 import FastSpeech2

        model = FastSpeech2(fs2_config)
        model.eval()

        batch = _make_synthetic_batch(
            batch_size=2,
            n_phones=4,
            duration_per_phone=5,
            n_mel=fs2_config.preprocess.n_mel,
        )

        with torch.no_grad():
            outputs = model(batch)

        output = outputs["output"]
        postnet_output = outputs["postnet_output"]

        assert postnet_output.shape == output.shape, (
            f"PostNet output shape {postnet_output.shape} must match "
            f"mel output shape {output.shape}"
        )

    def test_forward_is_deterministic_in_eval_mode(self, fs2_config):
        """Two forward passes in eval mode with the same input must be identical."""
        from models.tts.fastspeech2.fs2 import FastSpeech2

        model = FastSpeech2(fs2_config)
        model.eval()

        batch = _make_synthetic_batch(
            batch_size=1,
            n_phones=4,
            duration_per_phone=3,
            n_mel=fs2_config.preprocess.n_mel,
        )

        with torch.no_grad():
            outputs_a = model(batch)
            outputs_b = model(batch)

        assert torch.allclose(outputs_a["output"], outputs_b["output"]), (
            "Forward pass is non-deterministic in eval mode"
        )

    def test_output_is_finite(self, fs2_config):
        """All elements of the mel output must be finite (no NaN or Inf)."""
        from models.tts.fastspeech2.fs2 import FastSpeech2

        model = FastSpeech2(fs2_config)
        model.eval()

        batch = _make_synthetic_batch(
            batch_size=2,
            n_phones=5,
            duration_per_phone=3,
            n_mel=fs2_config.preprocess.n_mel,
        )

        with torch.no_grad():
            outputs = model(batch)

        output = outputs["output"]
        assert torch.isfinite(output).all(), (
            "Mel output contains NaN or Inf values"
        )
