# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the TransformerSVC model forward pass.

Tests verify that:
- A ConditionEncoder and Transformer can be instantiated with a minimal CPU config
- The combined forward pass (condition_encoder → acoustic_mapper) runs without errors
- The output mel tensor has the correct shape [batch, T, n_mel]
- Forward pass is deterministic in eval mode
- All output values are finite (no NaN or Inf)

These tests are marked @pytest.mark.integration and require only CPU.
No GPU or pretrained weights are needed.
"""

import pytest
import torch

from utils.util import JsonHParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_MEL = 20
_CONTENTVEC_DIM = 32
_HIDDEN_DIM = 64  # content_encoder_dim == output_singer_dim for "add" merge


def _build_svc_transformer_cfg(n_mel: int = _N_MEL):
    """Return minimal (condition_encoder_cfg, transformer_cfg) for testing.

    Uses only ContentVec content features and a speaker-ID embedding so the
    tests run without any pretrained model checkpoints.  Both encoder and
    transformer are tiny (hidden=64, 2 layers) to keep CPU runtime short.

    The ``merge_mode="add"`` strategy requires every encoder output to share
    the same dimensionality, which is satisfied by setting both
    ``content_encoder_dim`` and ``output_singer_dim`` to ``_HIDDEN_DIM``.
    """
    condition_encoder_cfg = JsonHParams(
        # Content features — ContentVec only
        use_whisper=False,
        use_contentvec=True,
        use_mert=False,
        use_wenet=False,
        contentvec_dim=_CONTENTVEC_DIM,
        content_encoder_dim=_HIDDEN_DIM,
        # Prosody features — disabled for minimal test
        use_f0=False,
        use_energy=False,
        use_uv=False,
        # Speaker identity
        use_spkid=True,
        output_singer_dim=_HIDDEN_DIM,
        singer_table_size=16,
        # Merge strategy
        merge_mode="add",
    )

    transformer_cfg = JsonHParams(
        type="transformer",
        input_dim=_HIDDEN_DIM,
        output_dim=n_mel,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
    )

    return condition_encoder_cfg, transformer_cfg


def _make_svc_batch(
    batch_size: int,
    seq_len: int,
    contentvec_dim: int = _CONTENTVEC_DIM,
    n_mel: int = _N_MEL,
) -> dict:
    """Create a synthetic SVC batch for TransformerSVC integration tests.

    Args:
        batch_size: Number of samples in the batch.
        seq_len: Number of acoustic frames per sample.
        contentvec_dim: Dimensionality of the ContentVec feature vectors.
        n_mel: Number of mel frequency bins in the target spectrogram.

    Returns:
        A dict with the keys consumed by ConditionEncoder and used as mel
        target during loss computation (``mask`` and ``mel``).
    """
    return {
        # ContentVec features: (B, T, contentvec_dim)
        "contentvec_feat": torch.randn(batch_size, seq_len, contentvec_dim),
        # Speaker IDs: (B, 1) — all zeros (single speaker)
        "spk_id": torch.zeros(batch_size, 1, dtype=torch.long),
        # Sequence lengths for masking
        "target_len": torch.full((batch_size,), seq_len, dtype=torch.long),
        # Frame-level mask: (B, T, 1)
        "mask": torch.ones(batch_size, seq_len, 1),
        # Target mel spectrogram: (B, T, n_mel)
        "mel": torch.randn(batch_size, seq_len, n_mel),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def svc_transformer_cfg():
    """Return minimal (condition_encoder_cfg, transformer_cfg) for SVC tests."""
    return _build_svc_transformer_cfg(n_mel=_N_MEL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTransformerSVCForwardPass:
    """Integration tests for the TransformerSVC model forward pass on CPU."""

    def test_condition_encoder_instantiation(self, svc_transformer_cfg):
        """ConditionEncoder should instantiate without errors given a minimal config."""
        from modules.encoder.condition_encoder import ConditionEncoder

        cond_cfg, _ = svc_transformer_cfg
        encoder = ConditionEncoder(cond_cfg)
        assert encoder is not None

    def test_transformer_instantiation(self, svc_transformer_cfg):
        """Transformer should instantiate without errors given a minimal config."""
        from models.svc.transformer.transformer import Transformer

        _, tr_cfg = svc_transformer_cfg
        model = Transformer(tr_cfg)
        assert model is not None

    def test_condition_encoder_output_shape(self, svc_transformer_cfg):
        """ConditionEncoder output must have shape [batch, T, hidden_dim]."""
        from modules.encoder.condition_encoder import ConditionEncoder

        cond_cfg, tr_cfg = svc_transformer_cfg
        encoder = ConditionEncoder(cond_cfg)
        encoder.eval()

        batch_size, seq_len = 2, 20
        batch = _make_svc_batch(batch_size, seq_len)

        with torch.no_grad():
            condition = encoder(batch)

        expected_shape = (batch_size, seq_len, tr_cfg.input_dim)
        assert condition.shape == expected_shape, (
            f"Expected ConditionEncoder output shape {expected_shape}, "
            f"got {condition.shape}"
        )

    def test_forward_output_shape(self, svc_transformer_cfg):
        """Combined forward pass must produce output of shape [batch, T, n_mel]."""
        from modules.encoder.condition_encoder import ConditionEncoder
        from models.svc.transformer.transformer import Transformer

        cond_cfg, tr_cfg = svc_transformer_cfg
        n_mel = tr_cfg.output_dim

        condition_encoder = ConditionEncoder(cond_cfg)
        acoustic_mapper = Transformer(tr_cfg)
        condition_encoder.eval()
        acoustic_mapper.eval()

        batch_size, seq_len = 2, 20
        batch = _make_svc_batch(batch_size, seq_len, n_mel=n_mel)

        with torch.no_grad():
            condition = condition_encoder(batch)
            output = acoustic_mapper(condition, mask=batch["mask"])

        assert output.ndim == 3, (
            f"Expected 3-D output [batch, T, n_mel], got {output.ndim}-D"
        )
        assert output.shape == (batch_size, seq_len, n_mel), (
            f"Expected shape ({batch_size}, {seq_len}, {n_mel}), "
            f"got {output.shape}"
        )

    def test_forward_is_deterministic_in_eval_mode(self, svc_transformer_cfg):
        """Two forward passes in eval mode with the same input must be identical."""
        from modules.encoder.condition_encoder import ConditionEncoder
        from models.svc.transformer.transformer import Transformer

        cond_cfg, tr_cfg = svc_transformer_cfg
        n_mel = tr_cfg.output_dim

        condition_encoder = ConditionEncoder(cond_cfg)
        acoustic_mapper = Transformer(tr_cfg)
        condition_encoder.eval()
        acoustic_mapper.eval()

        batch = _make_svc_batch(1, 10, n_mel=n_mel)

        with torch.no_grad():
            cond_a = condition_encoder(batch)
            out_a = acoustic_mapper(cond_a)
            cond_b = condition_encoder(batch)
            out_b = acoustic_mapper(cond_b)

        assert torch.allclose(out_a, out_b), (
            "TransformerSVC forward pass is non-deterministic in eval mode"
        )

    def test_output_is_finite(self, svc_transformer_cfg):
        """All elements of the model output must be finite (no NaN or Inf)."""
        from modules.encoder.condition_encoder import ConditionEncoder
        from models.svc.transformer.transformer import Transformer

        cond_cfg, tr_cfg = svc_transformer_cfg
        n_mel = tr_cfg.output_dim

        condition_encoder = ConditionEncoder(cond_cfg)
        acoustic_mapper = Transformer(tr_cfg)
        condition_encoder.eval()
        acoustic_mapper.eval()

        batch = _make_svc_batch(2, 15, n_mel=n_mel)

        with torch.no_grad():
            condition = condition_encoder(batch)
            output = acoustic_mapper(condition)

        assert torch.isfinite(output).all(), (
            "TransformerSVC output contains NaN or Inf values"
        )

    def test_batch_size_independence(self, svc_transformer_cfg):
        """Model should handle different batch sizes without errors."""
        from modules.encoder.condition_encoder import ConditionEncoder
        from models.svc.transformer.transformer import Transformer

        cond_cfg, tr_cfg = svc_transformer_cfg
        n_mel = tr_cfg.output_dim

        condition_encoder = ConditionEncoder(cond_cfg)
        acoustic_mapper = Transformer(tr_cfg)
        condition_encoder.eval()
        acoustic_mapper.eval()

        for batch_size in [1, 3, 4]:
            batch = _make_svc_batch(batch_size, 20, n_mel=n_mel)
            with torch.no_grad():
                condition = condition_encoder(batch)
                output = acoustic_mapper(condition)

            assert output.shape[0] == batch_size, (
                f"Batch dimension mismatch for batch_size={batch_size}: "
                f"got {output.shape[0]}"
            )

    def test_variable_sequence_lengths(self, svc_transformer_cfg):
        """Model should produce correct output for various sequence lengths."""
        from modules.encoder.condition_encoder import ConditionEncoder
        from models.svc.transformer.transformer import Transformer

        cond_cfg, tr_cfg = svc_transformer_cfg
        n_mel = tr_cfg.output_dim

        condition_encoder = ConditionEncoder(cond_cfg)
        acoustic_mapper = Transformer(tr_cfg)
        condition_encoder.eval()
        acoustic_mapper.eval()

        for seq_len in [5, 10, 50]:
            batch = _make_svc_batch(2, seq_len, n_mel=n_mel)
            with torch.no_grad():
                condition = condition_encoder(batch)
                output = acoustic_mapper(condition)

            assert output.shape == (2, seq_len, n_mel), (
                f"Unexpected output shape for seq_len={seq_len}: {output.shape}"
            )
