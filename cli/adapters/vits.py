# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI adapter for VITS – variational TTS requiring a local checkpoint."""

import argparse
import os

from cli.adapters.base import ModelAdapter


class VITSAdapter(ModelAdapter):
    """Adapter wrapping :class:`~models.tts.vits.vits_inference.VitsInference`
    for ``amphion infer``.

    Requires a locally trained checkpoint and experiment config (produced by
    ``sh egs/tts/vits/run.sh --stage 2 ...``).  No automatic weight download
    is performed – point ``--acoustics-dir`` or ``--checkpoint-path`` at a
    trained experiment directory.
    """

    # ------------------------------------------------------------------
    # Identity properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "vits"

    @property
    def description(self) -> str:
        return "VITS – variational TTS (requires local checkpoint)"

    @property
    def task_type(self) -> str:
        return "tts"

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register VITS-specific CLI arguments with *parser*."""
        parser.add_argument(
            "--text",
            required=True,
            metavar="TEXT",
            help="Text to synthesise.",
        )
        parser.add_argument(
            "--config",
            required=True,
            metavar="PATH",
            help=(
                "Path to the experiment config JSON file "
                "(e.g. egs/tts/vits/exp_config.json)."
            ),
        )

        ckpt_group = parser.add_mutually_exclusive_group(required=True)
        ckpt_group.add_argument(
            "--checkpoint-path",
            metavar="PATH",
            help=(
                "Path to a specific checkpoint step directory "
                "(e.g. ckpts/my_exp/checkpoint/step-000100000_loss-…)."
            ),
        )
        ckpt_group.add_argument(
            "--acoustics-dir",
            metavar="DIR",
            help=(
                "Experiment directory containing a 'checkpoint/' sub-directory.  "
                "The latest checkpoint is loaded automatically."
            ),
        )

        parser.add_argument(
            "--speaker-name",
            default=None,
            metavar="NAME",
            help=(
                "Speaker name for multi-speaker models.  "
                "Required when the model was trained with speaker IDs."
            ),
        )
        parser.add_argument(
            "--noise-scale",
            type=float,
            default=0.667,
            metavar="S",
            help="Noise scale for the flow (default: 0.667).",
        )
        parser.add_argument(
            "--noise-scale-w",
            type=float,
            default=0.8,
            metavar="S",
            help=(
                "Noise scale for the stochastic duration predictor (default: 0.8)."
            ),
        )
        parser.add_argument(
            "--length-scale",
            type=float,
            default=1.0,
            metavar="S",
            help=(
                "Length scale controlling speaking rate "
                "(default: 1.0; higher values produce slower speech)."
            ),
        )

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    def run(self, args: argparse.Namespace) -> None:
        """Load a local VITS checkpoint and synthesise speech from *args.text*.

        Runs in single-utterance mode, writing the synthesised audio to the
        path specified by ``args.output``.

        Args:
            args: Parsed namespace.  Expected attributes (beyond the shared
                ``--output`` / ``--device``):

                * ``text``            – target synthesis text
                * ``config``          – path to the experiment config JSON
                * ``checkpoint_path`` – specific checkpoint directory, or ``None``
                * ``acoustics_dir``   – experiment root directory, or ``None``
                * ``speaker_name``    – speaker name for multi-speaker models, or ``None``
                * ``noise_scale``     – flow noise scale
                * ``noise_scale_w``   – duration predictor noise scale
                * ``length_scale``    – speaking rate multiplier
        """
        import shutil
        import tempfile

        from models.tts.vits.vits_inference import VitsInference
        from utils.util import load_config

        cfg = load_config(args.config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build the synthetic args namespace expected by TTSInference.
            infer_args = argparse.Namespace(
                mode="single",
                text=args.text,
                acoustics_dir=getattr(args, "acoustics_dir", None),
                checkpoint_path=getattr(args, "checkpoint_path", None),
                output_dir=tmp_dir,
                vocoder_dir=None,
                speaker_name=getattr(args, "speaker_name", None),
                log_level="warning",
                dataset=None,
                testing_set="test",
                test_list_file=None,
                pitch_control=1.0,
                energy_control=1.0,
                duration_control=1.0,
            )

            inferencer = VitsInference(infer_args, cfg)
            inferencer.inference()

            generated = os.path.join(tmp_dir, "single", "test_pred.wav")
            if not os.path.exists(generated):
                raise RuntimeError(
                    "VITS inference completed but expected output file was not "
                    f"found at: {generated}"
                )

            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(generated, args.output)
