# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI adapter for FastSpeech2 – non-autoregressive TTS requiring a local checkpoint."""

import argparse
import os

from cli.adapters.base import ModelAdapter


class FastSpeech2Adapter(ModelAdapter):
    """Adapter wrapping :class:`~models.tts.fastspeech2.fs2_inference.FastSpeech2Inference`
    for ``amphion infer``.

    Requires a locally trained checkpoint, experiment config, and a matching
    vocoder checkpoint (produced by the ``egs/tts/fastspeech2/run.sh`` recipe).
    No automatic weight download is performed – point ``--acoustics-dir`` or
    ``--checkpoint-path`` and ``--vocoder-dir`` at the appropriate directories.
    """

    # ------------------------------------------------------------------
    # Identity properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "fastspeech2"

    @property
    def description(self) -> str:
        return "FastSpeech2 – non-autoregressive TTS (requires local checkpoint)"

    @property
    def task_type(self) -> str:
        return "tts"

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register FastSpeech2-specific CLI arguments with *parser*."""
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
                "(e.g. egs/tts/fastspeech2/exp_config.json)."
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
            "--vocoder-dir",
            required=True,
            metavar="DIR",
            help=(
                "Directory containing a trained vocoder checkpoint (.pt file) "
                "and its args.json config.  FastSpeech2 predicts mel-spectrograms "
                "which are converted to waveforms by the vocoder."
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
            "--pitch-control",
            type=float,
            default=1.0,
            metavar="F",
            help=(
                "Multiplier for predicted pitch "
                "(default: 1.0; values >1 raise pitch, <1 lower it)."
            ),
        )
        parser.add_argument(
            "--energy-control",
            type=float,
            default=1.0,
            metavar="F",
            help=(
                "Multiplier for predicted energy / volume "
                "(default: 1.0; values >1 increase loudness)."
            ),
        )
        parser.add_argument(
            "--duration-control",
            type=float,
            default=1.0,
            metavar="F",
            help=(
                "Multiplier for predicted phoneme durations "
                "(default: 1.0; values >1 slow speech, <1 speed it up)."
            ),
        )

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    def run(self, args: argparse.Namespace) -> None:
        """Load a local FastSpeech2 checkpoint and synthesise speech from *args.text*.

        Runs in single-utterance mode, writing the synthesised audio to the
        path specified by ``args.output``.

        Args:
            args: Parsed namespace.  Expected attributes (beyond the shared
                ``--output`` / ``--device``):

                * ``text``             – target synthesis text
                * ``config``           – path to the experiment config JSON
                * ``checkpoint_path``  – specific checkpoint directory, or ``None``
                * ``acoustics_dir``    – experiment root directory, or ``None``
                * ``vocoder_dir``      – directory with the vocoder checkpoint
                * ``speaker_name``     – speaker name for multi-speaker models, or ``None``
                * ``pitch_control``    – pitch multiplier
                * ``energy_control``   – energy multiplier
                * ``duration_control`` – duration multiplier
        """
        import shutil
        import tempfile

        from models.tts.fastspeech2.fs2_inference import FastSpeech2Inference
        from utils.util import load_config

        cfg = load_config(args.config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build the synthetic args namespace expected by TTSInference.
            infer_args = argparse.Namespace(
                mode="single",
                text=args.text,
                acoustics_dir=getattr(args, "acoustics_dir", None),
                checkpoint_path=getattr(args, "checkpoint_path", None),
                vocoder_dir=getattr(args, "vocoder_dir", None),
                output_dir=tmp_dir,
                speaker_name=getattr(args, "speaker_name", None),
                log_level="warning",
                dataset=None,
                testing_set="test",
                test_list_file=None,
                pitch_control=getattr(args, "pitch_control", 1.0),
                energy_control=getattr(args, "energy_control", 1.0),
                duration_control=getattr(args, "duration_control", 1.0),
            )

            inferencer = FastSpeech2Inference(infer_args, cfg)
            inferencer.inference()

            generated = os.path.join(tmp_dir, "single", "test_pred.wav")
            if not os.path.exists(generated):
                raise RuntimeError(
                    "FastSpeech2 inference completed but expected output file was not "
                    f"found at: {generated}"
                )

            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(generated, args.output)
