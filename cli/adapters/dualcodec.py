# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI adapter for DualCodec-VALLE – neural codec-based zero-shot TTS."""

import argparse

from cli.adapters.base import ModelAdapter, resolve_device


class DualCodecAdapter(ModelAdapter):
    """Adapter wrapping DualCodec-VALLE for ``amphion infer``.

    Lazily downloads pretrained weights from HuggingFace via the ``dualcodec``
    package on the first call to :meth:`run`.  Subsequent calls within the same
    process re-use the already-loaded models.
    """

    # ------------------------------------------------------------------
    # Identity properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dualcodec"

    @property
    def description(self) -> str:
        return "DualCodec-VALLE TTS – zero-shot TTS via neural codec language model"

    @property
    def task_type(self) -> str:
        return "tts"

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register DualCodec-VALLE-specific CLI arguments with *parser*."""
        parser.add_argument(
            "--text",
            required=True,
            metavar="TEXT",
            help="Target text to synthesise.",
        )
        parser.add_argument(
            "--ref-audio",
            required=True,
            metavar="PATH",
            help="Reference (prompt) audio file for voice cloning.",
        )
        parser.add_argument(
            "--ref-text",
            default=None,
            metavar="TEXT",
            help=(
                "Transcript of the reference audio clip.  "
                "When omitted, Whisper is used to transcribe the reference audio automatically."
            ),
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            metavar="T",
            help="Sampling temperature for the AR model (default: 1.0).",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=15,
            metavar="K",
            help="Top-k sampling parameter for the AR model (default: 15).",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=0.85,
            metavar="P",
            help="Top-p (nucleus) sampling parameter for the AR model (default: 0.85).",
        )
        parser.add_argument(
            "--repeat-penalty",
            type=float,
            default=1.1,
            metavar="W",
            help="Repetition penalty for the AR model (default: 1.1).",
        )
        parser.add_argument(
            "--cross-fade-duration",
            type=float,
            default=0.15,
            metavar="SECONDS",
            help="Cross-fade duration in seconds for chunk stitching (default: 0.15).",
        )

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    def run(self, args: argparse.Namespace) -> None:
        """Load DualCodec-VALLE (downloading weights if needed) and run zero-shot TTS.

        Args:
            args: Parsed namespace.  Expected attributes (beyond the shared
                ``--output`` / ``--device``):

                * ``text``               – target synthesis text
                * ``ref_audio``          – path to the prompt/reference audio
                * ``ref_text``           – transcript of the reference audio (or ``None``)
                * ``temperature``        – AR sampling temperature
                * ``top_k``              – AR top-k sampling parameter
                * ``top_p``              – AR nucleus sampling parameter
                * ``repeat_penalty``     – AR repetition penalty
                * ``cross_fade_duration`` – cross-fade duration in seconds
        """
        import soundfile as sf

        device = resolve_device(args)
        ar_model, nar_model, dualcodec_inference, tokenizer = self._load_models(device)

        from dualcodec.utils.utils_infer import preprocess_ref_audio_text
        from dualcodec.infer.valle.utils_valle_infer import infer_process

        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            args.ref_audio, args.ref_text or ""
        )

        final_wave, final_sample_rate, _ = infer_process(
            ar_model_obj=ar_model,
            nar_model_obj=nar_model,
            dualcodec_inference_obj=dualcodec_inference,
            tokenizer_obj=tokenizer,
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=args.text,
            cross_fade_duration=args.cross_fade_duration,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )

        sf.write(args.output, final_wave, final_sample_rate)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_models(self, device):
        """Load DualCodec-VALLE model components.

        Downloads all required checkpoints from HuggingFace via the ``dualcodec``
        package on the first invocation; subsequent calls are cached by the
        HuggingFace hub.

        Args:
            device: ``torch.device`` on which to place all model tensors.

        Returns:
            Tuple of ``(ar_model, nar_model, dualcodec_inference, tokenizer)``.
        """
        import dualcodec
        from dualcodec.infer.valle.utils_valle_infer import (
            load_dualcodec_valle_ar_12hzv1,
            load_dualcodec_valle_nar_12hzv1,
        )
        from dualcodec.utils import get_whisper_tokenizer

        ar_model = load_dualcodec_valle_ar_12hzv1()
        nar_model = load_dualcodec_valle_nar_12hzv1()
        tokenizer = get_whisper_tokenizer()
        dualcodec_model = dualcodec.get_model("12hz_v1")
        dualcodec_inference = dualcodec.Inference(
            dualcodec_model=dualcodec_model, device=device, autocast=True
        )

        return ar_model, nar_model, dualcodec_inference, tokenizer
