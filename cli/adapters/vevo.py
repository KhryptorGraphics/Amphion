# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI adapters for Vevo voice conversion and TTS models.

Provides four adapter classes:

* :class:`VevoVoiceAdapter`  – full voice conversion (content + style + timbre)
* :class:`VevoTimbreAdapter` – timbre-only voice conversion
* :class:`VevoStyleAdapter`  – style-only voice conversion (preserves source timbre)
* :class:`VevoTTSAdapter`    – zero-shot TTS with optional separate timbre reference
"""

import argparse
import os

from cli.adapters.base import ModelAdapter, resolve_device


# ---------------------------------------------------------------------------
# Shared model-download helpers
# ---------------------------------------------------------------------------

def _download_vevo_patterns(patterns, cache_dir="./ckpts/Vevo"):
    """Download Vevo checkpoint files matching *patterns* from HuggingFace.

    Args:
        patterns: List of glob patterns to pass to ``snapshot_download``.
        cache_dir: Local cache directory for the downloaded files.

    Returns:
        Local directory path returned by ``snapshot_download``.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=patterns,
    )


def _build_vevo_pipeline(device, *, use_content_tokenizer=False, tts_mode=False):
    """Construct a :class:`VevoInferencePipeline` with appropriate components.

    Args:
        device: ``torch.device`` for all model tensors.
        use_content_tokenizer: When ``True``, download and pass the vq32
            content tokenizer (needed for voice and style conversion).
        tts_mode: When ``True``, use the ``PhoneToVq8192`` AR model instead of
            ``Vq32ToVq8192`` (needed for TTS and timbre-only conversion uses
            this path when ``use_content_tokenizer=False``).

    Returns:
        A ready-to-use :class:`VevoInferencePipeline` instance.
    """
    from models.vc.vevo.vevo_utils import VevoInferencePipeline

    # --- Content tokenizer (vq32) – only for VC modes that need it ---
    content_tokenizer_ckpt_path = None
    if use_content_tokenizer:
        local_dir = _download_vevo_patterns(["tokenizer/vq32/*"])
        content_tokenizer_ckpt_path = os.path.join(
            local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
        )

    # --- Content-style tokenizer (vq8192) ---
    local_dir = _download_vevo_patterns(["tokenizer/vq8192/*"])
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # --- Autoregressive transformer ---
    ar_cfg_path = None
    ar_ckpt_path = None
    if tts_mode:
        local_dir = _download_vevo_patterns(
            ["contentstyle_modeling/PhoneToVq8192/*"]
        )
        ar_cfg_path = "./models/vc/vevo/config/PhoneToVq8192.json"
        ar_ckpt_path = os.path.join(
            local_dir, "contentstyle_modeling/PhoneToVq8192"
        )
    elif use_content_tokenizer:
        local_dir = _download_vevo_patterns(
            ["contentstyle_modeling/Vq32ToVq8192/*"]
        )
        ar_cfg_path = "./models/vc/vevo/config/Vq32ToVq8192.json"
        ar_ckpt_path = os.path.join(
            local_dir, "contentstyle_modeling/Vq32ToVq8192"
        )

    # --- Flow matching transformer ---
    local_dir = _download_vevo_patterns(["acoustic_modeling/Vq8192ToMels/*"])
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # --- Vocoder ---
    local_dir = _download_vevo_patterns(["acoustic_modeling/Vocoder/*"])
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    return VevoInferencePipeline(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )


# ---------------------------------------------------------------------------
# VevoVoiceAdapter
# ---------------------------------------------------------------------------

class VevoVoiceAdapter(ModelAdapter):
    """Adapter for full Vevo voice conversion (style + timbre from reference).

    Downloads ``amphion/Vevo`` weights from HuggingFace on first use.
    Uses both the content tokenizer (vq32) and the AR model (Vq32ToVq8192).
    """

    @property
    def name(self) -> str:
        return "vevo-voice"

    @property
    def description(self) -> str:
        return "Vevo voice conversion – transfer style and timbre from a reference audio"

    @property
    def task_type(self) -> str:
        return "vc"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register vevo-voice CLI arguments."""
        parser.add_argument(
            "--source",
            required=True,
            metavar="PATH",
            help="Source audio file whose content (speech) will be converted.",
        )
        parser.add_argument(
            "--reference",
            required=True,
            metavar="PATH",
            help="Reference audio file providing the target voice style and timbre.",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run voice conversion and write output to ``args.output``.

        Args:
            args: Parsed namespace with ``source``, ``reference``, ``output``,
                and ``device`` attributes.
        """
        from models.vc.vevo.vevo_utils import save_audio

        device = resolve_device(args)
        pipeline = _build_vevo_pipeline(device, use_content_tokenizer=True)

        gen_audio = pipeline.inference_ar_and_fm(
            src_wav_path=args.source,
            src_text=None,
            style_ref_wav_path=args.reference,
            timbre_ref_wav_path=args.reference,
        )
        save_audio(gen_audio, output_path=args.output)


# ---------------------------------------------------------------------------
# VevoTimbreAdapter
# ---------------------------------------------------------------------------

class VevoTimbreAdapter(ModelAdapter):
    """Adapter for Vevo timbre-only voice conversion.

    Converts only the timbre (speaker identity) while preserving the content
    and style of the source audio.  Uses the flow matching transformer directly
    without the AR content-style model.
    """

    @property
    def name(self) -> str:
        return "vevo-timbre"

    @property
    def description(self) -> str:
        return "Vevo timbre conversion – transfer only timbre from a reference audio"

    @property
    def task_type(self) -> str:
        return "vc"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register vevo-timbre CLI arguments."""
        parser.add_argument(
            "--source",
            required=True,
            metavar="PATH",
            help="Source audio file whose timbre will be converted.",
        )
        parser.add_argument(
            "--reference",
            required=True,
            metavar="PATH",
            help="Reference audio file providing the target timbre.",
        )
        parser.add_argument(
            "--flow-matching-steps",
            type=int,
            default=32,
            metavar="N",
            help="Number of flow matching steps (default: 32).",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run timbre conversion and write output to ``args.output``.

        Args:
            args: Parsed namespace with ``source``, ``reference``,
                ``flow_matching_steps``, ``output``, and ``device`` attributes.
        """
        from models.vc.vevo.vevo_utils import save_audio

        device = resolve_device(args)
        pipeline = _build_vevo_pipeline(device, use_content_tokenizer=False)

        gen_audio = pipeline.inference_fm(
            src_wav_path=args.source,
            timbre_ref_wav_path=args.reference,
            flow_matching_steps=args.flow_matching_steps,
        )
        save_audio(gen_audio, output_path=args.output)


# ---------------------------------------------------------------------------
# VevoStyleAdapter
# ---------------------------------------------------------------------------

class VevoStyleAdapter(ModelAdapter):
    """Adapter for Vevo style-only voice conversion.

    Transfers the prosody and style from a reference audio while preserving
    the timbre of the source speaker.  Uses the vq32 content tokenizer and the
    ``Vq32ToVq8192`` AR model.
    """

    @property
    def name(self) -> str:
        return "vevo-style"

    @property
    def description(self) -> str:
        return "Vevo style conversion – transfer style while preserving source timbre"

    @property
    def task_type(self) -> str:
        return "vc"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register vevo-style CLI arguments."""
        parser.add_argument(
            "--source",
            required=True,
            metavar="PATH",
            help="Source audio file whose timbre will be preserved.",
        )
        parser.add_argument(
            "--style-ref",
            required=True,
            metavar="PATH",
            help="Reference audio file providing the target style/prosody.",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run style conversion and write output to ``args.output``.

        Args:
            args: Parsed namespace with ``source``, ``style_ref``, ``output``,
                and ``device`` attributes.
        """
        from models.vc.vevo.vevo_utils import save_audio

        device = resolve_device(args)
        pipeline = _build_vevo_pipeline(device, use_content_tokenizer=True)

        gen_audio = pipeline.inference_ar_and_fm(
            src_wav_path=args.source,
            src_text=None,
            style_ref_wav_path=args.style_ref,
            timbre_ref_wav_path=args.source,
        )
        save_audio(gen_audio, output_path=args.output)


# ---------------------------------------------------------------------------
# VevoTTSAdapter
# ---------------------------------------------------------------------------

class VevoTTSAdapter(ModelAdapter):
    """Adapter for Vevo zero-shot TTS.

    Synthesises speech from text using a reference audio for style and timbre.
    Optionally accepts a separate timbre reference for style/timbre decoupling.
    Uses the ``PhoneToVq8192`` AR model.
    """

    @property
    def name(self) -> str:
        return "vevo-tts"

    @property
    def description(self) -> str:
        return "Vevo TTS – zero-shot TTS with reference audio voice cloning"

    @property
    def task_type(self) -> str:
        return "tts"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register vevo-tts CLI arguments."""
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
            help="Reference audio file for style and timbre cloning.",
        )
        parser.add_argument(
            "--ref-text",
            default=None,
            metavar="TEXT",
            help="Transcript of the reference audio clip (optional but recommended).",
        )
        parser.add_argument(
            "--timbre-ref-audio",
            default=None,
            metavar="PATH",
            help=(
                "Separate reference audio for timbre when style and timbre should "
                "come from different sources.  Defaults to --ref-audio."
            ),
        )
        parser.add_argument(
            "--src-language",
            default="en",
            metavar="LANG",
            help="Language code for the target text (default: 'en').",
        )
        parser.add_argument(
            "--ref-language",
            default="en",
            metavar="LANG",
            help="Language code for the reference audio transcript (default: 'en').",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run zero-shot TTS and write output to ``args.output``.

        Args:
            args: Parsed namespace with ``text``, ``ref_audio``, ``ref_text``,
                ``timbre_ref_audio``, ``src_language``, ``ref_language``,
                ``output``, and ``device`` attributes.
        """
        from models.vc.vevo.vevo_utils import save_audio

        device = resolve_device(args)
        pipeline = _build_vevo_pipeline(device, tts_mode=True)

        timbre_ref = args.timbre_ref_audio if args.timbre_ref_audio else args.ref_audio

        gen_audio = pipeline.inference_ar_and_fm(
            src_wav_path=None,
            src_text=args.text,
            style_ref_wav_path=args.ref_audio,
            timbre_ref_wav_path=timbre_ref,
            style_ref_wav_text=args.ref_text,
            src_text_language=args.src_language,
            style_ref_wav_text_language=args.ref_language,
        )
        save_audio(gen_audio, output_path=args.output)
