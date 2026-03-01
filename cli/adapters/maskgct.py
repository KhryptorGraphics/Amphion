# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CLI adapter for MaskGCT – zero-shot TTS via masked generative codec transformer."""

import argparse

from cli.adapters.base import ModelAdapter, resolve_device


class MaskGCTAdapter(ModelAdapter):
    """Adapter wrapping :class:`MaskGCT_Inference_Pipeline` for ``amphion infer``.

    Lazily downloads pretrained weights from HuggingFace (``amphion/MaskGCT``)
    on the first call to :meth:`run`.  Subsequent calls within the same process
    re-use the already-loaded pipeline.
    """

    # ------------------------------------------------------------------
    # Identity properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "maskgct"

    @property
    def description(self) -> str:
        return "MaskGCT TTS – zero-shot TTS with reference audio (ICLR 2025)"

    @property
    def task_type(self) -> str:
        return "tts"

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register MaskGCT-specific CLI arguments with *parser*."""
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
            "--prompt-text",
            required=True,
            metavar="TEXT",
            help="Transcript of the reference audio clip.",
        )
        parser.add_argument(
            "--language",
            default="en",
            metavar="LANG",
            help=(
                "Language code for both prompt and target text (default: 'en').  "
                "Supported values: en, zh, fr, de, ja, ko."
            ),
        )
        parser.add_argument(
            "--target-language",
            default=None,
            metavar="LANG",
            help=(
                "Language code for the target text when it differs from "
                "--language.  Defaults to the value of --language."
            ),
        )
        parser.add_argument(
            "--target-len",
            type=float,
            default=None,
            metavar="SECONDS",
            help=(
                "Target audio duration in seconds.  "
                "When omitted the duration is estimated automatically."
            ),
        )
        parser.add_argument(
            "--n-timesteps",
            type=int,
            default=25,
            metavar="N",
            help="Number of diffusion timesteps for the T2S stage (default: 25).",
        )
        parser.add_argument(
            "--cfg-weight",
            type=float,
            default=2.5,
            metavar="W",
            help="Classifier-free guidance weight (default: 2.5).",
        )

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    def run(self, args: argparse.Namespace) -> None:
        """Load MaskGCT (downloading weights if needed) and run zero-shot TTS.

        Args:
            args: Parsed namespace.  Expected attributes (beyond the shared
                ``--output`` / ``--device``):

                * ``text``            – target synthesis text
                * ``ref_audio``       – path to the prompt/reference audio
                * ``prompt_text``     – transcript of the reference audio
                * ``language``        – prompt language code
                * ``target_language`` – target language code (or ``None``)
                * ``target_len``      – desired duration in seconds (or ``None``)
                * ``n_timesteps``     – T2S diffusion steps
                * ``cfg_weight``      – classifier-free guidance weight
        """
        import soundfile as sf

        device = resolve_device(args)
        pipeline = self._load_pipeline(device)

        target_language = args.target_language or args.language

        recovered_audio = pipeline.maskgct_inference(
            prompt_speech_path=args.ref_audio,
            prompt_text=args.prompt_text,
            target_text=args.text,
            language=args.language,
            target_language=target_language,
            target_len=args.target_len,
            n_timesteps=args.n_timesteps,
            cfg=args.cfg_weight,
        )

        sf.write(args.output, recovered_audio, 24000)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self, device):
        """Build and return a :class:`MaskGCT_Inference_Pipeline`.

        Downloads all required safetensors checkpoints from HuggingFace
        (``amphion/MaskGCT``) on the first invocation; subsequent calls are
        cached by the HuggingFace hub.

        Args:
            device: ``torch.device`` on which to place all model tensors.

        Returns:
            A ready-to-use :class:`MaskGCT_Inference_Pipeline` instance.
        """
        import safetensors.torch
        from huggingface_hub import hf_hub_download

        from models.tts.maskgct.maskgct_utils import (
            MaskGCT_Inference_Pipeline,
            build_acoustic_codec,
            build_s2a_model,
            build_semantic_codec,
            build_semantic_model,
            build_t2s_model,
        )
        from utils.util import load_config

        cfg_path = "./models/tts/maskgct/config/maskgct.json"
        cfg = load_config(cfg_path)

        # Build model components
        semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
        codec_encoder, codec_decoder = build_acoustic_codec(
            cfg.model.acoustic_codec, device
        )
        t2s_model = build_t2s_model(cfg.model.t2s_model, device)
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

        # Download checkpoints from HuggingFace (cached after first download)
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        codec_encoder_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="acoustic_codec/model.safetensors"
        )
        codec_decoder_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors"
        )
        t2s_model_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="t2s_model/model.safetensors"
        )
        s2a_1layer_ckpt = hf_hub_download(
            "amphion/MaskGCT",
            filename="s2a_model/s2a_model_1layer/model.safetensors",
        )
        s2a_full_ckpt = hf_hub_download(
            "amphion/MaskGCT",
            filename="s2a_model/s2a_model_full/model.safetensors",
        )

        # Load weights into models
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
        safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
        safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

        return MaskGCT_Inference_Pipeline(
            semantic_model,
            semantic_codec,
            codec_encoder,
            codec_decoder,
            t2s_model,
            s2a_model_1layer,
            s2a_model_full,
            semantic_mean,
            semantic_std,
            device,
        )
