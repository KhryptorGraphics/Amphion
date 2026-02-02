# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Amphion Unified Web Interface

A tabbed Gradio interface providing access to multiple TTS and VC models:
- MaskGCT TTS: Zero-shot text-to-speech with voice cloning
- DualCodec-VALLE TTS: Neural codec-based TTS
- Vevo Voice Conversion: Timbre, style, and voice conversion

Usage:
    python -m models.web.amphion_unified
"""

import gradio as gr
import torch
import numpy as np
import soundfile as sf
import os
import tempfile
from typing import Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================================
# Lazy Model Loading - Models are loaded on first use to save memory
# ============================================================================

class LazyModelManager:
    """Manages lazy loading of models to optimize memory usage."""

    def __init__(self):
        self._maskgct_loaded = False
        self._dualcodec_valle_loaded = False
        self._vevo_loaded = False

        # MaskGCT components
        self.maskgct_semantic_model = None
        self.maskgct_semantic_mean = None
        self.maskgct_semantic_std = None
        self.maskgct_semantic_codec = None
        self.maskgct_codec_encoder = None
        self.maskgct_codec_decoder = None
        self.maskgct_t2s_model = None
        self.maskgct_s2a_model_1layer = None
        self.maskgct_s2a_model_full = None
        self.maskgct_whisper = None
        self.maskgct_processor = None

        # DualCodec-VALLE components
        self.valle_ar_model = None
        self.valle_nar_model = None
        self.valle_tokenizer = None
        self.valle_dualcodec = None
        self.valle_dualcodec_inference = None

        # Vevo components
        self.vevo_pipeline = None

    def load_maskgct(self):
        """Load MaskGCT models."""
        if self._maskgct_loaded:
            return

        logger.info("Loading MaskGCT models...")

        from huggingface_hub import hf_hub_download
        import safetensors.torch
        from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
        import whisper

        from models.codec.kmeans.repcodec_model import RepCodec
        from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
        from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
        from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
        from utils.util import load_config

        # Load processor
        self.maskgct_processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        # Load config
        cfg_path = "./models/tts/maskgct/config/maskgct.json"
        cfg = load_config(cfg_path)

        # Build semantic model
        self.maskgct_semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.maskgct_semantic_model.to(device)
        stat_mean_var = torch.load("./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt")
        self.maskgct_semantic_mean = stat_mean_var["mean"].to(device)
        self.maskgct_semantic_std = torch.sqrt(stat_mean_var["var"]).to(device)

        # Build semantic codec
        self.maskgct_semantic_codec = RepCodec(cfg=cfg.model.semantic_codec)
        self.maskgct_semantic_codec.to(device)

        # Build acoustic codec
        self.maskgct_codec_encoder = CodecEncoder(cfg=cfg.model.acoustic_codec.encoder)
        self.maskgct_codec_decoder = CodecDecoder(cfg=cfg.model.acoustic_codec.decoder)
        self.maskgct_codec_encoder.to(device)
        self.maskgct_codec_decoder.to(device)

        # Build T2S model
        self.maskgct_t2s_model = MaskGCT_T2S(cfg=cfg.model.t2s_model)
        self.maskgct_t2s_model.to(device)

        # Build S2A models
        self.maskgct_s2a_model_1layer = MaskGCT_S2A(cfg=cfg.model.s2a_model.s2a_1layer)
        self.maskgct_s2a_model_1layer.to(device)
        self.maskgct_s2a_model_full = MaskGCT_S2A(cfg=cfg.model.s2a_model.s2a_full)
        self.maskgct_s2a_model_full.to(device)

        # Download and load checkpoints
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
        codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")
        t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")
        s2a_1layer_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
        s2a_full_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")

        safetensors.torch.load_model(self.maskgct_semantic_codec, semantic_code_ckpt)
        safetensors.torch.load_model(self.maskgct_codec_encoder, codec_encoder_ckpt)
        safetensors.torch.load_model(self.maskgct_codec_decoder, codec_decoder_ckpt)
        safetensors.torch.load_model(self.maskgct_t2s_model, t2s_model_ckpt)
        safetensors.torch.load_model(self.maskgct_s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(self.maskgct_s2a_model_full, s2a_full_ckpt)

        self._maskgct_loaded = True
        logger.info("MaskGCT models loaded successfully.")

    def load_dualcodec_valle(self):
        """Load DualCodec-VALLE models."""
        if self._dualcodec_valle_loaded:
            return

        logger.info("Loading DualCodec-VALLE models...")

        import dualcodec
        from dualcodec.infer.valle.utils_valle_infer import (
            load_dualcodec_valle_ar_12hzv1,
            load_dualcodec_valle_nar_12hzv1,
        )
        from dualcodec.utils import get_whisper_tokenizer

        self.valle_ar_model = load_dualcodec_valle_ar_12hzv1()
        self.valle_nar_model = load_dualcodec_valle_nar_12hzv1()
        self.valle_tokenizer = get_whisper_tokenizer()
        self.valle_dualcodec = dualcodec.get_model("12hz_v1")
        self.valle_dualcodec_inference = dualcodec.Inference(
            dualcodec_model=self.valle_dualcodec, device=device, autocast=True
        )

        self._dualcodec_valle_loaded = True
        logger.info("DualCodec-VALLE models loaded successfully.")

    def load_vevo(self):
        """Load Vevo models."""
        if self._vevo_loaded:
            return

        logger.info("Loading Vevo models...")

        from huggingface_hub import snapshot_download
        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        # Download tokenizer
        local_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            cache_dir="./ckpts/Vevo",
            allow_patterns=["tokenizer/vq8192/*"],
        )
        content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

        # Download AR model
        local_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            cache_dir="./ckpts/Vevo",
            allow_patterns=["contentstyle_modeling/PhoneToVq8192/*"],
        )
        ar_cfg_path = "./models/vc/vevo/config/PhoneToVq8192.json"
        ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/PhoneToVq8192")

        # Download FMT model
        local_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            cache_dir="./ckpts/Vevo",
            allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
        )
        fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
        fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

        # Download Vocoder
        local_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            cache_dir="./ckpts/Vevo",
            allow_patterns=["acoustic_modeling/Vocoder/*"],
        )
        vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
        vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

        self.vevo_pipeline = VevoInferencePipeline(
            content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
            ar_cfg_path=ar_cfg_path,
            ar_ckpt_path=ar_ckpt_path,
            fmt_cfg_path=fmt_cfg_path,
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=vocoder_cfg_path,
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=device,
        )

        self._vevo_loaded = True
        logger.info("Vevo models loaded successfully.")


# Global model manager
model_manager = LazyModelManager()

# ============================================================================
# MaskGCT Interface
# ============================================================================

def maskgct_inference(
    prompt_wav_path: str,
    target_text: str,
    target_len: float = -1,
    n_timesteps: int = 25,
    progress=gr.Progress()
) -> Optional[str]:
    """Run MaskGCT inference."""
    if not prompt_wav_path:
        gr.Warning("Please upload a reference audio file.")
        return None

    if not target_text.strip():
        gr.Warning("Please enter text to generate.")
        return None

    progress(0.1, desc="Loading models...")
    model_manager.load_maskgct()

    import librosa
    import whisper
    import py3langid as langid
    from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

    progress(0.2, desc="Processing audio...")
    speech_16k = librosa.load(prompt_wav_path, sr=16000)[0]
    speech_24k = librosa.load(prompt_wav_path, sr=24000)[0]

    # Detect language and get prompt text using Whisper
    progress(0.3, desc="Detecting language...")
    if model_manager.maskgct_whisper is None:
        model_manager.maskgct_whisper = whisper.load_model("turbo")

    # Detect spoken language
    audio = whisper.load_audio(prompt_wav_path)
    audio_padded = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128).to(model_manager.maskgct_whisper.device)
    _, probs = model_manager.maskgct_whisper.detect_language(mel)
    prompt_language = max(probs, key=probs.get)

    # Get prompt text
    asr_result = model_manager.maskgct_whisper.transcribe(prompt_wav_path, language=prompt_language)

    # Use first 4+ seconds as prompt
    short_prompt_text = ""
    short_prompt_end_ts = 0.0
    for segment in asr_result["segments"]:
        short_prompt_text += segment["text"]
        short_prompt_end_ts = segment["end"]
        if short_prompt_end_ts >= 4:
            break

    speech_24k = speech_24k[0:int(short_prompt_end_ts * 24000)]
    speech_16k = speech_16k[0:int(short_prompt_end_ts * 16000)]

    # Detect target language
    target_language = langid.classify(target_text)[0]

    progress(0.4, desc="Processing phonemes...")

    # G2P conversion
    def g2p_fn(text, language):
        if language in ["zh", "en"]:
            return chn_eng_g2p(text)
        else:
            return g2p(text, sentence=None, language=language)

    prompt_phone_id = g2p_fn(short_prompt_text, prompt_language)[1]
    target_phone_id = g2p_fn(target_text, target_language)[1]

    # Calculate target length
    if target_len < 0:
        target_len_frames = int(
            (len(speech_16k) * len(target_phone_id) / len(prompt_phone_id)) / 16000 * 50
        )
    else:
        target_len_frames = int(target_len * 50)

    progress(0.5, desc="Extracting semantic features...")

    # Extract features
    with torch.no_grad():
        inputs = model_manager.maskgct_processor(speech_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0].unsqueeze(0).to(device)
        attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(device)

        vq_emb = model_manager.maskgct_semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]
        feat = (feat - model_manager.maskgct_semantic_mean) / model_manager.maskgct_semantic_std
        semantic_code, _ = model_manager.maskgct_semantic_codec.quantize(feat)

        # Extract acoustic code
        speech_tensor = torch.tensor(speech_24k).unsqueeze(0).to(device)
        vq_emb = model_manager.maskgct_codec_encoder(speech_tensor.unsqueeze(1))
        _, vq, _, _, _ = model_manager.maskgct_codec_decoder.quantizer(vq_emb)
        acoustic_code = vq.permute(1, 2, 0)

    progress(0.6, desc="Generating semantic tokens...")

    # Text to semantic
    prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(device)
    target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)
    phone_id = torch.cat([prompt_phone_id, target_phone_id])

    with torch.no_grad():
        predict_semantic = model_manager.maskgct_t2s_model.reverse_diffusion(
            semantic_code[:, :],
            target_len_frames,
            phone_id.unsqueeze(0),
            n_timesteps=n_timesteps,
            cfg=2.5,
            rescale_cfg=0.75,
        )
        combine_semantic_code = torch.cat([semantic_code[:, :], predict_semantic], dim=-1)

    progress(0.8, desc="Generating audio...")

    # Semantic to acoustic
    n_timesteps_s2a = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    with torch.no_grad():
        cond = model_manager.maskgct_s2a_model_1layer.cond_emb(combine_semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_1layer = model_manager.maskgct_s2a_model_1layer.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps_s2a[:1],
            cfg=2.5,
            rescale_cfg=0.75,
        )

        cond = model_manager.maskgct_s2a_model_full.cond_emb(combine_semantic_code)
        predict_full = model_manager.maskgct_s2a_model_full.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps_s2a,
            cfg=2.5,
            rescale_cfg=0.75,
            gt_code=predict_1layer,
        )

        vq_emb = model_manager.maskgct_codec_decoder.vq2emb(predict_full.permute(2, 0, 1), n_quantizers=12)
        recovered_audio = model_manager.maskgct_codec_decoder(vq_emb)
        recovered_audio = recovered_audio[0][0].cpu().numpy()

    progress(0.95, desc="Saving audio...")

    # Save output
    os.makedirs("./output", exist_ok=True)
    output_path = f"./output/maskgct_{np.random.randint(10000)}.wav"
    sf.write(output_path, recovered_audio, 24000)

    progress(1.0, desc="Complete!")
    return output_path


# ============================================================================
# DualCodec-VALLE Interface
# ============================================================================

def dualcodec_valle_inference(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    temperature: float = 1.0,
    top_k: int = 15,
    top_p: float = 0.85,
    repeat_penalty: float = 1.1,
    progress=gr.Progress()
) -> Optional[Tuple[int, np.ndarray]]:
    """Run DualCodec-VALLE inference."""
    if not ref_audio:
        gr.Warning("Please upload a reference audio file.")
        return None

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None

    progress(0.1, desc="Loading models...")
    model_manager.load_dualcodec_valle()

    from dualcodec.utils.utils_infer import preprocess_ref_audio_text
    from dualcodec.infer.valle.utils_valle_infer import infer_process

    progress(0.3, desc="Preprocessing audio...")
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio, ref_text)

    progress(0.5, desc="Generating speech...")
    final_wave, final_sample_rate, _ = infer_process(
        ar_model_obj=model_manager.valle_ar_model,
        nar_model_obj=model_manager.valle_nar_model,
        dualcodec_inference_obj=model_manager.valle_dualcodec_inference,
        tokenizer_obj=model_manager.valle_tokenizer,
        ref_audio=ref_audio_processed,
        ref_text=ref_text_processed,
        gen_text=gen_text,
        cross_fade_duration=0.15,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

    progress(1.0, desc="Complete!")

    if final_wave is not None:
        return (final_sample_rate, final_wave)
    return None


# ============================================================================
# Vevo Interface
# ============================================================================

def vevo_tts_inference(
    src_text: str,
    ref_wav: str,
    timbre_ref_wav: Optional[str] = None,
    ref_text: Optional[str] = None,
    src_language: str = "en",
    ref_language: str = "en",
    progress=gr.Progress()
) -> Optional[Tuple[int, np.ndarray]]:
    """Run Vevo TTS inference."""
    if not ref_wav:
        gr.Warning("Please upload a reference audio file.")
        return None

    if not src_text.strip():
        gr.Warning("Please enter text to generate.")
        return None

    progress(0.1, desc="Loading Vevo models...")
    model_manager.load_vevo()

    if timbre_ref_wav is None or timbre_ref_wav == "":
        timbre_ref_wav = ref_wav

    progress(0.5, desc="Generating speech...")
    gen_audio = model_manager.vevo_pipeline.inference_ar_and_fm(
        src_wav_path=None,
        src_text=src_text,
        style_ref_wav_path=ref_wav,
        timbre_ref_wav_path=timbre_ref_wav,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
    )

    progress(1.0, desc="Complete!")
    return (24000, gen_audio.cpu().numpy())


def vevo_vc_inference(
    src_wav: str,
    style_ref_wav: str,
    timbre_ref_wav: Optional[str] = None,
    progress=gr.Progress()
) -> Optional[Tuple[int, np.ndarray]]:
    """Run Vevo Voice Conversion inference."""
    if not src_wav:
        gr.Warning("Please upload a source audio file.")
        return None

    if not style_ref_wav:
        gr.Warning("Please upload a style reference audio file.")
        return None

    progress(0.1, desc="Loading Vevo models...")
    model_manager.load_vevo()

    if timbre_ref_wav is None or timbre_ref_wav == "":
        timbre_ref_wav = style_ref_wav

    progress(0.5, desc="Converting voice...")
    gen_audio = model_manager.vevo_pipeline.inference_ar_and_fm(
        src_wav_path=src_wav,
        src_text=None,
        style_ref_wav_path=style_ref_wav,
        timbre_ref_wav_path=timbre_ref_wav,
    )

    progress(1.0, desc="Complete!")
    return (24000, gen_audio.cpu().numpy())


# ============================================================================
# Build Gradio Interface
# ============================================================================

def create_app():
    """Create the unified Gradio app."""

    with gr.Blocks(
        title="Amphion TTS/VC Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        .tab-nav button { font-size: 16px; font-weight: 600; }
        """
    ) as app:
        gr.Markdown("""
        # Amphion Audio Generation Demo

        Welcome to the Amphion unified demo interface. Select a model from the tabs below:

        - **MaskGCT**: Zero-shot TTS with voice cloning (ICLR 2025)
        - **DualCodec-VALLE**: Neural codec-based TTS
        - **Vevo TTS**: Text-to-speech with style transfer
        - **Vevo VC**: Voice conversion with timbre/style control

        [![GitHub](https://img.shields.io/badge/GitHub-Amphion-blue)](https://github.com/open-mmlab/Amphion)
        """)

        with gr.Tabs():
            # ===== MaskGCT Tab =====
            with gr.TabItem("MaskGCT TTS", id="maskgct"):
                gr.Markdown("""
                ### MaskGCT: Masked Generative Codec Transformer

                Upload a reference audio to clone the voice. The model will automatically
                detect the language and transcribe the reference audio.
                """)

                with gr.Row():
                    with gr.Column():
                        maskgct_audio = gr.Audio(
                            label="Reference Audio (voice to clone)",
                            type="filepath"
                        )
                        maskgct_text = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to synthesize...",
                            lines=4
                        )
                        with gr.Row():
                            maskgct_duration = gr.Number(
                                label="Target Duration (seconds)",
                                value=-1,
                                info="Set to -1 for automatic duration"
                            )
                            maskgct_steps = gr.Slider(
                                label="Diffusion Steps",
                                minimum=15,
                                maximum=100,
                                value=25,
                                step=1
                            )
                        maskgct_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column():
                        maskgct_output = gr.Audio(label="Generated Audio")

                maskgct_btn.click(
                    fn=maskgct_inference,
                    inputs=[maskgct_audio, maskgct_text, maskgct_duration, maskgct_steps],
                    outputs=[maskgct_output]
                )

            # ===== DualCodec-VALLE Tab =====
            with gr.TabItem("DualCodec-VALLE", id="dualcodec"):
                gr.Markdown("""
                ### DualCodec-VALLE TTS

                A neural codec-based text-to-speech model with high-quality voice cloning.
                Provide reference audio and its transcript for best results.
                """)

                with gr.Row():
                    with gr.Column():
                        valle_ref_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath"
                        )
                        valle_ref_text = gr.Textbox(
                            label="Reference Text (transcript)",
                            placeholder="Enter the transcript of the reference audio...",
                            lines=2
                        )
                        valle_gen_text = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to synthesize...",
                            lines=4
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                valle_temp = gr.Slider(
                                    label="Temperature",
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1
                                )
                                valle_top_k = gr.Slider(
                                    label="Top-K",
                                    minimum=1,
                                    maximum=50,
                                    value=15,
                                    step=1
                                )
                            with gr.Row():
                                valle_top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.85,
                                    step=0.05
                                )
                                valle_repeat = gr.Slider(
                                    label="Repeat Penalty",
                                    minimum=1.0,
                                    maximum=2.0,
                                    value=1.1,
                                    step=0.1
                                )

                        valle_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column():
                        valle_output = gr.Audio(label="Generated Audio", type="numpy")

                valle_btn.click(
                    fn=dualcodec_valle_inference,
                    inputs=[
                        valle_ref_audio, valle_ref_text, valle_gen_text,
                        valle_temp, valle_top_k, valle_top_p, valle_repeat
                    ],
                    outputs=[valle_output]
                )

            # ===== Vevo TTS Tab =====
            with gr.TabItem("Vevo TTS", id="vevo_tts"):
                gr.Markdown("""
                ### Vevo Text-to-Speech

                Generate speech from text with style and timbre control.
                Use separate reference audios for style and timbre, or use the same for both.
                """)

                with gr.Row():
                    with gr.Column():
                        vevo_tts_text = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to synthesize...",
                            lines=4
                        )
                        vevo_tts_ref = gr.Audio(
                            label="Style Reference Audio",
                            type="filepath"
                        )
                        vevo_tts_ref_text = gr.Textbox(
                            label="Reference Text (optional)",
                            placeholder="Transcript of reference audio...",
                            lines=2
                        )
                        vevo_tts_timbre = gr.Audio(
                            label="Timbre Reference (optional, uses style ref if empty)",
                            type="filepath"
                        )
                        with gr.Row():
                            vevo_tts_src_lang = gr.Dropdown(
                                label="Source Language",
                                choices=["en", "zh"],
                                value="en"
                            )
                            vevo_tts_ref_lang = gr.Dropdown(
                                label="Reference Language",
                                choices=["en", "zh"],
                                value="en"
                            )
                        vevo_tts_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column():
                        vevo_tts_output = gr.Audio(label="Generated Audio", type="numpy")

                vevo_tts_btn.click(
                    fn=vevo_tts_inference,
                    inputs=[
                        vevo_tts_text, vevo_tts_ref, vevo_tts_timbre,
                        vevo_tts_ref_text, vevo_tts_src_lang, vevo_tts_ref_lang
                    ],
                    outputs=[vevo_tts_output]
                )

            # ===== Vevo VC Tab =====
            with gr.TabItem("Vevo Voice Conversion", id="vevo_vc"):
                gr.Markdown("""
                ### Vevo Voice Conversion

                Convert the voice of source audio to match a reference speaker.
                Upload source audio and reference audio for style/timbre.
                """)

                with gr.Row():
                    with gr.Column():
                        vevo_vc_src = gr.Audio(
                            label="Source Audio (voice to convert)",
                            type="filepath"
                        )
                        vevo_vc_style = gr.Audio(
                            label="Style Reference Audio",
                            type="filepath"
                        )
                        vevo_vc_timbre = gr.Audio(
                            label="Timbre Reference (optional, uses style ref if empty)",
                            type="filepath"
                        )
                        vevo_vc_btn = gr.Button("Convert Voice", variant="primary")

                    with gr.Column():
                        vevo_vc_output = gr.Audio(label="Converted Audio", type="numpy")

                vevo_vc_btn.click(
                    fn=vevo_vc_inference,
                    inputs=[vevo_vc_src, vevo_vc_style, vevo_vc_timbre],
                    outputs=[vevo_vc_output]
                )

        gr.Markdown("""
        ---
        **Note**: Models are loaded on first use. Initial inference may take longer due to model loading.

        For issues or feedback, visit [Amphion GitHub](https://github.com/open-mmlab/Amphion).
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=14558,
        root_path="/unified",
        allowed_paths=["./output"]
    )
