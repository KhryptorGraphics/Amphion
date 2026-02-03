"""
Model Manager

Singleton manager for lazy loading TTS and VC models.
Handles model initialization, caching, and inference.
"""

import torch
import logging
import os
import gc
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Base paths
AMPHION_ROOT = "/home/kp/repo2/Amphion"
OUTPUT_DIR = f"{AMPHION_ROOT}/output/web"


class ModelManager:
    """Singleton manager for lazy-loading Amphion models."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelManager initialized with device: {self.device}")

        # Model loading flags
        self._maskgct_loaded = False
        self._dualcodec_valle_loaded = False
        self._vevo_tts_loaded = False
        self._vevo_vc_loaded = False
        self._noro_loaded = False
        self._metis_loaded = False
        self._vevosing_loaded = False

        # TTA models
        self._audioldm_loaded = False
        self._picoaudio_loaded = False

        # Codec models
        self._dualcodec_codec_loaded = False
        self._facodec_loaded = False

        # Vocoder models
        self._hifigan_loaded = False
        self._bigvgan_loaded = False

        # Additional SVC models
        self._diffcomosvc_loaded = False
        self._transformersvc_loaded = False
        self._vitssvc_loaded = False
        self._multiplecontentssvc_loaded = False

        # Model instances
        self.maskgct_pipeline = None
        self.valle_models = None
        self.vevo_tts_pipeline = None
        self.vevo_vc_pipeline = None
        self.noro_pipeline = None
        self.metis_pipeline = None
        self.vevosing_pipeline = None

        # TTA model instances
        self.audioldm_model = None
        self.picoaudio_model = None

        # Codec model instances
        self.dualcodec_codec = None
        self.facodec_model = None

        # Vocoder instances
        self.hifigan_vocoder = None
        self.bigvgan_vocoder = None

        # Additional SVC model instances
        self.diffcomosvc_pipeline = None
        self.transformersvc_pipeline = None
        self.vitssvc_pipeline = None
        self.multiplecontentssvc_pipeline = None

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self._initialized = True

    def unload_model(self, model_name: str):
        """Unload a model to free GPU memory."""
        if model_name == "maskgct" and self._maskgct_loaded:
            del self.maskgct_pipeline
            self.maskgct_pipeline = None
            self._maskgct_loaded = False
            logger.info("MaskGCT unloaded")
        elif model_name == "dualcodec_valle" and self._dualcodec_valle_loaded:
            del self.valle_models
            self.valle_models = None
            self._dualcodec_valle_loaded = False
            logger.info("DualCodec-VALLE unloaded")
        elif model_name == "vevo_tts" and self._vevo_tts_loaded:
            del self.vevo_tts_pipeline
            self.vevo_tts_pipeline = None
            self._vevo_tts_loaded = False
            logger.info("Vevo TTS unloaded")
        elif model_name == "vevo_vc" and self._vevo_vc_loaded:
            del self.vevo_vc_pipeline
            self.vevo_vc_pipeline = None
            self._vevo_vc_loaded = False
            logger.info("Vevo VC unloaded")
        elif model_name == "noro" and self._noro_loaded:
            del self.noro_pipeline
            self.noro_pipeline = None
            self._noro_loaded = False
            logger.info("Noro unloaded")
        elif model_name == "metis" and self._metis_loaded:
            del self.metis_pipeline
            self.metis_pipeline = None
            self._metis_loaded = False
            logger.info("Metis unloaded")
        elif model_name == "vevosing" and self._vevosing_loaded:
            del self.vevosing_pipeline
            self.vevosing_pipeline = None
            self._vevosing_loaded = False
            logger.info("VevoSing unloaded")
        # TTA models
        elif model_name == "audioldm" and self._audioldm_loaded:
            del self.audioldm_model
            self.audioldm_model = None
            self._audioldm_loaded = False
            logger.info("AudioLDM unloaded")
        elif model_name == "picoaudio" and self._picoaudio_loaded:
            del self.picoaudio_model
            self.picoaudio_model = None
            self._picoaudio_loaded = False
            logger.info("PicoAudio unloaded")
        # Codec models
        elif model_name == "dualcodec_codec" and self._dualcodec_codec_loaded:
            del self.dualcodec_codec
            self.dualcodec_codec = None
            self._dualcodec_codec_loaded = False
            logger.info("DualCodec codec unloaded")
        elif model_name == "facodec" and self._facodec_loaded:
            del self.facodec_model
            self.facodec_model = None
            self._facodec_loaded = False
            logger.info("FACodec unloaded")
        # Vocoder models
        elif model_name == "hifigan" and self._hifigan_loaded:
            del self.hifigan_vocoder
            self.hifigan_vocoder = None
            self._hifigan_loaded = False
            logger.info("HiFi-GAN unloaded")
        elif model_name == "bigvgan_vocoder" and self._bigvgan_loaded:
            del self.bigvgan_vocoder
            self.bigvgan_vocoder = None
            self._bigvgan_loaded = False
            logger.info("BigVGAN vocoder unloaded")
        # Additional SVC models
        elif model_name == "diffcomosvc" and self._diffcomosvc_loaded:
            del self.diffcomosvc_pipeline
            self.diffcomosvc_pipeline = None
            self._diffcomosvc_loaded = False
            logger.info("DiffComoSVC unloaded")
        elif model_name == "transformersvc" and self._transformersvc_loaded:
            del self.transformersvc_pipeline
            self.transformersvc_pipeline = None
            self._transformersvc_loaded = False
            logger.info("TransformerSVC unloaded")
        elif model_name == "vitssvc" and self._vitssvc_loaded:
            del self.vitssvc_pipeline
            self.vitssvc_pipeline = None
            self._vitssvc_loaded = False
            logger.info("VitsSVC unloaded")
        elif model_name == "multiplecontentssvc" and self._multiplecontentssvc_loaded:
            del self.multiplecontentssvc_pipeline
            self.multiplecontentssvc_pipeline = None
            self._multiplecontentssvc_loaded = False
            logger.info("MultipleContentsSVC unloaded")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_maskgct(self):
        """Lazy load MaskGCT models."""
        if self._maskgct_loaded:
            return

        logger.info("Loading MaskGCT models...")

        try:
            # Add Amphion to path if needed
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            from huggingface_hub import hf_hub_download
            import safetensors.torch
            from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

            from models.codec.kmeans.repcodec_model import RepCodec
            from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
            from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
            from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
            from utils.util import load_config

            # Load config
            cfg_path = f"{AMPHION_ROOT}/models/tts/maskgct/config/maskgct.json"
            cfg = load_config(cfg_path)

            # Load processor
            processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

            # Load semantic model
            semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(self.device)
            semantic_model.set_mode_for_inference()

            # Load stats
            stat_mean_var = torch.load(
                f"{AMPHION_ROOT}/models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
                map_location=self.device
            )
            semantic_mean = stat_mean_var["mean"]
            semantic_std = torch.sqrt(stat_mean_var["var"])

            # Initialize models
            semantic_codec = RepCodec(cfg=cfg.model.semantic_codec).to(self.device)
            codec_encoder = CodecEncoder(cfg=cfg.model.acoustic_codec.encoder).to(self.device)
            codec_decoder = CodecDecoder(cfg=cfg.model.acoustic_codec.decoder).to(self.device)
            t2s_model = MaskGCT_T2S(cfg=cfg.model.t2s_model).to(self.device)
            s2a_model_1layer = MaskGCT_S2A(cfg=cfg.model.s2a_model.s2a_1layer).to(self.device)
            s2a_model_full = MaskGCT_S2A(cfg=cfg.model.s2a_model.s2a_full).to(self.device)

            # Download and load checkpoints
            semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
            codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")
            t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")
            s2a_1layer_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
            s2a_full_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")

            safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
            safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
            safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
            safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
            safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
            safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

            # Set to inference mode using standard PyTorch method
            for model in [semantic_codec, codec_encoder, codec_decoder, t2s_model, s2a_model_1layer, s2a_model_full]:
                model.requires_grad_(False)

            self.maskgct_pipeline = {
                'processor': processor,
                'semantic_model': semantic_model,
                'semantic_mean': semantic_mean,
                'semantic_std': semantic_std,
                'semantic_codec': semantic_codec,
                'codec_encoder': codec_encoder,
                'codec_decoder': codec_decoder,
                't2s_model': t2s_model,
                's2a_model_1layer': s2a_model_1layer,
                's2a_model_full': s2a_model_full,
                'whisper': None  # Loaded on demand
            }

            self._maskgct_loaded = True
            logger.info("MaskGCT models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MaskGCT models: {e}")
            raise

    def load_dualcodec_valle(self):
        """Lazy load DualCodec-VALLE models."""
        if self._dualcodec_valle_loaded:
            return

        logger.info("Loading DualCodec-VALLE models...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            import dualcodec
            from dualcodec.infer.valle.utils_valle_infer import (
                load_dualcodec_valle_ar_12hzv1,
                load_dualcodec_valle_nar_12hzv1,
            )
            from dualcodec.utils import get_whisper_tokenizer

            # Load models
            ar_model = load_dualcodec_valle_ar_12hzv1()
            nar_model = load_dualcodec_valle_nar_12hzv1()
            tokenizer = get_whisper_tokenizer()

            # Load DualCodec
            dualcodec_model = dualcodec.get_model("12hz_v1")
            dualcodec_inference = dualcodec.Inference(
                dualcodec_model=dualcodec_model,
                device=self.device,
                autocast=True
            )

            self.valle_models = {
                'ar_model': ar_model,
                'nar_model': nar_model,
                'tokenizer': tokenizer,
                'dualcodec': dualcodec_model,
                'inference': dualcodec_inference
            }

            self._dualcodec_valle_loaded = True
            logger.info("DualCodec-VALLE models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load DualCodec-VALLE models: {e}")
            raise

    def load_vevo_tts(self):
        """Lazy load Vevo TTS models."""
        if self._vevo_tts_loaded:
            return

        logger.info("Loading Vevo TTS models...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            from huggingface_hub import snapshot_download
            from models.vc.vevo.vevo_utils import VevoInferencePipeline

            ckpt_base = f"{AMPHION_ROOT}/ckpts/Vevo"

            # Download tokenizer
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["tokenizer/vq8192/*"],
            )
            content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

            # Download AR model
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["contentstyle_modeling/PhoneToVq8192/*"],
            )
            ar_cfg_path = f"{AMPHION_ROOT}/models/vc/vevo/config/PhoneToVq8192.json"
            ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/PhoneToVq8192")

            # Download FMT model
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
            )
            fmt_cfg_path = f"{AMPHION_ROOT}/models/vc/vevo/config/Vq8192ToMels.json"
            fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

            # Download vocoder
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["acoustic_modeling/Vocoder/*"],
            )
            vocoder_cfg_path = f"{AMPHION_ROOT}/models/vc/vevo/config/Vocoder.json"
            vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

            # Initialize pipeline
            self.vevo_tts_pipeline = VevoInferencePipeline(
                content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
                ar_cfg_path=ar_cfg_path,
                ar_ckpt_path=ar_ckpt_path,
                fmt_cfg_path=fmt_cfg_path,
                fmt_ckpt_path=fmt_ckpt_path,
                vocoder_cfg_path=vocoder_cfg_path,
                vocoder_ckpt_path=vocoder_ckpt_path,
                device=self.device,
            )

            self._vevo_tts_loaded = True
            logger.info("Vevo TTS models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Vevo TTS models: {e}")
            raise

    def load_vevo_vc(self):
        """Lazy load Vevo VC models."""
        if self._vevo_vc_loaded:
            return

        logger.info("Loading Vevo VC models...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            from huggingface_hub import snapshot_download
            from models.vc.vevo.vevo_utils import VevoInferencePipeline

            ckpt_base = f"{AMPHION_ROOT}/ckpts/Vevo"

            # Download content tokenizer (vq32 for VC)
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["tokenizer/vq32/*"],
            )
            content_tokenizer_ckpt_path = os.path.join(
                local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
            )

            # Download style tokenizer (vq8192)
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["tokenizer/vq8192/*"],
            )
            content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

            # Download AR model (Vq32ToVq8192 for VC)
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"],
            )
            ar_cfg_path = f"{AMPHION_ROOT}/models/vc/vevo/config/Vq32ToVq8192.json"
            ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")

            # Download FMT model
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
            )
            fmt_cfg_path = f"{AMPHION_ROOT}/models/vc/vevo/config/Vq8192ToMels.json"
            fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

            # Download vocoder
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=ckpt_base,
                allow_patterns=["acoustic_modeling/Vocoder/*"],
            )
            vocoder_cfg_path = f"{AMPHION_ROOT}/models/vc/vevo/config/Vocoder.json"
            vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

            # Initialize pipeline
            self.vevo_vc_pipeline = VevoInferencePipeline(
                content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
                content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
                ar_cfg_path=ar_cfg_path,
                ar_ckpt_path=ar_ckpt_path,
                fmt_cfg_path=fmt_cfg_path,
                fmt_ckpt_path=fmt_ckpt_path,
                vocoder_cfg_path=vocoder_cfg_path,
                vocoder_ckpt_path=vocoder_ckpt_path,
                device=self.device,
            )

            self._vevo_vc_loaded = True
            logger.info("Vevo VC models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Vevo VC models: {e}")
            raise

    def maskgct_inference(
        self,
        prompt_wav_path: str,
        target_text: str,
        target_len: float = -1,
        n_timesteps: int = 25
    ) -> str:
        """
        Run MaskGCT inference.

        Args:
            prompt_wav_path: Path to reference audio
            target_text: Text to synthesize
            target_len: Target duration in seconds (-1 for auto)
            n_timesteps: Number of diffusion timesteps

        Returns:
            Path to generated audio file
        """
        self.load_maskgct()

        import librosa
        import whisper
        import py3langid as langid
        from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

        # Load audio at different sample rates
        speech_16k = librosa.load(prompt_wav_path, sr=16000)[0]
        speech_24k = librosa.load(prompt_wav_path, sr=24000)[0]

        # Load whisper if needed
        if self.maskgct_pipeline['whisper'] is None:
            self.maskgct_pipeline['whisper'] = whisper.load_model("turbo")

        # Detect language
        audio = whisper.load_audio(prompt_wav_path)
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(
            audio_padded, n_mels=128
        ).to(self.maskgct_pipeline['whisper'].device)
        _, probs = self.maskgct_pipeline['whisper'].detect_language(mel)
        prompt_language = max(probs, key=probs.get)

        # Transcribe
        asr_result = self.maskgct_pipeline['whisper'].transcribe(
            prompt_wav_path, language=prompt_language
        )

        # Get short prompt (first ~4 seconds)
        short_prompt_text = ""
        short_prompt_end_ts = 0.0
        for segment in asr_result["segments"]:
            short_prompt_text += segment["text"]
            short_prompt_end_ts = segment["end"]
            if short_prompt_end_ts >= 4:
                break

        # Trim audio
        speech_24k = speech_24k[0:int(short_prompt_end_ts * 24000)]
        speech_16k = speech_16k[0:int(short_prompt_end_ts * 16000)]

        # Detect target language
        target_language = langid.classify(target_text)[0]

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

        # Run inference
        with torch.no_grad():
            # Semantic encoding
            inputs = self.maskgct_pipeline['processor'](
                speech_16k, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs["input_features"][0].unsqueeze(0).to(self.device)
            attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(self.device)

            vq_emb = self.maskgct_pipeline['semantic_model'](
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17]
            feat = (feat - self.maskgct_pipeline['semantic_mean']) / self.maskgct_pipeline['semantic_std']
            semantic_code, _ = self.maskgct_pipeline['semantic_codec'].quantize(feat)

            # Acoustic encoding
            speech_tensor = torch.tensor(speech_24k).unsqueeze(0).to(self.device)
            vq_emb = self.maskgct_pipeline['codec_encoder'](speech_tensor.unsqueeze(1))
            _, vq, _, _, _ = self.maskgct_pipeline['codec_decoder'].quantizer(vq_emb)
            acoustic_code = vq.permute(1, 2, 0)

        # Prepare phone IDs
        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(self.device)
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(self.device)
        phone_id = torch.cat([prompt_phone_id, target_phone_id])

        # T2S: Generate semantic tokens
        with torch.no_grad():
            predict_semantic = self.maskgct_pipeline['t2s_model'].reverse_diffusion(
                semantic_code[:, :],
                target_len_frames,
                phone_id.unsqueeze(0),
                n_timesteps=n_timesteps,
                cfg=2.5,
                rescale_cfg=0.75,
            )
            combine_semantic_code = torch.cat([semantic_code[:, :], predict_semantic], dim=-1)

        # S2A: Generate acoustic tokens
        n_timesteps_s2a = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        with torch.no_grad():
            # First layer
            cond = self.maskgct_pipeline['s2a_model_1layer'].cond_emb(combine_semantic_code)
            prompt = acoustic_code[:, :, :]
            predict_1layer = self.maskgct_pipeline['s2a_model_1layer'].reverse_diffusion(
                cond=cond,
                prompt=prompt,
                temp=1.5,
                filter_thres=0.98,
                n_timesteps=n_timesteps_s2a[:1],
                cfg=2.5,
                rescale_cfg=0.75,
            )

            # Full layers
            cond = self.maskgct_pipeline['s2a_model_full'].cond_emb(combine_semantic_code)
            predict_full = self.maskgct_pipeline['s2a_model_full'].reverse_diffusion(
                cond=cond,
                prompt=prompt,
                temp=1.5,
                filter_thres=0.98,
                n_timesteps=n_timesteps_s2a,
                cfg=2.5,
                rescale_cfg=0.75,
                gt_code=predict_1layer,
            )

            # Decode to audio
            vq_emb = self.maskgct_pipeline['codec_decoder'].vq2emb(
                predict_full.permute(2, 0, 1), n_quantizers=12
            )
            recovered_audio = self.maskgct_pipeline['codec_decoder'](vq_emb)
            recovered_audio = recovered_audio[0][0].cpu().numpy()

        # Save output
        output_path = f"{OUTPUT_DIR}/maskgct_{os.urandom(8).hex()}.wav"
        sf.write(output_path, recovered_audio, 24000)

        return output_path

    def dualcodec_valle_inference(
        self,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        temperature: float = 1.0,
        top_k: int = 15,
        top_p: float = 0.85,
        repeat_penalty: float = 1.1
    ) -> Tuple[int, np.ndarray]:
        """
        Run DualCodec-VALLE inference.

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_dualcodec_valle()

        from dualcodec.utils.utils_infer import preprocess_ref_audio_text
        from dualcodec.infer.valle.utils_valle_infer import infer_process

        # Preprocess reference
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio, ref_text)

        # Run inference
        final_wave, final_sample_rate, _ = infer_process(
            ar_model_obj=self.valle_models['ar_model'],
            nar_model_obj=self.valle_models['nar_model'],
            dualcodec_inference_obj=self.valle_models['inference'],
            tokenizer_obj=self.valle_models['tokenizer'],
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=gen_text,
            cross_fade_duration=0.15,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )

        return final_sample_rate, final_wave

    def vevo_tts_inference(
        self,
        src_text: str,
        ref_wav: str,
        timbre_ref_wav: Optional[str] = None,
        ref_text: Optional[str] = None,
        src_language: str = "en",
        ref_language: str = "en"
    ) -> Tuple[int, np.ndarray]:
        """
        Run Vevo TTS inference.

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_vevo_tts()

        if timbre_ref_wav is None or timbre_ref_wav == "":
            timbre_ref_wav = ref_wav

        gen_audio = self.vevo_tts_pipeline.inference_ar_and_fm(
            src_wav_path=None,
            src_text=src_text,
            style_ref_wav_path=ref_wav,
            timbre_ref_wav_path=timbre_ref_wav,
            style_ref_wav_text=ref_text,
            src_text_language=src_language,
            style_ref_wav_text_language=ref_language,
        )

        return 24000, gen_audio.cpu().numpy()

    def vevo_vc_inference(
        self,
        src_wav: str,
        style_ref_wav: str,
        timbre_ref_wav: Optional[str] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Run Vevo Voice Conversion inference.

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_vevo_vc()

        if timbre_ref_wav is None or timbre_ref_wav == "":
            timbre_ref_wav = style_ref_wav

        gen_audio = self.vevo_vc_pipeline.inference_ar_and_fm(
            src_wav_path=src_wav,
            src_text=None,
            style_ref_wav_path=style_ref_wav,
            timbre_ref_wav_path=timbre_ref_wav,
        )

        return 24000, gen_audio.cpu().numpy()

    def vevo_timbre_inference(
        self,
        src_wav: str,
        timbre_ref_wav: str
    ) -> Tuple[int, np.ndarray]:
        """
        Run Vevo Timbre conversion inference.

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_vevo_vc()

        gen_audio = self.vevo_vc_pipeline.inference_fm(
            src_wav_path=src_wav,
            timbre_ref_wav_path=timbre_ref_wav,
            flow_matching_steps=32,
        )

        return 24000, gen_audio.cpu().numpy()

    def vevo_style_inference(
        self,
        src_wav: str,
        style_ref_wav: str
    ) -> Tuple[int, np.ndarray]:
        """
        Run Vevo Style conversion inference.

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_vevo_vc()

        gen_audio = self.vevo_vc_pipeline.inference_ar_and_fm(
            src_wav_path=src_wav,
            src_text=None,
            style_ref_wav_path=style_ref_wav,
            timbre_ref_wav_path=src_wav,  # Keep original timbre
        )

        return 24000, gen_audio.cpu().numpy()

    def load_noro(self):
        """Lazy load Noro (noise-robust VC) models."""
        if self._noro_loaded:
            return

        logger.info("Loading Noro models...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            from safetensors.torch import load_model
            from utils.util import load_config
            from models.vc.Noro.noro_model import Noro_VCmodel
            from .hubert_loader import load_hubert_extractor
            from .bigvgan_loader import load_bigvgan

            # Paths
            checkpoint_path = f"{AMPHION_ROOT}/ckpts/Noro/model.safetensors"
            config_path = f"{AMPHION_ROOT}/egs/vc/Noro/exp_config_base.json"
            bigvgan_dir = f"{AMPHION_ROOT}/ckpts/bigvgan_22khz_80band"
            hubert_model_path = f"{AMPHION_ROOT}/ckpts/hubert/hubert_base_ls960.pt"
            kmeans_model_path = f"{AMPHION_ROOT}/ckpts/hubert/hubert_base_ls960_L9_km500.bin"

            # Check for checkpoint
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Noro checkpoint not found at {checkpoint_path}. "
                    "Download from: https://drive.google.com/drive/folders/1NPzSIuSKO8o87g5ImNzpw_BgbhsZaxNg"
                )

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Noro config not found at {config_path}")

            if not os.path.exists(hubert_model_path):
                raise FileNotFoundError(
                    f"Hubert model not found at {hubert_model_path}. "
                    "Download from: https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
                )

            if not os.path.exists(kmeans_model_path):
                raise FileNotFoundError(
                    f"KMeans model not found at {kmeans_model_path}. "
                    "Download from: https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin"
                )

            # Load config
            cfg = load_config(config_path)

            # Load Hubert extractor using torchaudio (avoids fairseq dependency)
            logger.info("Loading Hubert extractor (torchaudio)...")
            hubert = load_hubert_extractor(kmeans_model_path, str(self.device))

            # Load Noro model
            logger.info("Loading Noro model...")
            model = Noro_VCmodel(cfg=cfg.model)
            load_model(model, checkpoint_path)
            model.cuda(self.device)
            model.train(False)

            # Load BigVGAN vocoder
            vocoder = None
            vocoder_config = None
            if os.path.exists(bigvgan_dir):
                logger.info("Loading BigVGAN vocoder...")
                vocoder, vocoder_config = load_bigvgan(bigvgan_dir, str(self.device))
                logger.info(f"BigVGAN vocoder loaded (sample_rate={vocoder_config.get('sampling_rate', 22050)})")
            else:
                logger.warning(
                    f"BigVGAN vocoder not found at {bigvgan_dir}. "
                    "Noro will output mel spectrograms only."
                )

            self.noro_pipeline = {
                'config': cfg,
                'hubert': hubert,
                'model': model,
                'vocoder': vocoder,
                'vocoder_config': vocoder_config,
            }

            self._noro_loaded = True
            logger.info("Noro models loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"Noro setup incomplete: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Noro models: {e}")
            import traceback
            traceback.print_exc()
            raise

    def noro_inference(
        self,
        source_wav: str,
        reference_wav: str,
        inference_steps: int = 200,
        sigma: float = 1.2
    ) -> Tuple[int, np.ndarray]:
        """
        Run Noro (noise-robust) voice conversion inference.

        Args:
            source_wav: Path to source audio
            reference_wav: Path to reference audio (target voice)
            inference_steps: Number of diffusion steps (150-300 recommended)
            sigma: Sigma parameter (0.95-1.5 recommended)

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_noro()

        import librosa
        from utils.mel import mel_spectrogram_torch
        from utils.f0 import get_f0_features_using_dio, interpolate
        from torch.nn.utils.rnn import pad_sequence

        cfg = self.noro_pipeline['config']

        # Load source audio at 16kHz
        wav, _ = librosa.load(source_wav, sr=16000)
        wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
        audio = torch.from_numpy(wav).float().to(self.device)
        audio = audio[None, :]

        # Load reference audio at 16kHz
        ref_wav, _ = librosa.load(reference_wav, sr=16000)
        ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
        ref_audio = torch.from_numpy(ref_wav).float().to(self.device)
        ref_audio = ref_audio[None, :]

        with torch.no_grad():
            # Extract reference mel spectrogram
            ref_mel = mel_spectrogram_torch(ref_audio, cfg)
            ref_mel = ref_mel.transpose(1, 2).to(device=self.device)
            ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(self.device).bool()

            # Extract content features using Hubert
            logger.info("Extracting content features with Hubert...")
            _, content_feature = self.noro_pipeline['hubert'].extract_content_features(audio)
            content_feature = content_feature.to(device=self.device)

            # Extract F0 (pitch)
            logger.info("Extracting F0 features...")
            wav_np = audio.cpu().numpy()[0, :]
            pitch_raw = get_f0_features_using_dio(wav_np, cfg.preprocess)
            pitch_raw, _ = interpolate(pitch_raw)
            frame_num = len(wav_np) // cfg.preprocess.hop_size
            pitch_raw = torch.from_numpy(pitch_raw[:frame_num]).float()
            pitch = pad_sequence([pitch_raw], batch_first=True, padding_value=0).float()
            pitch = (pitch - pitch.mean(dim=1, keepdim=True)) / (
                pitch.std(dim=1, keepdim=True) + 1e-6
            )
            pitch = pitch.to(device=self.device)

            # Run Noro diffusion inference
            logger.info(f"Running Noro inference (steps={inference_steps}, sigma={sigma})...")
            x0 = self.noro_pipeline['model'].inference(
                content_feature=content_feature,
                pitch=pitch,
                x_ref=ref_mel,
                x_ref_mask=ref_mask,
                inference_steps=inference_steps,
                sigma=sigma,
            )

            # Convert mel to audio using BigVGAN vocoder
            if self.noro_pipeline['vocoder'] is not None:
                logger.info("Converting mel to audio with BigVGAN...")
                # x0 is [B, T, 80], need [B, 80, T] for BigVGAN
                mel_output = x0.transpose(1, 2)
                audio_output = self.noro_pipeline['vocoder'](mel_output)
                audio_output = audio_output.squeeze().cpu().numpy()

                # BigVGAN outputs at 22050 Hz, resample to 16000 Hz
                vocoder_sr = self.noro_pipeline['vocoder_config'].get('sampling_rate', 22050)
                if vocoder_sr != 16000:
                    logger.info(f"Resampling from {vocoder_sr}Hz to 16000Hz...")
                    audio_output = librosa.resample(
                        audio_output,
                        orig_sr=vocoder_sr,
                        target_sr=16000
                    )

                return 16000, audio_output
            else:
                # No vocoder available - return silence
                logger.warning("BigVGAN vocoder not available, returning silence")
                audio_length = x0.shape[1] * cfg.preprocess.hop_size
                return 16000, np.zeros(audio_length)

    def load_metis(self, model_type: str = "tts"):
        """Lazy load Metis foundation model.

        Args:
            model_type: One of "tts", "vc", "se", "tse" (default: "tts")
        """
        if self._metis_loaded:
            return

        logger.info(f"Loading Metis model (type={model_type})...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            from huggingface_hub import snapshot_download
            from models.tts.metis.metis import Metis
            from utils.util import load_config

            # Load config based on model type
            config_path = f"{AMPHION_ROOT}/models/tts/metis/config/{model_type}.json"
            if not os.path.exists(config_path):
                # Fallback to TTS config if specific one doesn't exist
                config_path = f"{AMPHION_ROOT}/models/tts/metis/config/tts.json"

            metis_cfg = load_config(config_path)

            # Download checkpoint from HuggingFace
            ckpt_dir = snapshot_download(
                repo_id="amphion/maskgct",
                repo_type="model",
                local_dir=f"{AMPHION_ROOT}/models/tts/maskgct/ckpt",
                allow_patterns=["t2s/model.safetensors"],
            )
            ckpt_path = os.path.join(ckpt_dir, "t2s/model.safetensors")

            # Initialize Metis
            metis = Metis(
                ckpt_path=ckpt_path,
                cfg=metis_cfg,
                device=str(self.device),
                model_type=model_type,
            )

            self.metis_pipeline = {
                'model': metis,
                'config': metis_cfg,
                'model_type': model_type,
            }

            self._metis_loaded = True
            logger.info("Metis model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Metis model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def metis_inference(
        self,
        prompt_wav_path: str,
        target_text: str,
        prompt_text: str,
        model_type: str = "tts",
        target_len: Optional[float] = None,
        n_timesteps: int = 25,
        cfg_scale: float = 2.5,
        source_wav_path: Optional[str] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Run Metis inference.

        Args:
            prompt_wav_path: Path to reference/prompt audio
            target_text: Text to synthesize (for TTS)
            prompt_text: Transcript of prompt audio
            model_type: One of "tts", "vc", "se", "tse"
            target_len: Target duration in seconds (None for auto)
            n_timesteps: Number of diffusion timesteps
            cfg_scale: Classifier-free guidance scale
            source_wav_path: Path to source audio (for VC/SE/TSE tasks)

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_metis(model_type)

        logger.info(f"Running Metis {model_type} inference...")

        metis = self.metis_pipeline['model']

        # Run inference based on model type
        if model_type == "tts":
            gen_speech = metis(
                prompt_speech_path=prompt_wav_path,
                text=target_text,
                prompt_text=prompt_text,
                target_len=target_len,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                model_type=model_type,
            )
        elif model_type in ["vc", "tse"]:
            # Voice conversion or target speaker extraction
            gen_speech = metis(
                prompt_speech_path=prompt_wav_path,
                source_speech_path=source_wav_path,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                model_type=model_type,
            )
        elif model_type == "se":
            # Speech enhancement (no prompt needed)
            gen_speech = metis(
                source_speech_path=source_wav_path,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                model_type=model_type,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return 24000, gen_speech

    def load_vevosing(self, mode: str = "fm"):
        """
        Load VevoSing (Vevo1.5) model for singing voice conversion.

        Args:
            mode: 'fm' for flow-matching only, 'ar' for AR+FM
        """
        if self._vevosing_loaded:
            return

        logger.info("Loading VevoSing model...")

        try:
            import sys
            sys.path.insert(0, AMPHION_ROOT)

            from huggingface_hub import snapshot_download
            from models.svc.vevosing.vevosing_utils import VevosingInferencePipeline

            # Download Content-Style Tokenizer
            local_dir = snapshot_download(
                repo_id="amphion/Vevo1.5",
                repo_type="model",
                cache_dir="./ckpts/Vevo1.5",
                allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
            )
            contentstyle_tokenizer_path = os.path.join(
                local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
            )

            # Download Flow Matching Transformer
            model_name = "fm_emilia101k_singnet7k"
            local_dir = snapshot_download(
                repo_id="amphion/Vevo1.5",
                repo_type="model",
                cache_dir="./ckpts/Vevo1.5",
                allow_patterns=[f"acoustic_modeling/{model_name}/*"],
            )
            fmt_cfg_path = f"{AMPHION_ROOT}/models/svc/vevosing/config/{model_name}.json"
            fmt_ckpt_path = os.path.join(local_dir, f"acoustic_modeling/{model_name}")

            # Download Vocoder
            local_dir = snapshot_download(
                repo_id="amphion/Vevo1.5",
                repo_type="model",
                cache_dir="./ckpts/Vevo1.5",
                allow_patterns=["acoustic_modeling/Vocoder/*"],
            )
            vocoder_cfg_path = f"{AMPHION_ROOT}/models/svc/vevosing/config/vocoder.json"
            vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

            # For AR mode, also download prosody tokenizer and AR model
            prosody_tokenizer_path = None
            ar_cfg_path = None
            ar_ckpt_path = None

            if mode == "ar":
                # Download Prosody Tokenizer
                local_dir = snapshot_download(
                    repo_id="amphion/Vevo1.5",
                    repo_type="model",
                    cache_dir="./ckpts/Vevo1.5",
                    allow_patterns=["tokenizer/prosody_fvq512_6.25hz/*"],
                )
                prosody_tokenizer_path = os.path.join(
                    local_dir, "tokenizer/prosody_fvq512_6.25hz"
                )

                # Download AR model
                ar_model_name = "ar_emilia101k_singnet7k"
                local_dir = snapshot_download(
                    repo_id="amphion/Vevo1.5",
                    repo_type="model",
                    cache_dir="./ckpts/Vevo1.5",
                    allow_patterns=[f"contentstyle_modeling/{ar_model_name}/*"],
                )
                ar_cfg_path = f"{AMPHION_ROOT}/models/svc/vevosing/config/{ar_model_name}.json"
                ar_ckpt_path = os.path.join(local_dir, f"contentstyle_modeling/{ar_model_name}")

            # Create inference pipeline
            self.vevosing_pipeline = {
                'inference': VevosingInferencePipeline(
                    content_style_tokenizer_ckpt_path=contentstyle_tokenizer_path,
                    fmt_cfg_path=fmt_cfg_path,
                    fmt_ckpt_path=fmt_ckpt_path,
                    vocoder_cfg_path=vocoder_cfg_path,
                    vocoder_ckpt_path=vocoder_ckpt_path,
                    prosody_tokenizer_ckpt_path=prosody_tokenizer_path,
                    ar_cfg_path=ar_cfg_path,
                    ar_ckpt_path=ar_ckpt_path,
                    device=self.device,
                ),
                'mode': mode,
            }

            self._vevosing_loaded = True
            logger.info("VevoSing model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VevoSing model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def vevosing_inference(
        self,
        content_wav_path: str,
        reference_wav_path: str,
        mode: str = "fm",
        use_shifted_src: bool = True,
        flow_matching_steps: int = 32,
    ) -> Tuple[int, np.ndarray]:
        """
        Run VevoSing singing voice conversion inference.

        Args:
            content_wav_path: Path to source audio (content/melody)
            reference_wav_path: Path to reference audio (timbre)
            mode: 'fm' for flow-matching only (timbre), 'ar' for AR+FM (full control)
            use_shifted_src: Use pitch-shifted source for prosody extraction
            flow_matching_steps: Number of flow matching steps

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        self.load_vevosing(mode)

        logger.info(f"Running VevoSing {mode} inference...")

        inference_pipeline = self.vevosing_pipeline['inference']

        if mode == "fm":
            gen_audio = inference_pipeline.inference_fm(
                src_wav_path=content_wav_path,
                timbre_ref_wav_path=reference_wav_path,
                use_shifted_src_to_extract_prosody=use_shifted_src,
                flow_matching_steps=flow_matching_steps,
            )
        else:
            # AR+FM mode with full control
            gen_audio = inference_pipeline.inference_ar(
                src_wav_path=content_wav_path,
                timbre_ref_wav_path=reference_wav_path,
                flow_matching_steps=flow_matching_steps,
            )

        return 24000, gen_audio

    # ===========================
    # TTA (Text-to-Audio) Methods
    # ===========================

    def load_audioldm(self):
        """Lazy load AudioLDM model for text-to-audio generation."""
        if self._audioldm_loaded:
            return

        logger.info("Loading AudioLDM model...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            # AudioLDM uses latent diffusion for audio generation
            # Placeholder implementation - actual implementation would load from HuggingFace
            logger.info("AudioLDM model loaded (placeholder)")
            self._audioldm_loaded = True

        except Exception as e:
            logger.error(f"Failed to load AudioLDM model: {e}")
            raise

    def tta_audioldm_inference(
        self,
        text_prompt: str,
        duration: float = 10.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> str:
        """
        Run AudioLDM inference for text-to-audio generation.

        Args:
            text_prompt: Text description of desired audio
            duration: Target duration in seconds
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            Path to generated audio file
        """
        self.load_audioldm()

        logger.info(f"Running AudioLDM inference: '{text_prompt[:50]}...'")

        # Placeholder implementation
        # In actual implementation, this would:
        # 1. Load AudioLDM pipeline from HuggingFace (amphion/audioldm)
        # 2. Generate audio using the text prompt
        # 3. Save to output directory

        output_path = f"{OUTPUT_DIR}/audioldm_{os.urandom(8).hex()}.wav"

        # Generate silent audio as placeholder
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)
        sf.write(output_path, audio, sample_rate)

        logger.info(f"AudioLDM generated audio saved to {output_path}")
        return output_path

    def load_picoaudio(self):
        """Lazy load PicoAudio model for text-to-audio generation."""
        if self._picoaudio_loaded:
            return

        logger.info("Loading PicoAudio model...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            # PicoAudio is a lightweight TTA model
            logger.info("PicoAudio model loaded (placeholder)")
            self._picoaudio_loaded = True

        except Exception as e:
            logger.error(f"Failed to load PicoAudio model: {e}")
            raise

    def tta_picoaudio_inference(
        self,
        text_prompt: str,
        duration: float = 10.0,
        num_inference_steps: int = 25,
    ) -> str:
        """
        Run PicoAudio inference for text-to-audio generation.

        Args:
            text_prompt: Text description of desired audio
            duration: Target duration in seconds
            num_inference_steps: Number of generation steps

        Returns:
            Path to generated audio file
        """
        self.load_picoaudio()

        logger.info(f"Running PicoAudio inference: '{text_prompt[:50]}...'")

        output_path = f"{OUTPUT_DIR}/picoaudio_{os.urandom(8).hex()}.wav"

        # Generate silent audio as placeholder
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)
        sf.write(output_path, audio, sample_rate)

        logger.info(f"PicoAudio generated audio saved to {output_path}")
        return output_path

    # ===========================
    # Codec Methods
    # ===========================

    def load_dualcodec_codec(self):
        """Lazy load DualCodec for encoding/decoding audio."""
        if self._dualcodec_codec_loaded:
            return

        logger.info("Loading DualCodec codec...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            import dualcodec

            # Load DualCodec model
            dualcodec_model = dualcodec.get_model("12hz_v1")
            dualcodec_inference = dualcodec.Inference(
                dualcodec_model=dualcodec_model,
                device=self.device,
                autocast=True
            )

            self.dualcodec_codec = {
                'model': dualcodec_model,
                'inference': dualcodec_inference,
            }

            self._dualcodec_codec_loaded = True
            logger.info("DualCodec codec loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load DualCodec codec: {e}")
            raise

    def codec_encode(self, audio_path: str, codec_type: str = "dualcodec") -> dict:
        """
        Encode audio to discrete tokens using neural codec.

        Args:
            audio_path: Path to input audio file
            codec_type: Type of codec ("dualcodec" or "facodec")

        Returns:
            Dictionary containing tokens and metadata
        """
        logger.info(f"Encoding audio with {codec_type}: {audio_path}")

        if codec_type == "dualcodec":
            self.load_dualcodec_codec()

            # Load audio
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Encode using DualCodec
            inference = self.dualcodec_codec['inference']
            with torch.no_grad():
                tokens = inference.encode(audio_tensor)

            return {
                "tokens": tokens.cpu().numpy().tolist(),
                "codec_type": codec_type,
                "sample_rate": sr,
            }

        elif codec_type == "facodec":
            # Placeholder for FACodec encoding
            logger.warning("FACodec encoding not yet implemented, returning dummy tokens")
            return {
                "tokens": [[0] * 100],
                "codec_type": codec_type,
                "sample_rate": 16000,
                "note": "FACodec not implemented - placeholder"
            }

        else:
            raise ValueError(f"Unknown codec type: {codec_type}")

    def codec_decode(self, tokens: list, codec_type: str = "dualcodec") -> str:
        """
        Decode discrete tokens to audio using neural codec.

        Args:
            tokens: List of token arrays or single token array
            codec_type: Type of codec ("dualcodec" or "facodec")

        Returns:
            Path to decoded audio file
        """
        logger.info(f"Decoding tokens with {codec_type}")

        output_path = f"{OUTPUT_DIR}/codec_decode_{os.urandom(8).hex()}.wav"

        if codec_type == "dualcodec":
            self.load_dualcodec_codec()

            # Convert tokens to tensor
            tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(self.device)
            if tokens_tensor.ndim == 1:
                tokens_tensor = tokens_tensor.unsqueeze(0)

            # Decode using DualCodec
            inference = self.dualcodec_codec['inference']
            with torch.no_grad():
                audio = inference.decode(tokens_tensor)

            # Save audio
            audio_np = audio.cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            sf.write(output_path, audio_np, 16000)

        elif codec_type == "facodec":
            # Placeholder for FACodec decoding
            logger.warning("FACodec decoding not yet implemented, returning silent audio")
            audio = np.zeros(16000, dtype=np.float32)
            sf.write(output_path, audio, 16000)

        else:
            raise ValueError(f"Unknown codec type: {codec_type}")

        return output_path

    def load_facodec(self):
        """Lazy load FACodec model."""
        if self._facodec_loaded:
            return

        logger.info("Loading FACodec model...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            # FACodec implementation
            logger.info("FACodec model loaded (placeholder)")
            self._facodec_loaded = True

        except Exception as e:
            logger.error(f"Failed to load FACodec model: {e}")
            raise

    # ===========================
    # Vocoder Methods
    # ===========================

    def load_hifigan(self):
        """Lazy load HiFi-GAN vocoder."""
        if self._hifigan_loaded:
            return

        logger.info("Loading HiFi-GAN vocoder...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            # HiFi-GAN vocoder implementation
            logger.info("HiFi-GAN vocoder loaded (placeholder)")
            self._hifigan_loaded = True

        except Exception as e:
            logger.error(f"Failed to load HiFi-GAN vocoder: {e}")
            raise

    def load_bigvgan_vocoder(self):
        """Lazy load BigVGAN vocoder."""
        if self._bigvgan_loaded:
            return

        logger.info("Loading BigVGAN vocoder...")

        try:
            import sys
            if AMPHION_ROOT not in sys.path:
                sys.path.insert(0, AMPHION_ROOT)

            from .bigvgan_loader import load_bigvgan

            bigvgan_dir = f"{AMPHION_ROOT}/ckpts/bigvgan_22khz_80band"
            vocoder, vocoder_config = load_bigvgan(bigvgan_dir, str(self.device))

            self.bigvgan_vocoder = {
                'model': vocoder,
                'config': vocoder_config,
            }

            self._bigvgan_loaded = True
            logger.info("BigVGAN vocoder loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load BigVGAN vocoder: {e}")
            raise

    def vocoder_inference(self, mel_path: str, vocoder_type: str = "hifigan") -> str:
        """
        Convert mel-spectrogram to audio using neural vocoder.

        Args:
            mel_path: Path to mel-spectrogram numpy file
            vocoder_type: Type of vocoder ("hifigan" or "bigvgan")

        Returns:
            Path to generated audio file
        """
        logger.info(f"Running {vocoder_type} vocoder inference...")

        # Load mel spectrogram
        mel = np.load(mel_path)
        mel_tensor = torch.from_numpy(mel).float().to(self.device)
        if mel_tensor.ndim == 2:
            mel_tensor = mel_tensor.unsqueeze(0)

        output_path = f"{OUTPUT_DIR}/vocoder_{vocoder_type}_{os.urandom(8).hex()}.wav"

        if vocoder_type == "hifigan":
            self.load_hifigan()

            # Placeholder for HiFi-GAN inference
            logger.warning("HiFi-GAN inference not yet implemented, returning silent audio")
            audio_length = mel.shape[-1] * 256  # Approximate hop_size
            audio = np.zeros(audio_length, dtype=np.float32)
            sf.write(output_path, audio, 22050)

        elif vocoder_type == "bigvgan":
            self.load_bigvgan_vocoder()

            # BigVGAN inference
            vocoder = self.bigvgan_vocoder['model']
            with torch.no_grad():
                audio = vocoder(mel_tensor)
            audio_np = audio.squeeze().cpu().numpy()
            sf.write(output_path, audio_np, 22050)

        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        return output_path

    # ===========================
    # Additional SVC Methods
    # ===========================

    def diffcomosvc_inference(
        self,
        content_wav_path: str,
        reference_wav_path: str,
    ) -> Tuple[int, np.ndarray]:
        """
        Run DiffComoSVC singing voice conversion.

        Args:
            content_wav_path: Path to source audio (content/melody)
            reference_wav_path: Path to reference audio (timbre)

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        logger.info("Running DiffComoSVC inference (experimental)...")

        # Placeholder implementation - DiffComoSVC may not have pretrained checkpoints
        logger.warning("DiffComoSVC is experimental and may not have pretrained checkpoints")

        # Return copy of source audio as placeholder
        audio, sr = sf.read(content_wav_path)
        return sr, audio

    def transformersvc_inference(
        self,
        content_wav_path: str,
        reference_wav_path: str,
    ) -> Tuple[int, np.ndarray]:
        """
        Run TransformerSVC singing voice conversion.

        Args:
            content_wav_path: Path to source audio (content/melody)
            reference_wav_path: Path to reference audio (timbre)

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        logger.info("Running TransformerSVC inference (experimental)...")

        logger.warning("TransformerSVC is experimental and may not have pretrained checkpoints")

        # Return copy of source audio as placeholder
        audio, sr = sf.read(content_wav_path)
        return sr, audio

    def vitssvc_inference(
        self,
        content_wav_path: str,
        reference_wav_path: str,
    ) -> Tuple[int, np.ndarray]:
        """
        Run VitsSVC singing voice conversion.

        Args:
            content_wav_path: Path to source audio (content/melody)
            reference_wav_path: Path to reference audio (timbre)

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        logger.info("Running VitsSVC inference (experimental)...")

        logger.warning("VitsSVC is experimental and may not have pretrained checkpoints")

        # Return copy of source audio as placeholder
        audio, sr = sf.read(content_wav_path)
        return sr, audio

    def multiplecontentssvc_inference(
        self,
        content_wav_path: str,
        reference_wav_path: str,
    ) -> Tuple[int, np.ndarray]:
        """
        Run MultipleContentsSVC singing voice conversion.

        Args:
            content_wav_path: Path to source audio (content/melody)
            reference_wav_path: Path to reference audio (timbre)

        Returns:
            Tuple of (sample_rate, audio_data)
        """
        logger.info("Running MultipleContentsSVC inference (experimental)...")

        logger.warning("MultipleContentsSVC is experimental and may not have pretrained checkpoints")

        # Return copy of source audio as placeholder
        audio, sr = sf.read(content_wav_path)
        return sr, audio
