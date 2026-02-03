#!/usr/bin/env python3
# Convert vocals between two artists using VevoSing

import os
import sys
from huggingface_hub import snapshot_download
from vevosing_utils import *

def load_inference_pipeline():
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
    )
    contentstyle_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )

    # ===== Flow Matching Transformer =====
    model_name = "fm_emilia101k_singnet7k"
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=[f"acoustic_modeling/{model_name}/*"],
    )
    fm_cfg_path = f"./models/svc/vevosing/config/{model_name}.json"
    fm_ckpt_path = os.path.join(local_dir, f"acoustic_modeling/{model_name}")

    # ===== Vocoder =====
    vocoder_cfg_path = "./models/svc/vevosing/config/vocoder.json"
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference Pipeline =====
    inference_pipeline = VevosingInferencePipeline(
        content_style_tokenizer_ckpt_path=contentstyle_tokenizer_ckpt_path,
        fmt_cfg_path=fm_cfg_path,
        fmt_ckpt_path=fm_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    return inference_pipeline

def convert_voice(pipeline, src_path, ref_path, output_path):
    print(f"Converting: {os.path.basename(src_path)} -> {os.path.basename(ref_path)} voice", flush=True)
    print(f"Starting inference with 32 flow matching steps (better quality)...", flush=True)
    gen_audio = pipeline.inference_fm(
        src_wav_path=src_path,
        timbre_ref_wav_path=ref_path,
        use_shifted_src_to_extract_prosody=True,
        flow_matching_steps=32,
    )
    print(f"Inference complete! Saving audio...", flush=True)
    save_audio(gen_audio, output_path=output_path)
    print(f"✓ Saved: {output_path}", flush=True)

if __name__ == "__main__":
    print("Loading VevoSing pipeline...")
    pipeline = load_inference_pipeline()

    # Define paths
    conor_vocals = "/home/kp/Music/1/Conor_Maynard-vocals.wav"
    william_vocals = "/home/kp/Music/1/William_Singe-vocals.wav"
    output_dir = "/home/kp/Music/1/converted"
    os.makedirs(output_dir, exist_ok=True)

    # Convert Conor to William's voice
    convert_voice(
        pipeline,
        conor_vocals,
        william_vocals,
        os.path.join(output_dir, "Conor_as_William.wav")
    )

    print("\nDone! Conversion complete.", flush=True)
