"""
Evaluation Routes

Provides API endpoints for audio evaluation metrics including:
- F0 analysis (RMSE, correlation, V/UV F1)
- Spectral metrics (MCD, PESQ, STOI, SI-SDR)
- Energy analysis (RMSE, correlation)

Note: Speaker similarity requires directory-based batch evaluation
which is not yet exposed via this API.
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

# Amphion paths
AMPHION_ROOT = "/home/kp/repo2/Amphion"
OUTPUT_DIR = f"{AMPHION_ROOT}/output/web"


async def save_upload_to_temp(upload: UploadFile) -> str:
    """Save uploaded file to temporary location."""
    suffix = os.path.splitext(upload.filename)[1] if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_DIR) as tmp:
        content = await upload.read()
        tmp.write(content)
        return tmp.name


def cleanup_files(*paths):
    """Remove temporary files."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")


@router.post("/f0")
async def evaluate_f0(
    background_tasks: BackgroundTasks,
    reference_audio: UploadFile = File(..., description="Reference audio file"),
    generated_audio: UploadFile = File(..., description="Generated audio file"),
    fs: int = Form(24000, description="Sample rate for analysis"),
    need_mean: bool = Form(True, description="Subtract mean from F0"),
) -> Dict[str, Any]:
    """
    Compute F0-based evaluation metrics.

    Returns F0 RMSE, Pearson correlation, and V/UV F1 score.
    """
    ref_path = None
    gen_path = None

    try:
        # Save uploads
        ref_path = await save_upload_to_temp(reference_audio)
        gen_path = await save_upload_to_temp(generated_audio)

        import sys
        if AMPHION_ROOT not in sys.path:
            sys.path.insert(0, AMPHION_ROOT)

        # Import evaluation functions
        from evaluation.metrics.f0.f0_rmse import extract_f0rmse
        from evaluation.metrics.f0.f0_pearson_coefficients import extract_fpc
        from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv

        kwargs = {"kwargs": {"fs": fs, "method": "cut", "need_mean": need_mean}}

        # Calculate metrics
        results = {}

        try:
            f0_rmse = extract_f0rmse(ref_path, gen_path, **kwargs)
            results["f0_rmse"] = float(f0_rmse)
        except Exception as e:
            logger.warning(f"F0 RMSE failed: {e}")
            results["f0_rmse"] = None

        try:
            fpc = extract_fpc(ref_path, gen_path, **kwargs)
            results["f0_pearson_correlation"] = float(fpc)
        except Exception as e:
            logger.warning(f"FPC failed: {e}")
            results["f0_pearson_correlation"] = None

        try:
            v_uv_f1 = extract_f1_v_uv(ref_path, gen_path, **kwargs)
            results["v_uv_f1_score"] = float(v_uv_f1)
        except Exception as e:
            logger.warning(f"V/UV F1 failed: {e}")
            results["v_uv_f1_score"] = None

        # Schedule cleanup
        background_tasks.add_task(cleanup_files, ref_path, gen_path)

        return {
            "status": "success",
            "metrics": results
        }

    except Exception as e:
        logger.error(f"F0 evaluation failed: {e}")
        if ref_path:
            cleanup_files(ref_path)
        if gen_path:
            cleanup_files(gen_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spectral")
async def evaluate_spectral(
    background_tasks: BackgroundTasks,
    reference_audio: UploadFile = File(..., description="Reference audio file"),
    generated_audio: UploadFile = File(..., description="Generated audio file"),
    fs: int = Form(16000, description="Sample rate for analysis"),
) -> Dict[str, Any]:
    """
    Compute spectral evaluation metrics.

    Returns PESQ, STOI, MCD, SI-SDR, and MRSTFT distance.
    """
    ref_path = None
    gen_path = None

    try:
        # Save uploads
        ref_path = await save_upload_to_temp(reference_audio)
        gen_path = await save_upload_to_temp(generated_audio)

        import sys
        if AMPHION_ROOT not in sys.path:
            sys.path.insert(0, AMPHION_ROOT)

        kwargs = {"kwargs": {"fs": fs, "method": "cut"}}

        results = {}

        # PESQ
        try:
            from evaluation.metrics.spectrogram.pesq import extract_pesq
            results["pesq"] = float(extract_pesq(ref_path, gen_path, **kwargs))
        except Exception as e:
            logger.warning(f"PESQ failed: {e}")
            results["pesq"] = None

        # STOI
        try:
            from evaluation.metrics.spectrogram.short_time_objective_intelligibility import extract_stoi
            results["stoi"] = float(extract_stoi(ref_path, gen_path, **kwargs))
        except Exception as e:
            logger.warning(f"STOI failed: {e}")
            results["stoi"] = None

        # MCD
        try:
            from evaluation.metrics.spectrogram.mel_cepstral_distortion import extract_mcd
            results["mcd"] = float(extract_mcd(ref_path, gen_path, **kwargs))
        except Exception as e:
            logger.warning(f"MCD failed: {e}")
            results["mcd"] = None

        # SI-SDR
        try:
            from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import extract_si_sdr
            results["si_sdr"] = float(extract_si_sdr(ref_path, gen_path, **kwargs))
        except Exception as e:
            logger.warning(f"SI-SDR failed: {e}")
            results["si_sdr"] = None

        # MRSTFT Distance
        try:
            from evaluation.metrics.spectrogram.multi_resolution_stft_distance import extract_mstft
            results["mrstft_distance"] = float(extract_mstft(ref_path, gen_path, **kwargs))
        except Exception as e:
            logger.warning(f"MRSTFT failed: {e}")
            results["mrstft_distance"] = None

        # Schedule cleanup
        background_tasks.add_task(cleanup_files, ref_path, gen_path)

        return {
            "status": "success",
            "metrics": results
        }

    except Exception as e:
        logger.error(f"Spectral evaluation failed: {e}")
        if ref_path:
            cleanup_files(ref_path)
        if gen_path:
            cleanup_files(gen_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/energy")
async def evaluate_energy(
    background_tasks: BackgroundTasks,
    reference_audio: UploadFile = File(..., description="Reference audio file"),
    generated_audio: UploadFile = File(..., description="Generated audio file"),
    fs: int = Form(24000, description="Sample rate for analysis"),
) -> Dict[str, Any]:
    """
    Compute energy-based evaluation metrics.

    Returns Energy RMSE and Energy Pearson correlation.
    """
    ref_path = None
    gen_path = None

    try:
        # Save uploads
        ref_path = await save_upload_to_temp(reference_audio)
        gen_path = await save_upload_to_temp(generated_audio)

        import sys
        if AMPHION_ROOT not in sys.path:
            sys.path.insert(0, AMPHION_ROOT)

        from evaluation.metrics.energy.energy_rmse import extract_energy_rmse
        from evaluation.metrics.energy.energy_pearson_coefficients import extract_energy_pearson_coeffcients

        kwargs = {"kwargs": {"fs": fs, "method": "cut"}}

        results = {}

        try:
            energy_rmse = extract_energy_rmse(ref_path, gen_path, **kwargs)
            results["energy_rmse"] = float(energy_rmse)
        except Exception as e:
            logger.warning(f"Energy RMSE failed: {e}")
            results["energy_rmse"] = None

        try:
            energy_corr = extract_energy_pearson_coeffcients(ref_path, gen_path, **kwargs)
            results["energy_pearson_correlation"] = float(energy_corr)
        except Exception as e:
            logger.warning(f"Energy Pearson failed: {e}")
            results["energy_pearson_correlation"] = None

        # Schedule cleanup
        background_tasks.add_task(cleanup_files, ref_path, gen_path)

        return {
            "status": "success",
            "metrics": results
        }

    except Exception as e:
        logger.error(f"Energy evaluation failed: {e}")
        if ref_path:
            cleanup_files(ref_path)
        if gen_path:
            cleanup_files(gen_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intelligibility")
async def evaluate_intelligibility(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    reference_text: str = Form(..., description="Reference transcription"),
) -> Dict[str, Any]:
    """
    Compute intelligibility metrics using Whisper ASR.

    Returns Character Error Rate (CER) and Word Error Rate (WER).
    """
    audio_path = None

    try:
        # Save upload
        audio_path = await save_upload_to_temp(audio)

        import sys
        if AMPHION_ROOT not in sys.path:
            sys.path.insert(0, AMPHION_ROOT)

        # Use Whisper for transcription
        import whisper

        # Load Whisper model (cached after first load)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        hypothesis = result["text"].strip()

        # Calculate CER
        from evaluation.metrics.intelligibility.character_error_rate import extract_cer
        cer = extract_cer(reference_text, hypothesis)

        # Calculate WER
        from evaluation.metrics.intelligibility.word_error_rate import extract_wer
        wer = extract_wer(reference_text, hypothesis)

        # Schedule cleanup
        background_tasks.add_task(cleanup_files, audio_path)

        return {
            "status": "success",
            "transcription": hypothesis,
            "reference": reference_text,
            "metrics": {
                "cer": float(cer),
                "wer": float(wer),
            }
        }

    except Exception as e:
        logger.error(f"Intelligibility evaluation failed: {e}")
        if audio_path:
            cleanup_files(audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def evaluate_batch(
    background_tasks: BackgroundTasks,
    reference_audio: UploadFile = File(..., description="Reference audio file"),
    generated_audio: UploadFile = File(..., description="Generated audio file"),
    fs: int = Form(16000, description="Sample rate for analysis"),
    metrics: str = Form("pesq,stoi,mcd,f0_rmse", description="Comma-separated list of metrics"),
    need_mean: bool = Form(True, description="Subtract mean from F0 (for F0 metrics)"),
) -> Dict[str, Any]:
    """
    Compute multiple evaluation metrics in a single request.

    Available metrics: pesq, stoi, mcd, si_sdr, mrstft, f0_rmse, fpc, v_uv_f1,
                       energy_rmse, energy_corr
    """
    ref_path = None
    gen_path = None

    try:
        # Save uploads
        ref_path = await save_upload_to_temp(reference_audio)
        gen_path = await save_upload_to_temp(generated_audio)

        import sys
        if AMPHION_ROOT not in sys.path:
            sys.path.insert(0, AMPHION_ROOT)

        kwargs = {"kwargs": {"fs": fs, "method": "cut", "need_mean": need_mean}}
        requested_metrics = [m.strip().lower() for m in metrics.split(",")]

        results = {}

        # Spectral metrics
        if "pesq" in requested_metrics:
            try:
                from evaluation.metrics.spectrogram.pesq import extract_pesq
                results["pesq"] = float(extract_pesq(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"PESQ failed: {e}")
                results["pesq"] = None

        if "stoi" in requested_metrics:
            try:
                from evaluation.metrics.spectrogram.short_time_objective_intelligibility import extract_stoi
                results["stoi"] = float(extract_stoi(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"STOI failed: {e}")
                results["stoi"] = None

        if "mcd" in requested_metrics:
            try:
                from evaluation.metrics.spectrogram.mel_cepstral_distortion import extract_mcd
                results["mcd"] = float(extract_mcd(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"MCD failed: {e}")
                results["mcd"] = None

        if "si_sdr" in requested_metrics:
            try:
                from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import extract_si_sdr
                results["si_sdr"] = float(extract_si_sdr(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"SI-SDR failed: {e}")
                results["si_sdr"] = None

        if "mrstft" in requested_metrics:
            try:
                from evaluation.metrics.spectrogram.multi_resolution_stft_distance import extract_mstft
                results["mrstft_distance"] = float(extract_mstft(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"MRSTFT failed: {e}")
                results["mrstft_distance"] = None

        # F0 metrics
        if "f0_rmse" in requested_metrics:
            try:
                from evaluation.metrics.f0.f0_rmse import extract_f0rmse
                results["f0_rmse"] = float(extract_f0rmse(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"F0 RMSE failed: {e}")
                results["f0_rmse"] = None

        if "fpc" in requested_metrics:
            try:
                from evaluation.metrics.f0.f0_pearson_coefficients import extract_fpc
                results["f0_pearson_correlation"] = float(extract_fpc(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"FPC failed: {e}")
                results["f0_pearson_correlation"] = None

        if "v_uv_f1" in requested_metrics:
            try:
                from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv
                results["v_uv_f1_score"] = float(extract_f1_v_uv(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"V/UV F1 failed: {e}")
                results["v_uv_f1_score"] = None

        # Energy metrics
        if "energy_rmse" in requested_metrics:
            try:
                from evaluation.metrics.energy.energy_rmse import extract_energy_rmse
                results["energy_rmse"] = float(extract_energy_rmse(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"Energy RMSE failed: {e}")
                results["energy_rmse"] = None

        if "energy_corr" in requested_metrics:
            try:
                from evaluation.metrics.energy.energy_pearson_coefficients import extract_energy_pearson_coeffcients
                results["energy_pearson_correlation"] = float(extract_energy_pearson_coeffcients(ref_path, gen_path, **kwargs))
            except Exception as e:
                logger.warning(f"Energy Pearson failed: {e}")
                results["energy_pearson_correlation"] = None

        # Schedule cleanup
        background_tasks.add_task(cleanup_files, ref_path, gen_path)

        return {
            "status": "success",
            "requested_metrics": requested_metrics,
            "metrics": results
        }

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        if ref_path:
            cleanup_files(ref_path)
        if gen_path:
            cleanup_files(gen_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def list_available_metrics() -> Dict[str, Any]:
    """
    List all available evaluation metrics.
    """
    return {
        "status": "success",
        "metrics": {
            "f0": {
                "f0_rmse": "F0 Root Mean Square Error",
                "fpc": "F0 Pearson Correlation",
                "v_uv_f1": "Voiced/Unvoiced F1 Score"
            },
            "spectral": {
                "pesq": "Perceptual Evaluation of Speech Quality (-0.5 to 4.5)",
                "stoi": "Short-Time Objective Intelligibility (0 to 1)",
                "mcd": "Mel Cepstral Distortion (lower is better)",
                "si_sdr": "Scale-Invariant Signal-to-Distortion Ratio (dB)",
                "mrstft": "Multi-Resolution STFT Distance"
            },
            "energy": {
                "energy_rmse": "Energy Root Mean Square Error",
                "energy_corr": "Energy Pearson Correlation"
            },
            "intelligibility": {
                "cer": "Character Error Rate (0 to 1)",
                "wer": "Word Error Rate (0 to 1)"
            }
        },
        "endpoints": [
            "POST /api/evaluation/f0 - F0 metrics",
            "POST /api/evaluation/spectral - Spectral metrics",
            "POST /api/evaluation/energy - Energy metrics",
            "POST /api/evaluation/intelligibility - ASR-based metrics",
            "POST /api/evaluation/batch - Multiple metrics at once"
        ]
    }
