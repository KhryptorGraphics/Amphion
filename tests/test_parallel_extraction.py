#!/usr/bin/env python
# Copyright (c) 2023 Amphion.
# Integration test for parallel acoustic feature extraction

import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.acoustic_extractor import (
    extract_utt_acoustic_features_parallel,
    extract_utt_acoustic_features_serial,
)
from utils import audio


@dataclass
class PreprocessConfig:
    """Minimal config for acoustic feature extraction testing."""
    sample_rate: int = 24000
    n_mel: int = 80
    win_size: int = 480
    hop_size: int = 120
    n_fft: int = 1024
    fmin: int = 0
    fmax: int = 12000
    extract_mel: bool = True
    mel_extract_mode: str = "librosa"  # Use librosa instead of taco to avoid TacotronSTFT issues
    extract_linear_spec: bool = False
    extract_pitch: bool = False  # Disabled to avoid parselmouth dependency
    pitch_extractor: str = "parselmouth"
    extract_uv: bool = False
    extract_energy: bool = True
    energy_extract_mode: str = "from_mel"
    extract_audio: bool = False
    extract_label: bool = False
    extract_duration: bool = False
    extract_acoustic_token: bool = False
    mel_dir: str = "mels"
    pitch_dir: str = "pitches"
    energy_dir: str = "energys"
    uv_dir: str = "uvs"
    audio_dir: str = "audios"
    # Additional attributes needed by __extract_utt_acoustic_features
    raw_data: str = "raw_data"
    duration_dir: str = "durations"
    lab_dir: str = "labs"
    linear_dir: str = "linears"
    phone_pitch_dir: str = "phone_pitches"
    phone_energy_dir: str = "phone_energys"
    label_dir: str = "labels"
    acoustic_token_dir: str = "acoustic_tokens"
    acoustic_token_extractor: str = "Encodec"
    bits: int = 8
    is_mu_law: bool = False
    log_amplitude_dir: str = "log_amplitudes"
    phase_dir: str = "phases"
    real_dir: str = "reals"
    imaginary_dir: str = "imaginarys"
    extract_amplitude_phase: bool = False


@dataclass
class Config:
    """Minimal config wrapper."""
    task_type: str = "svc"
    preprocess: PreprocessConfig = None

    def __post_init__(self):
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()


def create_synthetic_audio(output_path, sample_rate=24000, duration=2.0):
    """Create a synthetic audio file with a simple sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Create a simple harmonic signal (A4 note: 440 Hz with harmonics)
    frequency = 440.0
    audio_data = (
        0.5 * np.sin(2 * np.pi * frequency * t) +
        0.25 * np.sin(2 * np.pi * 2 * frequency * t) +
        0.125 * np.sin(2 * np.pi * 3 * frequency * t)
    )
    # Apply envelope to avoid clicks
    envelope = np.ones_like(audio_data)
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    audio_data = audio_data * envelope * 0.8  # Normalize

    # Save as WAV using scipy
    import scipy.io.wavfile as wavfile
    wavfile.write(output_path, sample_rate, (audio_data * 32767).astype(np.int16))
    return output_path


def create_test_dataset(dataset_dir, num_utterances=10, sample_rate=24000, duration=2.0):
    """Create a minimal test dataset with synthetic audio files."""
    os.makedirs(dataset_dir, exist_ok=True)

    metadata = []
    for i in range(num_utterances):
        uid = f"test_utterance_{i:03d}"
        wav_path = os.path.join(dataset_dir, f"{uid}.wav")

        # Create synthetic audio
        create_synthetic_audio(wav_path, sample_rate, duration)

        metadata.append({
            "Dataset": "test_dataset",
            "Singer": "test_singer",
            "Uid": uid,
            "Path": wav_path,
            "Duration": duration,
            "Index": i,
        })

    # Save metadata
    with open(os.path.join(dataset_dir, "train.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def run_extraction_test(
    metadata,
    output_dir,
    cfg,
    n_workers=1,
    parallel=False
):
    """Run acoustic feature extraction and measure time."""
    start_time = time.time()

    if parallel:
        results = extract_utt_acoustic_features_parallel(
            metadata, output_dir, cfg, n_workers=n_workers
        )
    else:
        extract_utt_acoustic_features_serial(metadata, output_dir, cfg)
        results = None

    elapsed_time = time.time() - start_time
    return elapsed_time, results


def verify_features(output_dir, metadata, cfg):
    """Verify that extracted features exist and are valid."""
    errors = []
    features_checked = 0

    for utt in metadata:
        uid = utt["Uid"]

        # Check mel features
        mel_path = os.path.join(output_dir, cfg.preprocess.mel_dir, f"{uid}.npy")
        if os.path.exists(mel_path):
            mel = np.load(mel_path)
            if mel.shape[0] != cfg.preprocess.n_mel:
                errors.append(f"{uid}: mel shape mismatch {mel.shape[0]} != {cfg.preprocess.n_mel}")
            features_checked += 1
        else:
            errors.append(f"{uid}: mel file not found: {mel_path}")

        # Check pitch features (only if extraction is enabled)
        if cfg.preprocess.extract_pitch:
            pitch_path = os.path.join(output_dir, cfg.preprocess.pitch_dir, f"{uid}.npy")
            if os.path.exists(pitch_path):
                pitch = np.load(pitch_path)
                if len(pitch) == 0:
                    errors.append(f"{uid}: empty pitch")
                features_checked += 1
            else:
                errors.append(f"{uid}: pitch file not found: {pitch_path}")

        # Check energy features
        energy_path = os.path.join(output_dir, cfg.preprocess.energy_dir, f"{uid}.npy")
        if os.path.exists(energy_path):
            energy = np.load(energy_path)
            if len(energy) == 0:
                errors.append(f"{uid}: empty energy")
            features_checked += 1
        else:
            errors.append(f"{uid}: energy file not found: {energy_path}")

    return errors, features_checked


def compare_features(dir_a, dir_b, metadata, feature_dirs):
    """Compare features extracted by two different methods."""
    differences = []

    for utt in metadata:
        uid = utt["Uid"]

        for feature_dir in feature_dirs:
            path_a = os.path.join(dir_a, feature_dir, f"{uid}.npy")
            path_b = os.path.join(dir_b, feature_dir, f"{uid}.npy")

            if not os.path.exists(path_a) or not os.path.exists(path_b):
                differences.append(f"{uid}/{feature_dir}: missing file")
                continue

            feat_a = np.load(path_a)
            feat_b = np.load(path_b)

            if feat_a.shape != feat_b.shape:
                differences.append(f"{uid}/{feature_dir}: shape mismatch {feat_a.shape} vs {feat_b.shape}")
                continue

            # Allow small floating point differences
            max_diff = np.max(np.abs(feat_a - feat_b))
            if max_diff > 1e-6:
                differences.append(f"{uid}/{feature_dir}: value diff max={max_diff:.2e}")

    return differences


def main():
    parser = argparse.ArgumentParser(description="Test parallel acoustic feature extraction")
    parser.add_argument("--num_utterances", type=int, default=10, help="Number of test utterances")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each test audio in seconds")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files after test")
    args = parser.parse_args()

    print("=" * 60)
    print("Parallel Acoustic Feature Extraction Integration Test")
    print("=" * 60)
    print(f"Utterances: {args.num_utterances}")
    print(f"Workers: {args.num_workers}")
    print(f"Duration per audio: {args.duration}s")
    print()

    # Create temporary directories
    temp_base = tempfile.mkdtemp(prefix="parallel_test_")
    dataset_dir = os.path.join(temp_base, "dataset")
    serial_output = os.path.join(temp_base, "serial_output")
    parallel_output = os.path.join(temp_base, "parallel_output")

    print(f"Temp directory: {temp_base}")

    try:
        # Create config
        cfg = Config(task_type="svc")

        # Create test dataset
        print("\n--- Creating test dataset ---")
        metadata = create_test_dataset(
            dataset_dir,
            num_utterances=args.num_utterances,
            duration=args.duration
        )
        print(f"Created {len(metadata)} synthetic audio files")

        # Test 1: Serial extraction
        print("\n--- Test 1: Serial Extraction (n_workers=1) ---")
        serial_time, _ = run_extraction_test(
            metadata, serial_output, cfg, n_workers=1, parallel=False
        )
        print(f"Serial extraction time: {serial_time:.2f}s")

        # Verify serial features
        errors, features_checked = verify_features(serial_output, metadata, cfg)
        if errors:
            print(f"ERRORS in serial extraction:")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        else:
            print(f"Serial extraction verified: {features_checked} feature files OK")

        # Test 2: Parallel extraction
        print(f"\n--- Test 2: Parallel Extraction (n_workers={args.num_workers}) ---")
        parallel_time, results = run_extraction_test(
            metadata, parallel_output, cfg, n_workers=args.num_workers, parallel=True
        )
        print(f"Parallel extraction time: {parallel_time:.2f}s")

        # Check results from parallel extraction
        if results:
            failed = [r for r in results if not r[1]]
            if failed:
                print(f"WARNING: {len(failed)} utterances failed:")
                for uid, _, error in failed[:5]:
                    print(f"  - {uid}: {error}")
            else:
                print(f"All {len(results)} utterances processed successfully")

        # Verify parallel features
        errors, features_checked = verify_features(parallel_output, metadata, cfg)
        if errors:
            print(f"ERRORS in parallel extraction:")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        else:
            print(f"Parallel extraction verified: {features_checked} feature files OK")

        # Test 3: Compare outputs
        print("\n--- Test 3: Comparing Serial vs Parallel Outputs ---")
        feature_dirs = [cfg.preprocess.mel_dir, cfg.preprocess.energy_dir]
        differences = compare_features(serial_output, parallel_output, metadata, feature_dirs)

        if differences:
            print(f"DIFFERENCES found ({len(differences)}):")
            for diff in differences[:10]:
                print(f"  - {diff}")
        else:
            print("PASS: Serial and parallel outputs are identical (within tolerance)")

        # Test 4: Performance comparison
        print("\n--- Test 4: Performance Summary ---")
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"Serial time:   {serial_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup:       {speedup:.2f}x (with {args.num_workers} workers)")

        # Overall result
        print("\n" + "=" * 60)
        if not errors and not differences:
            print("RESULT: ALL TESTS PASSED")
            print("  - No errors during extraction")
            print("  - Output features match between serial and parallel")
            print(f"  - Parallel extraction is {speedup:.2f}x faster")
            return 0
        else:
            print("RESULT: TESTS FAILED")
            return 1

    finally:
        if not args.keep_temp:
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(temp_base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())