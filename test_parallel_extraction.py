# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
import time
from pathlib import Path
from unittest.mock import patch
from multiprocessing import Pool

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processors.acoustic_extractor import extract_utt_acoustic_features_parallel


def mock_extract_features(dataset_output, cfg, utt):
    """Mock feature extraction function that writes dummy features

    Args:
        dataset_output (str): directory to store features
        cfg: configuration object
        utt (dict): utterance information

    """
    uid = utt["Uid"]

    # Create feature directories
    if cfg.preprocess.extract_mel:
        mel_dir = os.path.join(dataset_output, cfg.preprocess.mel_dir)
        os.makedirs(mel_dir, exist_ok=True)
        mel_path = os.path.join(mel_dir, f"{uid}.npy")
        # Create mock mel features (n_mel, time_steps)
        mel = np.random.randn(cfg.preprocess.n_mel, 100)
        np.save(mel_path, mel)

    if cfg.preprocess.extract_pitch:
        pitch_dir = os.path.join(dataset_output, cfg.preprocess.pitch_dir)
        os.makedirs(pitch_dir, exist_ok=True)
        pitch_path = os.path.join(pitch_dir, f"{uid}.npy")
        # Create mock pitch features
        pitch = np.random.randn(100) * 100 + 200
        np.save(pitch_path, pitch)

    if cfg.preprocess.extract_energy:
        energy_dir = os.path.join(dataset_output, cfg.preprocess.energy_dir)
        os.makedirs(energy_dir, exist_ok=True)
        energy_path = os.path.join(energy_dir, f"{uid}.npy")
        # Create mock energy features
        energy = np.random.randn(100)
        np.save(energy_path, energy)

    # Simulate some processing time
    time.sleep(0.01)


class MockConfig:
    """Mock configuration object for testing"""

    def __init__(self):
        self.task_type = "svc"
        self.preprocess = MockPreprocess()


class MockPreprocess:
    """Mock preprocessing configuration"""

    def __init__(self):
        self.sample_rate = 24000
        self.n_fft = 1024
        self.win_size = 480
        self.hop_size = 120
        self.n_mel = 80
        self.fmin = 0
        self.fmax = 12000

        # Feature extraction flags
        self.extract_mel = True
        self.extract_pitch = True
        self.extract_energy = True
        self.extract_audio = True
        self.extract_linear_spec = False
        self.extract_duration = False
        self.extract_uv = False
        self.extract_label = False
        self.extract_acoustic_token = False

        # Extract modes
        self.mel_extract_mode = "librosa"
        self.energy_extract_mode = "from_mel"
        self.pitch_extractor = "parselmouth"

        # Directory names
        self.mel_dir = "mels"
        self.pitch_dir = "pitches"
        self.energy_dir = "energys"
        self.audio_dir = "audios"
        self.raw_data = "raw_data"


def create_test_audio_file(file_path, duration=1.0, sample_rate=24000):
    """Create a synthetic audio file for testing

    Args:
        file_path (str): path to save the audio file
        duration (float): duration in seconds
        sample_rate (int): sample rate

    """
    # Create a simple sine wave
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    frequency = 440.0  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Save as wav using scipy
    from scipy.io import wavfile

    # Convert to int16 for wav format
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(file_path, sample_rate, audio_int16)


def create_test_metadata(output_dir, num_utterances=10):
    """Create test metadata and audio files

    Args:
        output_dir (str): directory to store test data
        num_utterances (int): number of test utterances to create

    Returns:
        list: metadata list with utterance information

    """
    metadata = []
    raw_data_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    for i in range(num_utterances):
        uid = f"test_utt_{i:04d}"
        singer = "test_singer"
        dataset = "test_dataset"

        # Create directory for singer
        singer_dir = os.path.join(raw_data_dir, singer)
        os.makedirs(singer_dir, exist_ok=True)

        # Create audio file
        audio_path = os.path.join(singer_dir, f"{uid}.wav")
        create_test_audio_file(audio_path, duration=0.5)

        # Add to metadata
        metadata.append(
            {
                "Dataset": dataset,
                "Singer": singer,
                "Uid": uid,
                "Path": audio_path,
                "Duration": 0.5,
                "index": i,
            }
        )

    return metadata


def verify_features_extracted(output_dir, metadata, cfg):
    """Verify that features were extracted successfully

    Args:
        output_dir (str): directory containing extracted features
        metadata (list): list of utterances
        cfg: configuration object

    Returns:
        bool: True if all features exist, False otherwise

    """
    success = True
    missing_features = []

    for utt in metadata:
        uid = utt["Uid"]

        # Check mel features
        if cfg.preprocess.extract_mel:
            mel_path = os.path.join(output_dir, cfg.preprocess.mel_dir, f"{uid}.npy")
            if not os.path.exists(mel_path):
                missing_features.append(f"Mel: {mel_path}")
                success = False
            else:
                # Verify mel can be loaded and has correct shape
                mel = np.load(mel_path)
                if mel.shape[0] != cfg.preprocess.n_mel:
                    missing_features.append(
                        f"Mel shape incorrect: {mel.shape} for {uid}"
                    )
                    success = False

        # Check pitch features
        if cfg.preprocess.extract_pitch:
            pitch_path = os.path.join(
                output_dir, cfg.preprocess.pitch_dir, f"{uid}.npy"
            )
            if not os.path.exists(pitch_path):
                missing_features.append(f"Pitch: {pitch_path}")
                success = False

        # Check energy features
        if cfg.preprocess.extract_energy:
            energy_path = os.path.join(
                output_dir, cfg.preprocess.energy_dir, f"{uid}.npy"
            )
            if not os.path.exists(energy_path):
                missing_features.append(f"Energy: {energy_path}")
                success = False

    if not success:
        print("\nMissing or invalid features:")
        for feature in missing_features[:10]:  # Show first 10
            print(f"  - {feature}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")

    return success


def test_parallel_extraction():
    """Test parallel feature extraction"""
    print("=" * 80)
    print("Testing Parallel Feature Extraction")
    print("=" * 80)

    # Create temporary directory for test
    test_dir = tempfile.mkdtemp(prefix="test_parallel_extraction_")
    print(f"\nTest directory: {test_dir}")

    try:
        # Create test configuration
        cfg = MockConfig()

        # Create test data
        num_utterances = 20
        print(f"\nCreating {num_utterances} test utterances...")
        metadata = create_test_metadata(test_dir, num_utterances)
        print(f"Created {len(metadata)} test utterances")

        # Patch the actual extraction functions with our mock
        with patch(
            "processors.acoustic_extractor.extract_utt_acoustic_features_tts",
            mock_extract_features,
        ), patch(
            "processors.acoustic_extractor.extract_utt_acoustic_features_svc",
            mock_extract_features,
        ), patch(
            "processors.acoustic_extractor.extract_utt_acoustic_features_vocoder",
            mock_extract_features,
        ), patch(
            "processors.acoustic_extractor.extract_utt_acoustic_features_tta",
            mock_extract_features,
        ):

            # Test 1: Serial processing (n_workers=1)
            print("\n" + "-" * 80)
            print("Test 1: Serial Processing (n_workers=1)")
            print("-" * 80)
            serial_dir = os.path.join(test_dir, "serial")
            os.makedirs(serial_dir, exist_ok=True)

            start_time = time.time()
            extract_utt_acoustic_features_parallel(
                metadata, serial_dir, cfg, n_workers=1
            )
            serial_time = time.time() - start_time

            print(f"\nSerial processing completed in {serial_time:.2f} seconds")

            # Verify serial extraction
            if verify_features_extracted(serial_dir, metadata, cfg):
                print("✓ Serial extraction: All features extracted successfully")
            else:
                print("✗ Serial extraction: Some features missing")
                return False

            # Test 2: Parallel processing (n_workers=4)
            print("\n" + "-" * 80)
            print("Test 2: Parallel Processing (n_workers=4)")
            print("-" * 80)
            parallel_dir = os.path.join(test_dir, "parallel")
            os.makedirs(parallel_dir, exist_ok=True)

            start_time = time.time()
            extract_utt_acoustic_features_parallel(
                metadata, parallel_dir, cfg, n_workers=4
            )
            parallel_time = time.time() - start_time

            print(f"\nParallel processing completed in {parallel_time:.2f} seconds")

            # Verify parallel extraction
            if verify_features_extracted(parallel_dir, metadata, cfg):
                print("✓ Parallel extraction: All features extracted successfully")
            else:
                print("✗ Parallel extraction: Some features missing")
                return False

            # Compare performance
            print("\n" + "=" * 80)
            print("Performance Comparison")
            print("=" * 80)
            print(f"Serial time:   {serial_time:.2f}s")
            print(f"Parallel time: {parallel_time:.2f}s")
            speedup = serial_time / parallel_time if parallel_time > 0 else 0
            print(f"Speedup:       {speedup:.2f}x")

            # Test 3: Verify feature consistency
            print("\n" + "-" * 80)
            print("Test 3: Feature Consistency Check")
            print("-" * 80)
            consistent = True
            for utt in metadata[:5]:  # Check first 5 utterances
                uid = utt["Uid"]

                # Compare mel features (they won't match exactly due to random generation)
                serial_mel_path = os.path.join(
                    serial_dir, cfg.preprocess.mel_dir, f"{uid}.npy"
                )
                parallel_mel_path = os.path.join(
                    parallel_dir, cfg.preprocess.mel_dir, f"{uid}.npy"
                )

                if os.path.exists(serial_mel_path) and os.path.exists(
                    parallel_mel_path
                ):
                    serial_mel = np.load(serial_mel_path)
                    parallel_mel = np.load(parallel_mel_path)
                    # Just verify they have the same shape
                    if serial_mel.shape == parallel_mel.shape:
                        print(f"✓ Features have consistent shape for {uid}")
                    else:
                        print(f"✗ Features have different shapes for {uid}")
                        consistent = False
                else:
                    print(f"✗ Features missing for {uid}")
                    consistent = False

            if consistent:
                print(
                    "\n✓ All checked features have consistent shapes between serial and parallel"
                )
            else:
                print(
                    "\n✗ Some features have inconsistent shapes between serial and parallel"
                )
                return False

            print("\n" + "=" * 80)
            print("ALL TESTS PASSED!")
            print("=" * 80)
            print(f"\n✓ Features extracted successfully with multiple workers")
            print(f"✓ Parallel processing works correctly (speedup: {speedup:.2f}x)")
            print(f"✓ Results have consistent shapes between serial and parallel processing")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_parallel_extraction()
    sys.exit(0 if success else 1)
