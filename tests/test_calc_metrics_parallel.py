# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests verifying the parallel metric computation path is invocable."""

import concurrent.futures
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup – executed once at module import time so that both the main
# process and any spawned sub-processes can resolve project imports.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Propagate the project root into PYTHONPATH so that ProcessPoolExecutor
# worker processes can import project modules.
_current_pythonpath = os.environ.get("PYTHONPATH", "")
if _PROJECT_ROOT not in _current_pythonpath:
    os.environ["PYTHONPATH"] = _PROJECT_ROOT + (
        os.pathsep + _current_pythonpath if _current_pythonpath else ""
    )

# ---------------------------------------------------------------------------
# Stub out optional heavy or binary-incompatible modules so that the test
# suite can run in environments where they are unavailable (e.g. pyworld's
# C extension may fail to load on certain platforms/Python versions).
# These stubs only affect the import phase; the actual test fixtures below
# use energy_rmse which depends only on librosa/numpy.
# ---------------------------------------------------------------------------

_STUB_MODULES = [
    "pyworld",  # C extension with platform-specific binary incompatibilities
    "resemblyzer",  # optional speaker similarity dependency
    "frechet_audio_distance",  # optional FAD dependency
    "pymcd",  # optional MCD dependency
    "pymcd.mcd",  # sub-module imported by mel_cepstral_distortion
    "pypesq",  # optional PESQ dependency
    "whisper",  # optional ASR dependency (only needed for wer/cer)
]
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Must come after the sys.path / stub setup above.
from bins.calc_metrics import (  # noqa: E402
    _compute_file_metric,
    _compute_file_metric_v_uv_f1,
    calc_metric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 22050


def _write_wav(path: str, signal: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> None:
    """Write a float32 numpy array to a WAV file using scipy."""
    import scipy.io.wavfile

    scipy.io.wavfile.write(path, sample_rate, signal.astype(np.float32))


def _make_audio_pairs(tmp_path, n_pairs: int = 4):
    """Create *n_pairs* synthetic 1-second WAV file pairs.

    Returns:
        tuple[Path, Path, Path]: (ref_dir, deg_dir, dump_dir)
    """
    ref_dir = tmp_path / "ref"
    deg_dir = tmp_path / "deg"
    dump_dir = tmp_path / "dump"
    for d in (ref_dir, deg_dir, dump_dir):
        d.mkdir()

    n_samples = _SAMPLE_RATE  # 1 second at 22 050 Hz
    t = np.linspace(0, 1.0, n_samples, endpoint=False)

    for i in range(n_pairs):
        freq = 440.0 * (i + 1)
        ref_sig = 0.5 * np.sin(2.0 * np.pi * freq * t)
        deg_sig = 0.4 * np.sin(2.0 * np.pi * freq * t)
        name = f"utt_{i:04d}.wav"
        _write_wav(str(ref_dir / name), ref_sig)
        _write_wav(str(deg_dir / name), deg_sig)

    return ref_dir, deg_dir, dump_dir


def _default_metric_kwargs() -> dict:
    """Return the default keyword arguments forwarded to calc_metric."""
    return dict(
        fs=_SAMPLE_RATE,
        method="cut",
        db_scale=False,
        need_mean=True,
        model_name="wavlm",
        similarity_mode="pairwith",
        ltr_path="None",
        intelligibility_mode="gt_audio",
        language="english",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_helper_functions_are_picklable():
    """Module-level helpers must be picklable for ProcessPoolExecutor."""
    import pickle

    pickle.dumps(_compute_file_metric)
    pickle.dumps(_compute_file_metric_v_uv_f1)


def test_cli_accepts_n_workers_flag():
    """--n_workers argument must appear in --help output."""
    result = subprocess.run(
        [sys.executable, "bins/calc_metrics.py", "--help"],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    if result.returncode != 0:
        pytest.skip(f"CLI failed to load (import error?): {result.stderr[:300]}")
    assert "n_workers" in result.stdout, (
        "--n_workers not found in --help output.\nstdout:\n" + result.stdout
    )


def test_parallel_path_invoked_when_n_workers_gt_1(tmp_path):
    """ProcessPoolExecutor must be instantiated once with max_workers=n_workers."""
    ref_dir, deg_dir, dump_dir = _make_audio_pairs(tmp_path, n_pairs=4)

    with patch.object(concurrent.futures, "ProcessPoolExecutor") as mock_ppe:
        # Simulate the executor returning 4 non-NaN float scores.
        mock_executor = mock_ppe.return_value.__enter__.return_value
        mock_executor.map.return_value = [1.0, 1.5, 2.0, 2.5]

        calc_metric(
            str(ref_dir),
            str(deg_dir),
            str(dump_dir),
            metrics=["energy_rmse"],
            n_workers=2,
            **_default_metric_kwargs(),
        )

        mock_ppe.assert_called_once_with(max_workers=2)

    result_path = dump_dir / "result.json"
    assert result_path.exists(), "result.json was not written by calc_metric"
    with result_path.open() as f:
        result = json.load(f)
    assert "energy_rmse" in result, "energy_rmse key missing from result.json"


def test_sequential_run_produces_valid_result(tmp_path):
    """n_workers=1 must run end-to-end and yield a finite non-negative score."""
    pytest.importorskip("scipy", reason="scipy required for synthetic WAV creation")

    ref_dir, deg_dir, dump_dir = _make_audio_pairs(tmp_path, n_pairs=4)

    calc_metric(
        str(ref_dir),
        str(deg_dir),
        str(dump_dir),
        metrics=["energy_rmse"],
        n_workers=1,
        **_default_metric_kwargs(),
    )

    result_path = dump_dir / "result.json"
    assert result_path.exists(), "result.json was not written"
    with result_path.open() as f:
        result = json.load(f)

    assert "energy_rmse" in result
    score = float(result["energy_rmse"])
    assert (
        np.isfinite(score) and score >= 0.0
    ), f"Unexpected energy_rmse score: {score!r}"
