# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration test configuration and shared fixtures.

Fixtures provided:
- integration_device: torch.device for CPU (integration tests run on CPU only)
- small_batch_size: small batch size (2) for integration tests
- integration_seq_len: short sequence length (20 frames) for fast tests
- torch_seed: sets a fixed random seed and restores the previous RNG state

Module-level stubs
------------------
``pyworld`` is registered as a lightweight stub in ``sys.modules`` before any
test module is collected.  The ``condition_encoder`` module imports
``utils.f0``, which in turn does ``import pyworld as pw`` at the top level.
The stub prevents an ``ImportError`` when the ``pyworld`` C-extension is not
available (e.g. it was compiled against a different Python ABI).  No
``pyworld`` functionality is exercised by these tests — it is only used by
``MelodyEncoder`` when ``use_f0=True``, which is disabled in all configs here.
"""

import sys
import types

import pytest
import torch

# ---------------------------------------------------------------------------
# Environment stubs
# ---------------------------------------------------------------------------

# Install a minimal pyworld stub so that ``utils.f0`` can be imported on
# systems where the pyworld native extension is unavailable or broken.
if "pyworld" not in sys.modules:
    sys.modules["pyworld"] = types.ModuleType("pyworld")


@pytest.fixture
def integration_device():
    """Return torch.device('cpu') for integration tests.

    Integration model tests run on CPU only — no GPU required.
    """
    return torch.device("cpu")


@pytest.fixture
def small_batch_size():
    """Return a small batch size (2) suitable for integration tests."""
    return 2


@pytest.fixture
def integration_seq_len():
    """Return a short sequence length (20 frames) for fast integration tests."""
    return 20


@pytest.fixture
def torch_seed():
    """Set a fixed random seed (42) for the duration of a test.

    Saves and restores the PyTorch RNG state so tests are isolated from one
    another regardless of execution order.
    """
    rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    yield 42
    torch.set_rng_state(rng_state)
