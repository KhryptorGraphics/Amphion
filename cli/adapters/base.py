# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract base class for all Amphion model adapters.

    Each adapter wraps one model and exposes a uniform interface to the
    ``amphion infer`` subcommand.  Concrete adapters must implement:

    * :attr:`name`        – CLI model identifier (e.g. ``'maskgct'``)
    * :attr:`description` – One-line description shown in ``--help``
    * :attr:`task_type`   – ``'tts'`` or ``'vc'``
    * :meth:`add_arguments` – Populate model-specific argparse arguments
    * :meth:`run`           – Execute inference given parsed ``args``
    """

    # ------------------------------------------------------------------
    # Identity properties (must be overridden by subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """CLI model identifier used with ``--model`` (e.g. ``'maskgct'``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short one-line description shown in model listing and ``--help``."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task category: ``'tts'`` (text-to-speech) or ``'vc'`` (voice conversion)."""

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add model-specific arguments to *parser*.

        This is called by the two-pass ``amphion infer`` dispatcher after the
        ``--model`` flag has been resolved.  Add any positional arguments,
        optional flags, or argument groups that this model requires.

        Args:
            parser: The subparser for ``amphion infer`` to which arguments
                should be added.
        """

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, args: argparse.Namespace) -> None:
        """Run model inference using the parsed *args* namespace.

        Implementations should:
        1. Load / download model weights (lazy, only when ``run`` is called).
        2. Execute inference.
        3. Write the output audio file to ``args.output``.

        Args:
            args: Fully parsed argument namespace containing both shared
                arguments (``--output``, ``--device``) and the model-specific
                arguments added by :meth:`add_arguments`.
        """


def resolve_device(args: argparse.Namespace):
    """Return a ``torch.device`` based on ``args.device``.

    When ``args.device`` is ``'auto'`` (the default), CUDA is selected if
    available, otherwise the CPU is used.  When an explicit device is
    requested (e.g. ``'cuda'``, ``'cuda:1'``, ``'cpu'``), the function
    validates that CUDA is accessible before returning.

    Args:
        args: Parsed argument namespace.  Must have a ``device`` attribute
            (str).  ``'auto'`` triggers automatic selection.

    Returns:
        ``torch.device`` for the resolved device.

    Raises:
        SystemExit: With exit code 1 and a human-readable message when a CUDA
            device is requested but is not available.
    """
    try:
        import torch
    except ImportError:
        sys.stderr.write(
            "error: PyTorch is not installed.\n"
            "Install it from https://pytorch.org before running inference.\n"
        )
        sys.exit(1)

    device_str = getattr(args, "device", "auto")

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explicit device requested – validate CUDA availability.
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        sys.stderr.write(
            f"error: Device '{device_str}' was requested, but no CUDA-capable "
            "GPU was found.\n"
            "Use '--device cpu' to run on CPU (slower), or check that your "
            "CUDA drivers and PyTorch CUDA build are compatible.\n"
        )
        sys.exit(1)

    return torch.device(device_str)
