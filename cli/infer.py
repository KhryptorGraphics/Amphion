# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""'amphion infer' subcommand: dispatch inference to the appropriate model adapter.

This module is automatically imported by ``cli/main.py`` via
``_register_infer_subcommand()``.  A two-pass argument-parsing strategy
allows each adapter to declare its own model-specific flags while still
providing a unified ``--help`` experience:

* ``amphion infer --help``              – shows available models and examples.
* ``amphion infer --model MODEL --help``– shows model-specific arguments.
"""

import argparse
import sys
from typing import List, Optional

from cli.model_registry import get_adapter_class, list_models


# ---------------------------------------------------------------------------
# Subcommand registration
# ---------------------------------------------------------------------------


def add_infer_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> argparse.ArgumentParser:
    """Register the ``infer`` subcommand with *subparsers*.

    The infer subparser recognises ``--model`` in the first parse pass.
    All other model-specific flags (e.g. ``--text``, ``--ref-audio``,
    ``--output``) are treated as *unknown* arguments by the top-level parser
    (``cli/main.py`` uses :meth:`~argparse.ArgumentParser.parse_known_args`)
    and are made available via ``args._extra_argv``.  The full set of
    arguments is then parsed in a second pass inside :func:`run_infer`.

    Args:
        subparsers: The ``_SubParsersAction`` object obtained from
            ``parser.add_subparsers()``.

    Returns:
        The configured ``infer`` :class:`argparse.ArgumentParser`.
    """
    models_table = list_models()

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on a supported model.",
        description=(
            "Run inference using any Amphion model.\n\n"
            f"Available models:\n{models_table}\n\n"
            "For model-specific options run:\n"
            "  amphion infer --model MODEL --help\n\n"
            "Examples:\n"
            "  amphion infer --model maskgct \\\n"
            "      --text 'Hello world' --ref-audio speaker.wav --output out.wav\n"
            "  amphion infer --model vevo-voice \\\n"
            "      --source src.wav --ref ref.wav --output out.wav"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # First-pass: only --model is declared here.  Model-specific flags are
    # resolved in run_infer() after the adapter class is loaded.
    infer_parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=(
            "Model to use for inference.  "
            f"One of: {', '.join(_model_names())}."
        ),
    )

    infer_parser.set_defaults(func=run_infer)
    return infer_parser


def _model_names() -> List[str]:
    """Return a sorted list of registered model names (for help text)."""
    from cli.model_registry import MODEL_REGISTRY

    return list(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Subcommand handler
# ---------------------------------------------------------------------------


def run_infer(args: argparse.Namespace) -> None:
    """Dispatch the ``amphion infer`` subcommand to the appropriate adapter.

    Implements a two-pass argument-parsing strategy:

    1. **First pass** – ``--model`` is already resolved by the top-level
       parser (stored in ``args.model``).  Any remaining, model-specific
       flags are available in ``args._extra_argv`` (a list stashed by
       ``cli/main.py``).
    2. **Second pass** – a full :class:`~argparse.ArgumentParser` is built
       combining shared flags (``--output``, ``--device``) with model-specific
       flags provided by
       :meth:`~cli.adapters.base.ModelAdapter.add_arguments`, then the
       complete argument list is re-parsed.

    Args:
        args: Namespace produced by the top-level ``amphion`` parser.
            Must contain:

            * ``args.model``       – model name string (from first pass)
            * ``args._extra_argv`` – remaining unparsed flags (list)
    """
    model_name: Optional[str] = getattr(args, "model", None)
    extra_argv: List[str] = list(getattr(args, "_extra_argv", None) or [])

    if not model_name:
        sys.stderr.write(
            "error: the --model argument is required.\n\n"
            f"Available models:\n{list_models()}\n\n"
            "Run 'amphion infer --help' for usage information.\n"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load adapter class
    # ------------------------------------------------------------------
    try:
        adapter_cls = get_adapter_class(model_name)
    except KeyError:
        sys.stderr.write(
            f"error: unknown model '{model_name}'.\n\n"
            f"Available models:\n{list_models()}\n"
        )
        sys.exit(1)
    except ImportError as exc:
        sys.stderr.write(
            f"error: could not load adapter for model '{model_name}':\n"
            f"  {exc}\n\n"
            "Make sure all required packages are installed.  Refer to the\n"
            "model-specific requirements.txt for details.\n"
        )
        sys.exit(1)

    adapter = adapter_cls()

    # ------------------------------------------------------------------
    # Second pass: full parse with shared + model-specific args
    # ------------------------------------------------------------------
    model_parser = argparse.ArgumentParser(
        prog=f"amphion infer --model {model_name}",
        description=(
            f"{adapter.description}\n\n"
            f"Task type: {adapter.task_type}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model to use for inference (already provided).",
    )
    model_parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output audio file path (e.g. out.wav).",
    )
    model_parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help=(
            "Compute device to run inference on.  "
            "One of: 'auto' (default), 'cpu', 'cuda', or 'cuda:N' "
            "where N is the GPU index."
        ),
    )

    # Delegate model-specific argument registration to the adapter.
    adapter.add_arguments(model_parser)

    # Reconstruct the full argument list for this invocation:
    # re-inject --model (parsed in the first pass, not in extra_argv) so the
    # second-pass parser validates it and populates full_args.model correctly.
    argv: List[str] = ["--model", model_name] + extra_argv
    full_args = model_parser.parse_args(argv)

    adapter.run(full_args)
