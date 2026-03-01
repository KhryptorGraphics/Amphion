# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

__version__ = "0.1.0"

_DESCRIPTION = """\
Amphion: Open-Source Audio, Music, and Speech Generation toolkit.

Available subcommands:
  infer     Run inference on a supported model.

Examples:
  amphion infer --model maskgct --text 'Hello world' --ref-audio speaker.wav --output out.wav
  amphion infer --model vevo-voice --source src.wav --ref ref.wav --output out.wav
  amphion infer --help
"""


def build_parser():
    """Build and return the top-level argument parser for the 'amphion' command."""
    parser = argparse.ArgumentParser(
        prog="amphion",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"amphion {__version__}",
        help="Show the Amphion version and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
    )

    # Register the 'infer' subcommand.
    # The full argument specification is added in cli/infer.py
    # (subtask-2-3); here we register a minimal placeholder so that
    # 'amphion infer --help' already routes correctly once cli.infer
    # is available.
    _register_infer_subcommand(subparsers)

    return parser, subparsers


def _register_infer_subcommand(subparsers):
    """Register the 'infer' subcommand.

    When cli.infer is available, its add_infer_subcommand() is used to
    populate the full set of arguments.  During the bootstrap phase
    (before cli/infer.py exists), a minimal placeholder is registered so
    that the top-level parser is importable and usable.
    """
    try:
        from cli.infer import add_infer_subcommand  # noqa: F401

        add_infer_subcommand(subparsers)
    except ImportError:
        # cli/infer.py not yet available – register a placeholder parser.
        infer_parser = subparsers.add_parser(
            "infer",
            help="Run inference on a supported model (see 'amphion infer --help').",
        )
        infer_parser.set_defaults(func=_infer_placeholder)


def _infer_placeholder(args):
    """Placeholder handler for the 'infer' subcommand before cli/infer.py exists."""
    sys.stderr.write(
        "error: 'amphion infer' is not fully initialised yet.\n"
        "Make sure all Amphion CLI modules are installed.\n"
    )
    sys.exit(1)


def main():
    """Entry point for the 'amphion' command-line interface."""
    parser, _ = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stdout)
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        # Subcommand was recognised but has no handler yet.
        parser.print_help(sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
