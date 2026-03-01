# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model registry for the Amphion CLI.

Maps model name strings to lazy-loading adapter descriptors and provides
helper utilities for listing and looking up adapter classes.  All imports
of concrete adapter modules are deferred so that missing optional packages
(e.g. ``dualcodec``) do not crash the CLI at startup.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Internal specification table
# ---------------------------------------------------------------------------

#: Registry specification: ordered mapping of CLI model names to metadata.
#: Keys must match exactly what users pass to ``--model``.
_REGISTRY_SPEC: Dict[str, Dict[str, Any]] = {
    "maskgct": {
        "module": "cli.adapters.maskgct",
        "class": "MaskGCTAdapter",
        "description": "MaskGCT TTS – zero-shot TTS with reference audio (ICLR 2025)",
        "task_type": "tts",
    },
    "vevo-voice": {
        "module": "cli.adapters.vevo",
        "class": "VevoVoiceAdapter",
        "description": "Vevo – voice conversion (voice identity transfer)",
        "task_type": "vc",
    },
    "vevo-timbre": {
        "module": "cli.adapters.vevo",
        "class": "VevoTimbreAdapter",
        "description": "Vevo – timbre conversion (style-free voice conversion)",
        "task_type": "vc",
    },
    "vevo-style": {
        "module": "cli.adapters.vevo",
        "class": "VevoStyleAdapter",
        "description": "Vevo – style conversion (prosody / speaking style transfer)",
        "task_type": "vc",
    },
    "vevo-tts": {
        "module": "cli.adapters.vevo",
        "class": "VevoTTSAdapter",
        "description": "Vevo – zero-shot TTS with voice and style reference",
        "task_type": "tts",
    },
    "dualcodec": {
        "module": "cli.adapters.dualcodec",
        "class": "DualCodecAdapter",
        "description": "DualCodec-VALLE – zero-shot TTS via dual-codebook codec",
        "task_type": "tts",
    },
    "vits": {
        "module": "cli.adapters.vits",
        "class": "VITSAdapter",
        "description": "VITS – variational TTS (requires local checkpoint)",
        "task_type": "tts",
    },
    "fastspeech2": {
        "module": "cli.adapters.fastspeech2",
        "class": "FastSpeech2Adapter",
        "description": "FastSpeech 2 – non-autoregressive TTS (requires local checkpoint)",
        "task_type": "tts",
    },
}


# ---------------------------------------------------------------------------
# Lazy adapter loader
# ---------------------------------------------------------------------------


class _LazyAdapterLoader:
    """Callable that lazily imports and caches an adapter class.

    Storing loaders in :data:`MODEL_REGISTRY` instead of importing the
    adapter classes directly means the registry can be populated at module
    import time without triggering imports of heavy optional dependencies
    (e.g. ``torch``, ``dualcodec``, or Vevo's HuggingFace weights).

    Usage::

        adapter_cls = MODEL_REGISTRY["maskgct"].get_class()
        adapter = MODEL_REGISTRY["maskgct"]()  # also valid (callable)
    """

    def __init__(self, module_path: str, class_name: str) -> None:
        self._module_path = module_path
        self._class_name = class_name
        self._cls: Optional[type] = None

    # --- Class-like callable interface so callers can do MODEL_REGISTRY[name]() ---

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the adapter class with the given arguments."""
        return self._load()(*args, **kwargs)

    def _load(self) -> type:
        """Import and return the underlying adapter class (cached after first load)."""
        if self._cls is None:
            module = importlib.import_module(self._module_path)
            self._cls = getattr(module, self._class_name)
        return self._cls

    def get_class(self) -> type:
        """Return the underlying adapter class (triggers the import on first call).

        Returns:
            The concrete :class:`~cli.adapters.base.ModelAdapter` subclass.

        Raises:
            ImportError: If the adapter module cannot be imported (e.g. because
                an optional dependency is not installed).
        """
        return self._load()

    def __repr__(self) -> str:
        return f"<LazyAdapterLoader {self._module_path}.{self._class_name}>"


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

#: Mapping from CLI model name to a lazy-loading adapter descriptor.
#:
#: Example usage::
#:
#:     from cli.model_registry import MODEL_REGISTRY
#:     adapter = MODEL_REGISTRY["maskgct"]()   # instantiate via __call__
#:     adapter_cls = MODEL_REGISTRY["maskgct"].get_class()  # or load class
MODEL_REGISTRY: Dict[str, _LazyAdapterLoader] = {
    name: _LazyAdapterLoader(spec["module"], spec["class"])
    for name, spec in _REGISTRY_SPEC.items()
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def list_models() -> str:
    """Return a formatted table of all registered models.

    The table includes the model name, task type (``tts`` / ``vc``), and a
    short description.  This output is used in ``amphion infer --help`` and
    in error messages when an unknown model name is given.

    Returns:
        Multi-line string suitable for printing to the terminal.

    Example output::

        Model          Task  Description
        -------------------------------------------------
        maskgct        tts   MaskGCT TTS – zero-shot TTS with reference audio
        vevo-voice     vc    Vevo – voice conversion (voice identity transfer)
        ...
    """
    col_name = max(len(n) for n in _REGISTRY_SPEC) + 2
    col_task = max(len(s["task_type"]) for s in _REGISTRY_SPEC.values()) + 2

    header = f"{'Model':<{col_name}}{'Task':<{col_task}}Description"
    separator = "-" * (col_name + col_task + 40)

    rows = [header, separator]
    for name, spec in _REGISTRY_SPEC.items():
        rows.append(
            f"{name:<{col_name}}{spec['task_type']:<{col_task}}{spec['description']}"
        )

    return "\n".join(rows)


def get_adapter_class(model_name: str) -> type:
    """Look up and import the adapter class for *model_name*.

    This is a convenience wrapper around :data:`MODEL_REGISTRY` that also
    provides a clear error message when the requested model is not found.

    Args:
        model_name: One of the keys in :data:`MODEL_REGISTRY` (e.g.
            ``'maskgct'``, ``'vevo-voice'``).

    Returns:
        The concrete :class:`~cli.adapters.base.ModelAdapter` subclass.

    Raises:
        KeyError: If *model_name* is not registered in :data:`MODEL_REGISTRY`.
        ImportError: If the adapter module cannot be imported (e.g. because
            an optional dependency such as ``dualcodec`` is not installed).
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'.\n\nAvailable models:\n{list_models()}"
        )
    return MODEL_REGISTRY[model_name].get_class()
