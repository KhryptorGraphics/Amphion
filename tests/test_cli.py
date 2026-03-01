# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Amphion CLI.

These tests verify:

- Model registry completeness (all 8 models registered).
- Adapter ``name``, ``task_type``, and ``description`` properties.
- Adapter argument registration (``add_arguments`` does not crash).
- Argparse error handling for missing required arguments.
- :func:`~cli.model_registry.get_adapter_class` error handling for unknown names.

No GPU, model downloads, or heavy optional packages are required.  All heavy
imports inside adapter ``run()`` / ``_load_pipeline()`` methods are deferred and
are never triggered by these tests.
"""

import argparse

import pytest

from cli.model_registry import MODEL_REGISTRY, get_adapter_class, list_models


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All eight model names that must appear in the registry.
_EXPECTED_MODELS = [
    "maskgct",
    "vevo-voice",
    "vevo-timbre",
    "vevo-style",
    "vevo-tts",
    "dualcodec",
    "vits",
    "fastspeech2",
]

#: Expected ``task_type`` for each model.
_EXPECTED_TASK_TYPES = {
    "maskgct": "tts",
    "vevo-voice": "vc",
    "vevo-timbre": "vc",
    "vevo-style": "vc",
    "vevo-tts": "tts",
    "dualcodec": "tts",
    "vits": "tts",
    "fastspeech2": "tts",
}


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestModelRegistryCompleteness:
    """MODEL_REGISTRY must contain all expected models."""

    def test_all_eight_models_registered(self):
        """All 8 expected model names must be keys in MODEL_REGISTRY."""
        for model_name in _EXPECTED_MODELS:
            assert model_name in MODEL_REGISTRY, (
                f"Expected model '{model_name}' not found in MODEL_REGISTRY. "
                f"Registered models: {list(MODEL_REGISTRY.keys())}"
            )

    def test_registry_has_exactly_eight_models(self):
        """MODEL_REGISTRY should have exactly 8 entries."""
        assert len(MODEL_REGISTRY) == 8, (
            f"Expected 8 models in registry, found {len(MODEL_REGISTRY)}: "
            f"{list(MODEL_REGISTRY.keys())}"
        )

    def test_list_models_contains_all_names(self):
        """list_models() output must contain every registered model name."""
        table = list_models()
        for model_name in _EXPECTED_MODELS:
            assert model_name in table, (
                f"'{model_name}' not found in list_models() output:\n{table}"
            )

    def test_list_models_contains_task_type_labels(self):
        """list_models() table must include both 'tts' and 'vc' task labels."""
        table = list_models()
        assert "tts" in table, "list_models() output does not mention 'tts'."
        assert "vc" in table, "list_models() output does not mention 'vc'."

    def test_list_models_returns_string(self):
        """list_models() must return a non-empty string."""
        table = list_models()
        assert isinstance(table, str) and table.strip(), (
            "list_models() did not return a non-empty string."
        )


# ---------------------------------------------------------------------------
# Adapter properties
# ---------------------------------------------------------------------------


class TestAdapterProperties:
    """Each adapter must expose correct name, task_type, and description."""

    @pytest.mark.parametrize("model_name", _EXPECTED_MODELS)
    def test_adapter_name_matches_registry_key(self, model_name):
        """adapter.name must match the registry key used to load it."""
        adapter_cls = get_adapter_class(model_name)
        adapter = adapter_cls()
        assert adapter.name == model_name, (
            f"Adapter loaded under key '{model_name}' reports name='{adapter.name}'."
        )

    @pytest.mark.parametrize("model_name", _EXPECTED_MODELS)
    def test_adapter_task_type_is_valid(self, model_name):
        """adapter.task_type must be 'tts' or 'vc'."""
        adapter_cls = get_adapter_class(model_name)
        adapter = adapter_cls()
        assert adapter.task_type in ("tts", "vc"), (
            f"Adapter '{model_name}' has unexpected task_type: '{adapter.task_type}'."
        )

    @pytest.mark.parametrize(
        "model_name,expected_task",
        list(_EXPECTED_TASK_TYPES.items()),
    )
    def test_adapter_task_type_correct(self, model_name, expected_task):
        """Each adapter must return the correct expected task_type value."""
        adapter_cls = get_adapter_class(model_name)
        adapter = adapter_cls()
        assert adapter.task_type == expected_task, (
            f"Adapter '{model_name}': expected task_type='{expected_task}', "
            f"got '{adapter.task_type}'."
        )

    @pytest.mark.parametrize("model_name", _EXPECTED_MODELS)
    def test_adapter_description_is_nonempty_string(self, model_name):
        """adapter.description must be a non-empty string."""
        adapter_cls = get_adapter_class(model_name)
        adapter = adapter_cls()
        assert isinstance(adapter.description, str) and adapter.description.strip(), (
            f"Adapter '{model_name}' has an empty or non-string description."
        )


# ---------------------------------------------------------------------------
# Adapter argument registration
# ---------------------------------------------------------------------------


class TestAdapterAddArguments:
    """add_arguments() on each adapter must run without raising."""

    @pytest.mark.parametrize("model_name", _EXPECTED_MODELS)
    def test_add_arguments_does_not_crash(self, model_name):
        """Calling add_arguments() on a fresh ArgumentParser must not raise."""
        adapter_cls = get_adapter_class(model_name)
        adapter = adapter_cls()
        parser = argparse.ArgumentParser(prog=f"test-{model_name}")
        # Should not raise any exception
        adapter.add_arguments(parser)
        # Verify the parser received at least one action beyond the default -h
        assert len(parser._actions) > 1, (
            f"Adapter '{model_name}' registered no arguments on the parser."
        )

    @pytest.mark.parametrize("model_name", _EXPECTED_MODELS)
    def test_add_arguments_registers_output_compatible_parser(self, model_name):
        """After add_arguments(), the parser must have at least one required argument."""
        adapter_cls = get_adapter_class(model_name)
        adapter = adapter_cls()
        parser = argparse.ArgumentParser(prog=f"test-{model_name}")
        adapter.add_arguments(parser)
        # At least one action should be required (i.e. have no default / be required=True)
        required_actions = [
            a
            for a in parser._actions
            if getattr(a, "required", False)
        ]
        assert required_actions, (
            f"Adapter '{model_name}' registered no required arguments.  "
            "Every adapter should have at least one required flag."
        )


# ---------------------------------------------------------------------------
# MaskGCT required argument handling
# ---------------------------------------------------------------------------


class TestMaskGCTRequiredArgs:
    """MaskGCT adapter must enforce all required arguments via argparse."""

    def _make_parser(self):
        """Return a parser with shared and MaskGCT-specific args registered."""
        from cli.adapters.maskgct import MaskGCTAdapter

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="maskgct")
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", default="auto")
        MaskGCTAdapter().add_arguments(parser)
        return parser

    def test_missing_text_causes_system_exit(self):
        """Omitting --text must cause a non-zero SystemExit (argparse error)."""
        parser = self._make_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "--output", "out.wav",
                "--ref-audio", "ref.wav",
                "--prompt-text", "Hello",
                # --text intentionally omitted
            ])
        assert exc_info.value.code != 0

    def test_missing_ref_audio_causes_system_exit(self):
        """Omitting --ref-audio must cause a non-zero SystemExit (argparse error)."""
        parser = self._make_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "--output", "out.wav",
                "--text", "Hello world",
                "--prompt-text", "Hello",
                # --ref-audio intentionally omitted
            ])
        assert exc_info.value.code != 0

    def test_missing_prompt_text_causes_system_exit(self):
        """Omitting --prompt-text must cause a non-zero SystemExit (argparse error)."""
        parser = self._make_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "--output", "out.wav",
                "--text", "Hello world",
                "--ref-audio", "ref.wav",
                # --prompt-text intentionally omitted
            ])
        assert exc_info.value.code != 0

    def test_all_required_args_parse_successfully(self):
        """Providing all required args must parse without error."""
        parser = self._make_parser()
        args = parser.parse_args([
            "--output", "out.wav",
            "--text", "Hello world",
            "--ref-audio", "ref.wav",
            "--prompt-text", "Hello",
        ])
        assert args.text == "Hello world"
        assert args.ref_audio == "ref.wav"
        assert args.output == "out.wav"
        # Default language should be 'en'
        assert args.language == "en"


# ---------------------------------------------------------------------------
# Vevo VC required argument handling
# ---------------------------------------------------------------------------


class TestVevoVoiceRequiredArgs:
    """VevoVoiceAdapter must enforce --source and --reference."""

    def _make_parser(self):
        """Return a parser with shared and vevo-voice-specific args registered."""
        from cli.adapters.vevo import VevoVoiceAdapter

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="vevo-voice")
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", default="auto")
        VevoVoiceAdapter().add_arguments(parser)
        return parser

    def test_missing_source_causes_system_exit(self):
        """Omitting --source must cause a non-zero SystemExit (argparse error)."""
        parser = self._make_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "--output", "out.wav",
                "--reference", "ref.wav",
                # --source intentionally omitted
            ])
        assert exc_info.value.code != 0

    def test_missing_reference_causes_system_exit(self):
        """Omitting --reference must cause a non-zero SystemExit (argparse error)."""
        parser = self._make_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "--output", "out.wav",
                "--source", "source.wav",
                # --reference intentionally omitted
            ])
        assert exc_info.value.code != 0

    def test_all_required_args_parse_successfully(self):
        """Providing all required args must parse without error."""
        parser = self._make_parser()
        args = parser.parse_args([
            "--output", "out.wav",
            "--source", "source.wav",
            "--reference", "ref.wav",
        ])
        assert args.source == "source.wav"
        assert args.reference == "ref.wav"
        assert args.output == "out.wav"


# ---------------------------------------------------------------------------
# get_adapter_class error handling
# ---------------------------------------------------------------------------


class TestGetAdapterClass:
    """get_adapter_class() must provide correct behaviour for known/unknown models."""

    def test_known_model_returns_callable_class(self):
        """get_adapter_class() must return a callable class for every known model."""
        for model_name in _EXPECTED_MODELS:
            cls = get_adapter_class(model_name)
            assert callable(cls), (
                f"get_adapter_class('{model_name}') returned a non-callable: {cls!r}."
            )

    def test_known_model_class_is_instantiable(self):
        """The returned class must be instantiable without arguments."""
        for model_name in _EXPECTED_MODELS:
            cls = get_adapter_class(model_name)
            instance = cls()
            assert instance is not None

    def test_unknown_model_raises_key_error(self):
        """get_adapter_class() must raise KeyError for an unregistered model name."""
        with pytest.raises(KeyError):
            get_adapter_class("nonexistent-model-xyz")

    def test_key_error_message_mentions_model_name(self):
        """The KeyError message should include the bad model name."""
        bad_name = "totally-fake-model"
        with pytest.raises(KeyError) as exc_info:
            get_adapter_class(bad_name)
        assert bad_name in str(exc_info.value), (
            f"KeyError message does not mention the bad model name '{bad_name}'. "
            f"Got: {exc_info.value!r}"
        )
