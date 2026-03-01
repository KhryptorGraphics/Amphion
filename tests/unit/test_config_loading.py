# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for JsonHParams (from utils/util.py) and config/base.json loading.

Covers:
- JsonHParams.__init__: basic attribute access from kwargs
- JsonHParams nested dict support: dict values are recursively converted
- JsonHParams dict interface: keys(), items(), values()
- JsonHParams.__len__: number of stored attributes
- JsonHParams.__contains__: 'in' operator
- JsonHParams.__setitem__ / __getitem__: bracket-style access
- JsonHParams.__repr__: string representation
- load_config / json5 parsing: config/base.json loads without error and
  contains expected top-level keys (preprocess, train, supported_model_type)
"""

import os

import json5
import pytest

from utils.util import JsonHParams


# ---------------------------------------------------------------------------
# JsonHParams basic attribute access
# ---------------------------------------------------------------------------


class TestJsonHParamsAttributeAccess:
    def test_single_scalar_int(self):
        """Integer keyword arg is accessible as attribute."""
        cfg = JsonHParams(batch_size=16)
        assert cfg.batch_size == 16

    def test_single_scalar_float(self):
        """Float keyword arg is accessible as attribute."""
        cfg = JsonHParams(lr=0.001)
        assert abs(cfg.lr - 0.001) < 1e-9

    def test_single_scalar_bool(self):
        """Bool keyword arg is accessible as attribute."""
        cfg = JsonHParams(use_dropout=True)
        assert cfg.use_dropout is True

    def test_single_scalar_string(self):
        """String keyword arg is accessible as attribute."""
        cfg = JsonHParams(optimizer="adam")
        assert cfg.optimizer == "adam"

    def test_multiple_scalars(self):
        """Multiple keyword args are all accessible as attributes."""
        cfg = JsonHParams(lr=0.01, epochs=100, model_name="resnet")
        assert abs(cfg.lr - 0.01) < 1e-9
        assert cfg.epochs == 100
        assert cfg.model_name == "resnet"

    def test_list_value(self):
        """List values are stored as-is."""
        cfg = JsonHParams(hidden_dims=[64, 128, 256])
        assert cfg.hidden_dims == [64, 128, 256]

    def test_empty_construction(self):
        """JsonHParams with no kwargs creates an empty object."""
        cfg = JsonHParams()
        assert len(cfg) == 0


# ---------------------------------------------------------------------------
# JsonHParams nested dict support
# ---------------------------------------------------------------------------


class TestJsonHParamsNestedDict:
    def test_nested_dict_becomes_json_hparams(self):
        """A dict value is recursively converted to a JsonHParams instance."""
        cfg = JsonHParams(preprocess={"n_mel": 80, "sample_rate": 24000})
        assert isinstance(cfg.preprocess, JsonHParams)

    def test_nested_dict_attribute_access(self):
        """Nested JsonHParams attributes are accessible via dot notation."""
        cfg = JsonHParams(preprocess={"n_mel": 80, "hop_size": 120})
        assert cfg.preprocess.n_mel == 80
        assert cfg.preprocess.hop_size == 120

    def test_doubly_nested_dict(self):
        """Deeply nested dicts are also converted recursively."""
        cfg = JsonHParams(outer={"inner": {"value": 42}})
        assert isinstance(cfg.outer, JsonHParams)
        assert isinstance(cfg.outer.inner, JsonHParams)
        assert cfg.outer.inner.value == 42

    def test_non_dict_value_not_converted(self):
        """Non-dict values (list, int, str) are stored as-is."""
        cfg = JsonHParams(sizes=[1, 2, 3], count=5, name="test")
        assert cfg.sizes == [1, 2, 3]
        assert cfg.count == 5
        assert cfg.name == "test"

    def test_mixed_nested_and_scalar(self):
        """A mix of nested dicts and scalar values works correctly."""
        cfg = JsonHParams(
            lr=0.001,
            model={"hidden": 256, "layers": 4},
        )
        assert abs(cfg.lr - 0.001) < 1e-9
        assert cfg.model.hidden == 256
        assert cfg.model.layers == 4


# ---------------------------------------------------------------------------
# JsonHParams dict interface: keys(), items(), values()
# ---------------------------------------------------------------------------


class TestJsonHParamsDictInterface:
    def test_keys_returns_all_keys(self):
        """keys() returns all top-level attribute names."""
        cfg = JsonHParams(a=1, b=2, c=3)
        assert set(cfg.keys()) == {"a", "b", "c"}

    def test_keys_empty(self):
        """keys() on an empty JsonHParams returns an empty view."""
        cfg = JsonHParams()
        assert set(cfg.keys()) == set()

    def test_values_returns_all_values(self):
        """values() returns all stored values."""
        cfg = JsonHParams(x=10, y=20)
        assert set(cfg.values()) == {10, 20}

    def test_values_empty(self):
        """values() on an empty JsonHParams returns an empty view."""
        cfg = JsonHParams()
        assert list(cfg.values()) == []

    def test_items_returns_key_value_pairs(self):
        """items() returns all (key, value) pairs."""
        cfg = JsonHParams(p=1, q=2)
        items = dict(cfg.items())
        assert items == {"p": 1, "q": 2}

    def test_items_can_be_iterated(self):
        """items() can be iterated to reconstruct a plain dict."""
        cfg = JsonHParams(alpha=0.1, beta=0.2)
        reconstructed = {k: v for k, v in cfg.items()}
        assert abs(reconstructed["alpha"] - 0.1) < 1e-9
        assert abs(reconstructed["beta"] - 0.2) < 1e-9

    def test_keys_items_values_consistent(self):
        """keys(), items(), and values() are internally consistent."""
        cfg = JsonHParams(a=1, b=2)
        keys = list(cfg.keys())
        vals = list(cfg.values())
        items = list(cfg.items())
        assert len(keys) == len(vals) == len(items)
        for k, v in items:
            assert k in keys
            assert v in vals


# ---------------------------------------------------------------------------
# JsonHParams.__len__
# ---------------------------------------------------------------------------


class TestJsonHParamsLen:
    def test_len_empty(self):
        """len() of empty JsonHParams is 0."""
        cfg = JsonHParams()
        assert len(cfg) == 0

    def test_len_one_item(self):
        """len() returns 1 for a single kwarg."""
        cfg = JsonHParams(only_one=42)
        assert len(cfg) == 1

    def test_len_multiple_items(self):
        """len() returns the total number of top-level kwargs."""
        cfg = JsonHParams(a=1, b=2, c=3, d=4)
        assert len(cfg) == 4

    def test_len_nested_counts_only_top_level(self):
        """len() counts only top-level keys, not nested ones."""
        cfg = JsonHParams(outer={"inner_a": 1, "inner_b": 2})
        # Only 'outer' is a top-level key
        assert len(cfg) == 1


# ---------------------------------------------------------------------------
# JsonHParams.__contains__
# ---------------------------------------------------------------------------


class TestJsonHParamsContains:
    def test_existing_key_found(self):
        """'in' operator returns True for a stored key."""
        cfg = JsonHParams(learning_rate=0.01)
        assert "learning_rate" in cfg

    def test_missing_key_not_found(self):
        """'in' operator returns False for a key that was not added."""
        cfg = JsonHParams(learning_rate=0.01)
        assert "nonexistent" not in cfg

    def test_empty_cfg_contains_nothing(self):
        """'in' returns False for any key on an empty JsonHParams."""
        cfg = JsonHParams()
        assert "anything" not in cfg

    def test_nested_key_not_in_top_level(self):
        """Keys of a nested dict are NOT contained at the top level."""
        cfg = JsonHParams(model={"hidden": 256})
        assert "model" in cfg
        assert "hidden" not in cfg


# ---------------------------------------------------------------------------
# JsonHParams.__setitem__ / __getitem__
# ---------------------------------------------------------------------------


class TestJsonHParamsSetGetItem:
    def test_getitem_retrieves_value(self):
        """cfg['key'] returns the value set at 'key'."""
        cfg = JsonHParams(n_mel=80)
        assert cfg["n_mel"] == 80

    def test_setitem_stores_value(self):
        """cfg['key'] = value stores a new attribute."""
        cfg = JsonHParams()
        cfg["sample_rate"] = 24000
        assert cfg.sample_rate == 24000

    def test_setitem_then_getitem(self):
        """Set via bracket notation, then retrieve via bracket notation."""
        cfg = JsonHParams()
        cfg["hop_size"] = 120
        assert cfg["hop_size"] == 120

    def test_setitem_overrides_existing(self):
        """cfg['key'] = new_value overrides a previously stored attribute."""
        cfg = JsonHParams(n_fft=1024)
        cfg["n_fft"] = 2048
        assert cfg["n_fft"] == 2048
        assert cfg.n_fft == 2048

    def test_getitem_equals_attribute_access(self):
        """cfg['key'] and cfg.key return the same value."""
        cfg = JsonHParams(fmin=0, fmax=12000)
        assert cfg["fmin"] == cfg.fmin
        assert cfg["fmax"] == cfg.fmax

    def test_setitem_non_dict_value(self):
        """Setting a list value via bracket notation works."""
        cfg = JsonHParams()
        cfg["tracker"] = ["tensorboard", "wandb"]
        assert cfg["tracker"] == ["tensorboard", "wandb"]


# ---------------------------------------------------------------------------
# JsonHParams.__repr__
# ---------------------------------------------------------------------------


class TestJsonHParamsRepr:
    def test_repr_is_string(self):
        """repr() returns a string."""
        cfg = JsonHParams(a=1)
        assert isinstance(repr(cfg), str)

    def test_repr_contains_key(self):
        """repr() output contains the attribute name."""
        cfg = JsonHParams(my_key=42)
        assert "my_key" in repr(cfg)

    def test_repr_empty(self):
        """repr() of empty JsonHParams produces '{}'."""
        cfg = JsonHParams()
        assert repr(cfg) == "{}"


# ---------------------------------------------------------------------------
# config/base.json loading with json5
# ---------------------------------------------------------------------------


class TestBaseJsonLoading:
    """Tests that config/base.json can be loaded via json5 and has the
    expected structure."""

    @pytest.fixture
    def base_config_path(self):
        """Return the path to config/base.json relative to the repo root."""
        # The test runner is expected to run from the repo root (PYTHONPATH=.)
        return os.path.join("config", "base.json")

    @pytest.fixture
    def base_config(self, base_config_path):
        """Load config/base.json using json5 and return the resulting dict."""
        with open(base_config_path, "r") as f:
            return json5.load(f)

    def test_file_exists(self, base_config_path):
        """config/base.json must exist on disk."""
        assert os.path.isfile(base_config_path), (
            f"Expected config/base.json at {base_config_path!r}"
        )

    def test_parses_without_error(self, base_config):
        """json5.load() must succeed and return a dict."""
        assert isinstance(base_config, dict)

    def test_has_preprocess_key(self, base_config):
        """Top-level key 'preprocess' is present."""
        assert "preprocess" in base_config

    def test_has_train_key(self, base_config):
        """Top-level key 'train' is present."""
        assert "train" in base_config

    def test_has_supported_model_type_key(self, base_config):
        """Top-level key 'supported_model_type' is present."""
        assert "supported_model_type" in base_config

    def test_preprocess_is_dict(self, base_config):
        """The 'preprocess' value is a dict (mapping of settings)."""
        assert isinstance(base_config["preprocess"], dict)

    def test_train_is_dict(self, base_config):
        """The 'train' value is a dict (mapping of settings)."""
        assert isinstance(base_config["train"], dict)

    def test_preprocess_has_n_mel(self, base_config):
        """preprocess section contains 'n_mel'."""
        assert "n_mel" in base_config["preprocess"]

    def test_preprocess_n_mel_value(self, base_config):
        """preprocess.n_mel equals 80 (the default in base.json)."""
        assert base_config["preprocess"]["n_mel"] == 80

    def test_preprocess_has_sample_rate(self, base_config):
        """preprocess section contains 'sample_rate'."""
        assert "sample_rate" in base_config["preprocess"]

    def test_preprocess_sample_rate_value(self, base_config):
        """preprocess.sample_rate equals 24000 (the default in base.json)."""
        assert base_config["preprocess"]["sample_rate"] == 24000

    def test_train_has_batch_size(self, base_config):
        """train section contains 'batch_size'."""
        assert "batch_size" in base_config["train"]

    def test_train_batch_size_is_positive(self, base_config):
        """train.batch_size is a positive integer."""
        batch_size = base_config["train"]["batch_size"]
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_config_loadable_as_json_hparams(self, base_config):
        """The parsed config dict can be wrapped in JsonHParams successfully."""
        cfg = JsonHParams(**base_config)
        assert "preprocess" in cfg
        assert "train" in cfg
        # Nested dicts should have been converted to JsonHParams
        assert isinstance(cfg.preprocess, JsonHParams)
        assert isinstance(cfg.train, JsonHParams)

    def test_json_hparams_nested_access(self, base_config):
        """After wrapping in JsonHParams, nested fields are dot-accessible."""
        cfg = JsonHParams(**base_config)
        assert cfg.preprocess.n_mel == 80
        assert cfg.preprocess.sample_rate == 24000
        assert cfg.train.batch_size == base_config["train"]["batch_size"]
