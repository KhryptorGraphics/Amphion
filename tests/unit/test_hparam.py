# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for utils/hparam.py

Covers:
- HParams.__init__: create with keyword args, type inference, empty construction
- HParams.add_hparam: add new scalar/list param, reserved name error, empty list error
- HParams.set_hparam: update existing param with type check, type mismatch errors
- HParams.parse: parse 'a=1,b=2.5' comma-separated strings, unknown param error
- HParams.override_from_dict: dict-based override of existing params
- HParams.to_json / parse_json: serialization round-trip
- HParams.get: key exists (with and without default), key missing (default returned)
- HParams.values: returns all params as dict
- HParams.__contains__: 'in' operator
- HParams.del_hparam: remove a param
- parse_values: parse a string into type-checked dictionary
"""

import json

import pytest

from utils.hparam import HParams, parse_values


# ---------------------------------------------------------------------------
# HParams.__init__
# ---------------------------------------------------------------------------


class TestHParamsInit:
    def test_empty_init(self):
        """HParams can be created with no keyword arguments."""
        hp = HParams()
        assert hp.values() == {}

    def test_scalar_int(self):
        hp = HParams(num_layers=4)
        assert hp.num_layers == 4

    def test_scalar_float(self):
        hp = HParams(learning_rate=0.001)
        assert abs(hp.learning_rate - 0.001) < 1e-9

    def test_scalar_bool(self):
        hp = HParams(use_dropout=True)
        assert hp.use_dropout is True

    def test_scalar_string(self):
        hp = HParams(activation="relu")
        assert hp.activation == "relu"

    def test_list_of_ints(self):
        hp = HParams(layers=[64, 128, 256])
        assert hp.layers == [64, 128, 256]

    def test_list_of_floats(self):
        hp = HParams(rates=[0.1, 0.01, 0.001])
        assert hp.rates == [0.1, 0.01, 0.001]

    def test_list_of_strings(self):
        hp = HParams(activations=["relu", "tanh"])
        assert hp.activations == ["relu", "tanh"]

    def test_multiple_params(self):
        hp = HParams(lr=0.01, epochs=100, model="resnet")
        assert hp.lr == 0.01
        assert hp.epochs == 100
        assert hp.model == "resnet"

    def test_type_map_populated(self):
        hp = HParams(lr=0.01, epochs=10)
        assert "lr" in hp._hparam_types
        assert "epochs" in hp._hparam_types
        # float type, not a list
        assert hp._hparam_types["lr"] == (float, False)
        # int type, not a list
        assert hp._hparam_types["epochs"] == (int, False)

    def test_list_type_map_populated(self):
        hp = HParams(sizes=[32, 64])
        assert hp._hparam_types["sizes"] == (int, True)

    def test_model_structure_none_by_default(self):
        hp = HParams(lr=0.1)
        assert hp._model_structure is None

    def test_model_structure_stored(self):
        sentinel = object()
        hp = HParams(model_structure=sentinel)
        assert hp._model_structure is sentinel


# ---------------------------------------------------------------------------
# HParams.add_hparam
# ---------------------------------------------------------------------------


class TestHParamsAddHparam:
    def test_add_scalar_int(self):
        hp = HParams()
        hp.add_hparam("batch_size", 32)
        assert hp.batch_size == 32

    def test_add_scalar_float(self):
        hp = HParams()
        hp.add_hparam("dropout", 0.5)
        assert abs(hp.dropout - 0.5) < 1e-9

    def test_add_scalar_string(self):
        hp = HParams()
        hp.add_hparam("optimizer", "adam")
        assert hp.optimizer == "adam"

    def test_add_scalar_bool(self):
        hp = HParams()
        hp.add_hparam("verbose", False)
        assert hp.verbose is False

    def test_add_list_param(self):
        hp = HParams()
        hp.add_hparam("hidden_units", [128, 256, 512])
        assert hp.hidden_units == [128, 256, 512]

    def test_add_list_type_recorded(self):
        hp = HParams()
        hp.add_hparam("sizes", [16, 32])
        assert hp._hparam_types["sizes"] == (int, True)

    def test_add_empty_list_raises(self):
        hp = HParams()
        with pytest.raises(ValueError, match="cannot be empty"):
            hp.add_hparam("empty", [])

    def test_add_duplicate_raises(self):
        """Adding a param that conflicts with an existing attribute should raise."""
        hp = HParams(lr=0.01)
        with pytest.raises(ValueError, match="reserved"):
            hp.add_hparam("lr", 0.02)

    def test_add_after_construction(self):
        hp = HParams(lr=0.01)
        hp.add_hparam("momentum", 0.9)
        assert hp.momentum == 0.9
        assert "momentum" in hp._hparam_types


# ---------------------------------------------------------------------------
# HParams.set_hparam
# ---------------------------------------------------------------------------


class TestHParamsSetHparam:
    def test_set_int_param(self):
        hp = HParams(epochs=10)
        hp.set_hparam("epochs", 20)
        assert hp.epochs == 20

    def test_set_float_param(self):
        hp = HParams(lr=0.01)
        hp.set_hparam("lr", 0.001)
        assert abs(hp.lr - 0.001) < 1e-9

    def test_set_string_param(self):
        hp = HParams(optimizer="sgd")
        hp.set_hparam("optimizer", "adam")
        assert hp.optimizer == "adam"

    def test_set_bool_param(self):
        hp = HParams(use_bn=False)
        hp.set_hparam("use_bn", True)
        assert hp.use_bn is True

    def test_set_list_param(self):
        hp = HParams(sizes=[32, 64])
        hp.set_hparam("sizes", [128, 256])
        assert hp.sizes == [128, 256]

    def test_set_int_from_float_raises(self):
        """Cannot set an integer param with a float value."""
        hp = HParams(epochs=10)
        with pytest.raises(ValueError):
            hp.set_hparam("epochs", 10.5)

    def test_set_scalar_with_list_raises(self):
        """Cannot set a scalar param with a list."""
        hp = HParams(lr=0.01)
        with pytest.raises(ValueError, match="single-valued"):
            hp.set_hparam("lr", [0.01, 0.001])

    def test_set_list_with_scalar_raises(self):
        """Cannot set a list param with a scalar."""
        hp = HParams(sizes=[32, 64])
        with pytest.raises(ValueError, match="multi-valued"):
            hp.set_hparam("sizes", 128)

    def test_set_unknown_param_raises(self):
        """Setting a param that doesn't exist should raise KeyError."""
        hp = HParams(lr=0.01)
        with pytest.raises(KeyError):
            hp.set_hparam("nonexistent", 42)

    def test_set_int_from_int(self):
        """Integer-to-integer (same type) should work fine."""
        hp = HParams(steps=100)
        hp.set_hparam("steps", 200)
        assert hp.steps == 200


# ---------------------------------------------------------------------------
# HParams.parse
# ---------------------------------------------------------------------------


class TestHParamsParse:
    def test_parse_single_int(self):
        hp = HParams(epochs=10)
        hp.parse("epochs=20")
        assert hp.epochs == 20

    def test_parse_single_float(self):
        hp = HParams(lr=0.01)
        hp.parse("lr=0.001")
        assert abs(hp.lr - 0.001) < 1e-9

    def test_parse_multiple_params(self):
        hp = HParams(epochs=10, lr=0.01, model="mlp")
        hp.parse("epochs=50,lr=0.001,model=resnet")
        assert hp.epochs == 50
        assert abs(hp.lr - 0.001) < 1e-9
        assert hp.model == "resnet"

    def test_parse_returns_self(self):
        hp = HParams(epochs=10)
        result = hp.parse("epochs=20")
        assert result is hp

    def test_parse_bool_true(self):
        hp = HParams(verbose=False)
        hp.parse("verbose=true")
        assert hp.verbose is True

    def test_parse_bool_false(self):
        hp = HParams(verbose=True)
        hp.parse("verbose=false")
        assert hp.verbose is False

    def test_parse_list_of_ints(self):
        hp = HParams(sizes=[32, 64])
        hp.parse("sizes=[128,256,512]")
        assert hp.sizes == [128, 256, 512]

    def test_parse_unknown_param_raises(self):
        hp = HParams(lr=0.01)
        with pytest.raises(ValueError):
            hp.parse("unknown=5")

    def test_parse_empty_string(self):
        hp = HParams(lr=0.01)
        # Empty string should not change anything
        hp.parse("")
        assert abs(hp.lr - 0.01) < 1e-9

    def test_parse_preserves_unmentioned_params(self):
        hp = HParams(lr=0.01, epochs=10)
        hp.parse("epochs=50")
        assert abs(hp.lr - 0.01) < 1e-9  # lr unchanged
        assert hp.epochs == 50


# ---------------------------------------------------------------------------
# HParams.override_from_dict
# ---------------------------------------------------------------------------


class TestHParamsOverrideFromDict:
    def test_override_single_param(self):
        hp = HParams(lr=0.01)
        hp.override_from_dict({"lr": 0.001})
        assert abs(hp.lr - 0.001) < 1e-9

    def test_override_multiple_params(self):
        hp = HParams(lr=0.01, epochs=10, model="mlp")
        hp.override_from_dict({"lr": 0.1, "epochs": 100, "model": "cnn"})
        assert abs(hp.lr - 0.1) < 1e-9
        assert hp.epochs == 100
        assert hp.model == "cnn"

    def test_override_returns_self(self):
        hp = HParams(lr=0.01)
        result = hp.override_from_dict({"lr": 0.001})
        assert result is hp

    def test_override_empty_dict(self):
        hp = HParams(lr=0.01)
        hp.override_from_dict({})
        assert abs(hp.lr - 0.01) < 1e-9

    def test_override_unknown_key_raises(self):
        hp = HParams(lr=0.01)
        with pytest.raises(KeyError):
            hp.override_from_dict({"nonexistent": 42})

    def test_override_type_mismatch_raises(self):
        """Dict override with wrong type should fail via set_hparam."""
        hp = HParams(epochs=10)
        with pytest.raises(ValueError):
            hp.override_from_dict({"epochs": 10.5})


# ---------------------------------------------------------------------------
# HParams.to_json / parse_json
# ---------------------------------------------------------------------------


class TestHParamsJsonRoundtrip:
    def test_to_json_produces_valid_json(self):
        hp = HParams(lr=0.01, epochs=100)
        j = hp.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_json_contains_all_params(self):
        hp = HParams(lr=0.01, epochs=100, model="mlp")
        j = hp.to_json()
        parsed = json.loads(j)
        assert "lr" in parsed
        assert "epochs" in parsed
        assert "model" in parsed

    def test_to_json_values_correct(self):
        hp = HParams(lr=0.01, epochs=100, model="mlp")
        j = hp.to_json()
        parsed = json.loads(j)
        assert abs(parsed["lr"] - 0.01) < 1e-9
        assert parsed["epochs"] == 100
        assert parsed["model"] == "mlp"

    def test_roundtrip_scalar_types(self):
        hp = HParams(lr=0.01, epochs=100, label="test", flag=True)
        j = hp.to_json()
        # Create fresh HParams and override from JSON
        hp2 = HParams(lr=0.0, epochs=0, label="", flag=False)
        hp2.parse_json(j)
        assert abs(hp2.lr - 0.01) < 1e-9
        assert hp2.epochs == 100
        assert hp2.label == "test"
        assert hp2.flag is True

    def test_roundtrip_list_param(self):
        hp = HParams(sizes=[32, 64, 128])
        j = hp.to_json()
        hp2 = HParams(sizes=[1, 1, 1])
        hp2.parse_json(j)
        assert hp2.sizes == [32, 64, 128]

    def test_to_json_with_indent(self):
        hp = HParams(lr=0.01)
        j = hp.to_json(indent=2)
        # Should be pretty-printed (contain newlines)
        assert "\n" in j

    def test_to_json_sort_keys(self):
        hp = HParams(z_param=1, a_param=2)
        j = hp.to_json(sort_keys=True)
        parsed = json.loads(j)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_parse_json_returns_self(self):
        hp = HParams(lr=0.01)
        j = '{"lr": 0.001}'
        result = hp.parse_json(j)
        assert result is hp

    def test_parse_json_unknown_key_raises(self):
        hp = HParams(lr=0.01)
        with pytest.raises(KeyError):
            hp.parse_json('{"nonexistent": 42}')


# ---------------------------------------------------------------------------
# HParams.get
# ---------------------------------------------------------------------------


class TestHParamsGet:
    def test_get_existing_param_no_default(self):
        hp = HParams(lr=0.01)
        assert abs(hp.get("lr") - 0.01) < 1e-9

    def test_get_existing_param_with_compatible_default(self):
        hp = HParams(lr=0.01)
        # Default is compatible (same type) — should return the actual value
        result = hp.get("lr", 0.1)
        assert abs(result - 0.01) < 1e-9

    def test_get_missing_param_returns_none(self):
        hp = HParams(lr=0.01)
        result = hp.get("nonexistent")
        assert result is None

    def test_get_missing_param_returns_default(self):
        hp = HParams(lr=0.01)
        result = hp.get("nonexistent", 42)
        assert result == 42

    def test_get_missing_param_with_string_default(self):
        hp = HParams(lr=0.01)
        result = hp.get("missing_key", "fallback")
        assert result == "fallback"

    def test_get_list_param(self):
        hp = HParams(sizes=[32, 64])
        result = hp.get("sizes")
        assert result == [32, 64]

    def test_get_incompatible_default_raises(self):
        """A default value of wrong type should raise ValueError."""
        hp = HParams(epochs=10)
        with pytest.raises(ValueError):
            hp.get("epochs", "not_an_int")

    def test_get_list_param_with_scalar_default_raises(self):
        """Passing a scalar default for a list param should raise ValueError."""
        hp = HParams(sizes=[32, 64])
        with pytest.raises(ValueError):
            hp.get("sizes", 32)


# ---------------------------------------------------------------------------
# HParams.values
# ---------------------------------------------------------------------------


class TestHParamsValues:
    def test_values_empty(self):
        hp = HParams()
        assert hp.values() == {}

    def test_values_returns_all_params(self):
        hp = HParams(lr=0.01, epochs=100)
        v = hp.values()
        assert set(v.keys()) == {"lr", "epochs"}

    def test_values_correct_values(self):
        hp = HParams(lr=0.01, epochs=100, model="mlp")
        v = hp.values()
        assert abs(v["lr"] - 0.01) < 1e-9
        assert v["epochs"] == 100
        assert v["model"] == "mlp"

    def test_values_updated_after_set(self):
        hp = HParams(lr=0.01)
        hp.set_hparam("lr", 0.1)
        assert abs(hp.values()["lr"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# HParams.__contains__
# ---------------------------------------------------------------------------


class TestHParamsContains:
    def test_existing_key_in_hparams(self):
        hp = HParams(lr=0.01)
        assert "lr" in hp

    def test_missing_key_not_in_hparams(self):
        hp = HParams(lr=0.01)
        assert "nonexistent" not in hp

    def test_empty_hparams_contains_nothing(self):
        hp = HParams()
        assert "lr" not in hp


# ---------------------------------------------------------------------------
# HParams.del_hparam
# ---------------------------------------------------------------------------


class TestHParamsDelHparam:
    def test_del_existing_param(self):
        hp = HParams(lr=0.01, epochs=10)
        hp.del_hparam("lr")
        assert "lr" not in hp
        assert not hasattr(hp, "lr")

    def test_del_removes_from_type_map(self):
        hp = HParams(lr=0.01)
        hp.del_hparam("lr")
        assert "lr" not in hp._hparam_types

    def test_del_nonexistent_param_noop(self):
        """del_hparam should silently do nothing for non-existent params."""
        hp = HParams(lr=0.01)
        hp.del_hparam("nonexistent")  # should not raise
        assert "lr" in hp

    def test_del_then_readd(self):
        hp = HParams(lr=0.01)
        hp.del_hparam("lr")
        hp.add_hparam("lr", 0.001)
        assert abs(hp.lr - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# parse_values
# ---------------------------------------------------------------------------


class TestParseValues:
    def test_parse_single_int(self):
        result = parse_values("a=1", {"a": int})
        assert result == {"a": 1}

    def test_parse_single_float(self):
        result = parse_values("b=2.5", {"b": float})
        assert abs(result["b"] - 2.5) < 1e-9

    def test_parse_multiple_params(self):
        result = parse_values("a=1,b=2.5", {"a": int, "b": float})
        assert result["a"] == 1
        assert abs(result["b"] - 2.5) < 1e-9

    def test_parse_string_value(self):
        result = parse_values("name=resnet", {"name": str})
        assert result["name"] == "resnet"

    def test_parse_bool_true(self):
        result = parse_values("flag=true", {"flag": bool})
        assert result["flag"] is True

    def test_parse_bool_false(self):
        result = parse_values("flag=false", {"flag": bool})
        assert result["flag"] is False

    def test_parse_bool_True_capital(self):
        result = parse_values("flag=True", {"flag": bool})
        assert result["flag"] is True

    def test_parse_list_of_ints(self):
        result = parse_values("L=[1,2,3]", {"L": int})
        assert result["L"] == [1, 2, 3]

    def test_parse_list_of_floats(self):
        result = parse_values("rates=[0.1,0.01]", {"rates": float})
        assert len(result["rates"]) == 2
        assert abs(result["rates"][0] - 0.1) < 1e-9
        assert abs(result["rates"][1] - 0.01) < 1e-9

    def test_parse_mixed_scalars(self):
        result = parse_values("x=5,L=[1,2],n=hello", {"x": int, "L": int, "n": str})
        assert result["x"] == 5
        assert result["L"] == [1, 2]
        assert result["n"] == "hello"

    def test_parse_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            parse_values("z=1", {"a": int})

    def test_parse_unknown_key_ignored_with_flag(self):
        result = parse_values("z=1,a=2", {"a": int}, ignore_unknown=True)
        assert result == {"a": 2}

    def test_parse_duplicate_assignment_raises(self):
        with pytest.raises(ValueError, match="Multiple assignments"):
            parse_values("a=1,a=2", {"a": int})

    def test_parse_empty_string_returns_empty_dict(self):
        result = parse_values("", {"a": int})
        assert result == {}

    def test_parse_malformed_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            parse_values("=1", {"a": int})

    def test_parse_index_assignment(self):
        """Indexed assignment should produce a dict mapping index to value."""
        result = parse_values("arr[0]=10,arr[1]=20", {"arr": int})
        assert result == {"arr": {0: 10, 1: 20}}
