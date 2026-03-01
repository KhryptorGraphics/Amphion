"""
Unit tests for preprocessors/dataset_validator.py.

Tests all public functions:
- validate_metadata_entry()
- validate_file_exists()
- validate_audio_duration()
- validate_dataset_split()
- validate_dataset()
"""

import json
import os
import tempfile

import pytest

from preprocessors.dataset_validator import (
    validate_audio_duration,
    validate_dataset,
    validate_dataset_split,
    validate_file_exists,
    validate_metadata_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(**kwargs):
    """Return a minimal valid metadata entry, with optional overrides."""
    base = {
        "Dataset": "test_dataset",
        "Singer": "speaker_01",
        "Uid": "test_dataset_speaker_01_0001",
        "Path": "/data/audio/file.wav",
        "Duration": 3.5,
    }
    base.update(kwargs)
    return base


def _write_json(path, data):
    """Write *data* as JSON to *path*."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# validate_metadata_entry
# ---------------------------------------------------------------------------

class TestValidateMetadataEntry:
    """Tests for validate_metadata_entry()."""

    def test_valid_entry_returns_true(self):
        entry = _make_entry()
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is True
        assert errors == []

    def test_missing_required_field_returns_false(self):
        entry = _make_entry()
        del entry["Uid"]
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is False
        assert any("Uid" in e for e in errors)

    def test_none_field_value_returns_false(self):
        entry = _make_entry(Singer=None)
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is False
        assert any("Singer" in e for e in errors)

    def test_empty_string_field_returns_false(self):
        entry = _make_entry(Dataset="   ")
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is False
        assert any("Dataset" in e for e in errors)

    def test_multiple_missing_fields_all_reported(self):
        entry = {}
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is False
        # Should report at least the five default required fields
        assert len(errors) >= 5

    def test_custom_required_fields(self):
        entry = {"Text": "hello world", "Path": "/audio/a.wav"}
        is_valid, errors = validate_metadata_entry(entry, required_fields=["Text", "Path"])
        assert is_valid is True
        assert errors == []

    def test_custom_required_fields_missing_one(self):
        entry = {"Text": "hello world"}
        is_valid, errors = validate_metadata_entry(entry, required_fields=["Text", "Path"])
        assert is_valid is False
        assert any("Path" in e for e in errors)

    def test_numeric_duration_field_is_valid(self):
        """Duration is a number, not a string; should not be flagged as empty."""
        entry = _make_entry(Duration=0.5)
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is True

    def test_zero_duration_is_valid_field_presence(self):
        """validate_metadata_entry only checks field presence/emptiness, not range."""
        entry = _make_entry(Duration=0)
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is True

    def test_whitespace_only_string_is_invalid(self):
        entry = _make_entry(Path="\t\n")
        is_valid, errors = validate_metadata_entry(entry)
        assert is_valid is False

    def test_returns_tuple_of_bool_and_list(self):
        entry = _make_entry()
        result = validate_metadata_entry(entry)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# validate_file_exists
# ---------------------------------------------------------------------------

class TestValidateFileExists:
    """Tests for validate_file_exists()."""

    def test_existing_file_returns_true(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.write_text("dummy")
        assert validate_file_exists(str(f)) is True

    def test_missing_file_returns_false(self, tmp_path):
        missing = str(tmp_path / "nonexistent.wav")
        assert validate_file_exists(missing) is False

    def test_missing_file_raise_on_missing_raises(self, tmp_path):
        missing = str(tmp_path / "nonexistent.wav")
        with pytest.raises(FileNotFoundError):
            validate_file_exists(missing, raise_on_missing=True)

    def test_existing_file_raise_on_missing_does_not_raise(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.write_text("dummy")
        result = validate_file_exists(str(f), raise_on_missing=True)
        assert result is True

    def test_existing_directory_returns_true(self, tmp_path):
        """os.path.exists is True for directories too."""
        assert validate_file_exists(str(tmp_path)) is True

    def test_error_message_contains_path(self, tmp_path):
        missing = str(tmp_path / "missing.wav")
        with pytest.raises(FileNotFoundError, match=str(missing)):
            validate_file_exists(missing, raise_on_missing=True)

    def test_default_raise_on_missing_is_false(self, tmp_path):
        """Should return False (not raise) by default when file is missing."""
        missing = str(tmp_path / "nope.wav")
        result = validate_file_exists(missing)
        assert result is False


# ---------------------------------------------------------------------------
# validate_audio_duration
# ---------------------------------------------------------------------------

class TestValidateAudioDuration:
    """Tests for validate_audio_duration()."""

    def test_duration_within_range_returns_true(self):
        assert validate_audio_duration(5.0, min_duration=1.0, max_duration=10.0) is True

    def test_duration_at_min_boundary_returns_true(self):
        assert validate_audio_duration(1.0, min_duration=1.0) is True

    def test_duration_below_min_returns_false(self):
        assert validate_audio_duration(0.5, min_duration=1.0) is False

    def test_duration_at_max_boundary_returns_true(self):
        assert validate_audio_duration(10.0, max_duration=10.0) is True

    def test_duration_above_max_returns_false(self):
        assert validate_audio_duration(11.0, max_duration=10.0) is False

    def test_no_max_duration_allows_any_large_value(self):
        assert validate_audio_duration(9999.0, min_duration=0.0) is True

    def test_zero_duration_with_default_min_returns_true(self):
        """Default min_duration is 0.0, so 0.0 is valid."""
        assert validate_audio_duration(0.0) is True

    def test_negative_duration_with_default_min_returns_false(self):
        assert validate_audio_duration(-1.0) is False

    def test_exact_range_boundaries(self):
        assert validate_audio_duration(3.0, min_duration=3.0, max_duration=3.0) is True

    def test_just_outside_both_boundaries(self):
        assert validate_audio_duration(2.9, min_duration=3.0, max_duration=10.0) is False
        assert validate_audio_duration(10.1, min_duration=3.0, max_duration=10.0) is False

    def test_returns_bool(self):
        result = validate_audio_duration(5.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# validate_dataset_split
# ---------------------------------------------------------------------------

class TestValidateDatasetSplit:
    """Tests for validate_dataset_split()."""

    def test_valid_split_all_entries_count(self, tmp_path):
        entries = [_make_entry(Uid="uid_{}".format(i)) for i in range(3)]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(json_path, check_files=False)
        assert valid_count == 3
        assert errors_list == []

    def test_missing_json_file_returns_zero_valid(self, tmp_path):
        json_path = str(tmp_path / "nonexistent.json")
        valid_count, errors_list = validate_dataset_split(json_path, check_files=False)
        assert valid_count == 0
        assert len(errors_list) == 1
        idx, uid, errs = errors_list[0]
        assert idx == -1

    def test_entry_with_missing_field_reported(self, tmp_path):
        entries = [_make_entry()]
        del entries[0]["Singer"]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(json_path, check_files=False)
        assert valid_count == 0
        assert len(errors_list) == 1
        idx, uid, errs = errors_list[0]
        assert idx == 0
        assert any("Singer" in e for e in errs)

    def test_mixed_valid_and_invalid_entries(self, tmp_path):
        entries = [
            _make_entry(Uid="uid_0"),
            _make_entry(Uid=""),  # empty Uid – invalid
            _make_entry(Uid="uid_2"),
        ]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(json_path, check_files=False)
        assert valid_count == 2
        assert len(errors_list) == 1
        idx, uid, errs = errors_list[0]
        assert idx == 1

    def test_check_files_true_reports_missing_path(self, tmp_path):
        entries = [_make_entry(Path=str(tmp_path / "missing.wav"))]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(json_path, check_files=True)
        assert valid_count == 0
        idx, uid, errs = errors_list[0]
        assert any("not found" in e.lower() or "missing" in e.lower() for e in errs)

    def test_check_files_true_valid_when_files_exist(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("dummy")
        entries = [_make_entry(Path=str(audio_file))]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(json_path, check_files=True)
        assert valid_count == 1
        assert errors_list == []

    def test_check_files_false_skips_file_presence(self, tmp_path):
        entries = [_make_entry(Path="/totally/nonexistent/path/audio.wav")]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(json_path, check_files=False)
        assert valid_count == 1
        assert errors_list == []

    def test_custom_required_fields(self, tmp_path):
        entries = [{"Text": "hello", "Uid": "uid_0"}]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        valid_count, errors_list = validate_dataset_split(
            json_path, required_fields=["Text", "Uid"], check_files=False
        )
        assert valid_count == 1
        assert errors_list == []

    def test_empty_json_array_returns_zero_valid_no_errors(self, tmp_path):
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, [])
        valid_count, errors_list = validate_dataset_split(json_path, check_files=False)
        assert valid_count == 0
        assert errors_list == []

    def test_returns_tuple_of_int_and_list(self, tmp_path):
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, [])
        result = validate_dataset_split(json_path, check_files=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], list)

    def test_error_tuple_structure(self, tmp_path):
        entries = [_make_entry(Uid=None)]
        json_path = str(tmp_path / "train.json")
        _write_json(json_path, entries)

        _, errors_list = validate_dataset_split(json_path, check_files=False)
        assert len(errors_list) == 1
        idx, uid, errs = errors_list[0]
        assert isinstance(idx, int)
        assert isinstance(errs, list)


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

class TestValidateDataset:
    """Tests for validate_dataset()."""

    def _make_audio_file(self, tmp_path, name="audio.wav"):
        """Create a dummy file on disk and return its path string."""
        f = tmp_path / name
        f.write_text("dummy")
        return str(f)

    def _build_dataset_dir(self, tmp_path, dataset_name, splits_data):
        """Create a dataset directory with split JSON files."""
        dataset_dir = tmp_path / dataset_name
        dataset_dir.mkdir()
        for split, entries in splits_data.items():
            _write_json(str(dataset_dir / "{}.json".format(split)), entries)
        return str(tmp_path)

    def test_all_splits_present_and_valid(self, tmp_path):
        audio = self._make_audio_file(tmp_path, "a.wav")
        entries = [_make_entry(Uid="uid_{}".format(i), Path=audio) for i in range(2)]
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"train": entries, "test": entries, "valid": entries},
        )

        results = validate_dataset(output_dir, "mydata")

        for split in ["train", "test", "valid"]:
            assert results[split]["file_found"] is True
            assert results[split]["valid_count"] == 2
            assert results[split]["total_count"] == 2
            assert results[split]["errors"] == []

    def test_missing_split_file_marked_not_found(self, tmp_path):
        audio = self._make_audio_file(tmp_path, "a.wav")
        entries = [_make_entry(Path=audio)]
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"train": entries},  # only train – no test/valid
        )

        results = validate_dataset(output_dir, "mydata")

        assert results["train"]["file_found"] is True
        assert results["test"]["file_found"] is False
        assert results["valid"]["file_found"] is False

    def test_invalid_entries_reported_per_split(self, tmp_path):
        audio = self._make_audio_file(tmp_path, "a.wav")
        bad_entry = _make_entry(Singer=None, Path=audio)
        good_entry = _make_entry(Uid="uid_good", Path=audio)
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"train": [bad_entry, good_entry], "test": [], "valid": []},
        )

        results = validate_dataset(output_dir, "mydata")

        assert results["train"]["valid_count"] == 1
        assert results["train"]["total_count"] == 2
        assert len(results["train"]["errors"]) == 1

    def test_custom_splits(self, tmp_path):
        audio = self._make_audio_file(tmp_path, "a.wav")
        entries = [_make_entry(Path=audio)]
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"dev": entries},
        )

        results = validate_dataset(output_dir, "mydata", splits=["dev"])

        assert "dev" in results
        assert results["dev"]["file_found"] is True
        assert results["dev"]["valid_count"] == 1

    def test_returns_dict_with_all_default_splits(self, tmp_path):
        audio = self._make_audio_file(tmp_path, "a.wav")
        entries = [_make_entry(Path=audio)]
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"train": entries, "test": entries, "valid": entries},
        )

        results = validate_dataset(output_dir, "mydata")

        assert isinstance(results, dict)
        for split in ["train", "test", "valid"]:
            assert split in results

    def test_result_dict_keys_present(self, tmp_path):
        audio = self._make_audio_file(tmp_path, "a.wav")
        entries = [_make_entry(Path=audio)]
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"train": entries, "test": entries, "valid": entries},
        )

        results = validate_dataset(output_dir, "mydata")

        for split in ["train", "test", "valid"]:
            assert "valid_count" in results[split]
            assert "total_count" in results[split]
            assert "errors" in results[split]
            assert "file_found" in results[split]

    def test_empty_split_files(self, tmp_path):
        output_dir = self._build_dataset_dir(
            tmp_path,
            "mydata",
            {"train": [], "test": [], "valid": []},
        )

        results = validate_dataset(output_dir, "mydata")

        for split in ["train", "test", "valid"]:
            assert results[split]["valid_count"] == 0
            assert results[split]["total_count"] == 0
            assert results[split]["errors"] == []
            assert results[split]["file_found"] is True

    def test_dataset_dir_does_not_exist(self, tmp_path):
        output_dir = str(tmp_path)
        # dataset subdir never created

        results = validate_dataset(output_dir, "nonexistent_dataset")

        for split in ["train", "test", "valid"]:
            assert results[split]["file_found"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
