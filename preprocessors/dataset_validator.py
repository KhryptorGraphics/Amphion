# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os


_DEFAULT_REQUIRED_FIELDS = ["Dataset", "Singer", "Uid", "Path", "Duration"]
_DEFAULT_SPLITS = ["train", "test", "valid"]


def validate_metadata_entry(entry, required_fields=None):
    """Validate a single metadata entry dict from a preprocessed JSON file.

    Checks that all required fields are present and non-empty.

    Args:
        entry (dict): A metadata entry as produced by preprocessors.
        required_fields (list, optional): Fields that must be present and non-empty.
            Defaults to ['Dataset', 'Singer', 'Uid', 'Path', 'Duration'].

    Returns:
        tuple: (is_valid, errors) where is_valid is bool and errors is a list of
            string descriptions of any validation failures.
    """
    if required_fields is None:
        required_fields = _DEFAULT_REQUIRED_FIELDS

    errors = []

    for field in required_fields:
        if field not in entry:
            errors.append("Missing required field: '{}'".format(field))
        elif entry[field] is None:
            errors.append("Field '{}' is None".format(field))
        elif isinstance(entry[field], str) and entry[field].strip() == "":
            errors.append("Field '{}' is empty string".format(field))

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_file_exists(path, raise_on_missing=False):
    """Check whether a file exists at the given path.

    Args:
        path (str): Filesystem path to check.
        raise_on_missing (bool): If True, raises FileNotFoundError when the file
            does not exist. Defaults to False.

    Returns:
        bool: True if the file exists, False otherwise (when raise_on_missing=False).

    Raises:
        FileNotFoundError: If the file does not exist and raise_on_missing=True.
    """
    exists = os.path.exists(path)
    if not exists and raise_on_missing:
        raise FileNotFoundError("File not found: {}".format(path))
    return exists


def validate_audio_duration(duration, min_duration=0.0, max_duration=None):
    """Validate that an audio duration falls within an acceptable range.

    Args:
        duration (float): Duration in seconds.
        min_duration (float): Minimum acceptable duration (inclusive). Defaults to 0.0.
        max_duration (float, optional): Maximum acceptable duration (inclusive).
            If None, no upper bound is enforced. Defaults to None.

    Returns:
        bool: True if the duration is within the specified range, False otherwise.
    """
    if duration < min_duration:
        return False
    if max_duration is not None and duration > max_duration:
        return False
    return True


def validate_dataset_split(json_path, required_fields=None, check_files=True):
    """Load and validate all entries in a preprocessed split JSON file.

    Args:
        json_path (str): Path to a train.json, test.json, or valid.json file.
        required_fields (list, optional): Fields required in each entry.
            Defaults to ['Dataset', 'Singer', 'Uid', 'Path', 'Duration'].
        check_files (bool): If True, also checks that each entry's 'Path' exists
            on disk. Defaults to True.

    Returns:
        tuple: (valid_count, errors_list) where valid_count is the number of fully
            valid entries and errors_list is a list of (index, uid, errors) tuples
            describing any validation failures.
    """
    if required_fields is None:
        required_fields = _DEFAULT_REQUIRED_FIELDS

    if not os.path.exists(json_path):
        return 0, [(-1, None, ["Split file not found: {}".format(json_path)])]

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    valid_count = 0
    errors_list = []

    for idx, entry in enumerate(entries):
        uid = entry.get("Uid", None)
        entry_errors = []

        is_valid, field_errors = validate_metadata_entry(entry, required_fields)
        entry_errors.extend(field_errors)

        if check_files and "Path" in entry and entry["Path"]:
            if not validate_file_exists(entry["Path"]):
                entry_errors.append("File not found: {}".format(entry["Path"]))

        if len(entry_errors) == 0:
            valid_count += 1
        else:
            errors_list.append((idx, uid, entry_errors))

    return valid_count, errors_list


def validate_dataset(output_dir, dataset_name, splits=None):
    """Validate all splits for a preprocessed dataset directory.

    Expects split files at: output_dir/dataset_name/{split}.json

    Args:
        output_dir (str): Base output directory (same as the output_path used by
            preprocessors).
        dataset_name (str): Name of the dataset subdirectory (e.g., 'libritts').
        splits (list, optional): Split names to validate. Defaults to
            ['train', 'test', 'valid'].

    Returns:
        dict: Mapping from split name to a dict with keys:
            - 'valid_count' (int): Number of valid entries.
            - 'total_count' (int): Total number of entries.
            - 'errors' (list): List of (index, uid, errors) tuples for invalid entries.
            - 'file_found' (bool): Whether the split JSON file was found.
    """
    if splits is None:
        splits = _DEFAULT_SPLITS

    dataset_dir = os.path.join(output_dir, dataset_name)
    results = {}

    print("-" * 10)
    print("Validating dataset: {}\n".format(dataset_name))

    for split in splits:
        json_path = os.path.join(dataset_dir, "{}.json".format(split))
        file_found = os.path.exists(json_path)

        if not file_found:
            print("[{}] Split file not found: {}".format(split, json_path))
            results[split] = {
                "valid_count": 0,
                "total_count": 0,
                "errors": [(-1, None, ["Split file not found: {}".format(json_path)])],
                "file_found": False,
            }
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        total_count = len(entries)

        valid_count, errors_list = validate_dataset_split(json_path)

        results[split] = {
            "valid_count": valid_count,
            "total_count": total_count,
            "errors": errors_list,
            "file_found": True,
        }

        print(
            "[{}] {}/{} entries valid, {} errors".format(
                split, valid_count, total_count, len(errors_list)
            )
        )

    return results
