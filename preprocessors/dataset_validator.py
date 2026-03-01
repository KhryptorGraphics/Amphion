# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


DEFAULT_REQUIRED_FIELDS = ["Dataset", "Uid", "Path", "Duration"]
DEFAULT_SPLITS = ["train", "test", "valid"]


@dataclass
class ValidationResult:
    """Result of a dataset validation pass.

    Attributes:
        is_valid: True when no errors were found (warnings are permitted).
        errors: Fatal problems that indicate corrupt or unusable data.
        warnings: Non-fatal issues worth reviewing (e.g. very short clips).
        stats: Summary statistics collected during validation.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


def validate_metadata_integrity(
    utterances: List[dict],
    required_fields: Optional[List[str]] = None,
) -> ValidationResult:
    """Check that every utterance dict contains all required fields with non-None values.

    Args:
        utterances: List of utterance dicts loaded from a metadata JSON file.
        required_fields: Fields that must be present and non-None in each entry.
            Defaults to DEFAULT_REQUIRED_FIELDS.

    Returns:
        ValidationResult with errors for each missing or None-valued field.
    """
    if required_fields is None:
        required_fields = DEFAULT_REQUIRED_FIELDS

    result = ValidationResult()
    missing_field_counts: Dict[str, int] = {f: 0 for f in required_fields}

    for idx, utt in enumerate(utterances):
        uid = utt.get("Uid", f"<index {idx}>")
        for field_name in required_fields:
            if field_name not in utt or utt[field_name] is None:
                result.errors.append(
                    f"Utterance '{uid}': missing or None value for required field '{field_name}'"
                )
                missing_field_counts[field_name] += 1

    if result.errors:
        result.is_valid = False

    result.stats["total_utterances"] = len(utterances)
    result.stats["missing_field_counts"] = {
        k: v for k, v in missing_field_counts.items() if v > 0
    }

    return result


def validate_file_existence(utterances: List[dict]) -> ValidationResult:
    """Check that the audio file path in each utterance exists on disk.

    Args:
        utterances: List of utterance dicts, each expected to have a 'Path' key.

    Returns:
        ValidationResult with errors for every missing file.
    """
    result = ValidationResult()
    missing_files = []

    for idx, utt in enumerate(utterances):
        uid = utt.get("Uid", f"<index {idx}>")
        path = utt.get("Path")
        if path is None:
            result.errors.append(
                f"Utterance '{uid}': 'Path' field is missing, cannot check file existence"
            )
        elif not os.path.exists(path):
            result.errors.append(
                f"Utterance '{uid}': file not found at path '{path}'"
            )
            missing_files.append(path)

    if result.errors:
        result.is_valid = False

    result.stats["total_utterances"] = len(utterances)
    result.stats["missing_files_count"] = len(missing_files)

    return result


def validate_audio_properties(
    utterances: List[dict],
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> ValidationResult:
    """Check Duration values for each utterance.

    Errors are raised for non-positive (zero or negative) durations.
    Warnings are raised when a duration falls outside [min_duration, max_duration].

    Args:
        utterances: List of utterance dicts, each expected to have a 'Duration' key.
        min_duration: Optional lower bound (seconds). Clips shorter than this get a warning.
        max_duration: Optional upper bound (seconds). Clips longer than this get a warning.

    Returns:
        ValidationResult with errors for invalid durations and warnings for suspicious ones.
    """
    result = ValidationResult()
    durations = []

    for idx, utt in enumerate(utterances):
        uid = utt.get("Uid", f"<index {idx}>")
        duration = utt.get("Duration")

        if duration is None:
            result.errors.append(
                f"Utterance '{uid}': 'Duration' field is missing"
            )
            result.is_valid = False
            continue

        try:
            duration = float(duration)
        except (TypeError, ValueError):
            result.errors.append(
                f"Utterance '{uid}': 'Duration' is not a valid number (got {duration!r})"
            )
            result.is_valid = False
            continue

        if duration <= 0:
            result.errors.append(
                f"Utterance '{uid}': 'Duration' must be positive (got {duration})"
            )
            result.is_valid = False
        else:
            durations.append(duration)
            if min_duration is not None and duration < min_duration:
                result.warnings.append(
                    f"Utterance '{uid}': duration {duration:.3f}s is below min_duration {min_duration}s"
                )
            if max_duration is not None and duration > max_duration:
                result.warnings.append(
                    f"Utterance '{uid}': duration {duration:.3f}s exceeds max_duration {max_duration}s"
                )

    result.stats["total_utterances"] = len(utterances)
    if durations:
        result.stats["duration_min"] = round(min(durations), 4)
        result.stats["duration_max"] = round(max(durations), 4)
        result.stats["duration_total_hours"] = round(sum(durations) / 3600, 4)

    return result


def validate_metadata_file(
    json_path: str,
    required_fields: Optional[List[str]] = None,
    check_files: bool = True,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> ValidationResult:
    """Load a metadata JSON file and run all validation checks.

    Args:
        json_path: Path to the metadata JSON file (e.g. train.json).
        required_fields: Fields required in each utterance entry. Defaults to
            DEFAULT_REQUIRED_FIELDS.
        check_files: When True, verify that every 'Path' exists on disk.
        min_duration: Optional minimum clip duration in seconds for warnings.
        max_duration: Optional maximum clip duration in seconds for warnings.

    Returns:
        A merged ValidationResult aggregating all individual check results.
    """
    result = ValidationResult()

    if not os.path.exists(json_path):
        result.errors.append(f"Metadata file not found: '{json_path}'")
        result.is_valid = False
        return result

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            utterances = json.load(f)
    except json.JSONDecodeError as exc:
        result.errors.append(f"Failed to parse JSON from '{json_path}': {exc}")
        result.is_valid = False
        return result

    if not isinstance(utterances, list):
        result.errors.append(
            f"Expected a JSON list in '{json_path}', got {type(utterances).__name__}"
        )
        result.is_valid = False
        return result

    # Run individual validation checks and merge results
    integrity_result = validate_metadata_integrity(utterances, required_fields)
    audio_result = validate_audio_properties(utterances, min_duration, max_duration)

    _merge_result(result, integrity_result)
    _merge_result(result, audio_result)

    if check_files:
        file_result = validate_file_existence(utterances)
        _merge_result(result, file_result)

    # Consolidate summary stats
    total = len(utterances)
    result.stats["total_utterances"] = total
    result.stats["json_path"] = json_path

    if "duration_min" in audio_result.stats:
        result.stats["duration_min"] = audio_result.stats["duration_min"]
    if "duration_max" in audio_result.stats:
        result.stats["duration_max"] = audio_result.stats["duration_max"]
    if "duration_total_hours" in audio_result.stats:
        result.stats["duration_total_hours"] = audio_result.stats["duration_total_hours"]
    if check_files and "missing_files_count" in result.stats:
        pass  # already merged from file_result

    _print_summary(json_path, result)

    return result


def validate_dataset(
    processed_dir: str,
    dataset_name: str,
    splits: Optional[List[str]] = None,
    required_fields: Optional[List[str]] = None,
    check_files: bool = True,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> Dict[str, ValidationResult]:
    """Validate all metadata splits for a preprocessed dataset.

    Looks for <split>.json files inside `processed_dir/<dataset_name>/` and
    runs full validation on each split that exists.

    Args:
        processed_dir: Root directory containing preprocessed dataset folders.
        dataset_name: Name of the dataset subdirectory (e.g. 'libritts').
        splits: List of split names to check. Defaults to ['train', 'test', 'valid'].
        required_fields: Fields required per utterance. Defaults to DEFAULT_REQUIRED_FIELDS.
        check_files: When True, verify every 'Path' exists on disk.
        min_duration: Optional minimum clip duration in seconds for warnings.
        max_duration: Optional maximum clip duration in seconds for warnings.

    Returns:
        A dict mapping split name to its ValidationResult.  Splits whose JSON
        file does not exist are omitted from the result.
    """
    if splits is None:
        splits = DEFAULT_SPLITS

    dataset_dir = os.path.join(processed_dir, dataset_name)
    results: Dict[str, ValidationResult] = {}

    for split in splits:
        json_path = os.path.join(dataset_dir, f"{split}.json")
        if not os.path.exists(json_path):
            continue
        results[split] = validate_metadata_file(
            json_path,
            required_fields=required_fields,
            check_files=check_files,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_result(target: ValidationResult, source: ValidationResult) -> None:
    """Merge errors, warnings, and validity from *source* into *target* in-place."""
    target.errors.extend(source.errors)
    target.warnings.extend(source.warnings)
    if not source.is_valid:
        target.is_valid = False
    target.stats.update(source.stats)


def _print_summary(json_path: str, result: ValidationResult) -> None:
    """Print a concise human-readable summary of the validation result."""
    total = result.stats.get("total_utterances", "?")
    missing = result.stats.get("missing_files_count", 0)
    dur_min = result.stats.get("duration_min", "N/A")
    dur_max = result.stats.get("duration_max", "N/A")
    hours = result.stats.get("duration_total_hours", "N/A")

    status = "PASS" if result.is_valid else "FAIL"
    print(f"[{status}] {json_path}")
    print(f"  Utterances : {total}")
    print(f"  Errors     : {len(result.errors)}")
    print(f"  Warnings   : {len(result.warnings)}")
    print(f"  Missing files: {missing}")
    if dur_min != "N/A":
        print(f"  Duration   : {dur_min}s – {dur_max}s  ({hours} hours total)")
