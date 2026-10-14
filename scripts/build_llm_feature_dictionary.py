from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FEATURE_FIELDS = [
    "source_feature_name",
    "definition",
    "calculation",
    "unit",
    "interpretation",
    "caution",
]

FEATURE_COLUMNS = [
    "phone_lock_episode_count",
    "night_phone_lock_episode_count",
    "first_phone_lock_start_time",
    "last_phone_lock_start_time",
    "phone_locked_duration_hours",
    "dark_duration_hours",
    "main_dark_start_time",
    "dark_episode_count",
    "activity_active_ratio",
    "activity_transition_count",
    "running_ratio",
    "conversation_minutes",
    "conversation_episode_count",
    "location_count",
    "primary_location_ratio",
    "location_transition_count",
    "location_entropy",
    "app_usage_record_count",
    "app_usage_timestamp_count",
    "app_diversity",
    "night_app_usage_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the feature dictionary used in LLM prompts."
    )
    parser.add_argument(
        "--histories",
        default=Path("outputs/phq9_post_14day_histories.csv"),
        type=Path,
        help="History CSV used to validate source feature names.",
    )
    parser.add_argument(
        "--output",
        default=Path("outputs/llm_feature_dictionary.json"),
        type=Path,
        help="Output JSON feature dictionary.",
    )
    return parser.parse_args()


def feature(
    source_feature_name: str,
    definition: str,
    calculation: str,
    unit: str,
    interpretation: str,
    caution: str,
) -> dict[str, str]:
    return {
        "source_feature_name": source_feature_name,
        "definition": definition,
        "calculation": calculation,
        "unit": unit,
        "interpretation": interpretation,
        "caution": caution,
    }


def build_dictionary() -> dict[str, Any]:
    return {
        "metadata": {
            "dictionary_name": "StudentLife daily behavioral feature dictionary for LLM input",
            "version": "v1",
            "language": "en",
            "feature_granularity": "participant-day",
        },
        "modalities": {
            "phonelock": {
                "description": (
                    "Features derived from observed phone-locked episodes. These features "
                    "describe periods in which the phone was locked, not phone unlock events."
                ),
                "features": {
                    "phone_locked_episode_count": feature(
                        "phone_lock_episode_count",
                        "Number of phone-locked episodes observed during the day.",
                        "Counts phone lock records for the participant on that day.",
                        "count",
                        "Higher values indicate more observed phone-locked episodes during the day.",
                        (
                            "Do not interpret this feature as a count of phone unlocks. It "
                            "represents observed phone-locked episodes and is a proxy for periods "
                            "without direct phone use."
                        ),
                    ),
                    "phone_locked_night_episode_count": feature(
                        "night_phone_lock_episode_count",
                        "Number of phone-locked episodes that started during the night period.",
                        "Counts phone lock records with start time between 00:00 and 06:00.",
                        "count",
                        "Higher values indicate more observed phone-locked episodes during the night.",
                        (
                            "Do not interpret this feature as a count of phone unlocks. It only "
                            "represents observed phone-locked episodes that started between 00:00 "
                            "and 06:00."
                        ),
                    ),
                    "first_phone_locked_episode_start_hour": feature(
                        "first_phone_lock_start_time",
                        "Start hour of the first phone-locked episode observed during the day.",
                        (
                            "Takes the earliest phone lock start time for the day. The technical "
                            "table already stores this value as decimal hour of day; the LLM JSON "
                            "keeps it as hour of day rounded to two decimal places."
                        ),
                        "hour of day from 0 to 24",
                        "Lower values indicate an earlier first observed phone-locked episode.",
                        (
                            "Do not interpret this feature as a phone unlock event. It only "
                            "describes the start hour of the first observed phone-locked episode."
                        ),
                    ),
                    "last_phone_locked_episode_start_hour": feature(
                        "last_phone_lock_start_time",
                        "Start hour of the last phone-locked episode observed during the day.",
                        (
                            "Takes the latest phone lock start time for the day. The technical "
                            "table already stores this value as decimal hour of day; the LLM JSON "
                            "keeps it as hour of day rounded to two decimal places."
                        ),
                        "hour of day from 0 to 24",
                        "Higher values indicate a later last observed phone-locked episode.",
                        (
                            "Do not interpret this feature as a phone unlock event. It only "
                            "describes the start hour of the last observed phone-locked episode."
                        ),
                    ),
                    "phone_locked_total_duration_hours": feature(
                        "phone_locked_duration_hours",
                        "Total duration during the day in which the phone was locked.",
                        (
                            "Calculates the union duration of phone-locked intervals for the day "
                            "and converts seconds to hours."
                        ),
                        "hours",
                        "Higher values indicate more total observed time with the phone locked.",
                        (
                            "Do not interpret this feature as phone unlock duration. It is a proxy "
                            "for observed time with the phone locked and without direct phone use."
                        ),
                    ),
                },
            },
            "dark": {
                "description": (
                    "Features derived from periods in which the phone was detected in a dark "
                    "environment."
                ),
                "features": {
                    "dark_environment_total_duration_hours": feature(
                        "dark_duration_hours",
                        "Total duration during the day in which the phone was in a dark environment.",
                        (
                            "Calculates the union duration of dark-environment intervals for the "
                            "day and converts seconds to hours."
                        ),
                        "hours",
                        (
                            "Higher values indicate more observed time with the phone in a dark "
                            "environment."
                        ),
                        (
                            "Do not interpret this feature as sleep. It only indicates that the "
                            "phone was detected in a dark environment."
                        ),
                    ),
                    "longest_dark_episode_start_hour": feature(
                        "main_dark_start_time",
                        "Start hour of the longest dark-environment episode during the day.",
                        (
                            "Finds the dark-environment episode with the longest duration and uses "
                            "its start time. The technical table already stores this value as "
                            "decimal hour of day; the LLM JSON keeps it as hour of day rounded to "
                            "two decimal places."
                        ),
                        "hour of day from 0 to 24",
                        (
                            "This may indicate when the main observed dark-environment period began."
                        ),
                        (
                            "Do not interpret this feature as sleep onset. It only indicates the "
                            "start hour of the longest detected dark-environment episode."
                        ),
                    ),
                    "dark_environment_episode_count": feature(
                        "dark_episode_count",
                        "Number of dark-environment episodes observed during the day.",
                        "Counts dark-environment records for the participant on that day.",
                        "count",
                        (
                            "Higher values indicate more fragmented observed periods in a dark "
                            "environment."
                        ),
                        (
                            "Do not interpret this feature as sleep fragmentation. It only reflects "
                            "observed dark-environment episodes."
                        ),
                    ),
                },
            },
            "activity": {
                "description": (
                    "Features derived from smartphone-inferred physical activity records."
                ),
                "features": {
                    "activity_active_record_ratio": feature(
                        "activity_active_ratio",
                        (
                            "Proportion of valid smartphone-inferred activity records classified "
                            "as active during the day."
                        ),
                        (
                            "Records classified as walking or running are counted as active. "
                            "Records classified as stationary are counted as non-active. Records "
                            "classified as unknown are excluded from both the numerator and the "
                            "denominator."
                        ),
                        "ratio from 0 to 1",
                        (
                            "Higher values indicate that a larger proportion of valid activity "
                            "records were classified as walking or running."
                        ),
                        (
                            "Do not interpret this ratio as exact chronological time spent active. "
                            "Unknown activity records are excluded and must not be treated as "
                            "inactive."
                        ),
                    ),
                    "activity_state_transition_count": feature(
                        "activity_transition_count",
                        (
                            "Number of changes between consecutive valid inferred activity states "
                            "during the day."
                        ),
                        (
                            "Orders valid activity records by timestamp, excludes unknown records, "
                            "and counts changes between consecutive states: stationary, walking, "
                            "or running."
                        ),
                        "count",
                        (
                            "Higher values indicate more alternation between inferred activity "
                            "states during the day."
                        ),
                        (
                            "This is based on inferred records and sampling frequency. It is not a "
                            "direct measure of exact movement count."
                        ),
                    ),
                    "activity_running_record_ratio": feature(
                        "running_ratio",
                        (
                            "Proportion of valid smartphone-inferred activity records classified "
                            "as running during the day."
                        ),
                        (
                            "Records classified as running are divided by all valid records "
                            "classified as stationary, walking, or running. Unknown records are "
                            "excluded."
                        ),
                        "ratio from 0 to 1",
                        (
                            "Higher values indicate that a larger proportion of valid activity "
                            "records were classified as running."
                        ),
                        (
                            "Do not interpret this ratio as exact time spent running or as a direct "
                            "exercise measurement. Unknown activity records are excluded."
                        ),
                    ),
                },
            },
            "conversation": {
                "description": (
                    "Features derived from episodes in which the phone detected conversation "
                    "around the participant."
                ),
                "features": {
                    "conversation_detected_total_duration_minutes": feature(
                        "conversation_minutes",
                        "Total duration of detected conversation episodes during the day.",
                        (
                            "Sums end_timestamp minus start_timestamp for conversation episodes "
                            "and converts seconds to minutes."
                        ),
                        "minutes",
                        (
                            "Higher values indicate more observed time around detected "
                            "conversation."
                        ),
                        (
                            "Do not infer conversation content, interlocutor, emotional quality, "
                            "social support, or whether the conversation was socially meaningful."
                        ),
                    ),
                    "conversation_detected_episode_count": feature(
                        "conversation_episode_count",
                        "Number of detected conversation episodes during the day.",
                        "Counts conversation episode records for the participant on that day.",
                        "count",
                        "Higher values indicate more detected conversation episodes.",
                        (
                            "Do not infer conversation content, interlocutor, emotional quality, or "
                            "social support. It is a proxy for exposure to detected in-person "
                            "conversation."
                        ),
                    ),
                },
            },
            "wifi_location": {
                "description": (
                    "Features derived from Wi-Fi-inferred location labels. These labels are used "
                    "as proxies for spatial context and mobility, not as GPS coordinates."
                ),
                "features": {
                    "wifi_location_unique_label_count": feature(
                        "location_count",
                        "Number of distinct Wi-Fi-inferred location labels observed during the day.",
                        "Counts distinct non-empty location labels for the participant on that day.",
                        "count",
                        (
                            "Higher values indicate a greater variety of observed Wi-Fi-inferred "
                            "locations."
                        ),
                        (
                            "Do not interpret this feature as GPS, precise geographic distance, or "
                            "a travel trajectory. The local data include labels such as in[...] and "
                            "near[...]."
                        ),
                    ),
                    "wifi_primary_location_record_ratio": feature(
                        "primary_location_ratio",
                        (
                            "Proportion of Wi-Fi location records concentrated in the most "
                            "frequent location label during the day."
                        ),
                        (
                            "Counts records by location label, selects the most frequent label, "
                            "and divides its count by the total number of Wi-Fi location records "
                            "that day."
                        ),
                        "ratio from 0 to 1",
                        (
                            "Higher values indicate stronger concentration of observed records in "
                            "one primary location."
                        ),
                        (
                            "Do not interpret this feature as precise time spent at a GPS location. "
                            "It reflects concentration of Wi-Fi-inferred records."
                        ),
                    ),
                    "wifi_location_transition_count": feature(
                        "location_transition_count",
                        "Number of changes between consecutive Wi-Fi-inferred location labels.",
                        (
                            "Orders Wi-Fi location records by time and counts changes between "
                            "consecutive location labels."
                        ),
                        "count",
                        (
                            "Higher values indicate more observed changes between Wi-Fi-inferred "
                            "location labels."
                        ),
                        (
                            "Do not interpret this feature as a precise travel trajectory. It is a "
                            "proxy for changes in Wi-Fi-inferred spatial context."
                        ),
                    ),
                    "wifi_location_entropy": feature(
                        "location_entropy",
                        (
                            "Entropy of the distribution of Wi-Fi location records across location "
                            "labels during the day."
                        ),
                        (
                            "Counts records per location label, converts counts to proportions, "
                            "and calculates -sum(p_i * ln(p_i))."
                        ),
                        "entropy value using natural logarithm",
                        (
                            "Lower values indicate concentration in fewer locations; higher values "
                            "indicate records distributed across more locations."
                        ),
                        (
                            "Do not interpret this feature as exact geographic distance, GPS "
                            "mobility, or a travel trajectory. It describes diversity of "
                            "Wi-Fi-inferred location records."
                        ),
                    ),
                },
            },
            "app_usage": {
                "description": (
                    "Features derived from observed app/task records. These features describe "
                    "records and timestamps, not exact app usage duration or screen time."
                ),
                "features": {
                    "app_usage_task_record_count": feature(
                        "app_usage_record_count",
                        "Number of app/task records observed during the day.",
                        "Counts rows in the app usage file for the participant on that day.",
                        "count",
                        (
                            "Higher values indicate more observed app/task records during the day."
                        ),
                        (
                            "Do not interpret this feature as screen time, app usage duration, or "
                            "intentional active phone use. Multiple app/task records can share the "
                            "same timestamp."
                        ),
                    ),
                    "app_usage_unique_timestamp_count": feature(
                        "app_usage_timestamp_count",
                        "Number of distinct timestamps with app/task records during the day.",
                        (
                            "Counts unique timestamp values among app usage records for the "
                            "participant on that day."
                        ),
                        "count",
                        (
                            "Higher values indicate more distinct observed moments with app/task "
                            "records."
                        ),
                        (
                            "Do not interpret this feature as app usage duration or screen time. It "
                            "complements the task record count because one timestamp may contain "
                            "multiple app/task records."
                        ),
                    ),
                    "app_usage_unique_app_package_count": feature(
                        "app_diversity",
                        "Number of distinct app package names observed during the day.",
                        (
                            "Counts distinct values of RUNNING_TASKS_topActivity_mPackage for the "
                            "participant on that day."
                        ),
                        "count",
                        (
                            "Higher values indicate a greater diversity of observed app packages."
                        ),
                        (
                            "Do not infer app categories, app purpose, or time spent in each app "
                            "from this feature. It only measures diversity of observed package names."
                        ),
                    ),
                    "app_usage_night_timestamp_ratio": feature(
                        "night_app_usage_ratio",
                        (
                            "Proportion of distinct app usage timestamps that occurred during the "
                            "night period."
                        ),
                        (
                            "Counts unique app usage timestamps between 00:00 and 06:00 and "
                            "divides by the total number of unique app usage timestamps that day."
                        ),
                        "ratio from 0 to 1",
                        (
                            "Higher values indicate a larger proportion of observed app/task "
                            "timestamps during the night."
                        ),
                        (
                            "Do not interpret this feature as screen time or app usage duration. It "
                            "is based on unique timestamps, not rows."
                        ),
                    ),
                },
            },
        },
    }


def iter_features(dictionary: dict[str, Any]) -> list[tuple[str, str, dict[str, str]]]:
    rows: list[tuple[str, str, dict[str, str]]] = []
    modalities = dictionary.get("modalities", {})
    for modality_name, modality in modalities.items():
        for feature_name, spec in modality.get("features", {}).items():
            rows.append((modality_name, feature_name, spec))
    return rows


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def validate_dictionary(dictionary: dict[str, Any], histories_path: Path) -> list[str]:
    errors: list[str] = []
    feature_rows = iter_features(dictionary)

    if len(feature_rows) != len(FEATURE_COLUMNS):
        errors.append(
            f"Expected {len(FEATURE_COLUMNS)} features, found {len(feature_rows)}."
        )

    source_names = [spec.get("source_feature_name") for _, _, spec in feature_rows]
    json_names = [feature_name for _, feature_name, _ in feature_rows]

    missing_sources = sorted(set(FEATURE_COLUMNS) - set(source_names))
    extra_sources = sorted(set(source_names) - set(FEATURE_COLUMNS))
    duplicate_sources = sorted({name for name in source_names if source_names.count(name) > 1})
    duplicate_json_names = sorted({name for name in json_names if json_names.count(name) > 1})

    if missing_sources:
        errors.append(f"Missing source feature definitions: {missing_sources}")
    if extra_sources:
        errors.append(f"Unexpected source feature definitions: {extra_sources}")
    if duplicate_sources:
        errors.append(f"Duplicated source feature names: {duplicate_sources}")
    if duplicate_json_names:
        errors.append(f"Duplicated JSON feature names: {duplicate_json_names}")

    for modality_name, feature_name, spec in feature_rows:
        missing_fields = [
            field
            for field in REQUIRED_FEATURE_FIELDS
            if not str(spec.get(field, "")).strip()
        ]
        if missing_fields:
            errors.append(
                f"{modality_name}.{feature_name}: missing fields {missing_fields}"
            )

    history_header = read_csv_header(histories_path)
    if history_header:
        missing_in_history = sorted(set(FEATURE_COLUMNS) - set(history_header))
        if missing_in_history:
            errors.append(
                f"Source features missing from {histories_path}: {missing_in_history}"
            )

    return errors


def main() -> None:
    args = parse_args()
    dictionary = build_dictionary()
    errors = validate_dictionary(dictionary, args.histories)

    if errors:
        for error in errors:
            print(f"ERROR {error}")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    feature_count = len(iter_features(dictionary))
    modality_count = len(dictionary["modalities"])
    print(f"Saved feature dictionary: {args.output}")
    print(f"Modalities: {modality_count}")
    print(f"Features: {feature_count}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
