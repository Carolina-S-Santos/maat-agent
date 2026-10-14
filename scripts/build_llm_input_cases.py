from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_MODALITIES = [
    "phonelock",
    "dark",
    "activity",
    "conversation",
    "wifi_location",
    "app_usage",
]

HISTORY_METADATA_COLUMNS = [
    "case_id",
    "uid",
    "type",
    "reference_date",
    "window_start",
    "window_end",
    "day_index",
    "date",
]

PHQ9_ITEM_TEXT_COLUMNS = [f"phq9_item_{index}_text" for index in range(1, 10)]
PHQ9_ITEM_SCORE_COLUMNS = [f"phq9_item_{index}_score" for index in range(1, 10)]

LABEL_COLUMNS = [
    "case_id",
    *PHQ9_ITEM_TEXT_COLUMNS,
    *PHQ9_ITEM_SCORE_COLUMNS,
    "phq9_score",
    "phq9_class_5",
    "phq9_class_3",
    "phq9_binary",
]

FORBIDDEN_INPUT_KEYS = {
    "uid",
    "type",
    "date",
    "reference_date",
    "window_start",
    "window_end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build JSONL input cases for the qualitative PHQ-9 and behavioral "
            "analysis experiment."
        )
    )
    parser.add_argument(
        "--histories",
        default=Path("outputs/phq9_post_14day_histories.csv"),
        type=Path,
        help="CSV with one row per case-day and no PHQ-9 labels.",
    )
    parser.add_argument(
        "--labels",
        default=Path("outputs/phq9_post_14day_labels.csv"),
        type=Path,
        help="CSV with PHQ-9 labels for each original case.",
    )
    parser.add_argument(
        "--feature-dictionary",
        default=Path("outputs/llm_feature_dictionary.json"),
        type=Path,
        help="JSON feature dictionary defining modality and feature order.",
    )
    parser.add_argument(
        "--cases-output",
        default=Path("outputs/llm_input_cases.jsonl"),
        type=Path,
        help="Output JSONL with one LLM input case per line.",
    )
    parser.add_argument(
        "--case-map-output",
        default=Path("outputs/llm_case_id_map.csv"),
        type=Path,
        help="Output CSV mapping neutral LLM case IDs to original case IDs.",
    )
    parser.add_argument(
        "--case-labels-output",
        default=Path("outputs/llm_case_labels.csv"),
        type=Path,
        help="Output CSV with labels keyed by neutral LLM case ID.",
    )
    parser.add_argument(
        "--round-digits",
        default=2,
        type=int,
        help="Number of decimal places for non-count numeric features.",
    )
    return parser.parse_args()


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_feature_specs(
    feature_dictionary: dict[str, Any],
) -> list[tuple[str, str, str, str]]:
    modalities = feature_dictionary.get("modalities")
    if not isinstance(modalities, dict):
        raise ValueError("Feature dictionary must contain a 'modalities' object.")

    modality_order = list(modalities.keys())
    if modality_order != EXPECTED_MODALITIES:
        raise ValueError(
            "Unexpected modality order in feature dictionary. "
            f"Expected {EXPECTED_MODALITIES}, found {modality_order}."
        )

    specs: list[tuple[str, str, str, str]] = []
    for modality_name in EXPECTED_MODALITIES:
        features = modalities[modality_name].get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError(f"Missing features for modality: {modality_name}")

        for json_feature_name, feature_spec in features.items():
            source_feature_name = str(feature_spec.get("source_feature_name", "")).strip()
            unit = str(feature_spec.get("unit", "")).strip()
            if not source_feature_name:
                raise ValueError(
                    f"{modality_name}.{json_feature_name} is missing source_feature_name."
                )
            if not unit:
                raise ValueError(f"{modality_name}.{json_feature_name} is missing unit.")
            specs.append((modality_name, json_feature_name, source_feature_name, unit))

    return specs


def validate_inputs(
    histories: pd.DataFrame,
    labels: pd.DataFrame,
    feature_specs: list[tuple[str, str, str, str]],
) -> None:
    required_history_columns = set(HISTORY_METADATA_COLUMNS) | {
        source_name for _, _, source_name, _ in feature_specs
    }
    missing_history_columns = sorted(required_history_columns - set(histories.columns))
    if missing_history_columns:
        raise ValueError(f"Missing history columns: {missing_history_columns}")

    missing_label_columns = sorted(set(LABEL_COLUMNS) - set(labels.columns))
    if missing_label_columns:
        raise ValueError(f"Missing label columns: {missing_label_columns}")

    duplicated_history_rows = int(histories.duplicated(["case_id", "day_index"]).sum())
    if duplicated_history_rows:
        raise ValueError(f"Duplicated case_id+day_index rows: {duplicated_history_rows}")

    duplicated_labels = int(labels.duplicated("case_id").sum())
    if duplicated_labels:
        raise ValueError(f"Duplicated label case_id rows: {duplicated_labels}")

    history_cases = set(histories["case_id"].astype(str))
    label_cases = set(labels["case_id"].astype(str))
    if history_cases != label_cases:
        missing_labels = sorted(history_cases - label_cases)
        missing_histories = sorted(label_cases - history_cases)
        raise ValueError(
            "History and label case IDs do not match. "
            f"Missing labels: {missing_labels}; missing histories: {missing_histories}"
        )

    for case_id, group in histories.groupby("case_id", sort=False):
        day_indices = sorted(pd.to_numeric(group["day_index"], errors="coerce").astype(int))
        if day_indices != list(range(1, 15)):
            raise ValueError(f"{case_id}: expected day_index 1..14, found {day_indices}")


def make_case_id(index: int) -> str:
    return f"case_{index:03d}"


def build_case_map(histories: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        histories[
            [
                "case_id",
                "uid",
                "type",
                "reference_date",
                "window_start",
                "window_end",
            ]
        ]
        .drop_duplicates("case_id", keep="first")
        .reset_index(drop=True)
    )
    metadata.insert(0, "llm_case_id", [make_case_id(i + 1) for i in range(len(metadata))])
    metadata = metadata.rename(columns={"case_id": "original_case_id"})
    return metadata[
        [
            "llm_case_id",
            "original_case_id",
            "uid",
            "type",
            "reference_date",
            "window_start",
            "window_end",
        ]
    ]


def build_case_labels(labels: pd.DataFrame, case_map: pd.DataFrame) -> pd.DataFrame:
    prepared_labels = labels[LABEL_COLUMNS].rename(
        columns={"case_id": "original_case_id"}
    )
    out = case_map[["llm_case_id", "original_case_id"]].merge(
        prepared_labels,
        on="original_case_id",
        how="left",
        validate="one_to_one",
    )

    required_phq9_columns = [
        *PHQ9_ITEM_TEXT_COLUMNS,
        *PHQ9_ITEM_SCORE_COLUMNS,
        "phq9_score",
        "phq9_class_5",
        "phq9_class_3",
        "phq9_binary",
    ]
    if out[required_phq9_columns].isna().any().any():
        raise ValueError("Found missing PHQ-9 labels after mapping LLM case IDs.")

    return out[["llm_case_id", "original_case_id", *LABEL_COLUMNS[1:]]]


def normalize_value(value: object, unit: str, round_digits: int) -> int | float | None:
    if pd.isna(value):
        return None

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None

    numeric_value = float(numeric)
    if unit == "count" and abs(numeric_value - round(numeric_value)) < 1e-9:
        return int(round(numeric_value))

    return round(numeric_value, round_digits)


def build_input_cases(
    histories: pd.DataFrame,
    case_map: pd.DataFrame,
    labels: pd.DataFrame,
    feature_specs: list[tuple[str, str, str, str]],
    round_digits: int,
) -> list[dict[str, Any]]:
    by_original_case = {
        str(case_id): group.sort_values("day_index").copy()
        for case_id, group in histories.groupby("case_id", sort=False)
    }

    specs_by_modality: dict[str, list[tuple[str, str, str]]] = {
        modality: [] for modality in EXPECTED_MODALITIES
    }
    for modality, json_feature_name, source_feature_name, unit in feature_specs:
        specs_by_modality[modality].append((json_feature_name, source_feature_name, unit))

    labels_by_case_id = {
        str(row["case_id"]): row for row in labels[LABEL_COLUMNS].to_dict("records")
    }

    cases: list[dict[str, Any]] = []
    for row in case_map.to_dict("records"):
        original_case_id = str(row["original_case_id"])
        llm_case_id = str(row["llm_case_id"])
        history = by_original_case[original_case_id]
        label = labels_by_case_id[original_case_id]

        phq9 = {
            "item_responses": {
                f"item_{index}": str(label[f"phq9_item_{index}_text"])
                for index in range(1, 10)
            },
            "item_scores": {
                f"item_{index}": int(pd.to_numeric(label[f"phq9_item_{index}_score"]))
                for index in range(1, 10)
            },
            "total_score": int(pd.to_numeric(label["phq9_score"])),
            "severity_class_5": str(label["phq9_class_5"]),
        }

        daily_history: list[dict[str, Any]] = []
        for history_row in history.to_dict("records"):
            day: dict[str, Any] = {"day_index": int(history_row["day_index"])}

            for modality in EXPECTED_MODALITIES:
                modality_features: dict[str, int | float | None] = {}
                for json_feature_name, source_feature_name, unit in specs_by_modality[
                    modality
                ]:
                    modality_features[json_feature_name] = normalize_value(
                        history_row[source_feature_name],
                        unit,
                        round_digits,
                    )
                day[modality] = modality_features

            daily_history.append(day)

        cases.append(
            {
                "case_id": llm_case_id,
                "phq9": phq9,
                "daily_history": daily_history,
            }
        )

    return cases


def find_forbidden_keys(obj: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}"
            if key in FORBIDDEN_INPUT_KEYS:
                findings.append(current_path)
            findings.extend(find_forbidden_keys(value, current_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            findings.extend(find_forbidden_keys(value, f"{path}[{index}]"))
    return findings


def validate_output_cases(
    cases: list[dict[str, Any]],
    feature_dictionary: dict[str, Any],
    case_map: pd.DataFrame,
    case_labels: pd.DataFrame,
) -> None:
    if len(cases) != len(case_map):
        raise ValueError(f"Expected {len(case_map)} cases, found {len(cases)}.")

    expected_case_ids = case_map["llm_case_id"].astype(str).tolist()
    actual_case_ids = [str(case["case_id"]) for case in cases]
    if actual_case_ids != expected_case_ids:
        raise ValueError("Case IDs in JSONL do not match llm_case_id_map order.")

    if case_labels["llm_case_id"].astype(str).tolist() != expected_case_ids:
        raise ValueError("Case labels do not match llm_case_id_map order.")

    dictionary_modalities = feature_dictionary["modalities"]
    expected_features_by_modality = {
        modality: list(dictionary_modalities[modality]["features"].keys())
        for modality in EXPECTED_MODALITIES
    }

    for case in cases:
        if list(case.keys()) != ["case_id", "phq9", "daily_history"]:
            raise ValueError(f"{case['case_id']}: unexpected top-level keys.")

        forbidden_keys = find_forbidden_keys(case)
        if forbidden_keys:
            raise ValueError(
                f"{case['case_id']}: forbidden keys found in JSON input: {forbidden_keys}"
            )

        phq9 = case["phq9"]
        if list(phq9.keys()) != [
            "item_responses",
            "item_scores",
            "total_score",
            "severity_class_5",
        ]:
            raise ValueError(f"{case['case_id']}: unexpected phq9 keys.")

        expected_item_keys = [f"item_{index}" for index in range(1, 10)]
        if list(phq9["item_responses"].keys()) != expected_item_keys:
            raise ValueError(f"{case['case_id']}: unexpected item_responses keys.")
        if list(phq9["item_scores"].keys()) != expected_item_keys:
            raise ValueError(f"{case['case_id']}: unexpected item_scores keys.")

        daily_history = case["daily_history"]
        if len(daily_history) != 14:
            raise ValueError(
                f"{case['case_id']}: expected 14 daily history rows, "
                f"found {len(daily_history)}."
            )

        for expected_day_index, day in enumerate(daily_history, start=1):
            expected_day_keys = ["day_index", *EXPECTED_MODALITIES]
            if list(day.keys()) != expected_day_keys:
                raise ValueError(
                    f"{case['case_id']} day {expected_day_index}: unexpected day keys."
                )

            if day["day_index"] != expected_day_index:
                raise ValueError(
                    f"{case['case_id']}: expected day_index={expected_day_index}, "
                    f"found {day['day_index']}."
                )

            for modality in EXPECTED_MODALITIES:
                actual_features = list(day[modality].keys())
                expected_features = expected_features_by_modality[modality]
                if actual_features != expected_features:
                    raise ValueError(
                        f"{case['case_id']} day {expected_day_index} {modality}: "
                        "feature order/content does not match dictionary."
                    )


def write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    histories = read_csv_clean(args.histories)
    labels = read_csv_clean(args.labels)
    feature_dictionary = read_json(args.feature_dictionary)
    feature_specs = extract_feature_specs(feature_dictionary)

    validate_inputs(histories, labels, feature_specs)

    case_map = build_case_map(histories)
    case_labels = build_case_labels(labels, case_map)
    cases = build_input_cases(
        histories,
        case_map,
        labels,
        feature_specs,
        args.round_digits,
    )

    validate_output_cases(cases, feature_dictionary, case_map, case_labels)

    write_jsonl(args.cases_output, cases)
    args.case_map_output.parent.mkdir(parents=True, exist_ok=True)
    case_map.to_csv(args.case_map_output, index=False, encoding="utf-8")
    case_labels.to_csv(args.case_labels_output, index=False, encoding="utf-8")

    null_count = sum(
        1
        for case in cases
        for day in case["daily_history"]
        for modality in EXPECTED_MODALITIES
        for value in day[modality].values()
        if value is None
    )
    print(f"Saved input cases: {args.cases_output}")
    print(f"Saved case map: {args.case_map_output}")
    print(f"Saved case labels: {args.case_labels_output}")
    print(f"Cases: {len(cases)}")
    print(f"Days per case: 14")
    print(f"Modalities: {len(EXPECTED_MODALITIES)}")
    print(f"Features per day: {len(feature_specs)}")
    print(f"Null feature values: {null_count}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
