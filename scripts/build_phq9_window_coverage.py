from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


MODALITY_COLUMNS = {
    "phonelock": [
        "phone_lock_episode_count",
        "night_phone_lock_episode_count",
        "first_phone_lock_start_time",
        "last_phone_lock_start_time",
        "phone_locked_duration_hours",
    ],
    "dark": [
        "dark_duration_hours",
        "main_dark_start_time",
        "dark_episode_count",
    ],
    "activity": [
        "activity_active_ratio",
        "activity_transition_count",
        "running_ratio",
    ],
    "conversation": [
        "conversation_minutes",
        "conversation_episode_count",
    ],
    "wifi_location": [
        "location_count",
        "primary_location_ratio",
        "location_transition_count",
        "location_entropy",
    ],
    "app_usage": [
        "app_usage_record_count",
        "app_usage_timestamp_count",
        "app_diversity",
        "night_app_usage_ratio",
    ],
}

OUTCOME_COLUMNS = [
    "uid",
    "type",
    "phq9_score",
    "phq9_class_5",
    "phq9_class_3",
    "phq9_binary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure 14-day feature coverage before each PHQ-9 outcome. "
            "Because StudentLife PHQ-9.csv has no response date, the default "
            "reference date is inferred from the participant feature range: "
            "pre = first feature date, post = last feature date."
        )
    )
    parser.add_argument(
        "--features",
        default=Path("outputs/studentlife_daily_features.csv"),
        type=Path,
        help="Participant-day feature table.",
    )
    parser.add_argument(
        "--outcomes",
        default=Path("outputs/phq9_outcomes.csv"),
        type=Path,
        help="PHQ-9 outcome table.",
    )
    parser.add_argument(
        "--reference-dates",
        default=None,
        type=Path,
        help=(
            "Optional CSV with uid,type,reference_date. Use this if exact "
            "PHQ-9 response dates are found later."
        ),
    )
    parser.add_argument(
        "--min-feature-date",
        default=None,
        help="Optional minimum feature date to keep before computing coverage.",
    )
    parser.add_argument(
        "--max-feature-date",
        default="2013-12-31",
        help=(
            "Optional maximum feature date to keep before computing coverage. "
            "The default removes the known isolated 2014 app_usage row from "
            "the StudentLife feature table."
        ),
    )
    parser.add_argument(
        "--output",
        default=Path("outputs/phq9_feature_window_coverage.csv"),
        type=Path,
        help="Output CSV with one row per PHQ-9 response.",
    )
    parser.add_argument(
        "--summary-output",
        default=Path("outputs/phq9_feature_window_coverage_summary.csv"),
        type=Path,
        help="Output CSV with coverage summary by PHQ-9 type and binary class.",
    )
    parser.add_argument(
        "--window-days",
        default=14,
        type=int,
        help="Number of days before PHQ-9 to inspect.",
    )
    parser.add_argument(
        "--min-available-days",
        default=13,
        type=int,
        help="Minimum number of available days required to mark coverage as sufficient.",
    )
    return parser.parse_args()


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def validate_inputs(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    reference_dates: pd.DataFrame | None,
) -> None:
    required_feature_columns = {"uid", "date"}
    missing_feature_columns = required_feature_columns - set(features.columns)
    if missing_feature_columns:
        raise ValueError(
            f"Missing required feature columns: {sorted(missing_feature_columns)}"
        )

    missing_outcome_columns = [column for column in OUTCOME_COLUMNS if column not in outcomes]
    if missing_outcome_columns:
        raise ValueError(f"Missing required outcome columns: {missing_outcome_columns}")

    missing_modality_columns = {
        modality: [column for column in columns if column not in features]
        for modality, columns in MODALITY_COLUMNS.items()
    }
    missing_modality_columns = {
        modality: columns
        for modality, columns in missing_modality_columns.items()
        if columns
    }
    if missing_modality_columns:
        raise ValueError(f"Missing modality feature columns: {missing_modality_columns}")

    if reference_dates is not None:
        required_reference_columns = {"uid", "type", "reference_date"}
        missing_reference_columns = required_reference_columns - set(reference_dates.columns)
        if missing_reference_columns:
            raise ValueError(
                "Missing required reference date columns: "
                f"{sorted(missing_reference_columns)}"
            )


def parse_optional_date(value: str | None, argument_name: str) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid {argument_name}: {value}")
    return parsed.date()


def prepare_features(
    features: pd.DataFrame,
    min_feature_date: date | None,
    max_feature_date: date | None,
) -> tuple[pd.DataFrame, int]:
    prepared = features.copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date

    invalid_dates = int(prepared["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found invalid feature dates: {invalid_dates}")

    duplicate_rows = int(prepared.duplicated(["uid", "date"]).sum())
    if duplicate_rows:
        raise ValueError(f"Found duplicated uid+date rows in feature table: {duplicate_rows}")

    before_filter = len(prepared)
    if min_feature_date is not None:
        prepared = prepared[prepared["date"] >= min_feature_date]
    if max_feature_date is not None:
        prepared = prepared[prepared["date"] <= max_feature_date]

    filtered_rows = before_filter - len(prepared)
    return prepared.reset_index(drop=True), filtered_rows


def prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    prepared = outcomes[OUTCOME_COLUMNS].copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["type"] = prepared["type"].astype("string").str.strip()

    duplicate_rows = int(prepared.duplicated(["uid", "type"]).sum())
    if duplicate_rows:
        raise ValueError(f"Found duplicated uid+type rows in outcome table: {duplicate_rows}")

    return prepared


def prepare_reference_dates(reference_dates: pd.DataFrame) -> pd.DataFrame:
    prepared = reference_dates.copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["type"] = prepared["type"].astype("string").str.strip()
    prepared["reference_date"] = pd.to_datetime(
        prepared["reference_date"], errors="coerce"
    ).dt.date

    invalid_dates = int(prepared["reference_date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found invalid custom reference dates: {invalid_dates}")

    duplicate_rows = int(prepared.duplicated(["uid", "type"]).sum())
    if duplicate_rows:
        raise ValueError(
            f"Found duplicated uid+type rows in reference date table: {duplicate_rows}"
        )

    return prepared[["uid", "type", "reference_date"]]


def infer_reference_dates(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    participant_ranges = (
        features.groupby("uid", as_index=False)
        .agg(
            participant_first_feature_date=("date", "min"),
            participant_last_feature_date=("date", "max"),
        )
    )
    inferred = outcomes[["uid", "type"]].merge(participant_ranges, on="uid", how="left")

    inferred["reference_date"] = pd.NaT
    inferred["reference_date_source"] = "missing"

    pre_mask = inferred["type"].eq("pre")
    post_mask = inferred["type"].eq("post")
    inferred.loc[pre_mask, "reference_date"] = inferred.loc[
        pre_mask, "participant_first_feature_date"
    ]
    inferred.loc[pre_mask, "reference_date_source"] = "inferred_first_feature_date"
    inferred.loc[post_mask, "reference_date"] = inferred.loc[
        post_mask, "participant_last_feature_date"
    ]
    inferred.loc[post_mask, "reference_date_source"] = "inferred_last_feature_date"

    return inferred


def merge_reference_dates(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    reference_dates: pd.DataFrame | None,
) -> pd.DataFrame:
    inferred = infer_reference_dates(features, outcomes)

    if reference_dates is None:
        return inferred

    custom = prepare_reference_dates(reference_dates)
    merged = inferred.drop(columns=["reference_date", "reference_date_source"]).merge(
        custom, on=["uid", "type"], how="left"
    )
    merged["reference_date_source"] = merged["reference_date"].notna().map(
        {True: "custom_reference_date", False: "missing_custom_reference_date"}
    )
    return merged


def count_days_with_any_value(window: pd.DataFrame, columns: list[str]) -> int:
    if window.empty:
        return 0
    return int(window[columns].notna().any(axis=1).sum())


def build_coverage(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    reference_dates: pd.DataFrame | None,
    window_days: int,
    min_available_days: int,
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("--window-days must be greater than zero")
    if min_available_days < 0:
        raise ValueError("--min-available-days must be zero or greater")
    if min_available_days > window_days:
        raise ValueError("--min-available-days cannot be greater than --window-days")

    references = merge_reference_dates(features, outcomes, reference_dates)
    rows: list[dict[str, object]] = []

    for outcome in outcomes.merge(references, on=["uid", "type"], how="left").to_dict("records"):
        uid = outcome["uid"]
        reference_date = outcome["reference_date"]

        if pd.isna(reference_date):
            row = {
                **{column: outcome[column] for column in OUTCOME_COLUMNS},
                "reference_date": None,
                "reference_date_source": outcome["reference_date_source"],
                "participant_first_feature_date": outcome["participant_first_feature_date"],
                "participant_last_feature_date": outcome["participant_last_feature_date"],
                "window_start": None,
                "window_end": None,
                "expected_days": window_days,
                "available_days": 0,
                "coverage_ratio": 0.0,
                "window_has_sufficient_coverage": False,
            }
            for modality in MODALITY_COLUMNS:
                row[f"{modality}_available_days"] = 0
            rows.append(row)
            continue

        reference_ts = pd.Timestamp(reference_date)
        window_start = (reference_ts - pd.Timedelta(days=window_days)).date()
        window_end = (reference_ts - pd.Timedelta(days=1)).date()

        participant_features = features[features["uid"].eq(uid)]
        window = participant_features[
            participant_features["date"].between(window_start, window_end)
        ]

        available_days = int(window["date"].nunique())
        row = {
            **{column: outcome[column] for column in OUTCOME_COLUMNS},
            "reference_date": reference_ts.date(),
            "reference_date_source": outcome["reference_date_source"],
            "participant_first_feature_date": outcome["participant_first_feature_date"],
            "participant_last_feature_date": outcome["participant_last_feature_date"],
            "window_start": window_start,
            "window_end": window_end,
            "expected_days": window_days,
            "available_days": available_days,
            "coverage_ratio": round(available_days / window_days, 4),
            "window_has_sufficient_coverage": available_days >= min_available_days,
        }
        for modality, columns in MODALITY_COLUMNS.items():
            row[f"{modality}_available_days"] = count_days_with_any_value(window, columns)
        rows.append(row)

    coverage = pd.DataFrame(rows)
    sort_order = {"pre": 0, "post": 1}
    coverage["_type_order"] = coverage["type"].map(sort_order).fillna(99)
    coverage = coverage.sort_values(["uid", "_type_order", "type"]).drop(
        columns=["_type_order"]
    )
    return coverage.reset_index(drop=True)


def build_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    group_specs = [
        ("overall", []),
        ("by_type", ["type"]),
        ("by_binary_class", ["phq9_binary"]),
        ("by_type_and_binary_class", ["type", "phq9_binary"]),
    ]

    for summary_level, columns in group_specs:
        grouped = [((), coverage)] if not columns else coverage.groupby(columns, dropna=False)
        for group_key, group in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            row = {
                "summary_level": summary_level,
                "rows": len(group),
                "valid_windows": int(group["window_has_sufficient_coverage"].sum()),
                "valid_window_ratio": round(
                    float(group["window_has_sufficient_coverage"].mean()), 4
                )
                if len(group)
                else 0.0,
                "mean_available_days": round(float(group["available_days"].mean()), 4)
                if len(group)
                else 0.0,
                "median_available_days": round(float(group["available_days"].median()), 4)
                if len(group)
                else 0.0,
            }
            for column, value in zip(columns, group_key):
                row[column] = value
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(f"Feature table not found: {args.features}")
    if not args.outcomes.exists():
        raise FileNotFoundError(f"PHQ-9 outcome table not found: {args.outcomes}")

    raw_features = read_csv_clean(args.features)
    raw_outcomes = read_csv_clean(args.outcomes)
    if args.reference_dates is not None and not args.reference_dates.exists():
        raise FileNotFoundError(f"Reference date table not found: {args.reference_dates}")
    raw_reference_dates = read_csv_clean(args.reference_dates) if args.reference_dates else None

    validate_inputs(raw_features, raw_outcomes, raw_reference_dates)
    min_feature_date = parse_optional_date(args.min_feature_date, "--min-feature-date")
    max_feature_date = parse_optional_date(args.max_feature_date, "--max-feature-date")
    features, filtered_feature_rows = prepare_features(
        raw_features,
        min_feature_date=min_feature_date,
        max_feature_date=max_feature_date,
    )
    outcomes = prepare_outcomes(raw_outcomes)

    coverage = build_coverage(
        features=features,
        outcomes=outcomes,
        reference_dates=raw_reference_dates,
        window_days=args.window_days,
        min_available_days=args.min_available_days,
    )
    summary = build_summary(coverage)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print("Saved:", args.output)
    print("Rows:", len(coverage))
    print("Window days:", args.window_days)
    print("Minimum available days:", args.min_available_days)
    print("Min feature date:", min_feature_date if min_feature_date else "none")
    print("Max feature date:", max_feature_date if max_feature_date else "none")
    print("Filtered feature rows:", filtered_feature_rows)
    print(
        "Valid windows:",
        int(coverage["window_has_sufficient_coverage"].sum()),
        "of",
        len(coverage),
    )
    print("Reference date sources:")
    print(coverage["reference_date_source"].value_counts(dropna=False).to_string())
    print("\nCoverage summary:")
    print(summary.to_string(index=False))
    print("\nSummary saved:", args.summary_output)


if __name__ == "__main__":
    main()
