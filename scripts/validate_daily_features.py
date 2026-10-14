from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


UID_RE = re.compile(r"u\d+")
TOLERANCE = 1e-5

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

MODALITY_ANCHORS = {
    "phonelock": "phone_locked_duration_hours",
    "dark": "dark_duration_hours",
    "activity": "activity_active_ratio",
    "conversation": "conversation_minutes",
    "wifi_location": "location_count",
    "app_usage": "app_usage_record_count",
}

COUNT_COLUMNS = [
    "phone_lock_episode_count",
    "night_phone_lock_episode_count",
    "dark_episode_count",
    "activity_transition_count",
    "conversation_episode_count",
    "location_count",
    "location_transition_count",
    "app_usage_record_count",
    "app_usage_timestamp_count",
    "app_diversity",
]

RATIO_COLUMNS = [
    "activity_active_ratio",
    "running_ratio",
    "primary_location_ratio",
    "night_app_usage_ratio",
]

TIME_COLUMNS = [
    "first_phone_lock_start_time",
    "last_phone_lock_start_time",
    "main_dark_start_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate outputs/studentlife_daily_features.csv against raw data."
    )
    parser.add_argument("--dataset-root", default=Path("dataset"), type=Path)
    parser.add_argument(
        "--features",
        default=Path("outputs/studentlife_daily_features.csv"),
        type=Path,
    )
    parser.add_argument(
        "--report",
        default=Path("outputs/daily_features_validation_report.txt"),
        type=Path,
    )
    parser.add_argument(
        "--failures",
        default=Path("outputs/daily_features_validation_failures.csv"),
        type=Path,
    )
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--sample-size", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def extract_uid(path: Path) -> str:
    match = UID_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not infer uid from filename: {path.name}")
    return match.group(0)


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        index_col=False,
        on_bad_lines="skip",
        skip_blank_lines=True,
    )
    df.columns = df.columns.astype(str).str.strip()
    return df


def unix_to_local_datetime(values: pd.Series, timezone: str) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(values, errors="coerce"),
        unit="s",
        utc=True,
        errors="coerce",
    ).dt.tz_convert(timezone)


def decimal_hour(datetimes: pd.Series) -> pd.Series:
    return (
        datetimes.dt.hour
        + datetimes.dt.minute / 60
        + datetimes.dt.second / 3600
        + datetimes.dt.microsecond / 3_600_000_000
    )


def is_night(datetimes: pd.Series) -> pd.Series:
    return (datetimes.dt.hour >= 0) & (datetimes.dt.hour < 6)


def union_duration_for_day(
    starts: pd.Series,
    ends: pd.Series,
    target_date: str,
    timezone: str,
    unit: str,
) -> float:
    day_start = pd.Timestamp(target_date).tz_localize(timezone)
    day_end = day_start + pd.Timedelta(days=1)
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for start, end in zip(starts, ends):
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        segment_start = max(start, day_start)
        segment_end = min(end, day_end)
        if segment_end > segment_start:
            segments.append((segment_start, segment_end))

    if not segments:
        return np.nan

    segments.sort(key=lambda item: item[0])
    total_seconds = 0.0
    current_start, current_end = segments[0]
    for start, end in segments[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total_seconds += (current_end - current_start).total_seconds()
            current_start, current_end = start, end
    total_seconds += (current_end - current_start).total_seconds()

    if unit == "hours":
        return total_seconds / 3600
    if unit == "minutes":
        return total_seconds / 60
    raise ValueError(f"Unsupported unit: {unit}")


def entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=True)
    if counts.empty:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def transition_count(values: pd.Series) -> int:
    if values.empty:
        return 0
    return max(int((values != values.shift()).sum()) - 1, 0)


def assert_check(
    failures: list[dict[str, object]],
    lines: list[str],
    name: str,
    ok: bool,
    details: str,
    severity: str = "ERROR",
) -> None:
    status = "PASS" if ok else severity
    lines.append(f"[{status}] {name}: {details}")
    if not ok:
        failures.append(
            {
                "severity": severity,
                "check": name,
                "details": details,
            }
        )


def compare_values(expected: float, actual: float) -> bool:
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) != pd.isna(actual):
        return False
    return math.isclose(float(expected), float(actual), rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def compare_expected_to_actual(
    failures: list[dict[str, object]],
    lines: list[str],
    modality: str,
    uid: str,
    date: str,
    expected: dict[str, float],
    actual_row: pd.Series,
) -> None:
    for column, expected_value in expected.items():
        actual_value = actual_row[column]
        ok = compare_values(expected_value, actual_value)
        details = (
            f"{uid} {date} {column}: expected={expected_value}, actual={actual_value}"
        )
        assert_check(
            failures,
            lines,
            f"raw_sample_{modality}_{column}",
            ok,
            details,
        )


def structural_checks(
    features: pd.DataFrame,
    dataset_root: Path,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    missing = [col for col in ["uid", "date", *FEATURE_COLUMNS] if col not in features]
    assert_check(
        failures,
        lines,
        "required_columns",
        not missing,
        f"missing={missing}",
    )

    duplicates = int(features.duplicated(["uid", "date"]).sum())
    assert_check(
        failures,
        lines,
        "unique_uid_date",
        duplicates == 0,
        f"duplicate uid+date rows={duplicates}",
    )

    parsed_dates = pd.to_datetime(features["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    assert_check(
        failures,
        lines,
        "valid_dates",
        invalid_dates == 0,
        f"invalid dates={invalid_dates}",
    )

    uid_is_valid = features["uid"].astype(str).str.match(r"^u\d+$")
    invalid_uids = int((~uid_is_valid).sum())
    assert_check(
        failures,
        lines,
        "uid_format",
        invalid_uids == 0,
        f"invalid uid values={invalid_uids}",
    )

    raw_uids = set()
    for folder, pattern in [
        ("sensing/phonelock", "phonelock_u*.csv"),
        ("sensing/dark", "dark_u*.csv"),
        ("sensing/activity", "activity_u*.csv"),
        ("sensing/conversation", "conversation_u*.csv"),
        ("sensing/wifi_location", "wifi_location_u*.csv"),
        ("app_usage", "running_app_u*.csv"),
    ]:
        for path in (dataset_root / folder).glob(pattern):
            raw_uids.add(extract_uid(path))
    table_uids = set(features["uid"].unique())
    assert_check(
        failures,
        lines,
        "uid_coverage",
        table_uids == raw_uids,
        f"table={len(table_uids)} raw={len(raw_uids)} missing={sorted(raw_uids - table_uids)} extra={sorted(table_uids - raw_uids)}",
    )

    numeric_failures = []
    for column in FEATURE_COLUMNS:
        converted = pd.to_numeric(features[column], errors="coerce")
        bad = features[column].notna() & converted.isna()
        if bad.any():
            numeric_failures.append(column)
    assert_check(
        failures,
        lines,
        "feature_columns_numeric",
        not numeric_failures,
        f"non numeric columns={numeric_failures}",
    )


def range_checks(
    features: pd.DataFrame,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    for column in RATIO_COLUMNS:
        values = features[column].dropna()
        invalid = int(((values < -TOLERANCE) | (values > 1 + TOLERANCE)).sum())
        assert_check(
            failures,
            lines,
            f"range_{column}",
            invalid == 0,
            f"invalid ratio values={invalid}",
        )

    for column in TIME_COLUMNS:
        values = features[column].dropna()
        invalid = int(((values < -TOLERANCE) | (values >= 24 + TOLERANCE)).sum())
        assert_check(
            failures,
            lines,
            f"range_{column}",
            invalid == 0,
            f"invalid decimal-hour values={invalid}",
        )

    duration_limits = {
        "phone_locked_duration_hours": 24,
        "dark_duration_hours": 24,
        "conversation_minutes": 1440,
    }
    for column, limit in duration_limits.items():
        values = features[column].dropna()
        invalid = int(((values < -TOLERANCE) | (values > limit + TOLERANCE)).sum())
        assert_check(
            failures,
            lines,
            f"range_{column}",
            invalid == 0,
            f"invalid duration values={invalid}, limit={limit}",
        )

    for column in COUNT_COLUMNS:
        values = features[column].dropna()
        negative = int((values < -TOLERANCE).sum())
        non_integer = int((np.abs(values - np.round(values)) > TOLERANCE).sum())
        assert_check(
            failures,
            lines,
            f"count_{column}",
            negative == 0 and non_integer == 0,
            f"negative={negative}, non_integer={non_integer}",
        )

    location_rows = features.dropna(subset=["location_count", "location_entropy"])
    if not location_rows.empty:
        entropy_invalid = int(
            (
                (location_rows["location_entropy"] < -TOLERANCE)
                | (
                    location_rows["location_entropy"]
                    > np.log(location_rows["location_count"]) + TOLERANCE
                )
            ).sum()
        )
        assert_check(
            failures,
            lines,
            "location_entropy_bounds",
            entropy_invalid == 0,
            f"invalid entropy values={entropy_invalid}",
        )

    dates = pd.to_datetime(features["date"], errors="coerce")
    after_2013 = int((dates.dt.year > 2013).sum())
    assert_check(
        failures,
        lines,
        "date_outliers_after_2013",
        after_2013 == 0,
        f"rows after 2013={after_2013}. Treat as warning because raw app_usage contains this timestamp.",
        severity="WARN",
    )


def calc_phonelock(dataset_root: Path, uid: str, date: str, timezone: str) -> dict[str, float]:
    path = dataset_root / "sensing" / "phonelock" / f"phonelock_{uid}.csv"
    df = read_csv_clean(path)
    start_dt = unix_to_local_datetime(df["start"], timezone)
    end_dt = unix_to_local_datetime(df["end"], timezone)
    valid = pd.DataFrame({"start_dt": start_dt, "end_dt": end_dt}).dropna()
    valid = valid[valid["end_dt"] > valid["start_dt"]].copy()
    valid["date"] = valid["start_dt"].dt.date.astype(str)
    day = valid[valid["date"] == date]

    result = {
        "phone_lock_episode_count": np.nan,
        "night_phone_lock_episode_count": np.nan,
        "first_phone_lock_start_time": np.nan,
        "last_phone_lock_start_time": np.nan,
        "phone_locked_duration_hours": union_duration_for_day(
            valid["start_dt"], valid["end_dt"], date, timezone, "hours"
        ),
    }
    if not day.empty:
        start_hours = decimal_hour(day["start_dt"])
        result.update(
            {
                "phone_lock_episode_count": len(day),
                "night_phone_lock_episode_count": int(is_night(day["start_dt"]).sum()),
                "first_phone_lock_start_time": float(start_hours.min()),
                "last_phone_lock_start_time": float(start_hours.max()),
            }
        )
    return result


def calc_dark(dataset_root: Path, uid: str, date: str, timezone: str) -> dict[str, float]:
    path = dataset_root / "sensing" / "dark" / f"dark_{uid}.csv"
    df = read_csv_clean(path)
    start_dt = unix_to_local_datetime(df["start"], timezone)
    end_dt = unix_to_local_datetime(df["end"], timezone)
    valid = pd.DataFrame({"start_dt": start_dt, "end_dt": end_dt}).dropna()
    valid = valid[valid["end_dt"] > valid["start_dt"]].copy()
    valid["date"] = valid["start_dt"].dt.date.astype(str)
    valid["duration_seconds"] = (valid["end_dt"] - valid["start_dt"]).dt.total_seconds()
    day = valid[valid["date"] == date]

    result = {
        "dark_duration_hours": union_duration_for_day(
            valid["start_dt"], valid["end_dt"], date, timezone, "hours"
        ),
        "main_dark_start_time": np.nan,
        "dark_episode_count": np.nan,
    }
    if not day.empty:
        longest = day.loc[day["duration_seconds"].idxmax()]
        result.update(
            {
                "main_dark_start_time": float(decimal_hour(pd.Series([longest["start_dt"]])).iloc[0]),
                "dark_episode_count": len(day),
            }
        )
    return result


def calc_activity(dataset_root: Path, uid: str, date: str, timezone: str) -> dict[str, float]:
    path = dataset_root / "sensing" / "activity" / f"activity_{uid}.csv"
    df = read_csv_clean(path)
    timestamp_dt = unix_to_local_datetime(df["timestamp"], timezone)
    activity = pd.to_numeric(df["activity inference"], errors="coerce")
    valid = pd.DataFrame({"timestamp_dt": timestamp_dt, "activity": activity}).dropna()
    valid = valid[valid["activity"].isin([0, 1, 2])].copy()
    valid["date"] = valid["timestamp_dt"].dt.date.astype(str)
    day = valid[valid["date"] == date].sort_values("timestamp_dt")
    if day.empty:
        return {
            "activity_active_ratio": np.nan,
            "activity_transition_count": np.nan,
            "running_ratio": np.nan,
        }
    return {
        "activity_active_ratio": float(day["activity"].isin([1, 2]).mean()),
        "activity_transition_count": transition_count(day["activity"]),
        "running_ratio": float((day["activity"] == 2).mean()),
    }


def calc_conversation(
    dataset_root: Path, uid: str, date: str, timezone: str
) -> dict[str, float]:
    path = dataset_root / "sensing" / "conversation" / f"conversation_{uid}.csv"
    df = read_csv_clean(path)
    start_dt = unix_to_local_datetime(df["start_timestamp"], timezone)
    end_dt = unix_to_local_datetime(df["end_timestamp"], timezone)
    valid = pd.DataFrame({"start_dt": start_dt, "end_dt": end_dt}).dropna()
    valid = valid[valid["end_dt"] > valid["start_dt"]].copy()
    valid["date"] = valid["start_dt"].dt.date.astype(str)
    day = valid[valid["date"] == date]
    return {
        "conversation_minutes": union_duration_for_day(
            valid["start_dt"], valid["end_dt"], date, timezone, "minutes"
        ),
        "conversation_episode_count": len(day) if not day.empty else np.nan,
    }


def calc_wifi_location(
    dataset_root: Path, uid: str, date: str, timezone: str
) -> dict[str, float]:
    path = dataset_root / "sensing" / "wifi_location" / f"wifi_location_{uid}.csv"
    df = read_csv_clean(path)
    time_dt = unix_to_local_datetime(df["time"], timezone)
    location = df["location"].astype("string").str.strip()
    valid = pd.DataFrame({"time_dt": time_dt, "location": location}).dropna()
    valid = valid[valid["location"] != ""].copy()
    valid["date"] = valid["time_dt"].dt.date.astype(str)
    day = valid[valid["date"] == date].sort_values("time_dt")
    if day.empty:
        return {
            "location_count": np.nan,
            "primary_location_ratio": np.nan,
            "location_transition_count": np.nan,
            "location_entropy": np.nan,
        }
    return {
        "location_count": int(day["location"].nunique()),
        "primary_location_ratio": float(day["location"].value_counts().iloc[0] / len(day)),
        "location_transition_count": transition_count(day["location"]),
        "location_entropy": entropy(day["location"]),
    }


def calc_app_usage(dataset_root: Path, uid: str, date: str, timezone: str) -> dict[str, float]:
    path = dataset_root / "app_usage" / f"running_app_{uid}.csv"
    df = read_csv_clean(path)
    timestamp_dt = unix_to_local_datetime(df["timestamp"], timezone)
    timestamp_value = pd.to_numeric(df["timestamp"], errors="coerce")
    app_package = df["RUNNING_TASKS_topActivity_mPackage"].astype("string")
    valid = pd.DataFrame(
        {
            "timestamp_dt": timestamp_dt,
            "timestamp_value": timestamp_value,
            "app_package": app_package,
        }
    ).dropna(subset=["timestamp_dt"])
    valid["date"] = valid["timestamp_dt"].dt.date.astype(str)
    day = valid[valid["date"] == date]
    if day.empty:
        return {
            "app_usage_record_count": np.nan,
            "app_usage_timestamp_count": np.nan,
            "app_diversity": np.nan,
            "night_app_usage_ratio": np.nan,
        }

    unique_timestamps = day.dropna(subset=["timestamp_value"]).drop_duplicates(
        subset=["timestamp_value"]
    )
    return {
        "app_usage_record_count": len(day),
        "app_usage_timestamp_count": int(day["timestamp_value"].nunique()),
        "app_diversity": int(day["app_package"].nunique()),
        "night_app_usage_ratio": float(is_night(unique_timestamps["timestamp_dt"]).mean()),
    }


def raw_sample_checks(
    features: pd.DataFrame,
    dataset_root: Path,
    timezone: str,
    sample_size: int,
    seed: int,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    calculators = {
        "phonelock": calc_phonelock,
        "dark": calc_dark,
        "activity": calc_activity,
        "conversation": calc_conversation,
        "wifi_location": calc_wifi_location,
        "app_usage": calc_app_usage,
    }

    for modality, anchor_column in MODALITY_ANCHORS.items():
        candidates = features[features[anchor_column].notna()][["uid", "date"]]
        if candidates.empty:
            assert_check(
                failures,
                lines,
                f"raw_sample_{modality}",
                False,
                f"no rows available for anchor column {anchor_column}",
            )
            continue
        sample = candidates.sample(
            n=min(sample_size, len(candidates)),
            random_state=seed,
            replace=False,
        )
        for _, item in sample.iterrows():
            uid = item["uid"]
            date = item["date"]
            expected = calculators[modality](dataset_root, uid, date, timezone)
            actual = features[(features["uid"] == uid) & (features["date"] == date)].iloc[0]
            compare_expected_to_actual(
                failures, lines, modality, uid, date, expected, actual
            )


def app_usage_source_checks(
    dataset_root: Path,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    duration_like_terms = [
        "duration",
        "usage_time",
        "time_spent",
        "screen_time",
        "foreground_time",
        "elapsed",
    ]
    app_usage_columns = set()
    for path in (dataset_root / "app_usage").glob("running_app_u*.csv"):
        df = read_csv_clean(path)
        app_usage_columns.update(df.columns)

    duration_like = [
        col
        for col in sorted(app_usage_columns)
        if any(term in col.lower() for term in duration_like_terms)
    ]
    assert_check(
        failures,
        lines,
        "app_usage_no_duration_column",
        not duration_like,
        f"duration-like columns in app_usage={duration_like}",
    )


def write_outputs(
    report_path: Path,
    failures_path: Path,
    lines: list[str],
    failures: list[dict[str, object]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(failures).to_csv(failures_path, index=False)


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    failures: list[dict[str, object]] = []
    lines: list[str] = [
        "Daily features validation report",
        f"features_file={args.features}",
        f"dataset_root={args.dataset_root}",
        f"timezone={args.timezone}",
        "",
        "Summary",
        f"rows={len(features)}",
        f"participants={features['uid'].nunique()}",
        f"date_range={features['date'].min()} to {features['date'].max()}",
        "",
        "Checks",
    ]

    structural_checks(features, args.dataset_root, failures, lines)
    range_checks(features, failures, lines)
    raw_sample_checks(
        features,
        args.dataset_root,
        args.timezone,
        args.sample_size,
        args.seed,
        failures,
        lines,
    )
    app_usage_source_checks(args.dataset_root, failures, lines)

    errors = [failure for failure in failures if failure["severity"] == "ERROR"]
    warnings = [failure for failure in failures if failure["severity"] == "WARN"]
    lines.extend(
        [
            "",
            "Final status",
            f"errors={len(errors)}",
            f"warnings={len(warnings)}",
            "result=PASS" if not errors else "result=FAIL",
        ]
    )
    write_outputs(args.report, args.failures, lines, failures)
    print("\n".join(lines))
    print(f"\nReport saved to: {args.report}")
    print(f"Failures saved to: {args.failures}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
