from __future__ import annotations

import argparse
import math
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


UID_RE = re.compile(r"u\d+")
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6

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
        description="Build the StudentLife participant-day feature table."
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        type=Path,
        help="Path to the StudentLife dataset directory.",
    )
    parser.add_argument(
        "--output",
        default=Path("outputs/studentlife_daily_features.csv"),
        type=Path,
        help="CSV file where the participant-day feature table will be saved.",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help=(
            "Timezone used to convert Unix timestamps into calendar days. "
            "StudentLife was collected at Dartmouth, so America/New_York is "
            "the default."
        ),
    )
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
    return (datetimes.dt.hour >= NIGHT_START_HOUR) & (
        datetimes.dt.hour < NIGHT_END_HOUR
    )


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["uid", "date", *columns])


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=["uid", "date", *FEATURE_COLUMNS])
    return reduce(
        lambda left, right: pd.merge(left, right, on=["uid", "date"], how="outer"),
        non_empty,
    )


def concat_participant_frames(
    frames: list[pd.DataFrame], columns: list[str]
) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return empty_frame(columns)
    return pd.concat(non_empty, ignore_index=True, sort=False)


def split_interval_duration_by_day(
    df: pd.DataFrame,
    uid: str,
    start_col: str,
    end_col: str,
    output_col: str,
    timezone: str,
    unit: str,
) -> pd.DataFrame:
    starts = unix_to_local_datetime(df[start_col], timezone)
    ends = unix_to_local_datetime(df[end_col], timezone)

    rows: list[dict[str, object]] = []
    for start, end in zip(starts, ends):
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue

        current = start
        while current < end:
            next_midnight = current.normalize() + pd.Timedelta(days=1)
            segment_end = min(end, next_midnight)
            rows.append(
                {
                    "uid": uid,
                    "date": str(current.date()),
                    "segment_start": current,
                    "segment_end": segment_end,
                }
            )
            current = segment_end

    if not rows:
        return empty_frame([output_col])

    segments = pd.DataFrame(rows)

    def union_seconds(group: pd.DataFrame) -> float:
        intervals = group.sort_values("segment_start")[
            ["segment_start", "segment_end"]
        ].itertuples(index=False)
        total = 0.0
        current_start = None
        current_end = None

        for start, end in intervals:
            if current_start is None:
                current_start = start
                current_end = end
                continue

            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += (current_end - current_start).total_seconds()
                current_start = start
                current_end = end

        if current_start is not None:
            total += (current_end - current_start).total_seconds()

        return total

    out = (
        segments.groupby(["uid", "date"])
        .apply(union_seconds, include_groups=False)
        .reset_index(name=output_col)
    )

    if unit == "hours":
        out[output_col] = out[output_col] / 3600
    elif unit == "minutes":
        out[output_col] = out[output_col] / 60
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

    return out


def build_phonelock_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((dataset_root / "sensing" / "phonelock").glob("phonelock_u*.csv")):
        uid = extract_uid(path)
        df = read_csv_clean(path)
        required = {"start", "end"}
        if not required.issubset(df.columns):
            continue

        df["start_dt"] = unix_to_local_datetime(df["start"], timezone)
        df["end_dt"] = unix_to_local_datetime(df["end"], timezone)
        df = df.dropna(subset=["start_dt", "end_dt"])
        df = df[df["end_dt"] > df["start_dt"]].copy()
        if df.empty:
            continue

        df["uid"] = uid
        df["date"] = df["start_dt"].dt.date.astype(str)
        df["start_decimal_hour"] = decimal_hour(df["start_dt"])
        df["night_start"] = is_night(df["start_dt"]).astype(int)

        starts = (
            df.groupby(["uid", "date"], as_index=False)
            .agg(
                phone_lock_episode_count=("start_dt", "size"),
                night_phone_lock_episode_count=("night_start", "sum"),
                first_phone_lock_start_time=("start_decimal_hour", "min"),
                last_phone_lock_start_time=("start_decimal_hour", "max"),
            )
        )
        duration = split_interval_duration_by_day(
            df,
            uid,
            "start",
            "end",
            "phone_locked_duration_hours",
            timezone,
            "hours",
        )
        frames.append(merge_frames([starts, duration]))

    return concat_participant_frames(
        frames,
        [
            "phone_lock_episode_count",
            "night_phone_lock_episode_count",
            "first_phone_lock_start_time",
            "last_phone_lock_start_time",
            "phone_locked_duration_hours",
        ],
    )


def build_dark_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((dataset_root / "sensing" / "dark").glob("dark_u*.csv")):
        uid = extract_uid(path)
        df = read_csv_clean(path)
        required = {"start", "end"}
        if not required.issubset(df.columns):
            continue

        df["start_dt"] = unix_to_local_datetime(df["start"], timezone)
        df["end_dt"] = unix_to_local_datetime(df["end"], timezone)
        df = df.dropna(subset=["start_dt", "end_dt"])
        df = df[df["end_dt"] > df["start_dt"]].copy()
        if df.empty:
            continue

        df["uid"] = uid
        df["date"] = df["start_dt"].dt.date.astype(str)
        df["duration_seconds"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds()
        df["start_decimal_hour"] = decimal_hour(df["start_dt"])

        counts = df.groupby(["uid", "date"], as_index=False).agg(
            dark_episode_count=("start_dt", "size")
        )

        longest_idx = df.groupby(["uid", "date"])["duration_seconds"].idxmax()
        longest = df.loc[
            longest_idx, ["uid", "date", "start_decimal_hour"]
        ].rename(columns={"start_decimal_hour": "main_dark_start_time"})

        duration = split_interval_duration_by_day(
            df,
            uid,
            "start",
            "end",
            "dark_duration_hours",
            timezone,
            "hours",
        )
        frames.append(merge_frames([duration, longest, counts]))

    return concat_participant_frames(
        frames,
        ["dark_duration_hours", "main_dark_start_time", "dark_episode_count"],
    )


def build_activity_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((dataset_root / "sensing" / "activity").glob("activity_u*.csv")):
        uid = extract_uid(path)
        df = read_csv_clean(path)
        required = {"timestamp", "activity inference"}
        if not required.issubset(df.columns):
            continue

        df["timestamp_dt"] = unix_to_local_datetime(df["timestamp"], timezone)
        df["activity"] = pd.to_numeric(df["activity inference"], errors="coerce")
        df = df.dropna(subset=["timestamp_dt", "activity"])
        df = df[df["activity"].isin([0, 1, 2])].copy()
        if df.empty:
            continue

        df["uid"] = uid
        df["date"] = df["timestamp_dt"].dt.date.astype(str)
        df["is_active"] = df["activity"].isin([1, 2]).astype(int)
        df["is_running"] = (df["activity"] == 2).astype(int)

        ratios = df.groupby(["uid", "date"], as_index=False).agg(
            valid_activity_records=("activity", "size"),
            active_records=("is_active", "sum"),
            running_records=("is_running", "sum"),
        )
        ratios["activity_active_ratio"] = (
            ratios["active_records"] / ratios["valid_activity_records"]
        )
        ratios["running_ratio"] = (
            ratios["running_records"] / ratios["valid_activity_records"]
        )
        ratios = ratios.drop(
            columns=["valid_activity_records", "active_records", "running_records"]
        )

        df = df.sort_values(["uid", "date", "timestamp_dt"])
        transitions = (
            df.groupby(["uid", "date"])["activity"]
            .apply(lambda s: max(int((s != s.shift()).sum()) - 1, 0))
            .reset_index(name="activity_transition_count")
        )
        frames.append(merge_frames([ratios, transitions]))

    return concat_participant_frames(
        frames,
        ["activity_active_ratio", "activity_transition_count", "running_ratio"],
    )


def build_conversation_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(
        (dataset_root / "sensing" / "conversation").glob("conversation_u*.csv")
    ):
        uid = extract_uid(path)
        df = read_csv_clean(path)
        required = {"start_timestamp", "end_timestamp"}
        if not required.issubset(df.columns):
            continue

        df["start_dt"] = unix_to_local_datetime(df["start_timestamp"], timezone)
        df["end_dt"] = unix_to_local_datetime(df["end_timestamp"], timezone)
        df = df.dropna(subset=["start_dt", "end_dt"])
        df = df[df["end_dt"] > df["start_dt"]].copy()
        if df.empty:
            continue

        df["uid"] = uid
        df["date"] = df["start_dt"].dt.date.astype(str)
        counts = df.groupby(["uid", "date"], as_index=False).agg(
            conversation_episode_count=("start_dt", "size")
        )

        duration = split_interval_duration_by_day(
            df,
            uid,
            "start_timestamp",
            "end_timestamp",
            "conversation_minutes",
            timezone,
            "minutes",
        )
        frames.append(merge_frames([duration, counts]))

    return concat_participant_frames(
        frames,
        ["conversation_minutes", "conversation_episode_count"],
    )


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


def build_wifi_location_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(
        (dataset_root / "sensing" / "wifi_location").glob("wifi_location_u*.csv")
    ):
        uid = extract_uid(path)
        df = read_csv_clean(path)
        required = {"time", "location"}
        if not required.issubset(df.columns):
            continue

        df["time_dt"] = unix_to_local_datetime(df["time"], timezone)
        df["location"] = df["location"].astype("string").str.strip()
        df = df.dropna(subset=["time_dt", "location"])
        df = df[df["location"] != ""].copy()
        if df.empty:
            continue

        df["uid"] = uid
        df["date"] = df["time_dt"].dt.date.astype(str)
        df = df.sort_values(["uid", "date", "time_dt"])

        grouped = df.groupby(["uid", "date"])["location"]
        out = grouped.agg(
            location_count="nunique",
            primary_location_ratio=lambda s: s.value_counts().iloc[0] / len(s),
            location_transition_count=transition_count,
            location_entropy=entropy,
        ).reset_index()
        frames.append(out)

    return concat_participant_frames(
        frames,
        [
            "location_count",
            "primary_location_ratio",
            "location_transition_count",
            "location_entropy",
        ],
    )


def build_app_usage_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((dataset_root / "app_usage").glob("running_app_u*.csv")):
        uid = extract_uid(path)
        df = read_csv_clean(path)
        required = {"timestamp", "RUNNING_TASKS_topActivity_mPackage"}
        if not required.issubset(df.columns):
            continue

        df["timestamp_dt"] = unix_to_local_datetime(df["timestamp"], timezone)
        df["app_package"] = df["RUNNING_TASKS_topActivity_mPackage"].astype("string")
        df = df.dropna(subset=["timestamp_dt"])
        if df.empty:
            continue

        df["uid"] = uid
        df["date"] = df["timestamp_dt"].dt.date.astype(str)
        df["timestamp_value"] = pd.to_numeric(df["timestamp"], errors="coerce")

        base = df.groupby(["uid", "date"], as_index=False).agg(
            app_usage_record_count=("timestamp_dt", "size"),
            app_usage_timestamp_count=("timestamp_value", "nunique"),
            app_diversity=("app_package", "nunique"),
        )

        unique_timestamps = df.dropna(subset=["timestamp_value"]).drop_duplicates(
            subset=["uid", "date", "timestamp_value"]
        )
        unique_timestamps["is_night_timestamp"] = is_night(
            unique_timestamps["timestamp_dt"]
        ).astype(int)
        night = unique_timestamps.groupby(["uid", "date"], as_index=False).agg(
            night_timestamps=("is_night_timestamp", "sum"),
            total_timestamps=("timestamp_value", "size"),
        )
        night["night_app_usage_ratio"] = (
            night["night_timestamps"] / night["total_timestamps"]
        )
        night = night.drop(columns=["night_timestamps", "total_timestamps"])

        frames.append(merge_frames([base, night]))

    return concat_participant_frames(
        frames,
        [
            "app_usage_record_count",
            "app_usage_timestamp_count",
            "app_diversity",
            "night_app_usage_ratio",
        ],
    )


def build_daily_features(dataset_root: Path, timezone: str) -> pd.DataFrame:
    builders = [
        ("phonelock", build_phonelock_features),
        ("dark", build_dark_features),
        ("activity", build_activity_features),
        ("conversation", build_conversation_features),
        ("wifi_location", build_wifi_location_features),
        ("app_usage", build_app_usage_features),
    ]

    frames: list[pd.DataFrame] = []
    for name, builder in builders:
        frame = builder(dataset_root, timezone)
        frames.append(frame)
        print(
            f"{name}: {len(frame):,} participant-day rows, "
            f"{frame['uid'].nunique() if not frame.empty else 0} participants"
        )

    daily = merge_frames(frames)
    for column in FEATURE_COLUMNS:
        if column not in daily.columns:
            daily[column] = np.nan

    daily = daily[["uid", "date", *FEATURE_COLUMNS]]
    daily = daily.sort_values(["uid", "date"]).reset_index(drop=True)

    numeric_cols = [col for col in FEATURE_COLUMNS if col in daily.columns]
    daily[numeric_cols] = daily[numeric_cols].round(6)
    return daily


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    daily = build_daily_features(dataset_root, args.timezone)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output, index=False)

    print("\nSaved:", args.output)
    print("Rows:", f"{len(daily):,}")
    print("Participants:", daily["uid"].nunique())
    if not daily.empty:
        print("Date range:", daily["date"].min(), "to", daily["date"].max())
        print("Feature columns:", len(FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
