from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PHQ9_ITEM_TEXT_COLUMNS = [f"phq9_item_{index}_text" for index in range(1, 10)]
PHQ9_ITEM_SCORE_COLUMNS = [f"phq9_item_{index}_score" for index in range(1, 10)]

PHQ9_COLUMNS = [
    *PHQ9_ITEM_TEXT_COLUMNS,
    *PHQ9_ITEM_SCORE_COLUMNS,
    "phq9_score",
    "phq9_class_5",
    "phq9_class_3",
    "phq9_binary",
]

COVERAGE_REQUIRED_COLUMNS = [
    "uid",
    "type",
    "reference_date",
    "window_start",
    "window_end",
    "expected_days",
    "available_days",
    "window_has_sufficient_coverage",
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

LABEL_COLUMNS = [
    "case_id",
    "uid",
    "type",
    "reference_date",
    *PHQ9_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 14-day post-PHQ-9 histories and a separate PHQ-9 label table "
            "from daily StudentLife features and window coverage."
        )
    )
    parser.add_argument(
        "--features",
        default=Path("outputs/studentlife_daily_features.csv"),
        type=Path,
        help="Participant-day feature table.",
    )
    parser.add_argument(
        "--coverage",
        default=Path("outputs/phq9_feature_window_coverage.csv"),
        type=Path,
        help="PHQ-9 14-day window coverage table.",
    )
    parser.add_argument(
        "--outcomes",
        default=Path("outputs/phq9_outcomes.csv"),
        type=Path,
        help="Enriched PHQ-9 outcomes table keyed by uid and type.",
    )
    parser.add_argument(
        "--histories-output",
        default=Path("outputs/phq9_post_14day_histories.csv"),
        type=Path,
        help="Output CSV with one row per case-day and no PHQ-9 labels.",
    )
    parser.add_argument(
        "--labels-output",
        default=Path("outputs/phq9_post_14day_labels.csv"),
        type=Path,
        help="Output CSV with one row per case and the PHQ-9 ground truth.",
    )
    parser.add_argument(
        "--target-type",
        default="post",
        help="PHQ-9 type to use when building histories. Default: post.",
    )
    return parser.parse_args()


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {"uid", "date"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    prepared = features.copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date

    invalid_dates = int(prepared["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found invalid feature dates: {invalid_dates}")

    duplicate_rows = int(prepared.duplicated(["uid", "date"]).sum())
    if duplicate_rows:
        raise ValueError(f"Found duplicated uid+date rows: {duplicate_rows}")

    return prepared


def prepare_coverage(coverage: pd.DataFrame, target_type: str) -> pd.DataFrame:
    missing = [column for column in COVERAGE_REQUIRED_COLUMNS if column not in coverage]
    if missing:
        raise ValueError(f"Missing required coverage columns: {missing}")

    prepared = coverage.copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["type"] = prepared["type"].astype("string").str.strip()
    prepared["reference_date"] = pd.to_datetime(
        prepared["reference_date"], errors="coerce"
    ).dt.date
    prepared["window_start"] = pd.to_datetime(
        prepared["window_start"], errors="coerce"
    ).dt.date
    prepared["window_end"] = pd.to_datetime(prepared["window_end"], errors="coerce").dt.date
    prepared["window_has_sufficient_coverage"] = parse_bool_series(
        prepared["window_has_sufficient_coverage"]
    )

    valid = prepared[
        prepared["type"].eq(target_type)
        & prepared["window_has_sufficient_coverage"].eq(True)
    ].copy()

    if valid.empty:
        raise ValueError(
            f"No rows found for type={target_type!r} with sufficient window coverage."
        )

    invalid_window_dates = int(valid[["reference_date", "window_start", "window_end"]].isna().any(axis=1).sum())
    if invalid_window_dates:
        raise ValueError(f"Found valid coverage rows with invalid dates: {invalid_window_dates}")

    duplicate_rows = int(valid.duplicated(["uid", "type"]).sum())
    if duplicate_rows:
        raise ValueError(f"Found duplicated uid+type valid windows: {duplicate_rows}")

    valid["expected_days"] = pd.to_numeric(valid["expected_days"], errors="coerce")
    invalid_expected_days = int(valid["expected_days"].isna().sum())
    if invalid_expected_days:
        raise ValueError(f"Found invalid expected_days values: {invalid_expected_days}")

    return valid.sort_values(["uid", "type"]).reset_index(drop=True)


def prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    required = ["uid", "type", *PHQ9_COLUMNS]
    missing = [column for column in required if column not in outcomes.columns]
    if missing:
        raise ValueError(f"Missing required outcomes columns: {missing}")

    prepared = outcomes.copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["type"] = prepared["type"].astype("string").str.strip()

    duplicate_rows = int(prepared.duplicated(["uid", "type"]).sum())
    if duplicate_rows:
        raise ValueError(f"Found duplicated uid+type rows in outcomes: {duplicate_rows}")

    return prepared[required].sort_values(["uid", "type"]).reset_index(drop=True)


def merge_coverage_with_outcomes(
    coverage: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    merged = coverage.merge(
        outcomes,
        on=["uid", "type"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcomes"),
    )

    missing_outcomes = int(merged[PHQ9_COLUMNS].isna().all(axis=1).sum())
    if missing_outcomes:
        rows = merged.loc[merged[PHQ9_COLUMNS].isna().all(axis=1), ["uid", "type"]]
        raise ValueError(
            "Missing PHQ-9 outcomes after merging coverage with outcomes: "
            f"{rows.to_dict('records')}"
        )

    return merged


def make_case_id(uid: str, phq9_type: str) -> str:
    return f"{uid}_{phq9_type}"


def build_histories_and_labels(
    features: pd.DataFrame,
    valid_windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in features.columns if column not in {"uid", "date"}]
    history_rows: list[pd.DataFrame] = []
    label_rows: list[dict[str, object]] = []

    for window in valid_windows.to_dict("records"):
        uid = str(window["uid"])
        phq9_type = str(window["type"])
        case_id = make_case_id(uid, phq9_type)
        window_start = pd.Timestamp(window["window_start"])
        window_end = pd.Timestamp(window["window_end"])
        expected_days = int(window["expected_days"])

        date_range = pd.date_range(window_start, window_end, freq="D").date
        if len(date_range) != expected_days:
            raise ValueError(
                f"{case_id}: expected_days={expected_days}, "
                f"but window contains {len(date_range)} dates."
            )

        base = pd.DataFrame(
            {
                "case_id": case_id,
                "uid": uid,
                "type": phq9_type,
                "reference_date": window["reference_date"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "day_index": range(1, expected_days + 1),
                "date": date_range,
            }
        )

        participant_features = features.loc[
            features["uid"].eq(uid),
            ["date", *feature_columns],
        ]
        history = base.merge(participant_features, on="date", how="left")
        history_rows.append(history[HISTORY_METADATA_COLUMNS + feature_columns])

        label_rows.append(
            {
                "case_id": case_id,
                "uid": uid,
                "type": phq9_type,
                "reference_date": window["reference_date"],
                **{column: window[column] for column in PHQ9_COLUMNS},
            }
        )

    histories = pd.concat(history_rows, ignore_index=True)
    labels = pd.DataFrame(label_rows)[LABEL_COLUMNS]
    return histories, labels


def summarize(histories: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    feature_columns = [
        column for column in histories.columns if column not in HISTORY_METADATA_COLUMNS
    ]
    missing_feature_days = int(histories[feature_columns].isna().all(axis=1).sum())
    return {
        "cases": labels["case_id"].nunique(),
        "history_rows": len(histories),
        "label_rows": len(labels),
        "days_per_case_min": int(histories.groupby("case_id")["date"].nunique().min()),
        "days_per_case_max": int(histories.groupby("case_id")["date"].nunique().max()),
        "empty_feature_days": missing_feature_days,
    }


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(f"Feature table not found: {args.features}")
    if not args.coverage.exists():
        raise FileNotFoundError(f"Coverage table not found: {args.coverage}")
    if not args.outcomes.exists():
        raise FileNotFoundError(f"PHQ-9 outcomes table not found: {args.outcomes}")

    features = prepare_features(read_csv_clean(args.features))
    valid_windows = prepare_coverage(read_csv_clean(args.coverage), args.target_type)
    outcomes = prepare_outcomes(read_csv_clean(args.outcomes))
    valid_windows = merge_coverage_with_outcomes(valid_windows, outcomes)
    histories, labels = build_histories_and_labels(features, valid_windows)

    args.histories_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    histories.to_csv(args.histories_output, index=False)
    labels.to_csv(args.labels_output, index=False)

    summary = summarize(histories, labels)
    print("Saved histories:", args.histories_output)
    print("Saved labels:", args.labels_output)
    print("Target type:", args.target_type)
    print("Cases:", summary["cases"])
    print("History rows:", summary["history_rows"])
    print("Label rows:", summary["label_rows"])
    print(
        "Days per case:",
        summary["days_per_case_min"],
        "to",
        summary["days_per_case_max"],
    )
    print("Days with no feature row in daily table:", summary["empty_feature_days"])
    print("\nBinary label distribution:")
    print(labels["phq9_binary"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
