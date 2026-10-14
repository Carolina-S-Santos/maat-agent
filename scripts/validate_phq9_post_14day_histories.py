from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PHQ9_COLUMNS = [
    "phq9_score",
    "phq9_class_5",
    "phq9_class_3",
    "phq9_binary",
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
            "Validate PHQ-9 post 14-day histories and labels against the daily "
            "feature table and the window coverage table."
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
        help="PHQ-9 window coverage table.",
    )
    parser.add_argument(
        "--histories",
        default=Path("outputs/phq9_post_14day_histories.csv"),
        type=Path,
        help="Generated 14-day histories table.",
    )
    parser.add_argument(
        "--labels",
        default=Path("outputs/phq9_post_14day_labels.csv"),
        type=Path,
        help="Generated labels table.",
    )
    parser.add_argument(
        "--report",
        default=Path("outputs/phq9_post_14day_histories_validation_report.txt"),
        type=Path,
        help="Path where validation report will be saved.",
    )
    parser.add_argument(
        "--failures",
        default=Path("outputs/phq9_post_14day_histories_validation_failures.csv"),
        type=Path,
        help="Path where validation failures will be saved.",
    )
    parser.add_argument(
        "--target-type",
        default="post",
        help="PHQ-9 type expected in the histories. Default: post.",
    )
    return parser.parse_args()


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def make_case_id(uid: str, phq9_type: str) -> str:
    return f"{uid}_{phq9_type}"


def add_check(
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
        failures.append({"severity": severity, "check": name, "details": details})


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    prepared = features.copy()
    prepared["uid"] = prepared["uid"].astype("string").str.strip()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date
    return prepared


def prepare_coverage(coverage: pd.DataFrame, target_type: str) -> pd.DataFrame:
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
    prepared["expected_days"] = pd.to_numeric(prepared["expected_days"], errors="coerce")
    prepared["available_days"] = pd.to_numeric(prepared["available_days"], errors="coerce")

    valid = prepared[
        prepared["type"].eq(target_type)
        & prepared["window_has_sufficient_coverage"].eq(True)
    ].copy()
    valid["case_id"] = valid.apply(
        lambda row: make_case_id(str(row["uid"]), str(row["type"])), axis=1
    )
    return valid.sort_values(["uid", "type"]).reset_index(drop=True)


def prepare_histories(histories: pd.DataFrame) -> pd.DataFrame:
    prepared = histories.copy()
    for column in ["uid", "type", "case_id"]:
        prepared[column] = prepared[column].astype("string").str.strip()
    for column in ["reference_date", "window_start", "window_end", "date"]:
        prepared[column] = pd.to_datetime(prepared[column], errors="coerce").dt.date
    prepared["day_index"] = pd.to_numeric(prepared["day_index"], errors="coerce")
    return prepared


def prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    prepared = labels.copy()
    for column in ["uid", "type", "case_id"]:
        prepared[column] = prepared[column].astype("string").str.strip()
    prepared["reference_date"] = pd.to_datetime(
        prepared["reference_date"], errors="coerce"
    ).dt.date
    return prepared


def validate_structure(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    histories: pd.DataFrame,
    labels: pd.DataFrame,
    failures: list[dict[str, object]],
    lines: list[str],
) -> list[str]:
    feature_columns = [column for column in features.columns if column not in {"uid", "date"}]

    missing_history_columns = [
        column
        for column in [*HISTORY_METADATA_COLUMNS, *feature_columns]
        if column not in histories
    ]
    add_check(
        failures,
        lines,
        "history_required_columns",
        not missing_history_columns,
        f"missing={missing_history_columns}",
    )

    missing_label_columns = [column for column in LABEL_COLUMNS if column not in labels]
    add_check(
        failures,
        lines,
        "label_required_columns",
        not missing_label_columns,
        f"missing={missing_label_columns}",
    )

    leaked_columns = [column for column in PHQ9_COLUMNS if column in histories]
    add_check(
        failures,
        lines,
        "no_phq9_leakage_in_histories",
        not leaked_columns,
        f"leaked_columns={leaked_columns}",
    )

    expected_cases = set(coverage["case_id"])
    history_cases = set(histories["case_id"])
    label_cases = set(labels["case_id"])
    add_check(
        failures,
        lines,
        "history_cases_match_coverage",
        history_cases == expected_cases,
        f"expected={len(expected_cases)}, actual={len(history_cases)}",
    )
    add_check(
        failures,
        lines,
        "label_cases_match_coverage",
        label_cases == expected_cases,
        f"expected={len(expected_cases)}, actual={len(label_cases)}",
    )

    duplicate_history_rows = int(histories.duplicated(["case_id", "day_index"]).sum())
    add_check(
        failures,
        lines,
        "unique_case_day_index",
        duplicate_history_rows == 0,
        f"duplicates={duplicate_history_rows}",
    )

    duplicate_label_rows = int(labels.duplicated(["case_id"]).sum())
    add_check(
        failures,
        lines,
        "unique_label_case_id",
        duplicate_label_rows == 0,
        f"duplicates={duplicate_label_rows}",
    )

    expected_history_rows = int(coverage["expected_days"].sum())
    add_check(
        failures,
        lines,
        "expected_history_row_count",
        len(histories) == expected_history_rows,
        f"expected={expected_history_rows}, actual={len(histories)}",
    )
    add_check(
        failures,
        lines,
        "expected_label_row_count",
        len(labels) == len(coverage),
        f"expected={len(coverage)}, actual={len(labels)}",
    )

    return feature_columns


def validate_case_windows(
    coverage: pd.DataFrame,
    histories: pd.DataFrame,
    feature_columns: list[str],
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    mismatches: list[dict[str, object]] = []
    availability_mismatches: list[dict[str, object]] = []

    for window in coverage.to_dict("records"):
        case_id = window["case_id"]
        case_history = histories[histories["case_id"].eq(case_id)].sort_values("day_index")
        start = pd.Timestamp(window["window_start"])
        end = pd.Timestamp(window["window_end"])
        expected_dates = list(pd.date_range(start, end, freq="D").date)
        expected_day_index = list(range(1, len(expected_dates) + 1))

        actual_dates = case_history["date"].tolist()
        actual_day_index = case_history["day_index"].astype("Int64").tolist()

        if actual_dates != expected_dates or actual_day_index != expected_day_index:
            mismatches.append(
                {
                    "case_id": case_id,
                    "expected_dates": expected_dates,
                    "actual_dates": actual_dates,
                    "expected_day_index": expected_day_index,
                    "actual_day_index": actual_day_index,
                }
            )

        available_days = int(case_history[feature_columns].notna().any(axis=1).sum())
        if available_days != int(window["available_days"]):
            availability_mismatches.append(
                {
                    "case_id": case_id,
                    "expected_available_days": int(window["available_days"]),
                    "actual_available_days": available_days,
                }
            )

    add_check(
        failures,
        lines,
        "case_dates_and_day_index",
        not mismatches,
        f"mismatched_cases={len(mismatches)}",
    )
    for mismatch in mismatches[:10]:
        failures.append(
            {
                "severity": "ERROR",
                "check": "case_dates_and_day_index_detail",
                "details": mismatch,
            }
        )

    add_check(
        failures,
        lines,
        "available_days_match_coverage",
        not availability_mismatches,
        f"mismatched_cases={len(availability_mismatches)}",
    )
    for mismatch in availability_mismatches[:10]:
        failures.append(
            {
                "severity": "ERROR",
                "check": "available_days_match_coverage_detail",
                "details": mismatch,
            }
        )


def validate_labels_against_coverage(
    coverage: pd.DataFrame,
    labels: pd.DataFrame,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    expected = coverage[LABEL_COLUMNS].sort_values("case_id").reset_index(drop=True)
    actual = labels[LABEL_COLUMNS].sort_values("case_id").reset_index(drop=True)

    same_shape = expected.shape == actual.shape
    add_check(
        failures,
        lines,
        "labels_same_shape_as_coverage",
        same_shape,
        f"expected_shape={expected.shape}, actual_shape={actual.shape}",
    )
    if not same_shape:
        return

    mismatches: list[dict[str, object]] = []
    for column in LABEL_COLUMNS:
        unequal = expected[column].astype(str) != actual[column].astype(str)
        for index in expected.index[unequal]:
            mismatches.append(
                {
                    "case_id": expected.loc[index, "case_id"],
                    "column": column,
                    "expected": expected.loc[index, column],
                    "actual": actual.loc[index, column],
                }
            )

    add_check(
        failures,
        lines,
        "labels_match_coverage_values",
        not mismatches,
        f"mismatch_count={len(mismatches)}",
    )
    for mismatch in mismatches[:20]:
        failures.append(
            {
                "severity": "ERROR",
                "check": "labels_match_coverage_values_detail",
                "details": mismatch,
            }
        )


def validate_histories_against_features(
    features: pd.DataFrame,
    histories: pd.DataFrame,
    feature_columns: list[str],
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    source = features[["uid", "date", *feature_columns]].copy()
    merged = histories[["case_id", "uid", "date", *feature_columns]].merge(
        source,
        on=["uid", "date"],
        how="left",
        suffixes=("_history", "_source"),
    )

    mismatches: list[dict[str, object]] = []
    for column in feature_columns:
        history_values = pd.to_numeric(merged[f"{column}_history"], errors="coerce")
        source_values = pd.to_numeric(merged[f"{column}_source"], errors="coerce")
        both_missing = history_values.isna() & source_values.isna()
        both_equal = (history_values - source_values).abs() <= 1e-9
        unequal = ~(both_missing | both_equal)

        for index in merged.index[unequal]:
            mismatches.append(
                {
                    "case_id": merged.loc[index, "case_id"],
                    "uid": merged.loc[index, "uid"],
                    "date": merged.loc[index, "date"],
                    "column": column,
                    "history": merged.loc[index, f"{column}_history"],
                    "source": merged.loc[index, f"{column}_source"],
                }
            )

    add_check(
        failures,
        lines,
        "history_values_match_daily_features",
        not mismatches,
        f"mismatch_count={len(mismatches)}",
    )
    for mismatch in mismatches[:20]:
        failures.append(
            {
                "severity": "ERROR",
                "check": "history_values_match_daily_features_detail",
                "details": mismatch,
            }
        )


def validate_known_outlier_excluded(
    histories: pd.DataFrame,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    outlier_date = pd.Timestamp("2014-01-16").date()
    outlier_rows = int(histories["date"].eq(outlier_date).sum())
    add_check(
        failures,
        lines,
        "known_2014_app_usage_outlier_excluded",
        outlier_rows == 0,
        f"rows_on_2014_01_16={outlier_rows}",
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
    for path in [args.features, args.coverage, args.histories, args.labels]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    features = prepare_features(read_csv_clean(args.features))
    coverage = prepare_coverage(read_csv_clean(args.coverage), args.target_type)
    histories = prepare_histories(read_csv_clean(args.histories))
    labels = prepare_labels(read_csv_clean(args.labels))

    failures: list[dict[str, object]] = []
    lines: list[str] = [
        "PHQ-9 post 14-day histories validation report",
        f"features_file={args.features}",
        f"coverage_file={args.coverage}",
        f"histories_file={args.histories}",
        f"labels_file={args.labels}",
        "",
        "Summary",
        f"coverage_cases={coverage['case_id'].nunique()}",
        f"history_rows={len(histories)}",
        f"history_cases={histories['case_id'].nunique()}",
        f"label_rows={len(labels)}",
        f"label_cases={labels['case_id'].nunique()}",
        "",
        "Checks",
    ]

    feature_columns = validate_structure(
        features, coverage, histories, labels, failures, lines
    )
    validate_case_windows(coverage, histories, feature_columns, failures, lines)
    validate_labels_against_coverage(coverage, labels, failures, lines)
    validate_histories_against_features(features, histories, feature_columns, failures, lines)
    validate_known_outlier_excluded(histories, failures, lines)

    errors = [failure for failure in failures if failure["severity"] == "ERROR"]
    warnings = [failure for failure in failures if failure["severity"] == "WARN"]
    lines.extend(
        [
            "",
            "Final status",
            f"errors={len(errors)}",
            f"warnings={len(warnings)}",
            "result=PASS" if not errors else "result=FAIL",
            "",
            "Label distribution",
            labels["phq9_binary"].value_counts().sort_index().to_string(),
        ]
    )

    write_outputs(args.report, args.failures, lines, failures)
    print("\n".join(lines))
    print(f"\nReport saved to: {args.report}")
    print(f"Failures saved to: {args.failures}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
