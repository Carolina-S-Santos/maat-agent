from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESPONSE_TO_SCORE = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

EXPECTED_COLUMNS = [
    "uid",
    "type",
    "phq9_score",
    "phq9_class_5",
    "phq9_class_3",
    "phq9_binary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate outputs/phq9_outcomes.csv against raw PHQ-9 data."
    )
    parser.add_argument(
        "--raw",
        default=Path("dataset/survey/PHQ-9.csv"),
        type=Path,
        help="Path to raw StudentLife PHQ-9 CSV.",
    )
    parser.add_argument(
        "--outcomes",
        default=Path("outputs/phq9_outcomes.csv"),
        type=Path,
        help="Path to generated PHQ-9 outcomes CSV.",
    )
    parser.add_argument(
        "--report",
        default=Path("outputs/phq9_validation_report.txt"),
        type=Path,
        help="Path where validation report will be saved.",
    )
    parser.add_argument(
        "--failures",
        default=Path("outputs/phq9_validation_failures.csv"),
        type=Path,
        help="Path where validation failures will be saved.",
    )
    return parser.parse_args()


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def classify_5(score: int) -> str:
    if 0 <= score <= 4:
        return "minimal"
    if 5 <= score <= 9:
        return "mild"
    if 10 <= score <= 14:
        return "moderate"
    if 15 <= score <= 19:
        return "moderately_severe"
    if 20 <= score <= 27:
        return "severe"
    raise ValueError(f"PHQ-9 score outside expected range 0-27: {score}")


def classify_3(score: int) -> str:
    if 0 <= score <= 4:
        return "low"
    if 5 <= score <= 14:
        return "medium"
    if 15 <= score <= 27:
        return "high"
    raise ValueError(f"PHQ-9 score outside expected range 0-27: {score}")


def classify_binary(score: int) -> str:
    if 0 <= score <= 9:
        return "low_or_none"
    if 10 <= score <= 27:
        return "clinically_relevant"
    raise ValueError(f"PHQ-9 score outside expected range 0-27: {score}")


def get_item_columns(raw: pd.DataFrame) -> list[str]:
    required = {"uid", "type", "Response"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required raw PHQ-9 columns: {sorted(missing)}")

    item_columns = [column for column in raw.columns if column not in required]
    if len(item_columns) != 9:
        raise ValueError(
            f"Expected 9 PHQ-9 item columns, found {len(item_columns)}: {item_columns}"
        )
    return item_columns


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


def recalculate_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    item_columns = get_item_columns(raw)
    scored_items = raw[item_columns].replace(RESPONSE_TO_SCORE)

    invalid_values: dict[str, list[str]] = {}
    for column in item_columns:
        numeric = pd.to_numeric(scored_items[column], errors="coerce")
        invalid = raw.loc[numeric.isna(), column].dropna().unique().tolist()
        if invalid:
            invalid_values[column] = invalid
        scored_items[column] = numeric

    if invalid_values:
        raise ValueError(f"Found unrecognized PHQ-9 item values: {invalid_values}")

    missing_items = scored_items.isna().sum(axis=1)
    if int(missing_items.sum()) > 0:
        rows = raw.loc[missing_items > 0, ["uid", "type"]].to_dict("records")
        raise ValueError(f"Found missing PHQ-9 item values in rows: {rows}")

    expected = raw[["uid", "type"]].copy()
    expected["uid"] = expected["uid"].astype("string").str.strip()
    expected["type"] = expected["type"].astype("string").str.strip()
    expected["phq9_score"] = scored_items.sum(axis=1).astype(int)
    expected["phq9_class_5"] = expected["phq9_score"].apply(classify_5)
    expected["phq9_class_3"] = expected["phq9_score"].apply(classify_3)
    expected["phq9_binary"] = expected["phq9_score"].apply(classify_binary)
    return expected[EXPECTED_COLUMNS]


def validate_structure(
    outcomes: pd.DataFrame, failures: list[dict[str, object]], lines: list[str]
) -> None:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in outcomes]
    add_check(
        failures,
        lines,
        "required_columns",
        not missing_columns,
        f"missing={missing_columns}",
    )

    duplicate_count = int(outcomes.duplicated(["uid", "type"]).sum())
    add_check(
        failures,
        lines,
        "unique_uid_type",
        duplicate_count == 0,
        f"duplicate uid+type rows={duplicate_count}",
    )

    missing_values = outcomes[EXPECTED_COLUMNS].isna().sum().to_dict()
    total_missing = int(sum(missing_values.values()))
    add_check(
        failures,
        lines,
        "no_missing_expected_columns",
        total_missing == 0,
        f"missing={missing_values}",
    )

    valid_types = {"pre", "post"}
    invalid_types = sorted(set(outcomes["type"].dropna()) - valid_types)
    add_check(
        failures,
        lines,
        "valid_type_values",
        not invalid_types,
        f"invalid={invalid_types}",
    )

    score_numeric = pd.to_numeric(outcomes["phq9_score"], errors="coerce")
    invalid_score = int(score_numeric.isna().sum())
    out_of_range = int(((score_numeric < 0) | (score_numeric > 27)).sum())
    non_integer = int(((score_numeric - score_numeric.round()).abs() > 0).sum())
    add_check(
        failures,
        lines,
        "valid_score_values",
        invalid_score == 0 and out_of_range == 0 and non_integer == 0,
        f"non_numeric={invalid_score}, out_of_range={out_of_range}, non_integer={non_integer}",
    )


def validate_against_raw(
    raw: pd.DataFrame,
    outcomes: pd.DataFrame,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    expected = recalculate_outcomes(raw)
    expected = expected.sort_values(["uid", "type"]).reset_index(drop=True)
    actual = outcomes[EXPECTED_COLUMNS].copy()
    actual["uid"] = actual["uid"].astype("string").str.strip()
    actual["type"] = actual["type"].astype("string").str.strip()
    actual = actual.sort_values(["uid", "type"]).reset_index(drop=True)

    same_shape = expected.shape == actual.shape
    add_check(
        failures,
        lines,
        "same_row_count_as_raw",
        same_shape,
        f"expected_shape={expected.shape}, actual_shape={actual.shape}",
    )

    if not same_shape:
        return

    mismatches: list[dict[str, object]] = []
    for column in EXPECTED_COLUMNS:
        unequal = expected[column].astype(str) != actual[column].astype(str)
        for index in expected.index[unequal]:
            mismatches.append(
                {
                    "uid": expected.loc[index, "uid"],
                    "type": expected.loc[index, "type"],
                    "column": column,
                    "expected": expected.loc[index, column],
                    "actual": actual.loc[index, column],
                }
            )

    add_check(
        failures,
        lines,
        "matches_recalculated_raw_values",
        not mismatches,
        f"mismatch_count={len(mismatches)}",
    )
    for mismatch in mismatches[:20]:
        failures.append(
            {
                "severity": "ERROR",
                "check": "value_mismatch_detail",
                "details": mismatch,
            }
        )


def validate_response_excluded(
    raw: pd.DataFrame,
    outcomes: pd.DataFrame,
    failures: list[dict[str, object]],
    lines: list[str],
) -> None:
    item_columns = get_item_columns(raw)
    response_values = set(raw["Response"].dropna().unique())
    score_values = set(RESPONSE_TO_SCORE)
    response_looks_like_item = sorted(response_values & score_values)

    add_check(
        failures,
        lines,
        "response_column_not_item",
        not response_looks_like_item,
        f"response_values_overlapping_item_values={response_looks_like_item}",
    )

    scored_with_response = raw[item_columns + ["Response"]].replace(RESPONSE_TO_SCORE)
    response_numeric = pd.to_numeric(scored_with_response["Response"], errors="coerce")
    response_would_change_score = int(response_numeric.notna().sum())
    add_check(
        failures,
        lines,
        "response_excluded_from_score",
        response_would_change_score == 0,
        f"rows_where_response_would_add_numeric_score={response_would_change_score}",
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
    raw = read_csv_clean(args.raw)
    outcomes = read_csv_clean(args.outcomes)

    failures: list[dict[str, object]] = []
    lines: list[str] = [
        "PHQ-9 outcomes validation report",
        f"raw_file={args.raw}",
        f"outcomes_file={args.outcomes}",
        "",
        "Summary",
        f"raw_rows={len(raw)}",
        f"outcome_rows={len(outcomes)}",
        f"participants={outcomes['uid'].nunique() if 'uid' in outcomes else 'NA'}",
        "",
        "Checks",
    ]

    validate_structure(outcomes, failures, lines)
    validate_against_raw(raw, outcomes, failures, lines)
    validate_response_excluded(raw, outcomes, failures, lines)

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
