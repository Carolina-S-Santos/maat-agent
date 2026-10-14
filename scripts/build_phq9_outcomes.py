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

ITEM_TEXT_COLUMNS = [f"phq9_item_{index}_text" for index in range(1, 10)]
ITEM_SCORE_COLUMNS = [f"phq9_item_{index}_score" for index in range(1, 10)]

OUTPUT_COLUMNS = [
    "uid",
    "type",
    *ITEM_TEXT_COLUMNS,
    *ITEM_SCORE_COLUMNS,
    "phq9_score",
    "phq9_class_5",
    "phq9_class_3",
    "phq9_binary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PHQ-9 outcome table from StudentLife survey data."
    )
    parser.add_argument(
        "--input",
        default=Path("dataset/survey/PHQ-9.csv"),
        type=Path,
        help="Path to the raw StudentLife PHQ-9 CSV file.",
    )
    parser.add_argument(
        "--output",
        default=Path("outputs/phq9_outcomes.csv"),
        type=Path,
        help="Path where the PHQ-9 outcome table will be saved.",
    )
    parser.add_argument(
        "--distribution-output",
        default=Path("outputs/phq9_class_distribution.csv"),
        type=Path,
        help="Path where class distribution summary will be saved.",
    )
    return parser.parse_args()


def read_phq9(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def get_item_columns(df: pd.DataFrame) -> list[str]:
    required = {"uid", "type", "Response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in PHQ-9 file: {sorted(missing)}")

    item_columns = [col for col in df.columns if col not in required]
    if len(item_columns) != 9:
        raise ValueError(
            f"Expected 9 PHQ-9 item columns, found {len(item_columns)}: {item_columns}"
        )
    return item_columns


def score_items(df: pd.DataFrame, item_columns: list[str]) -> pd.DataFrame:
    scored = df[item_columns].replace(RESPONSE_TO_SCORE)

    invalid_values: dict[str, list[str]] = {}
    for column in item_columns:
        numeric = pd.to_numeric(scored[column], errors="coerce")
        invalid = df.loc[numeric.isna(), column].dropna().unique().tolist()
        if invalid:
            invalid_values[column] = invalid
        scored[column] = numeric

    if invalid_values:
        raise ValueError(f"Found unrecognized PHQ-9 item values: {invalid_values}")

    missing_items = scored.isna().sum(axis=1)
    if int(missing_items.sum()) > 0:
        rows = df.loc[missing_items > 0, ["uid", "type"]].to_dict("records")
        raise ValueError(f"Found missing PHQ-9 item values in rows: {rows}")

    return scored


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


def build_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    item_columns = get_item_columns(raw)
    scored_items = score_items(raw, item_columns)

    outcomes = raw[["uid", "type"]].copy()
    outcomes["uid"] = outcomes["uid"].astype("string").str.strip()
    outcomes["type"] = outcomes["type"].astype("string").str.strip()

    for index, column in enumerate(item_columns, start=1):
        outcomes[f"phq9_item_{index}_text"] = raw[column].astype("string")
        outcomes[f"phq9_item_{index}_score"] = scored_items[column].astype(int)

    outcomes["phq9_score"] = scored_items.sum(axis=1).astype(int)
    outcomes["phq9_class_5"] = outcomes["phq9_score"].apply(classify_5)
    outcomes["phq9_class_3"] = outcomes["phq9_score"].apply(classify_3)
    outcomes["phq9_binary"] = outcomes["phq9_score"].apply(classify_binary)

    type_order = {"pre": 0, "post": 1}
    outcomes["_type_order"] = outcomes["type"].map(type_order).fillna(99)
    outcomes = outcomes.sort_values(["uid", "_type_order", "type"]).reset_index(drop=True)
    return outcomes[OUTPUT_COLUMNS]


def build_distribution(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in ["phq9_class_5", "phq9_class_3", "phq9_binary"]:
        counts = outcomes[target].value_counts(dropna=False).sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "target": target,
                    "label": label,
                    "count": int(count),
                    "percentage": round(float(count / len(outcomes)), 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"PHQ-9 input file not found: {args.input}")

    raw = read_phq9(args.input)
    outcomes = build_outcomes(raw)
    distribution = build_distribution(outcomes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.distribution_output.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.output, index=False)
    distribution.to_csv(args.distribution_output, index=False)

    print("Saved:", args.output)
    print("Rows:", len(outcomes))
    print("Participants:", outcomes["uid"].nunique())
    type_values = outcomes["type"].dropna().unique().tolist()
    ordered_types = [value for value in ["pre", "post"] if value in type_values]
    ordered_types.extend(sorted(value for value in type_values if value not in ordered_types))
    print("Types:", ", ".join(ordered_types))
    print("Score range:", outcomes["phq9_score"].min(), "to", outcomes["phq9_score"].max())
    print("\nClass distribution:")
    print(distribution.to_string(index=False))
    print("\nDistribution saved:", args.distribution_output)


if __name__ == "__main__":
    main()
